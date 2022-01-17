import shapely
import math
import pandas as pd
import numpy as np 
import pathlib
from scipy import stats, signal
from shapely import geometry
import skspatial.objects
from ellipse import LsqEllipse
import circle_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Ellipse
import pingouin as pg

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def filter(df,z):
    z_score = np.abs(stats.zscore(df, nan_policy='omit'))
    return df[(z_score<z).all(axis=1)]

def rot_matrix_3d(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def rot_matrix_2d(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])

def simplify(trace):
    x,y = trace[:,0],trace[:,1]
    theta =  np.array(np.rad2deg(np.arctan2(y,x))%360)
    df =pd.DataFrame({'x':x,'y':y,'theta':theta})

    df.theta = df.theta.round()
    df.theta[df.theta==360] = 0
    df = df.sort_values(by=['theta'])
    df = df.groupby(['theta']).mean()
    return df.iloc[:,0:2].values

def savgol_filter(trace, l , p):
    trace_x = trace[:,0]
    trace_y = trace[:,1]
    trace_x = signal.savgol_filter(trace_x,window_length=l, polyorder=p)
    trace_y = signal.savgol_filter(trace_y,window_length=l, polyorder=p)
    return np.column_stack((trace_x,trace_y))

def in_ellipse(data,major,minor,angle):
    theta = np.linspace(0, np.pi*2, 360)

    r = major * minor  / np.sqrt((minor * np.cos(theta))**2 + (major * np.sin(theta))**2)
    xy = np.stack([r * np.cos(theta), r * np.sin(theta)], 1)

    ellipse = shapely.affinity.rotate(shapely.geometry.Polygon(xy), angle, 'center')
    x, y = ellipse.exterior.xy
    remaining_data = np.array([p for p in data if ellipse.contains(shapely.geometry.Point(p))])

    return np.array(remaining_data)

def in_circle(data,r,xc,yc):
    circle = shapely.geometry.Point(xc,yc).buffer(r)
    remaining_data = np.array([p for p in data if circle.contains(shapely.geometry.Point(p))])
    return remaining_data

p = pathlib.Path("./trace")
files = [x for x in  p.glob('**/*6*') if x.is_file()]

# 8 plots for 2 rows
fig, axs = plt.subplots(2,8,figsize=(45/2.54, 25/2.54))
plt.rcParams['figure.dpi'] = 300
i=0

# create empty to list to append dataframes onto
df_data_list = []
df_scales_list = []
specimen_list = []
for file in files:
    if 'L' in file.name:
        lr = 0
    elif 'R' in file.name:
        lr =1
    
    if 'L' in file.name:
        specimen ='Specimen '+str(i+1)+'-L'
        axs[lr][i].set_title(specimen,loc='left', fontsize=12)
    else:
        specimen='Specimen '+str(i+1)+'-R'
        axs[lr][i].set_title(specimen,loc='left', fontsize=12)
    
    # read data
    six_d = pd.read_csv(file, delimiter=",",skiprows=4,index_col=False)
    three_d = pd.read_csv(pathlib.Path(str(file).replace('_6d','_3d')), delimiter=",",skiprows=4,index_col=False)

    # distribute data
    rot_hum = six_d.iloc[:,14:23].values.reshape(-1,3)[0:3,]
    hum_marker = six_d.iloc[1:,23:26].values[0:1,]
    gt_marker = three_d.iloc[1:,28:31].values[0:1,]
    trace = six_d.iloc[:,10:13].values
    


    # filter outliers
    trace = filter(pd.DataFrame(trace),2).to_numpy()

    # rotate perpendicular to resection plane
    points = skspatial.objects.Points(trace)
    plane = skspatial.objects.Plane.best_fit(points)
    rot_plane = rot_matrix_3d(plane.normal,[0,0,1])
    trace = np.dot(rot_plane,points.T).T

    # account for odd specimen that has different reference frame
    if file.stem.endswith('048R_6d'): 
        gt_hum = three_d.iloc[:,28:31].values[1,:].reshape(1,3)
        
    else:
        #translate GT onto humeral csys
        gt_point = gt_marker - hum_marker
        # rotate GT onto humeral csys
        gt_hum = np.dot(rot_hum,gt_point.T).T

    # rotate again to account for plane fitting
    gt_resect = np.dot(rot_plane,gt_hum.T).T

    # remove z -dimension
    trace = trace[:,0:2]
    gt_resect = gt_resect[:,0:2]

 
    # fit ellipse to find center of trace
    el_reg1 = LsqEllipse().fit(trace)
    center, width, height, phi = el_reg1.as_parameters()

    # move trace data to the center of fitted ellipse
    trace = trace - center
    gt_resect = gt_resect - center

    # account for odd specimen that has different reference frame
    if file.stem.endswith('048R_6d'): 
        rot = rot_matrix_2d(np.deg2rad(127))
        gt_resect = np.dot(rot,gt_resect.T).T

   # rotate until GT faces "up"
    angle = math.atan2(gt_resect[:,1],gt_resect[:,0])
    rot2 = rot_matrix_2d(-angle+np.deg2rad(90))
    trace = np.dot(rot2,trace.T).T
    gt_resect = np.dot(rot2,gt_resect.T).T

    
    # filter trace data
    trace_raw = trace # store unaltered trace
    trace = simplify(trace)
    trace = trace[::4]
    trace = savgol_filter(trace,11,2)
    trace = 1.01*trace #measurement taken at median of cortex, sacle to outside

    # fit final ellipse on the rotated and centered data
    el_reg2 = LsqEllipse().fit(trace)
    center, width, height, phi = el_reg2.as_parameters()
    if height>width:
        major = height
        minor = width
    else:
        major = width
        minor = height
    # fit circle 
    xc,yc, radius, s = circle_fit.least_squares_circle(trace)
    
    # plot all
    # plot filtered trace
    axs[lr][i].scatter(trace[:,0],trace[:,1], marker='.',linewidth=0.8,zorder=1,color='k')
    # plot raw trace
    axs[lr][i].scatter(trace_raw[:,0],trace_raw[:,1], marker='.',linewidth=0.8,zorder=0,color='silver',alpha=0.5)
    # plot ellipse    
    ellipse_in = Ellipse(
        xy=center, width=2*width, height=2*height, angle=np.rad2deg(phi),
        edgecolor='r', fc='r', lw=2, label='Fit', alpha=0.2,zorder=4)
    axs[lr][i].add_patch(ellipse_in)
    ellipse = Ellipse(
        xy=center, width=2*width, height=2*height, angle=np.rad2deg(phi),
        edgecolor='r', fc='None', ls='-',lw=2, label='Fit', zorder=5)
    el_verts = ellipse.get_verts() # store points that consitute ellipse, to create shapely object from
    axs[lr][i].add_patch(ellipse)
    axs[lr][i].plot([center[0]], [center[1]], marker='.', mec='r', mew=1)
    
    # add major and minor axes of ellipse
    b = 4
    s = 0.6
    # if the width is larger than the height then must be rotated 90 clockwise
    if width>height:
        vy = rotate(center,[center[0],center[1]+b],(phi+np.deg2rad(270)))
        vx = rotate(center,vy,np.deg2rad(-90)) 
    else:
        vy = rotate(center,[center[0],center[1]+b],(phi+np.deg2rad(0)))
        vx = rotate(center,vy,np.deg2rad(270))  
    axs[lr][i].arrow(center[0], center[1], (vy[0]-center[0]), (vy[1]-center[1]), color='r', ls=':')
    axs[lr][i].arrow(center[0], center[1], s*(vx[0]-center[0]), s*(vx[1]-center[1]), color='r', ls=':')
    
    # plot circle
    circlein = plt.Circle((xc,yc), radius, color='b', alpha=0.2, zorder=2)
    circle = plt.Circle((xc,yc), radius, edgecolor='b', fc='None', ls='-',lw=2, zorder=3 )
    ci_verts = circle.get_verts() # store points that consitute circle, to create shapely object from
    axs[lr][i].plot([xc], [yc], marker='.', mec='b', mew=1)
    axs[lr][i].add_patch(circlein)
    axs[lr][i].add_patch(circle)
    
    # add greater tuboristy onto plot as a point
    axs[lr][i].plot(gt_resect[:,0], gt_resect[:,1], marker='X', c='g',zorder=6)

    # set title
    # axs[lr][i].set_title(file.stem,loc='left', fontsize=12)
    # set increment of tick size
    axs[lr,i].yaxis.set_major_locator(MultipleLocator(10))
    axs[lr,i].xaxis.set_major_locator(MultipleLocator(10))
    # ensure aspect ratio is equal
    axs[lr,i].set_aspect('equal')
    # add grid
    axs[lr][i].grid()

    # create shapely objects
    el = geometry.Polygon(el_verts)
    ci = geometry.Polygon(ci_verts)

    # fit ground truth N-dimensional polygon
    trace_av = trace
    #plot ngon
    # axs[lr][i].scatter(trace_av[:,0],trace_av[:,1],c='r', zorder = 7 )
    ngon = geometry.Polygon(np.squeeze(trace_av))
    # ngon = geometry.Polygon(trace_av)
    
    # determine coverage
    ngon_el = el.intersection(ngon)
    ngon_ci = ci.intersection(ngon)
    cov_el = ngon_el.area/el.area
    cov_ci = ngon_ci.area/ci.area
    
    # determine cortical coverage
    ngon_edge = geometry.Polygon(np.squeeze(0.99*trace_av)).difference(geometry.Polygon(np.squeeze(0.97*trace_av)))
    ngon_edge_el = el.intersection(ngon_edge)
    ngon_edge_ci = ci.intersection(ngon_edge)
    cort_cov_el = ngon_edge_el.area/ngon_edge.area
    cort_cov_ci = ngon_edge_ci.area/ngon_edge.area

    # determine overhang wrt to implant
    over_el = el.difference(ngon).area/el.area
    over_ci = ci.difference(ngon).area/ci.area
    
    fscale_df_list = []
    # determine scale at which 100% coverage occurs and resulting overhang
    for scale in np.linspace(0.8,1.2,100):

        s_el = shapely.affinity.scale(geom=el,xfact=scale,yfact=scale)
        s_ci = shapely.affinity.scale(geom=ci,xfact=scale,yfact=scale)
        
        # determine coverage
        i_ngon_el = s_el.intersection(ngon)
        i_ngon_ci = s_ci.intersection(ngon)
        i_cov_el = i_ngon_el.area/ngon.area
        i_cov_ci = i_ngon_ci.area/ngon.area

        # determine cortical coverage
        ngon_edge = geometry.Polygon(np.squeeze(0.99*trace_av)).difference(geometry.Polygon(np.squeeze(0.97*trace_av)))
        i_ngon_edge_el = s_el.intersection(ngon_edge)
        i_ngon_edge_ci = s_ci.intersection(ngon_edge)
        i_cort_cov_el = i_ngon_edge_el.area/ngon_edge.area
        i_cort_cov_ci = i_ngon_edge_ci.area/ngon_edge.area

        # determine overhang wrt to implant
        i_over_el = s_el.difference(ngon).area/s_el.area
        i_over_ci = s_ci.difference(ngon).area/s_ci.area 

        iscale = pd.DataFrame({
            'el coverage': i_cov_el,
            'ci coverage': i_cov_ci,
            'el overhang':i_over_el,
            'ci overhang':i_over_ci,
            'el cort coverage':i_cort_cov_el,
            'ci cort coverage':i_cort_cov_ci,
            
        },index=[scale])
        fscale_df_list.append(iscale) 
    df_scale = pd.concat(fscale_df_list)
    
    #find where coverage is 100% and the corresponding scale,overhang
    scale_max_cov_el = df_scale[['el coverage']].idxmax()
    max_cov_overhang_el = df_scale.loc[scale_max_cov_el][['el overhang']]
    scale_max_cov_ci = df_scale[['ci coverage']].idxmax()
    max_cov_overhang_ci = df_scale.loc[scale_max_cov_ci][['ci overhang']]

    #find where overhang is 1% and the corresponding scale,coverage,and cortical coverage
    scale_no_overhang_el = df_scale[['el overhang']].iloc[::-1].idxmin() #change sort direction with iloc
    cort_cov_no_overhang_el = df_scale.loc[scale_no_overhang_el][['el cort coverage']]
    cov_no_overhang_el = df_scale.loc[scale_no_overhang_el][['el coverage']]
    scale_no_overhang_ci = df_scale[['ci overhang']].iloc[::-1].idxmin()
    cort_cov_no_overhang_ci = df_scale.loc[scale_no_overhang_ci][['ci cort coverage']]
    cov_no_overhang_ci = df_scale.loc[scale_no_overhang_ci][['ci coverage']]
    print('circle no overhang')
    print(scale_no_overhang_ci)
    print(df_scale.loc[scale_no_overhang_ci])
    print('ellipse no overhang')
    print(scale_no_overhang_el)
    print(df_scale.loc[scale_no_overhang_el])


    # add title with info onto subplot
    angle_name = r'$\angle$'
    
    # axs[lr][i].set_xlabel(f'\nellipse coverage: {cov_el*100:.2f}%\nmajor: {major:.2f} mm\nminor: {minor:.2f} mm\n{angle_name}: {np.rad2deg(phi):.2f}\xb0\ncircle coverage: {cov_ci*100:.2f}%\nr: {radius:.2f} mm',loc='right', fontsize=7, alpha=0.8)
    axs[lr][i].set_xlabel(
        f'\
        ellipse\n\
        coverage: {cov_el*100:.2f}%\n\
        cortical coverage: {cort_cov_el*100:.2f}%\n\
        overhang: {over_el*100:.2f}%\n\
        major: {major:.2f} mm\n\
        minor: {minor:.2f} mm\n\
        {angle_name}: {np.rad2deg(phi):.2f}\xb0\n\n\
        circle\n\
        coverage: {cov_ci*100:.2f}%\n\
        cortical coverage: {cort_cov_ci*100:.2f}%\n\
        overhang: {over_ci*100:.2f}%\n\
        r: {radius:.2f} mm'
        ,loc='center', fontsize=7, alpha=0.8
    )


    # print useful info
    print('\n')
    print(specimen)
    print('rotated by:{:.2f}.'.format(np.rad2deg(angle)))
    # print('ngon_area: {:.2f}, el_area: {:.2f}, ci_area: {:.2f}'.format(ngon.area, el.area, ci.area))
    print('ellipse_coverage: {:.2f}, circle_coverage: {:.2f}'.format(cov_el,cov_ci))
    print('ellipse_cortical_coverage: {:.2f}, circle_cortical_coverage: {:.2f}'.format(cort_cov_el,cort_cov_ci))
    print('ellipse_overhang: {:.2f}, circle_overhang: {:.2f}'.format(over_el,over_ci))
    pdf = df_scale[df_scale.index>0.95]
    pdf = df_scale[df_scale.index<1.05]
    # print(pdf) 
    # store info
    
    idata = pd.DataFrame(
        {
            'file':file.stem,
            'el coverage': cov_el,
            'el cort coverage':cort_cov_el,
            'el overhang':over_el,
            'el fscaled':scale_max_cov_el.values[0],
            'el foverhang':max_cov_overhang_el.values[0][0],
            'el nscaled':scale_no_overhang_el.values[0],
            'el ncover':cov_no_overhang_el.values[0][0],
            'el ncortcover':cort_cov_no_overhang_el.values[0][0],
            'ci coverage':cov_ci,
            'ci cort coverage':cort_cov_ci,
            'ci overhang':over_ci,
            'ci fscaled':scale_max_cov_ci.values[0],
            'ci foverhang':max_cov_overhang_ci.values[0][0],            
            'ci nscaled':scale_no_overhang_ci.values[0],
            'ci ncover':cov_no_overhang_ci.values[0][0],            
            'ci ncortcover':cort_cov_no_overhang_ci.values[0][0],            
            'el major':major,
            'el minor':minor,
            'el phi':np.rad2deg(phi),
            'el center x':center[0],
            'el center y':center[1],
            'ci radius':radius,
            'ci center x':xc,
            'ci center y':yc
        }, index=[specimen]
    )
    df_scales_list.append(df_scale)
    df_data_list.append(idata)
    specimen_list.append(specimen)
    if 'R' in file.name:
        i += 1

# find max convergance

# # print dataframe
print('\nfull data')
df = pd.concat(df_data_list)
df.to_csv('data.csv')
print(df)

print('\nbasic geometry stats')
print(df[['el major','ci radius','el minor','ci radius','el phi']])
print(df[['el major','ci radius','el minor','ci radius','el phi']].describe().round(3).to_latex())
print(df[['el major','ci radius','el minor','ci radius','el phi']].diff(axis=1).describe().round(3))#.to_latex())
# print(pg.ttest(df[['el foverhang']].values.flatten(),df[['ci foverhang']].values.flatten()))



print('\nmax')
print(df[['el fscaled','ci fscaled','el foverhang','ci foverhang']])
print(df[['el fscaled','ci fscaled','el foverhang','ci foverhang']].describe())#.multiply(100))#.to_latex())
print(df[['el fscaled','ci fscaled','el foverhang','ci foverhang']].diff(axis=1).describe().round(3).multiply(100))#.to_latex())
print(pg.ttest(df[['el foverhang']].values.flatten(),df[['ci foverhang']].values.flatten()))

print('\nmin')
print(df[['el nscaled','ci nscaled','el ncortcover','ci ncortcover','el ncover','ci ncover']])
print(df[['el nscaled','ci nscaled','el ncover','ci ncover','el ncortcover','ci ncortcover']].describe().round(3).multiply(100))#.to_latex())
print(df[['el nscaled','ci nscaled','el ncover','ci ncover','el ncortcover','ci ncortcover']].diff(axis=1).describe().round(3).multiply(100))#.to_latex())
print(pg.ttest(df[['el ncortcover']].values.flatten(),df[['ci ncortcover']].values.flatten()))
print(pg.ttest(df[['el ncover']].values.flatten(),df[['ci ncover']].values.flatten()))

# do stats
print('\nother stats')
basic_stats = df[['el coverage','ci coverage','el overhang','ci overhang','el cort coverage','ci cort coverage']].describe()
print(basic_stats.round(3))

# independent samples t-tests to determine if the groups have a statistically significant difference
ind_coverage = pg.ttest(df[['el coverage']].values.flatten(),df[['ci coverage']].values.flatten())
ind_overhang = pg.ttest(df[['el overhang']].values.flatten(),df[['ci overhang']].values.flatten())
ind_cort_coverage = pg.ttest(df[['el cort coverage']].values.flatten(),df[['ci cort coverage']].values.flatten())

df_stat = pd.concat([ind_coverage,ind_overhang,ind_cort_coverage],keys=['coverage','overhang','cortical coverage'])
print(df_stat.round(3))

plt.show()
# print(df_scales_list)
data = pd.concat(df_scales_list)
datas = data.groupby(data.index).mean()
datastd = data.groupby(data.index).std()
datas = datas[datas.index<1.1]
datastd = datastd[datastd.index<1.1]
datas = datas[datas.index>0.9]
datastd = datastd[datastd.index>0.9]

print(datas)
print(datastd)

plt.plot(datas.index, datas[['el coverage']],color='red')
plt.fill_between(datas.index, (datas[['el coverage']] - datastd[['el coverage']]).values.flatten(), (datas[['el coverage']] + datastd[['el coverage']]).values.flatten(), alpha=0.3, color='red',label='ellipse')
plt.plot(datas.index, datas[['ci coverage']], color='blue')
plt.fill_between(datas.index, (datas[['ci coverage']] - datastd[['ci coverage']]).values.flatten(), (datas[['ci coverage']] + datastd[['ci coverage']]).values.flatten(), alpha=0.3, color='blue', label='circle')
plt.axvline(x=0.949, color='red', linestyle='dashed',alpha=0.5)
plt.axvline(x=0.910, color='blue', linestyle='dashed',alpha=0.5)
plt.ylabel('Coverage')
plt.yticks([0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975,1.0],['80%','82.5%','85%','87.5%','90%','92.5%','95%','97.5%','100%'])
plt.xticks([0.9,0.95,1.0,1.05,1.1],['-10%','-5%','0%','5%','10%'])
plt.xlabel(r'$\longleftarrow Downsizing$''  |  'r'$Upsizing \longrightarrow$''     ')
plt.legend()
plt.tight_layout(pad=0.1)
# plt.savefig('coverage.png')
plt.show()

plt.plot(datas.index, datas[['el overhang']],color='red')
plt.fill_between(datas.index, (datas[['el overhang']] - datastd[['el overhang']]).values.flatten(), (datas[['el overhang']] + datastd[['el overhang']]).values.flatten(), alpha=0.3, color='red',label='ellipse')
plt.plot(datas.index, datas[['ci overhang']], color='blue')
plt.fill_between(datas.index, (datas[['ci overhang']] - datastd[['ci overhang']]).values.flatten(), (datas[['ci overhang']] + datastd[['ci overhang']]).values.flatten(), alpha=0.3, color='blue', label='circle')
plt.axvline(x=1.053, color='red', linestyle='dashed',alpha=0.5)
plt.axvline(x=1.088, color='blue', linestyle='dashed',alpha=0.5)
plt.ylabel('Overhang')
plt.yticks([0,0.025,0.05,0.075,0.10,0.125,0.150,0.175],['0%','2.5%','5%','7.5%','10%','12.5%','15%','17.5%'])
plt.xticks([0.9,0.95,1.0,1.05,1.1],['-10%','-5%','0%','5%','10%'])
plt.xlabel(r'$\longleftarrow Downsizing$''  |  'r'$Upsizing \longrightarrow$''     ')
plt.legend()
plt.tight_layout(pad=0.1)
# plt.savefig('overhang.png')
plt.show()

plt.plot(datas.index, datas[['el cort coverage']],color='red')
plt.fill_between(datas.index, (datas[['el cort coverage']] - datastd[['el cort coverage']]).values.flatten(), (datas[['el cort coverage']] + datastd[['el cort coverage']]).values.flatten(), alpha=0.3, color='red',label='ellipse')
plt.plot(datas.index, datas[['ci cort coverage']], color='blue')
plt.fill_between(datas.index, (datas[['ci cort coverage']] - datastd[['ci cort coverage']]).values.flatten(), (datas[['ci cort coverage']] + datastd[['ci cort coverage']]).values.flatten(), alpha=0.3, color='blue', label='circle')
plt.axvline(x=0.949, color='red', linestyle='dashed',alpha=0.5)
plt.axvline(x=0.910, color='blue', linestyle='dashed',alpha=0.5)
plt.ylabel('Cortical Coverage')
plt.yticks([0,0.2,0.4,0.6,0.8,1.0],['0%','20%','40%','60%','80%','100%'])
plt.xticks([0.9,0.95,1.0,1.05,1.1],['-10%','-5%','0%','5%','10%'])
plt.xlabel(r'$\longleftarrow Downsizing$''  |  'r'$Upsizing \longrightarrow$''     ')
plt.legend()
plt.tight_layout(pad=0.1)
# plt.savefig('cortcoverage.png')
plt.show()
# datas.plot()

# show plot

plt.show()