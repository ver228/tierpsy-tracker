# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 16:08:07 2015

@author: ajaver
"""
import tables
import itertools
import numpy as np
import matplotlib.pylab as plt
import time

class plate_worms(tables.IsDescription):
#class for the pytables 
    worm_index = tables.Int32Col(pos=0)
    frame_number = tables.Int32Col(pos=1)
    #label_image = tables.Int32Col(pos=2)
    coord_x = tables.Float32Col(pos=2)
    coord_y = tables.Float32Col(pos=3) 
    area = tables.Float32Col(pos=4) 
    perimeter = tables.Float32Col(pos=5) 
    major_axis = tables.Float32Col(pos=6) 
    minor_axis = tables.Float32Col(pos=7) 
    eccentricity = tables.Float32Col(pos=8) 
    compactness = tables.Float32Col(pos=9) 
    orientation = tables.Float32Col(pos=10) 
    solidity = tables.Float32Col(pos=11) 
    intensity_mean = tables.Float32Col(pos=12)
    intensity_std = tables.Float32Col(pos=13)
    speed = tables.Float32Col(pos=14)

#featuresFile = '/Users/ajaver/Desktop/Gecko_compressed/Features_CaptureTest_90pc_Ch2_18022015_230213.hdf5';

featuresFile = '/Users/ajaver/Desktop/Gecko_compressed/bFeatures_CaptureTest_90pc_Ch4_16022015_174636.hdf5';


feature_fid = tables.open_file(featuresFile, mode = 'r')
feature_table = feature_fid.get_node('/plate_worms')
track_size = np.bincount(feature_table.cols.worm_index)

indexes = np.arange(track_size.size);
indexes = indexes[track_size>=50]

last_frames = [];
first_frames = [];
for ii in indexes:
    min_frame = 1e32;
    max_frame = 0;
    
    for dd in feature_table.where('worm_index == %i'% ii):
        if dd['frame_number'] < min_frame:
            min_frame = dd['frame_number']
            min_row = (dd['worm_index'], dd['frame_number'], dd['coord_x'], dd['coord_y'], dd['area'], dd['minor_axis'])
        
        if dd['frame_number'] > max_frame:
            max_frame = dd['frame_number']
            max_row = (dd['worm_index'], dd['frame_number'], dd['coord_x'], dd['coord_y'], dd['area'], dd['minor_axis'])
    last_frames.append(max_row)
    first_frames.append(min_row)
feature_fid.close()

last_frames = np.array(last_frames)
first_frames = np.array(first_frames)

plt.figure()
plt.plot(first_frames[:,2], first_frames[:,3], '.b')
plt.plot(last_frames[:,2], last_frames[:,3], '.r')

#%%
#def make_link(G, node1, node2):
#    if node1 not in G:
#        G[node1] = {}
#    (G[node1])[node2] = 1
#    if node2 not in G:
#        G[node2] = {}
#    (G[node2])[node1] = 1
#    return G
#
#def make_graph(link_list):
#    G = dict()
#    for n1, n2 in link_list:
#        make_link(G, n1, n2)
#    return G

AREA_RATIO_LIM = (0.67, 1.5);
#MAX_DIST_GAP = 50;
MAX_DELTA_TIME = 25;



join_frames = [];
for kk in range(last_frames.shape[0]):
    
    row2check =  last_frames[kk,:]
    last_frame = row2check[1]
    
    possible_rows = first_frames[np.bitwise_and(first_frames[:,1] > last_frame, first_frames[:,1] < last_frame+MAX_DELTA_TIME),:]
    if possible_rows.size > 0:
        areaR = row2check[4]/possible_rows[:,4];
        
        good = np.bitwise_and(areaR>AREA_RATIO_LIM[0], areaR<AREA_RATIO_LIM[1])
        possible_rows = possible_rows[good,:]
        
        R = np.sqrt( (possible_rows[:,2]  - row2check[2]) ** 2 + (possible_rows[:,3]  - row2check[3]) ** 2)
        if R.shape[0] == 0:
            continue
        
        indmin = np.argmin(R)
        if R[indmin] <= row2check[5]: #only join trajectories that move at most one worm body
            join_frames.append((int(possible_rows[indmin, 0]),int(row2check[0])))
            #join_frames[int(last_row[1])] =  int(possible_rows[indmin, 1]

relations_dict = dict(join_frames)

feature_fid = tables.open_file(featuresFile, mode = 'r+')
feature_table = feature_fid.get_node('/plate_worms')


descr = feature_table.description._v_colObjects;
try:
    joined_table = feature_fid.create_table('/', "plate_worms_joined", descr,'')
except:
    feature_fid.removeNode('/', "plate_worms_joined")
    joined_table = feature_fid.create_table('/', "plate_worms_joined", descr,'')

#row_new = joinedTable.row
for ii in indexes:
    
    ind = ii;
    while ind in relations_dict:
        ind = relations_dict[ind]
    
    tot = 0
    all_dd = []
    for row in feature_table.where('worm_index == %i'% ii):
        tot += 1
        dd = list(row[:])
        dd[0] = ind
        all_dd += [tuple(dd)]
    joined_table.append(all_dd);
    if ii != ind:
        print ii, ind, tot
joined_table.flush()

feature_fid.close()
    
#%%

feature_fid = tables.open_file(featuresFile, mode = 'r')
joined_table = feature_fid.get_node('/plate_worms_joined')
#plt.figure()
#plt.imshow(image, interpolation = 'none', cmap = 'gray')
track_size2 = np.bincount(joined_table.cols.worm_index)
indexes2 = np.argsort(track_size2)[::-1]

#from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#plt.figure()
for ii in indexes2[0:20]:
    coord = [(row['coord_x'], row['coord_y'], row['frame_number']) for row in joined_table.where('worm_index == %i'% ii)]
    coord = np.array(coord).T
    plt.plot(coord[0,:], coord[1,:], '.-')
    #ax.plot(coord[0,:], coord[1,:], coord[2,:], '.-')
feature_fid.close()
#%%
#import numpy
#from mayavi import mlab
#
#def test_points3d():
#    t = numpy.linspace(0, 4 * numpy.pi, 20)
#    cos = numpy.cos
#    sin = numpy.sin
#
#    x = sin(2 * t)
#    y = cos(t)
#    z = cos(2 * t)
#    s = 2 + sin(t)
#
#    return mlab.points3d(x, y, z, s, colormap="copper", scale_factor=.25)
# Create the data.
#from numpy import pi, sin, cos, mgrid
#dphi, dtheta = pi/250.0, pi/250.0
#[phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
#m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
#r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
#x = r*sin(phi)*cos(theta)
#y = r*cos(phi)
#z = r*sin(phi)*sin(theta)
#
## View it.
#from mayavi import mlab
#s = mlab.mesh(x,y,z)
#mlab.show()
#
#        
#plt.figure()
#for ii in indexes:
#    if track_size[ii] >= 50000:
#        data = [(dd['frame_number'], dd['intensity_mean']) for dd in feature_table.where('worm_index == %i'% ii)]
#        data = np.array(data).T
#        plt.plot(data[0,:], data[1,:])
#
#plt.figure()
#for ii in indexes:
#    if track_size[ii] >= 50000:
#        data = [(dd['frame_number'], dd['area']) for dd in feature_table.where('worm_index == %i'% ii)]
#        data = np.array(data).T
#        plt.plot(data[0,:], data[1,:])

#plt.figure()
#for ii in indexes:
#    if track_size[ii] >= 50000:
#        data = [(dd['frame_number'], dd['intensity_mean']) for dd in feature_table.where('worm_index == %i'% ii)]
#        data = np.array(data).T
#        [bins, edges] = np.histogram(data[1,:],100);
#        plt.plot(edges[:-1], bins/float(data.shape[1]))
#
#feature_fid.close()

#%%
#max_frame = feature_table.cols.frame_number[-1]

#stats_str = ['median', 'mean', 'N', 'std', 'max', 'min'];

#area_stats = {}
#for stat in stats_str:
#    area_stats[stat] = np.zeros((max_frame+1, 2));

#tic = time.time();
#
#def frame_selector(row):
#    return row['frame_number']
#for frame_number, rows in itertools.groupby(feature_table, frame_selector):
#    if frame_number % 1000 == 0:
#        toc = time.time() 
#        print frame_number, toc-tic
#        tic = toc
#        
#    dat_list =[r['area'] for r in rows];
#    
#    for ii in range(2):
#        if ii == 1:
#            dat = np.array([x for x in dat_list if x>40])
#        else:
#            dat =  np.array(dat_list);
#        area_stats['N'][frame_number,ii] = len(dat)
#        area_stats['median'][frame_number,ii] = np.median(dat)
#        area_stats['mean'][frame_number,ii] = np.mean(dat)
#        area_stats['max'][frame_number,ii] = np.max(dat)
#        area_stats['min'][frame_number,ii] = np.min(dat)
#        area_stats['std'][frame_number,ii] = np.std(dat)
#
#for key in area_stats.keys():
#    fig = plt.figure()
#    fig.suptitle(key)
#    plt.plot(area_stats[key][:,0])
#
#for key in area_stats.keys():
#    fig = plt.figure()
#    fig.suptitle(key)
#    plt.plot(area_stats[key][:,1])
#    
#    
