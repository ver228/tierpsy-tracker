# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:03:15 2015

@author: ajaver
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:39:41 2015

@author: ajaver
"""
import pandas as pd
#import matplotlib.pylab as plt
import numpy as np
import h5py
import cv2
import time
import tables
import operator
import matplotlib.pylab as plt

import sys, os
sys.path.append('../videoCompression/')
from parallelProcHelper import timeCounterStr


from scipy.spatial.distance import cdist
from scipy.interpolate import RectBivariateSpline
#from calContrastMaps import calContrastMapsBinned
#from image_difference import image_difference



def angleWindow(x, y, windowSize):
    #given a series of x and y coordinates over time, calculates the angle
    #between each tangent vector over a given window making up the skeleton
    #and the x-axis.
    #arrays to build up and export
    dX = x[:-windowSize] - x[windowSize:];
    dY = y[:-windowSize] - y[windowSize:];
    
    #calculate angles
    skelAngles = np.arctan2(dY, dX)
    #%repeat final angle to make array the same length as skelX and skelY
    skelAngles = np.lib.pad(skelAngles, (angleSmoothSize/2, angleSmoothSize/2), 'edge')
    return skelAngles;

if __name__  == "__main__":
#    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/CaptureTest_90pc_Ch1_02022015_141431.hdf5';
#    trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_trajectories.hdf5';
#    segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_segworm.hdf5';
#    contrastmap_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_lmap.hdf5';

#    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Compressed/Capture_Ch3_11052015_195105.hdf5';
#    trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Trajectories/Capture_Ch3_11052015_195105_trajectories.hdf5';
#    segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Trajectories/Capture_Ch3_11052015_195105_segworm.hdf5';
#    contrastmap_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Trajectories/Capture_Ch3_11052015_195105_lmap.hdf5';

    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/20150512/Capture_Ch3_12052015_194303.hdf5';
    trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150512/Trajectories/Capture_Ch3_12052015_194303_trajectories.hdf5';
    segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150512/Trajectories/Capture_Ch3_12052015_194303_segworm.hdf5';
    contrastmap_file = '/Users/ajaver/Desktop/Gecko_compressed/20150512/Trajectories/Capture_Ch3_12052015_194303_lmap.hdf5';

    base_name = os.path.splitext(os.path.split(masked_image_file)[-1])[0]
    
    MAX_DELT = 1;
    ROI_SIZE = 128;
    
    WIDTH_RESAMPLING = 7
    
    angleSmoothSize = 6;
    if angleSmoothSize%2 == 1:
        angleSmoothSize += 1; 
    
    
    table_fid = pd.HDFStore(trajectories_file, 'r');
    df = table_fid['/plate_worms'];
    df =  df[df['worm_index_joined'] > 0]# and ('frame_index' > 147500)]
    #df =  df[df['worm_index_joined'] == 3]
    #df = df[df['segworm_id']>=0]; #select only rows with a valid segworm skeleton
    df = df.query('segworm_id>=0')# and frame_number>147500')
    table_fid.close()
    
    #track_counts = df['worm_index_joined'].value_counts()    
    #tot_ind_ini = 0;
    #tot_ind_last = 0;
    
    all_ind_block = [];
    
    progressTime = timeCounterStr('Writing individual worm videos.');
    for worm_ind in df['worm_index_joined'].unique():
        #select data from the same worm (trajectory)
        worm = df[(df['worm_index_joined']==worm_ind)].sort(columns='frame_number')
        
        #identify blocks as segments where the skeletons are separated by more than MAX_DELT frames    
        delTs = np.diff(worm['frame_number']); 
        block_ind = np.zeros(len(worm), dtype = np.int);    
        block_ind[0] = 1;
        for ii, delT in enumerate(delTs):
            if delT <= MAX_DELT:
                block_ind[ii+1] = block_ind[ii];
            else:
                block_ind[ii+1] = block_ind[ii]+1;
        
        #construct index for the blocks
        worm_segworm_id = worm['segworm_id'].values  
        plate_worms_id = worm.index.values;
            
        #identify segments that correspond to the begining or ending of a block        
        all_ind_block += zip(*[block_ind.size*[worm_ind], list(block_ind), list(plate_worms_id), list(worm_segworm_id)])
    
    all_ind_block = sorted(all_ind_block, key = operator.itemgetter(0, 1)); #sort by worm index first, and the by block index
    all_ind_block = zip(*all_ind_block)
    all_ind_block += [range(len(all_ind_block[0]))]
    all_ind_block = zip(*all_ind_block)
    #%%
    contrastmap_fid = tables.File(contrastmap_file, 'w');
    
    ind_block_table = contrastmap_fid.create_table('/', 'block_index', {
        'worm_index_joined':    tables.Int32Col(pos=0),
        'block_id'         : tables.Int32Col(pos=1),
        'plate_worms_id'   : tables.Int32Col(pos=2),
        'segworm_id'         : tables.Int32Col(pos=3),
        'lmap_id'     : tables.Int32Col(pos=4),
        }, filters = tables.Filters(complevel=5, complib='zlib', shuffle=True))
    
    ind_block_table.append(all_ind_block);
    
    contrastmap_fid.flush()
    contrastmap_fid.close()
    del(all_ind_block)
    #%%
    #read the valid contrastmap data (again)
    contrastmap_fid = pd.HDFStore(contrastmap_file, 'r');
    df_map_ind = contrastmap_fid['/block_index']
    contrastmap_fid.close()
    
    #read the trajectories data and add a column for the cmap index
    trajectories_fid = pd.HDFStore(trajectories_file, 'r');
    df = trajectories_fid['/plate_worms'];
    df = df.loc[df_map_ind['plate_worms_id'], ['frame_number', 'worm_index_joined', 'coord_x', 'coord_y', 'segworm_id']];
    df.loc[:, 'lmap_id'] = pd.Series(df_map_ind['lmap_id'].values, df_map_ind['plate_worms_id'].values)
    assert(len(df) == len(df_map_ind))
    #%%
    #open the hdf5 with the masked images
    mask_fid = h5py.File(masked_image_file, 'r');
    mask_dataset = mask_fid["/mask"]
    
    segworm_fid = h5py.File(segworm_file, 'r')
    
    tot_maps = len(df_map_ind)
    nsegments = segworm_fid['/segworm_results/skeleton'].shape[-1];
    
    contrastmap_fid = h5py.File(contrastmap_file, 'r+');
    #%%
    
    lmaps_data = contrastmap_fid.create_dataset("/block_lmap" , (tot_maps, WIDTH_RESAMPLING, nsegments), \
                    'float', maxshape = (None, WIDTH_RESAMPLING, nsegments), \
                    chunks = (1, WIDTH_RESAMPLING, nsegments),
                    compression="gzip", 
                                    compression_opts=4,
                                    shuffle=True);
    #%%
    tic = time.time()
    tic_first = tic
    
    for frame, wormsInFrame in df.groupby('frame_number'):
        img = mask_dataset[frame,:,:]
        for ii, worm in wormsInFrame.iterrows():
            worm_index = int(worm['worm_index_joined']); 
            lmap_id = worm['lmap_id']
            segworm_id = worm['segworm_id']
    
            range_x = np.round(worm['coord_x']) + [-ROI_SIZE/2, ROI_SIZE/2]
            range_y = np.round(worm['coord_y']) + [-ROI_SIZE/2, ROI_SIZE/2]
            
            if (range_y[0] <0) or (range_y[0]>= img.shape[0]) or \
            (range_x[0] <0) or (range_x[0]>= img.shape[1]):
                continue
            
            worm_img =  img[range_y[0]:range_y[1], range_x[0]:range_x[1]]
            worm_width = np.squeeze(segworm_fid['/segworm_results/width'][segworm_id,:,:]);
            
            skeleton = segworm_fid['/segworm_results/skeleton'][segworm_id,:,:]
            skelX = skeleton[1,:]-range_x[0]
            skelY = skeleton[0,:]-range_y[0]
  
            skelAngles = angleWindow(skelX, skelY, angleSmoothSize)
            #%get the perpendicular angles to define line scans (orientation doesn't
            #%matter here so subtracting pi/2 should always work)
            perpAngles = skelAngles - np.pi/2;
            
            #%for each skeleton point get the coordinates for two line scans: one in the
            #%positive direction along perpAngles and one in the negative direction (use
            #%two that both start on skeleton so that the intensities are the same in
            #%the line scan)
            
            #resample the points along the worm width
            halfWidth = (np.median(worm_width[10:-10])/2.) + 0.5 #add half a pixel to get part of the contour
            r_ind = np.linspace(-halfWidth,halfWidth, WIDTH_RESAMPLING)
            
            #create the grid of points to be interpolated (make use of numpy implicit broadcasting Nx1 + 1xM = NxM)
            endPointsX = skelX + r_ind[:, np.newaxis]*np.cos(perpAngles);
            endPointsY = skelY + r_ind[:, np.newaxis]*np.sin(perpAngles);
            
            
            f = RectBivariateSpline(np.arange(worm_img.shape[0]), np.arange(worm_img.shape[1]), worm_img)
            straightWorm = f.ev(endPointsY, endPointsX)
            
            lmaps_data[lmap_id,:,:] = straightWorm
#    ##    
        contrastmap_fid.flush()
        if frame % 500 == 0:
            progress_str = progressTime.getStr(frame)
            print(base_name + ' ' + progress_str);
        
    mask_fid.close()
    contrastmap_fid.close()
#%%
    
#plt.figure()
#plt.imshow(straightWorm, cmap = 'gray', interpolation = 'none'); 
#
#plt.figure()
#plt.imshow(worm_img, cmap = 'gray', interpolation = 'none'); 
#plt.plot(dat['contour_ventral'][1,:],dat['contour_ventral'][0,:], '.') 
#plt.plot(dat['contour_dorsal'][1,:],dat['contour_dorsal'][0,:], '.') 
#plt.plot(dat['skeleton'][1,:],dat['skeleton'][0,:], '.')
#plt.plot(endPointsX,endPointsY, '.k')



#%%
