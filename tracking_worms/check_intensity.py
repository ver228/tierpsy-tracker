# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 22:15:48 2015

@author: ajaver
"""

import pandas as pd
import numpy as np
import tables
import matplotlib.pylab as plt

import sys
sys.path.append('../tracking_worms/')
from getSkeletons import getWormROI

import sys
sys.path.append('../videoCompression/')
from parallelProcHelper import timeCounterStr

sys.path.append('../segworm_python/')
from main_segworm import getStraightenWormInt



base_name = 'Capture_Ch3_12052015_194303'
root_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150512/'    

#root_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150511/'
#base_name = 'Capture_Ch1_11052015_195105'

masked_image_file = root_dir + 'Compressed/' + base_name + '.hdf5'
trajectories_file = root_dir + 'Trajectories/' + base_name + '_trajectories.hdf5'
skeletons_file = root_dir + 'Trajectories/' + base_name + '_skeletons.hdf5'
intensities_file = root_dir + 'Trajectories/' + base_name + '_intensities.hdf5'

#MAKE VIDEOS
roi_size = 128
width_resampling = 13
length_resampling = 121


with pd.HDFStore(skeletons_file, 'r') as ske_file_id:
    #data to extract the ROI
    trajectories_df = ske_file_id['/trajectories_data']

    #get the first and last frame of each worm_index
    indexes_data = trajectories_df[['worm_index_joined', 'skeleton_id']]
    rows_indexes = indexes_data.groupby('worm_index_joined').agg([min, max])['skeleton_id']
    del indexes_data

#def getIntensitiesMap(masked_image_file, skeletons_file, intensities_file, roi_size = 128):
with tables.File(masked_image_file, 'r')  as mask_fid, \
     tables.File(skeletons_file, 'r') as ske_file_id, \
     tables.File(intensities_file, "w") as int_file_id:
    
    #pointer to the compressed videos
    mask_dataset = mask_fid.get_node("/mask")
    
    #obtain a fix length for the worm half width
    half_widths = {}
    mid_cnt = ske_file_id.get_node('/contour_width').shape[1]/2
    for worm_index, row_range in rows_indexes.iterrows():
        contour_width = ske_file_id.get_node('/contour_width')[row_range['min']:row_range['max']+1, :]
        half_widths[worm_index] = np.nanmedian(contour_width[:, mid_cnt])/2 + 0.5
    
    #pointer to skeletons
    skel_tab = ske_file_id.get_node('/skeleton')
    
    #get first and last frame for each worm
    worms_frame_range = trajectories_df.groupby('worm_index_joined').agg({'frame_number': [min, max]})['frame_number']
    tot_rows = len(trajectories_df)
    
    #create array to save the intensities
    filters = tables.Filters(complevel=5, complib='zlib', shuffle=True)
    worm_int_tab = int_file_id.create_carray("/", "straighten_worm_intensity", \
                               tables.Float16Atom(dflt=np.nan), \
                               (tot_rows, length_resampling, width_resampling), \
                                chunkshape = (1, length_resampling, width_resampling),\
                                filters = filters);
    
    progressTime = timeCounterStr('Obtaining intensity maps.');
    for frame, frame_data in trajectories_df.groupby('frame_number'):
        img = mask_dataset[frame,:,:]
        for segworm_id, row_data in frame_data.iterrows():
            worm_index = row_data['worm_index_joined']
            worm_img, roi_corner = getWormROI(img, row_data['coord_x'], row_data['coord_y'], roi_size)
            
            skeleton = skel_tab[segworm_id,:,:]-roi_corner
            
            if not np.any(np.isnan(skeleton)): 
                straighten_worm = getStraightenWormInt(worm_img, skeleton, half_width = half_widths[worm_index], \
                width_resampling = width_resampling, length_resampling = length_resampling)
                worm_int_tab[segworm_id, :, :]  = straighten_worm.T
                
        if frame % 500 == 0:
            progress_str = progressTime.getStr(frame)
            print(base_name + ' ' + progress_str);   
    
    mask_fid.close()
    int_file_id.close()
#%%

 
#plt.plot(grid_x, grid_y, '.')

#plt.figure()
#plt.imshow(straighten_img, cmap='gray', interpolation= 'none')    

#plt.figure()
#tt = worm_img[np.round(grid_y).astype(np.int), np.round(grid_x).astype(np.int)]
#plt.imshow(tt, cmap='gray', interpolation= 'none')    

#%%


    #if frame % 500 == 0:
    #    progress_str = progressTime.getStr(frame)
    #    print(base_name + ' ' + progress_str);

#for kk in [41, 42]:
#    plt.figure()
#    for frame in range(kk*1000, (kk+1)*1000, 100):
#        xx = np.nanmean(np.nanmean(worm_intensity[frame:(frame+ 1),:,2:5], axis=0),axis=1)
#        plt.plot(xx)
#
#mid_intensity = np.mean(worm_intensity[:,:,2:5], axis=2);
##%%
#
#for kk in range(0,51, 5):
#    plt.figure()
#    plt.plot(median_filter(mid_intensity[:, kk],1000))



#%%
#import tifffile
#N = 25.
#windows = np.ones(N)/N
#worm_intensity2 = np.zeros((worm_intensity.shape[0], worm_intensity.shape[2], worm_intensity.shape[1]))
#for ii in range(worm_intensity.shape[1]):
#    for jj in range(worm_intensity.shape[2]):
#        worm_intensity2[:,jj,ii] = np.convolve(worm_intensity[:,ii,jj], windows, 'same')
#
#tifffile.imsave('/Users/ajaver/Desktop/Gecko_compressed/image.tiff', worm_intensity2.astype(np.uint8), compress=4) 