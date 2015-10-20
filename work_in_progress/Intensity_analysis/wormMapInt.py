# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:42:09 2015

@author: ajaver
"""

import pandas as pd
import numpy as np
import tables
import os
import shutil
import matplotlib.pylab as plt

base_name = 'Capture_Ch3_12052015_194303'
mask_dir = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/20150512/'
results_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150512/'    


masked_image_file = mask_dir + base_name + '.hdf5'
trajectories_file = results_dir + base_name + '_trajectories.hdf5'
skeletons_file = results_dir + base_name + '_skeletons.hdf5'
intensities_file = results_dir + base_name + '_intensities.hdf5'
save_dir_int = (os.sep).join([results_dir, base_name+'_int_maps']) + os.sep

if os.path.exists(save_dir_int):
    shutil.rmtree(save_dir_int)
os.makedirs(save_dir_int)

#(rows_indexes['max'] - rows_indexes['min']).sort()

with pd.HDFStore(skeletons_file, 'r') as ske_file_id:
    #data to extract the ROI
    trajectories_df = ske_file_id['/trajectories_data']

    #get the first and last frame of each worm_index
    indexes_data = trajectories_df[['worm_index_joined', 'skeleton_id']]
    rows_indexes = indexes_data.groupby('worm_index_joined').agg([min, max])['skeleton_id']
    del indexes_data
    
with tables.File(skeletons_file, 'r') as ske_file_id, \
     tables.File(intensities_file, "r") as int_file_id:
         worm_int_tab = int_file_id.get_node('/straighten_worm_intensity')
         
         row_range = rows_indexes.loc[209]
         mapI = worm_int_tab[row_range.loc['min']:row_range.loc['max']+1]

mapI =  mapI.transpose((0,2,1)).copy();


import tifffile
tifffile.imsave('test.tif', mapI.astype(np.uint8))

mapI_smooth = mapI.copy()
smooth_lim = 12
for frame in range(mapI.shape[0]):
    top = frame+smooth_lim
    
    bot = frame-smooth_lim
    if top>=mapI.shape[0]: 
        top = mapI.shape[0]-1;
        bot = mapI.shape[0]-smooth_lim*2
    if bot<0:
        bot = 0
        top = smooth_lim*2
        
    mapI_smooth[frame] = np.nanmean(mapI[bot:top+1], axis=0)
tifffile.imsave('test_smooth2.tif', mapI_smooth.astype(np.uint8))    

 


#%%
#plt.imshow(aa)