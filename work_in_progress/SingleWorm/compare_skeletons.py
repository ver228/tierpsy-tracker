# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:00:00 2015

@author: ajaver
"""

import os
import h5py 
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

from scipy.io import loadmat

masked_file = '/Users/ajaver/Desktop/Videos/Check_Align_samples/MaskedVideos/npr-13 (tm1504)V on food L_2010_01_25__11_56_02___4___2.hdf5' 
#masked_file = '/Users/ajaver/Desktop/Videos/Check_Align_samples/MaskedVideos/npr-9 (tm1652) on food R_2010_01_25__16_24_16___4___13.hdf5' 

mask_dir, _, base_file =  masked_file.rpartition(os.sep)

#read segworm results
segworm_file = mask_dir.replace(os.sep + 'MaskedVideos', os.sep + 'Videos' + os.sep) + base_file[:-5] + '_features.mat'
ff = loadmat(segworm_file)
segworm_skel_x = ff['worm']['posture'][0][0]['skeleton'][0][0]['x'][0][0]
segworm_skel_y = ff['worm']['posture'][0][0]['skeleton'][0][0]['y'][0][0]

#get the microns per pixel in order to get a similar skeleton.
results_file = mask_dir.replace(os.sep + 'MaskedVideos', os.sep + 'Results' + os.sep) + base_file[:-5] + '_skeletons.hdf5'
micronsPerPixel = ff['info']['video'][0][0]['resolution'][0][0]['micronsPerPixels'][0][0]['x'][0][0][0,0]
micronsPerPixel = abs(micronsPerPixel)

#pass data from micros to pixels
segworm_skel_x /= micronsPerPixel
segworm_skel_y /= micronsPerPixel


#get the total number of frames
tot_frames = ff['info']['video'][0][0]['length'][0][0]['frames'][0][0][0,0]

with pd.HDFStore(masked_file, 'r') as mask_fid:
    video_metadata = mask_fid['/video_metadata']
    timestamps = video_metadata['best_effort_timestamp'].values 
    timestamps = timestamps.astype(np.int)

skel_x = np.full_like(segworm_skel_x, np.nan)
skel_y = np.full_like(segworm_skel_y, np.nan)

with h5py.File(results_file, 'r') as fid:
    skeletons = fid['/skeleton'][:]
    
    trajectories_data = fid['/trajectories_data'][:]    
    assert np.all(np.diff(trajectories_data['skeleton_id'])==1)
    frame_number = trajectories_data['frame_number']
    
    skel_x[:,timestamps[frame_number]] = np.rollaxis(skeletons[:,:,0],1)
    skel_y[:,timestamps[frame_number]] = np.rollaxis(skeletons[:,:,1],1)
    
    del skeletons
    del trajectories_data
#center data. The data is not stage corrected for the moment so we need to centre
#both (segworm and my algorithm) to be able to compare them
midpoint = np.round(segworm_skel_x.shape[0]/2)
segworm_skel_x += skel_x[midpoint,:] - segworm_skel_x[midpoint,:]
segworm_skel_y += skel_y[midpoint,:] - segworm_skel_y[midpoint,:]


#%%
with h5py.File(masked_file, 'r') as fid:
    #assert fid['/mask'].shape[0] == segworm_skel_x.shape[1]
    for ind in range(230, 250):
        image = fid['/mask'][ind]

        plt.figure()
        
        tstamp = timestamps[ind]
        print(ind, tstamp)
        #plt.imshow(image, interpolation='none', cmap='gray')
        plt.plot(segworm_skel_x[:,tstamp], segworm_skel_y[:,tstamp])
        plt.plot(skel_x[:,tstamp], skel_y[:,tstamp])
        



