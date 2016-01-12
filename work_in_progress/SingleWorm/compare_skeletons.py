# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:00:00 2015

@author: ajaver
"""

import os
import h5py 
import matplotlib.pylab as plt
import numpy as np

from scipy.io import loadmat

masked_file = '/Users/ajaver/Desktop/Videos/Check_Align_samples/MaskedVideos/npr-13 (tm1504)V on food L_2010_01_25__11_56_02___4___2.hdf5' 

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

#get the total number of frames
tot_frames = ff['info']['video'][0][0]['length'][0][0]['frames'][0][0][0,0]

skel_x = np.full_like(segworm_skel_x, np.nan)
skel_y = np.full_like(segworm_skel_y, np.nan)

with h5py.File(results_file, 'r') as fid:
    skeletons = fid['/skeleton'][:]*micronsPerPixel
    
    trajectories_data = fid['/trajectories_data'][:]    
    assert np.all(np.diff(trajectories_data['skeleton_id'])==1)
    frame_number = trajectories_data['frame_number']
    
    skel_x[:,frame_number] = np.rollaxis(skeletons[:,:,0],1)
    skel_y[:,frame_number] = np.rollaxis(skeletons[:,:,1],1)
    
    del skeletons
    del trajectories_data

#center data. The data is not stage corrected for the moment so we need to centre
#both (segworm and my algorithm) to be able to compare them
midpoint = np.round(segworm_skel_x.shape[0]/2)
segworm_skel_x -= segworm_skel_x[midpoint,:]
segworm_skel_y -= segworm_skel_y[midpoint,:]

midpoint = np.round(skel_x.shape[0]/2)
skel_x -= skel_x[midpoint,:]
skel_y -= skel_y[midpoint,:]

#%%
for ind in range(250, 260): 
    plt.figure()
    plt.plot(segworm_skel_x[:,ind], segworm_skel_y[:,ind])
    plt.plot(skel_x[:,ind], skel_y[:,ind])
    plt.xlim((-600, 600))
    plt.ylim((-600, 600))
    



