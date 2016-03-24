# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 22:01:59 2016

@author: ajaver
"""

import h5py
import os
import numpy as np
import matplotlib.pylab as plt
from scipy.io import loadmat
import glob
import os



delT = 15   

main_dir = '/Users/ajaver/Desktop/Videos/single_worm/agar_2/MaskedVideos/'

files = glob.glob(os.path.join(main_dir, '*.hdf5' ))
files = sorted(files)
#%%
for mask_id, masked_image_file in enumerate(files[0:15]):
    skeletons_file = masked_image_file.replace('MaskedVideos', 'Results')[:-5] + '_skeletons.hdf5'
    feat_file = masked_image_file.replace('MaskedVideos', 'Features')[:-5] + '_features.mat'

    #%%
    with h5py.File(skeletons_file, 'r') as fid:
        if not '/stage_movement/stage_vec' in fid:
            continue
        skeletons_ori = fid['/skeleton'][:]
        stage_vec = fid['/stage_movement/stage_vec'][:]
        
        timestamp_ind = fid['/timestamp/raw'][:].astype(np.int)
        
        max_ind = np.max(timestamp_ind)+1
        skeletons = np.full((max_ind, skeletons_ori.shape[1], skeletons_ori.shape[2]), np.nan)
        
        
        
        skeletons[timestamp_ind,:,:] = skeletons_ori + stage_vec[:, np.newaxis,:]
        
        

    #%%
    if os.path.exists(feat_file):
        fvars = loadmat(feat_file)
        micronsPerPixels_x = fvars['info']['video'][0][0]['resolution'][0][0]['micronsPerPixels'][0][0]['x'][0][0][0][0]
        micronsPerPixels_y = fvars['info']['video'][0][0]['resolution'][0][0]['micronsPerPixels'][0][0]['y'][0][0][0][0]
        
        segworm_x = fvars['worm']['posture'][0][0]['skeleton'][0][0]['x'][0][0].T
        segworm_x /= micronsPerPixels_x 
        
        segworm_y = fvars['worm']['posture'][0][0]['skeleton'][0][0]['y'][0][0].T
        segworm_y /=micronsPerPixels_y
        #%%
        max_n_skel = min(segworm_x.shape[0], skeletons.shape[0])   
        #%%
        seg_x = segworm_x
        seg_y = segworm_y
                
        skel_x_ori = skeletons[:,:,0];
        skel_y_ori = skeletons[:,:,1];
        
        pos_coord = ((x_sign*XX, y_sign*YY) for XX, YY in \
        [(skel_x_ori, skel_y_ori), (skel_y_ori, skel_x_ori)] \
        for x_sign in (1,-1) for y_sign in (1,-1))
        
        R_min = np.inf
        
        for XX, YY in pos_coord:
            dXo = XX[:max_n_skel] - seg_x[:max_n_skel]
            dYo = YY[:max_n_skel] - seg_y[:max_n_skel]
            
            shift_x = np.nanmedian(dXo)
            shift_y = np.nanmedian(dYo)
            
            #print(shift_x, shift_y)
            dX = dXo - shift_x
            dY = dYo - shift_y
            
            
            R = dX*dX + dY*dY
            
            R_tot = np.nanmean(R)
            if R_tot < R_min:
                skel_x, skel_y = XX - shift_x, YY - shift_y
                R_min = R_tot
                R_error = R
        
        #%%
            
        #for shift_m 
                
        
        #plt.figure()    
        #plt.plot(seg_x, seg_y, 'b')
        #plt.plot(skel_x, skel_y, 'r')
        #%%
        skel_error = np.mean(np.sqrt(R_error), axis=1)
                
        #%%
        plt.figure()
        plt.subplot(3,1,1)        
        plt.plot(skel_x[::delT].T, skel_y[::delT].T, 'b')    
        plt.plot(seg_x[::delT].T, seg_y[::delT].T, 'r')
        plt.axis('equal')
        plt.title(mask_id)
        
        plt.subplot(3,1,2)
        plt.plot(skel_error, '.')
        plt.ylim((0, np.nanmax(skel_error)))
        
        plt.subplot(3,1,3)
        plt.plot(skel_x[:,25]) 
        plt.plot(seg_x[:,25])
        #%%
    else:
        plt.figure()
        plt.title(mask_id)
        plt.plot(skeletons[:max_n_skel:delT,:, 0], skeletons[:max_n_skel:delT,:, 1], 'b') 
        #%%