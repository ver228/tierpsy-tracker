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

#44 has a shift...

delT = 15   

main_dir = '/Users/ajaver/Desktop/Videos/single_worm/agar_2/MaskedVideos/'

files = glob.glob(os.path.join(main_dir, '*.hdf5' ))
files = sorted(files)
#%%
for mask_id, masked_image_file in enumerate(files):
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
        
        rotation_matrix = fid['/stage_movement'].attrs['rotation_matrix']
        pixel_per_micron_scale = fid['/stage_movement'].attrs['pixel_per_micron_scale']
        
        for ind_ori, int_ts in enumerate(timestamp_ind):
            #%%
            rot_skel = np.dot(skeletons_ori[ind_ori], rotation_matrix)
            scale_skel = rot_skel*pixel_per_micron_scale;
            skeletons[int_ts,:,:]  = scale_skel - stage_vec[ind_ori]
    #%%
    
    #plt.figure()
    #plt.plot(skeletons[0, :, 0].T, skeletons[0, :, 1].T)
    #plt.plot(-segworm_x[0], -segworm_y[0])
    #%%
    if os.path.exists(feat_file):
        fvars = loadmat(feat_file)
        micronsPerPixels_x = fvars['info']['video'][0][0]['resolution'][0][0]['micronsPerPixels'][0][0]['x'][0][0][0][0]
        micronsPerPixels_y = fvars['info']['video'][0][0]['resolution'][0][0]['micronsPerPixels'][0][0]['y'][0][0][0][0]
        
        segworm_x = -fvars['worm']['posture'][0][0]['skeleton'][0][0]['x'][0][0].T
        segworm_y = -fvars['worm']['posture'][0][0]['skeleton'][0][0]['y'][0][0].T
        
        #%%
        # it seems there is a bug in segworm and the flag for dropped frames is the same as stage movements...
        #FLAG_DROPPED = 2;
        #FLAG_STAGE = 3;
        frame_annotations = fvars['info']['video'][0][0]['annotations'][0][0]['frames'][0][0][0];
        
        
        
        #%%
        
        max_n_skel = min(segworm_x.shape[0], skeletons.shape[0])
        
        #%%
        seg_x = segworm_x
        seg_y = segworm_y
                
        skel_x = skeletons[:,:,0];
        skel_y = skeletons[:,:,1];
        
        dXo = skel_x[:max_n_skel] - seg_x[:max_n_skel]
        dYo = skel_y[:max_n_skel] - seg_y[:max_n_skel]
        
        shift_x = np.nanmedian(dXo)
        shift_y = np.nanmedian(dYo)
        
        #skel_x -= shift_x
        #skel_y -= shift_y
        #%%        
#        pos_coord = ((x_sign*XX, y_sign*YY) for XX, YY in \
#        [(skel_x_ori, skel_y_ori), (skel_y_ori, skel_x_ori)] \
#        for x_sign in (1,-1) for y_sign in (1,-1))
#        
#        R_min = np.inf
#        
#        for XX, YY in pos_coord:
#            dXo = XX[:max_n_skel] - seg_x[:max_n_skel]
#            dYo = YY[:max_n_skel] - seg_y[:max_n_skel]
#            
#            shift_x = np.nanmedian(dXo)
#            shift_y = np.nanmedian(dYo)
#            
#            #print(shift_x, shift_y)
#            dX = dXo - shift_x
#            dY = dYo - shift_y
#            
#            
#            R = dX*dX + dY*dY
#            
#            R_tot = np.nanmean(R)
#            if R_tot < R_min:
#                skel_x, skel_y = XX - shift_x, YY - shift_y
#                R_min = R_tot
#                R_error = R
        
        #%%
        
        dX = skel_x[:max_n_skel] - seg_x[:max_n_skel]
        dY = skel_y[:max_n_skel] - seg_y[:max_n_skel]
        R_error = dX*dX + dY*dY
        
        skel_error = np.sqrt(np.mean(R_error, axis=1))
                
        
        #%%
        w_xlim = (-10, skel_error.size+10)
        
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(skel_error, '.')
        plt.ylim((0, np.nanmax(skel_error)))
        plt.xlim(w_xlim)
        plt.title(mask_id)
        
        plt.subplot(2,1,2)
        plt.plot(skel_x[:,1], 'b') 
        plt.plot(seg_x[:,1], 'r')
        plt.xlim(w_xlim)
        #%%
        plt.figure()
        #plt.subplot(3,1,1)        
        plt.plot(skel_x[::delT].T, skel_y[::delT].T, 'b')    
        plt.plot(seg_x[::delT].T, seg_y[::delT].T, 'r')
        plt.axis('equal')
        plt.title(mask_id)
        #%%
    else:
        plt.figure()
        plt.title(mask_id)
        plt.plot(skeletons[::delT,:, 0].T, skeletons[::delT,:, 1].T, 'b') 
        plt.axis('equal')
        #%%