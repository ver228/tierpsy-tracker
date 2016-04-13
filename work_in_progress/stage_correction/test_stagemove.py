# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 22:01:59 2016

@author: ajaver
"""

import h5py
import tables
import os
import numpy as np
import matplotlib.pylab as plt
from scipy.io import loadmat
import glob
import os
import pandas as pd

delT = 15

main_dir = '/Users/ajaver/Desktop/Videos/single_worm/agar_1/MaskedVideos/'

from MWTracker.featuresAnalysis.obtainFeaturesHelper import WormFromTable
#from MWTracker.featuresAnalysis.obtainFeatures import getMicroPerPixels, getFPS


def getFPS(skeletons_file, fps_expected):
    #try to infer the fps from the timestamp
    try:
        with tables.File(skeletons_file, 'r') as fid:
            timestamp_time = fid.get_node('/timestamp/time')[:]
            fps = 1/np.median(np.diff(timestamp_time))
            if np.isnan(fps): 
                raise ValueError
            is_default_timestamp = 0
    except (tables.exceptions.NoSuchNodeError, IOError, ValueError):
        fps = fps_expected
        is_default_timestamp = 1
    
    return fps, is_default_timestamp


def getMicroPerPixel(skeletons_file):
    try:
        with tables.File(skeletons_file, 'r') as fid:
            return fid.get_node('/stage_movement')._v_attrs['pixel_per_micron_scale']
    except (tables.exceptions.NoSuchNodeError, IOError, KeyError):
        #i need to change it to something better, but for the momement let's use 1 as default
        return 1

def correctSingleWorm(worm, skeletons_file):
    with h5py.File(skeletons_file, 'r') as fid:
        if not '/stage_movement/stage_vec' in fid:
            continue
        stage_vec_ori = fid['/stage_movement/stage_vec'][:]
        timestamp_ind = fid['/timestamp/raw'][:].astype(np.int)
        rotation_matrix = fid['/stage_movement'].attrs['rotation_matrix']

    #adjust the stage_vec to match the timestamps in the skeletons
    timestamp_ind = timestamp_ind - worm.first_frame
    good = (timestamp_ind>=0) & (timestamp_ind<=worm.timestamp[-1])
    
    timestamp_ind = timestamp_ind[good]
    stage_vec_ori = stage_vec_ori[good]
    
    stage_vec = np.full((worm.timestamp.size,2), np.nan)
    stage_vec[timestamp_ind, :] = stage_vec_ori
    
    tot_skel = worm.skeleton.shape[0]
    
    for field in ['skeleton', 'ventral_contour', 'dorsal_contour']:
        tmp_dat = getattr(worm, field)
        for ii in range(tot_skel):
            tmp_dat[ii] = np.dot(tmp_dat[ii], rotation_matrix) - stage_vec[ii]
        setattr(worm, field, tmp_dat)


files = glob.glob(os.path.join(main_dir, '*.hdf5' ))
files = sorted(files)[:1]
#%%
for mask_id, masked_image_file in enumerate(files):
    skeletons_file = masked_image_file.replace('MaskedVideos', 'Results')[:-5] + '_skeletons.hdf5'
    feat_file = masked_image_file.replace('MaskedVideos', 'Features')[:-5] + '_features.mat'
#%%
    print(mask_id)
    
    fps, is_default_timestamp = getFPS(skeletons_file, fps_expected=25)
    micronsPerPixel = pix2mum = getMicroPerPixel(skeletons_file)
    worm = WormFromTable(skeletons_file, 1, fps=fps, micronsPerPixel=micronsPerPixel)
    

    
    skeletons = worm.skeleton
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