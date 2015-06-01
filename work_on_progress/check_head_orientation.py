# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:35:04 2015

@author: ajaver
"""

import pandas as pd
import h5py
import numpy as np
import os
import cv2
import shutil

import matplotlib.pylab as plt


import sys
sys.path.append('../tracking_worms/')
from getSkeletons import drawWormContour, getWormROI, getWormMask

sys.path.append('../videoCompression/')
from parallelProcHelper import timeCounterStr

def getAnglesDelta(dx,dy):
    angles = np.arctan2(dx,dy)
    dAngles = np.diff(angles)
        
    
    positiveJumps = np.where(dAngles > np.pi)[0] + 1; #%+1 to cancel shift of diff
    negativeJumps = np.where(dAngles <-np.pi)[0] + 1;
        
    #% subtract 2pi from remainging data after positive jumps
    for jump in positiveJumps:
        angles[jump:] = angles[jump:] - 2*np.pi;
        
    #% add 2pi to remaining data after negative jumps
    for jump in negativeJumps:
        angles[jump:] = angles[jump:] + 2*np.pi;
    
    #% rotate skeleton angles so that mean orientation is zero
    meanAngle = np.mean(angles);
    angles = angles - meanAngle;

    return angles, meanAngle

def calculateHeadTailAng(skeletons, seg_skel_delta, good):
    angles_head = np.empty(skeletons.shape[0])
    angles_head.fill(np.nan)
    angles_tail = angles_head.copy()
    
    dx = skeletons[good,seg_skel_delta,0] - skeletons[good,0,0]
    dy = skeletons[good,seg_skel_delta,1] - skeletons[good,0,1]
    
    angles_head[good], _ = getAnglesDelta(dx,dy)
    
    dx = skeletons[good,-seg_skel_delta-1,0] - skeletons[good,-1,0]
    dy = skeletons[good,-seg_skel_delta-1,1] - skeletons[good,-1,1]
    
    angles_tail[good], _ = getAnglesDelta(dx,dy)
    return angles_head, angles_tail

def getBlocksIDs(skeletons, invalid, max_skel_lost = 10):
    good_ind = np.where(~invalid)[0];            
    delTs = np.diff(good_ind)
    
    block_ind = np.zeros_like(good_ind)
    block_ind[0] = 1;
    for ii, delT in enumerate(delTs):
        if delT <= max_skel_lost:
            block_ind[ii+1] = block_ind[ii];
        else:
            block_ind[ii+1] = block_ind[ii]+1;
    block_ids = np.zeros(skeletons.shape[0], dtype=np.int)
    block_ids[good_ind] = block_ind
    tot_blocks = block_ind[-1]
    return block_ids, tot_blocks

def isWormHTSwitched(df_worm, skeletons, seg_skel_delta = 5, max_skel_lost = 10, window_std = 25):
    invalid = np.isnan(skeletons[:,0,0])
    block_ids, tot_blocks = getBlocksIDs(invalid, max_skel_lost)
    angles_head, angles_tail = calculateHeadTailAng(skeletons, seg_skel_delta, block_ids!=0)
        
    ts = pd.DataFrame({'head_angle':angles_head, 'tail_angle':angles_tail}, \
    np.arange(df_worm['frame_number'].min(), df_worm['frame_number'].max()+1))
    
    roll_std = pd.rolling_std(ts, window = window_std, min_periods = window_std-max_skel_lost);
    
    roll_std["is_head"] = (roll_std['head_angle']>roll_std['tail_angle'])
    roll_std["block_id"] = block_ids
    
    #this function will return nan if the number of elements in the group is less than min_skel_block
    mean_relevant = lambda x: x.mean() if x.count() > min_skel_block else np.nan
    head_prob = roll_std.groupby('block_id').agg({'is_head': mean_relevant})
    
    head_prob.loc[0] = np.nan
    #fill nan, forward with the last valid observation, and then first backward with the next valid observation
    head_prob = head_prob.fillna(method = 'ffill').fillna(method = 'bfill')
    
    is_switch_block = np.squeeze(head_prob.values)<0.5
    is_switch_skel = is_switch_block[block_ids]
    return is_switch_skel, roll_std


if __name__ == "__main__":
    #root_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150511/'
    #base_name = 'Capture_Ch1_11052015_195105'
    root_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150512/'    
    base_name = 'Capture_Ch3_12052015_194303'

    
    masked_image_file = root_dir + '/Compressed/' + base_name + '.hdf5'
    trajectories_file = root_dir + '/Trajectories/' + base_name + '_trajectories.hdf5'
    skeletons_file = root_dir + '/Trajectories/' + base_name + '_segworm.hdf5'
    video_save_dir = root_dir + '/Worm_Movies_corrected/' + base_name + os.sep

    max_skel_lost = 10
    window_std = 25
    seg_skel_delta = 5; #separation between skeleton segments to calculate the angles
    min_skel_block = 250 #consider only around 10s intervals to determine if it is head or tail... 
    
    segworm_FILE = h5py.File(skeletons_file, 'r+');
    segworm_df = pd.DataFrame({'worm_index':segworm_FILE['/worm_index_joined'][:], \
    'frame_number':segworm_FILE['/frame_number'][:],
    'coord_x':segworm_FILE['/coord_x'][:], 'coord_y':segworm_FILE['/coord_y'][:],
    'threshold':segworm_FILE['/threshold'][:]}, 
    index = segworm_FILE['/segworm_id'][:])

    #segworm_df=segworm_df.query('worm_index==3')
    #%%
    progressTime = timeCounterStr('Correcting head_tail.');
    
    worm_counter = 0
    for worm_index, df_worm in segworm_df.groupby('worm_index'):
        assert np.all(np.diff(df_worm.index)==1)
        
        ini = df_worm.index[0]
        end = df_worm.index[-1]
        
        skeletons = segworm_FILE['/skeleton'][ini:end+1,:,:]
        invalid = np.all(skeletons[:,:,0]==0, axis=1)
        
        cnt_side1 = segworm_FILE['/contour_side1'][ini:end+1,:,:]
        cnt_side2 = segworm_FILE['/contour_side2'][ini:end+1,:,:]
        cnt_side1_len = segworm_FILE['/contour_side1_length'][ini:end+1]
        cnt_side2_len = segworm_FILE['/contour_side2_length'][ini:end+1]
        worm_intensity = segworm_FILE['/straighten_worm_intensity'][ini:end+1]

        
        skeletons[invalid] = np.nan
        cnt_side1[invalid] = np.nan
        cnt_side2[invalid] = np.nan
        cnt_side1_len[invalid] = np.nan
        cnt_side2_len[invalid] = np.nan
        worm_intensity[invalid] = np.nan

        if np.any(~invalid):
            is_switched_skel, roll_std = isWormHTSwitched(df_worm, skeletons, seg_skel_delta = seg_skel_delta, max_skel_lost = max_skel_lost, window_std = window_std)
            #roll_std[['head_angle','tail_angle']].plot()
            
            skeletons[is_switched_skel,:,:] = skeletons[is_switched_skel,::-1,:]
            #contours must be switched to keep clockwise orientation
            cnt_side1[is_switched_skel], cnt_side2[is_switched_skel ] = \
            cnt_side2[is_switched_skel,::-1,:], cnt_side1[is_switched_skel,::-1,:]
            cnt_side1[is_switched_skel], cnt_side2[is_switched_skel] = cnt_side2[is_switched_skel], cnt_side1[is_switched_skel]
            
        #is_switched_skel, roll_std = isWormHTSwitched(skeletons, seg_skel_delta = seg_skel_delta, max_skel_lost = max_skel_lost, window_std = window_std)
        #roll_std[['head_angle','tail_angle']].plot()
        
        segworm_FILE['/skeleton'][ini:end+1,:,:] = skeletons
        segworm_FILE['/contour_side1'][ini:end+1,:,:] = cnt_side1
        segworm_FILE['/contour_side2'][ini:end+1,:,:] = cnt_side2
        segworm_FILE['/contour_side1_length'][ini:end+1] = cnt_side1_len
        segworm_FILE['/contour_side2_length'][ini:end+1] = cnt_side2_len
        segworm_FILE['/straighten_worm_intensity'][ini:end+1, :, :] = worm_intensity
        del skeletons
        
        worm_counter += 1;
        
        if worm_counter % 10 == 1:
            progress_str = progressTime.getStr(worm_counter)
            print(base_name + ' ' + progress_str);
    
    segworm_FILE.close();
    
    
    #MAKE VIDEOS
    roi_size = 128
    
    if os.path.exists(video_save_dir):
        shutil.rmtree(video_save_dir)
    os.makedirs(video_save_dir)
    
    #create videos
    #re-open the file as read only. We do not want accidents...
    segworm_FILE = h5py.File(skeletons_file, 'r');
    
  
    #pointer to the compressed videos
    mask_fid = h5py.File(masked_image_file, 'r');
    mask_dataset = mask_fid["/mask"]
    
    #get first and last frame for each worm
    worms_frame_range = segworm_df.groupby('worm_index').agg({'frame_number': [min, max]})['frame_number']
    
    video_list = {}
    progressTime = timeCounterStr('Creating videos.');
    for frame, frame_data in segworm_df.groupby('frame_number'):
        #if frame >500: break

        img = mask_dataset[frame,:,:]
        for segworm_id, row_data in frame_data.iterrows():
            worm_index = row_data['worm_index']
            worm_img, roi_corner = getWormROI(img, row_data['coord_x'], row_data['coord_y'], roi_size)
            
            skeleton = segworm_FILE['/skeleton'][segworm_id,:,:]-roi_corner
            cnt_side1 = segworm_FILE['/contour_side1'][segworm_id,:,:]-roi_corner
            cnt_side2 = segworm_FILE['/contour_side2'][segworm_id,:,:]-roi_corner
            
            if np.all(np.isnan(skeleton)):
                worm_mask = getWormMask(worm_img, row_data['threshold'])
            else:
                worm_mask = np.zeros(0)
            
            if (worms_frame_range['min'][worm_index] == frame) or (not worm_index in video_list):
                movie_save_name = video_save_dir + ('worm_%i.avi' % worm_index)
                #gray pixels if no contour is drawn
                video_list[worm_index] = cv2.VideoWriter(movie_save_name, \
                cv2.cv.FOURCC('M','J','P','G'), 25, (roi_size, roi_size), isColor=True)
            
            
            
            worm_rgb = drawWormContour(worm_img, worm_mask, skeleton, cnt_side1, cnt_side2)
            assert (worm_rgb.shape[0] == worm_img.shape[0]) and (worm_rgb.shape[1] == worm_img.shape[1]) 
            video_list[worm_index].write(worm_rgb)
            
        
            if (worms_frame_range['max'][worm_index] == frame):
                video_list[worm_index].release();
                video_list.pop(worm_index, None)
        
        if (frame-1 % 10000) == 0:
            #reopen hdf5 to avoid a buffer overflow  https://github.com/h5py/h5py/issues/480
            segworm_FILE.close()
            segworm_FILE = h5py.File(skeletons_file, "r+");
            
            
        if frame % 500 == 0:
            progress_str = progressTime.getStr(frame)
            print(base_name + ' ' + progress_str);

