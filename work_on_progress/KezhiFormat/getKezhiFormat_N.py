# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:43:31 2015

@author: ajaver
"""

import matplotlib.pylab as plt
import os
import pandas as pd
import numpy as np
import h5py
import cv2

import sys
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking')
from MWTracker.helperFunctions.timeCounterStr import timeCounterStr
from MWTracker.trackWorms.getSkeletonsTables import getWormROI, getWormMask
from MWTracker.trackWorms.segWormPython.mainSegworm import binaryMask2Contour

class kezhi_structure:
    def __init__(self, fid, worm_group, total_frames, roi_size):
        self.fid = fid;
        main_group_str = '/' + worm_group

        if not main_group_str in self.fid:
            self.main_group = self.fid.create_group(main_group_str)
        
        self.masks = self.fid.create_dataset(main_group_str + "/masks", 
                                    (total_frames, roi_size,roi_size), 
                                    dtype = "u1", maxshape = (total_frames, roi_size,roi_size), 
                                    chunks = (1, roi_size,roi_size),
                                    compression="gzip", 
                                    compression_opts=4,
                                    shuffle=True);
    
        self.frames = self.fid.create_dataset(main_group_str + '/frames', 
                                              (total_frames,), 
                                          dtype = 'i', maxshape = (total_frames,))
        self.CMs = self.fid.create_dataset(main_group_str + '/CMs', 
                                           (total_frames,2), 
                                          dtype = 'i', maxshape = (total_frames,2))
        self.index = 0;
        
    def add(self, mask, frame, CM):
        assert np.all(mask.shape == self.masks.shape[1:])
        assert len(CM) == 2
        assert self.index < self.masks.shape[0]
        
        self.masks[self.index, :,:] = mask;
        self.frames[self.index] = frame;
        self.CMs[self.index,:] = CM;
        self.index += 1;
        
    def close(self):
        self.fid.close();
        
        
if __name__ == '__main__':
    
    masked_image_file = '/Volumes/behavgenom$/GeckoVideo/MaskedVideos/20150521_1115/Capture_Ch5_21052015_111806.hdf5'
    base_name = masked_image_file.rpartition('.')[0].rpartition(os.sep)[-1]
    
    skeletons_dir = '/Volumes/behavgenom$/GeckoVideo/Results/20150521_1115/'
    
    
    skeletons_file = skeletons_dir + base_name + '_skeletons.hdf5'
    kezhi_save_file = skeletons_dir + base_name + '_kezhi.hdf5'

    ##if len(smoothed_CM) == 0: #no valid trajectories identified
    ##    return;
        

    #open the file with the masked images
    with h5py.File(masked_image_file, 'r') as mask_fid, \
    h5py.File(kezhi_save_file, 'w') as kezhi_fid:

        mask_dataset = mask_fid["/mask"]

        #read trajectories data
        with pd.HDFStore(skeletons_file, 'r') as ske_file_id:
            trajectories_data = ske_file_id['/trajectories_data']
            #trajectories_data = trajectories_data[trajectories_data['frame_number']<1000]
        
        first_frame = trajectories_data['frame_number'].max()
        last_frame = trajectories_data['frame_number'].min()
        
        worm_grouped = trajectories_data[['worm_index_joined', 'skeleton_id', 'frame_number']].groupby('worm_index_joined')
        worm_range = worm_grouped.aggregate([min, max])['frame_number']
        worm_range['total'] = worm_range['max'] - worm_range['min'] + 1
        
        #save skeletons
        with h5py.File(skeletons_file, 'r') as ske_file_id:
            n_seg = ske_file_id['/skeleton'].shape[1]
            
            ii = 0
            progress_timer = timeCounterStr('');
            for ii, dat in enumerate(worm_grouped):
                worm_index, worm_data = dat
                
                if ii % 10 == 0:
                    dd = " Writing skeletons worm %i of %i." % (ii+1, len(worm_grouped))
                    dd = base_name + dd + ' Total time:' + progress_timer.getTimeStr()
                    print(dd)
            
                worm_group_name = ('worm_%i' % worm_index)
                
                first_w_frame = worm_data['frame_number'].min()
                last_w_frame = worm_data['frame_number'].max()
                tot_w_frames = last_w_frame - first_w_frame + 1
                
                ind_ff = worm_data['frame_number'].values - first_w_frame
                skel_id = worm_data['skeleton_id'].values
                
                w_skel = np.full((tot_w_frames, n_seg, 2), np.nan)
                w_skel[ind_ff] = ske_file_id['/skeleton'][skel_id,:,:]
                
                if not "/" + worm_group_name in kezhi_fid:
                    kezhi_fid.create_group("/" + worm_group_name)
                
                kezhi_fid.create_dataset("/" + worm_group_name + "/skeletons", 
                                        data = w_skel, compression="gzip", \
                                        compression_opts=4, shuffle=True);

        #save images            
        frame_grouped = trajectories_data.groupby('frame_number')
        #create a saved the id of the files being saved
        kezhi_list = {};

        progressTime = timeCounterStr('Exporting to Kezhi compatible format.');
        for frame, frame_data in frame_grouped:
             img = mask_dataset[frame,:,:]
             
             for ii, worm_row in frame_data.iterrows():
                 worm_index = int(worm_row['worm_index_joined'])
                 if frame == worm_range.loc[worm_index]['min']:
                     #intialize figure and movie recorder
                     total_frames = worm_range.loc[worm_index]['total']
                     roi_size = worm_row['roi_size']
                     worm_group = ('worm_%i' % worm_index)
                     kezhi_list[worm_index] = kezhi_structure(kezhi_fid, worm_group, total_frames, roi_size);
                
                 #obtain bounding box from the trajectories
                 worm_roi, roi_corner = getWormROI(img, worm_row['coord_x'], worm_row['coord_y'], worm_row['roi_size'])
                 worm_mask = getWormMask(worm_roi, worm_row['threshold']) 
                 
                 worm_cnt = binaryMask2Contour(worm_mask)
                 worm_mask = np.zeros_like(worm_mask)
                 cv2.drawContours(worm_mask, [worm_cnt.astype(np.int32)], 0, 1, -1)
                 
                 worm_reduced = worm_mask*worm_roi
                 worm_CM = np.round([worm_row['coord_x'], worm_row['coord_y']])
                 kezhi_list[worm_index].add(worm_reduced, frame, worm_CM)

             if frame >= worm_range.loc[worm_index]['max']:
                 kezhi_list.pop(worm_index)
         
                
             if frame % 500 == 0:
                 progress_str = progressTime.getStr(frame)
                 print(base_name + ' ' + progress_str);

