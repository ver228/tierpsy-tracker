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
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import cv2

import sys
sys.path.append('../videoCompression/')
from parallelProcHelper import timeCounterStr


def getSmoothedTraj(trajectories_file, smooth_window_size = 101):
    #get id of trajectories with calculated skeletons
    table_fid = pd.HDFStore(trajectories_file, 'r');
    df = table_fid['/plate_worms'];
    df =  df[df['worm_index_joined'] > 0]
    
    counts = df['worm_index_joined'].value_counts()
    good_index = counts[counts>25].index;
    
    df = df[df['worm_index_joined'].isin(good_index)];
    table_fid.close()
    
    #smooth trajectories (reduce jiggling from the CM to obtain a nicer video)
    smoothed_CM = {};
    for worm_index in good_index:
        dat = df[df['worm_index_joined']==worm_index][['coord_x', 'coord_y', 'frame_number', 'threshold']]
        x = dat['coord_x'].values
        y = dat['coord_y'].values
        t = dat['frame_number'].values
        thresh = dat['threshold'].values
        
        first_frame = np.min(t);
        last_frame = np.max(t);
        tnew = np.arange(first_frame, last_frame+1);
        if len(tnew) <= smooth_window_size:
            continue
        
        fx = interp1d(t, x)
        xnew = savgol_filter(fx(tnew), smooth_window_size, 3);
        fy = interp1d(t, y)
        ynew = savgol_filter(fy(tnew), smooth_window_size, 3);
    
        fthresh = interp1d(t, thresh)
        threshnew = fthresh(tnew);
        
        smoothed_CM[worm_index] = {}
        smoothed_CM[worm_index]['coord_x'] = xnew
        smoothed_CM[worm_index]['coord_y'] = ynew
        smoothed_CM[worm_index]['first_frame'] = first_frame
        smoothed_CM[worm_index]['last_frame'] = last_frame
        smoothed_CM[worm_index]['threshold'] = threshnew

    return smoothed_CM

class kezhi_structure:
    def __init__(self, file_name, total_frames, roi_size):
        self.fid = h5py.File(file_name, "w");
        self.masks = self.fid.create_dataset("/masks", (total_frames, roi_size,roi_size), 
                                    dtype = "u1", maxshape = (total_frames, roi_size,roi_size), 
                                    chunks = (1, roi_size,roi_size),
                                    compression="gzip", 
                                    compression_opts=4,
                                    shuffle=True);
    
        self.frames = self.fid.create_dataset('/frames', (total_frames,), 
                                          dtype = 'i', maxshape = (total_frames,))
        self.CMs = self.fid.create_dataset('/CMs', (total_frames,2), 
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
        
        


masked_movies_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Compressed/'
trajectories_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Trajectories/'
base_name = 'Capture_Ch1_11052015_195105'
main_video_save_dir = r'/Users/ajaver/Desktop/Gecko_compressed/20150511/kezhi_format/'

masked_image_file = masked_movies_dir + base_name + '.hdf5'
trajectories_file = trajectories_dir + base_name + '_trajectories.hdf5'
segworm_file = trajectories_dir + base_name + '_segworm_fix.hdf5'
kezhi_save_dir = main_video_save_dir + base_name + os.sep    


max_frame_number = -1
roi_size = 128
min_mask_area = 50

roi_center = roi_size/2
roi_window = [-roi_center, roi_center]


if not os.path.exists(kezhi_save_dir):
    os.makedirs(kezhi_save_dir)

smoothed_CM = getSmoothedTraj(trajectories_file, smooth_window_size = 101)


last_frame = np.max([smoothed_CM[key]['last_frame'] for key in smoothed_CM])
first_frame = np.min([smoothed_CM[key]['first_frame'] for key in smoothed_CM])
if max_frame_number <0 or max_frame_number > last_frame:
    max_frame_number = last_frame;


##if len(smoothed_CM) == 0: #no valid trajectories identified
##    return;
    

#open the file with the masked images
mask_fid = h5py.File(masked_image_file, 'r');
mask_dataset = mask_fid["/mask"]

#create a saved the id of the files being saved
kezhi_list = {};

progressTime = timeCounterStr('Exporting to Kezhi compatible format.');

for frame in range(first_frame,max_frame_number):
    
    img = mask_dataset[frame,:,:]
    
    index2search = []
    for worm_index in smoothed_CM:
        if (frame >= smoothed_CM[worm_index]['first_frame']) and \
        (frame <= smoothed_CM[worm_index]['last_frame']):
            index2search.append(worm_index)

    for worm_index in index2search:
        if frame == smoothed_CM[worm_index]['first_frame']:
            #intialize figure and movie recorder
            total_frames = smoothed_CM[worm_index]['last_frame'] - smoothed_CM[worm_index]['first_frame'] + 1
            file_name = kezhi_save_dir + ('worm_%i.hdf5' % worm_index)
            kezhi_list[worm_index] = kezhi_structure(file_name, total_frames, roi_size);
        
        #obtain bounding box from the trajectories
        ind = int(frame-smoothed_CM[worm_index]['first_frame'])
        worm_CM = np.round([smoothed_CM[worm_index]['coord_x'][ind], \
        smoothed_CM[worm_index]['coord_y'][ind]])
        range_x = worm_CM[0] + roi_window
        range_y = worm_CM[1] + roi_window
        
        if range_x[0]<0: range_x -= range_x[0]
        if range_y[0]<0: range_y -= range_y[0]
        
        if range_x[1]>img.shape[1]: range_x += img.shape[1]-range_x[1]-1
        if range_y[1]>img.shape[0]: range_y += img.shape[0]-range_y[1]-1
        #%%
        worm_img =  img[range_y[0]:range_y[1], range_x[0]:range_x[1]]
        worm_mask = ((worm_img < smoothed_CM[worm_index]['threshold'][ind]) & (worm_img!=0)).astype(np.uint8)        
        worm_mask = cv2.morphologyEx(worm_mask, cv2.MORPH_CLOSE,np.ones((5,5)))
        
        contours, _ = cv2.findContours(worm_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        #clean mask if there is more than one contour
        
        min_dist_center = np.inf;
        valid_ind = -1
        for ii, cnt in enumerate(contours):
            cnt_area = cv2.contourArea(cnt)
            if cnt_area <= min_mask_area:
                continue
            mm = cv2.moments(cnt)
            cm_x = mm['m10']/mm['m00']
            cm_y = mm['m01']/mm['m00']
            dist_center = (cm_x-roi_center)**2 + (cm_y-roi_center)**2
            if min_dist_center > dist_center:
                min_dist_center = dist_center
                valid_ind = ii
                    
            #write only the closest contour to the mask center
            worm_mask = np.zeros_like(worm_mask);
            cv2.drawContours(worm_mask, contours, valid_ind, 1, -1)
        #%%
        worm_reduced = worm_mask*worm_img
        kezhi_list[worm_index].add(worm_reduced, frame, worm_CM)

    if frame >= smoothed_CM[worm_index]['last_frame']:
        kezhi_list[worm_index].close()   
        kezhi_list.pop(worm_index)
 
        
    if frame % 500 == 0:
        progress_str = progressTime.getStr(frame)
        print(base_name + ' ' + progress_str);
        
for worm_index in kezhi_list:
    kezhi_list[worm_index].close()  
    
mask_fid.close()
