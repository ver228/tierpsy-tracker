# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:21:39 2015

@author: ajaver
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:39:41 2015

@author: ajaver
"""
import pandas as pd
import h5py #make sure you are using hdf5 1.8.14 or higher https://github.com/h5py/h5py/issues/480
import os
import shutil
import cv2
import numpy as np

from scipy.ndimage.filters import median_filter
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

import sys
sys.path.append('../videoCompression/')
from parallelProcHelper import timeCounterStr

sys.path.append('../segworm_python/')
from main_segworm import getSkeleton, getStraightenWormInt

sys.path.append('../work_on_progress/')
from check_head_orientation import correctHeadTail


def getSmoothTrajectories(trajectories_file, displacement_smooth_win = 101, min_displacement = 0, threshold_smooth_win = 501):
    #read that frame an select trajectories that were considered valid by join_trajectories
    table_fid = pd.HDFStore(trajectories_file, 'r')
    df = table_fid['/plate_worms'][['worm_index_joined', 'frame_number', 'coord_x', 'coord_y','threshold']]
    df =  df.query('worm_index_joined > 0')
    table_fid.close()
    
    tracks_data = df.groupby('worm_index_joined').aggregate(['max', 'min', 'count'])
    if min_displacement > 0:
        #filter for trajectories that move too little (static objects)     
        delX = tracks_data['coord_x']['max'] - tracks_data['coord_x']['min']
        delY = tracks_data['coord_y']['max'] - tracks_data['coord_y']['min']
        
        good_index = tracks_data[(delX>min_displacement) & (delY>min_displacement)].index
        df = df[df.worm_index_joined.isin(good_index)]
    
    track_lenghts = (tracks_data['frame_number']['max'] - tracks_data['frame_number']['min']+1)
    tot_rows_ini = track_lenghts[track_lenghts>displacement_smooth_win].sum()
    del track_lenghts
    
    trajectories_df = pd.DataFrame({\
    'frame_number': np.zeros(tot_rows_ini, np.int32), \
    'worm_index_joined': np.zeros(tot_rows_ini, np.int32), \
    'plate_worm_id': np.zeros(tot_rows_ini, np.int32), \
    'coord_x': np.zeros(tot_rows_ini, np.double), \
    'coord_y': np.zeros(tot_rows_ini, np.double), \
    'threshold': np.zeros(tot_rows_ini, np.double)
    })
    
    #store the maximum and minimum frame of each worm
    worms_frame_range = {}
    
    #smooth trajectories (reduce jiggling from the CM to obtain a nicer video)
    #interpolate for possible missing frames in the trajectories
    tot_rows = 0;    
    for worm_index, worm_data in df.groupby('worm_index_joined'):
        x = worm_data['coord_x'].values
        y = worm_data['coord_y'].values
        t = worm_data['frame_number'].values
        thresh = worm_data['threshold'].values
        
        first_frame = np.min(t);
        last_frame = np.max(t);
        worms_frame_range[worm_index] = (first_frame, last_frame)
        
        tnew = np.arange(first_frame, last_frame+1);
        if len(tnew) <= displacement_smooth_win:
            continue
        
        fx = interp1d(t, x)
        xnew = savgol_filter(fx(tnew), displacement_smooth_win, 3);
        fy = interp1d(t, y)
        ynew = savgol_filter(fy(tnew), displacement_smooth_win, 3);
    
        fthresh = interp1d(t, thresh)
        threshnew = median_filter(fthresh(tnew), threshold_smooth_win);
        
        new_total = tot_rows + xnew.size
        segworm_id = np.arange(tot_rows, new_total, dtype = np.int32);
        tot_rows = new_total
        
        plate_worm_id = np.empty(xnew.size, dtype = np.int32)
        plate_worm_id.fill(-1)
        plate_worm_id[t - first_frame] =  worm_data.index
    
        trajectories_df.loc[segworm_id,'worm_index_joined'] = worm_index
        trajectories_df.loc[segworm_id,'coord_x'] = xnew
        trajectories_df.loc[segworm_id,'coord_y'] = ynew
        trajectories_df.loc[segworm_id,'frame_number'] = np.arange(first_frame, last_frame+1, dtype=np.int32)
        trajectories_df.loc[segworm_id,'threshold'] = threshnew
        trajectories_df.loc[segworm_id,'plate_worm_id'] = plate_worm_id
        
    assert tot_rows == tot_rows_ini
    
    return trajectories_df, worms_frame_range, tot_rows

def getWormROI(img, CMx, CMy, roi_size = 128):
    roi_center = roi_size/2
    roi_range = np.array([-roi_center, roi_center])

    #obtain bounding box from the trajectories
    range_x = round(CMx) + roi_range
    range_y = round(CMy) + roi_range
    
    if range_x[0]<0: range_x -= range_x[0]
    if range_y[0]<0: range_y -= range_y[0]
    
    if range_x[1]>img.shape[1]: range_x += img.shape[1]-range_x[1]-1
    if range_y[1]>img.shape[0]: range_y += img.shape[0]-range_y[1]-1
    
    worm_img = img[range_y[0]:range_y[1], range_x[0]:range_x[1]]
    roi_corner = np.array([range_x[0]-1, range_y[0]-1])
    
    return worm_img, roi_corner

def getWormMask(worm_img, threshold):
    #make the worm more uniform
    #worm_img = cv2.morphologyEx(worm_img, cv2.MORPH_CLOSE, np.ones((2,2)));
    worm_img = cv2.medianBlur(worm_img, 3);
    #compute the threshold
    worm_mask = ((worm_img < threshold) & (worm_img!=0)).astype(np.uint8)        
    worm_mask = cv2.morphologyEx(worm_mask, cv2.MORPH_CLOSE,np.ones((3,3)))
    return worm_mask

def drawWormContour(worm_img, worm_mask, skeleton, cnt_side1, cnt_side2, \
colorpalette = [(119, 158,27 ), (2, 95, 217), (138, 41, 231)]):
    
    intensity_rescale = 255./min(1.1*np.max(worm_img),255.);
    
    if intensity_rescale == np.inf:
         #the image is likely to be all zeros
        return cv2.cvtColor(worm_img, cv2.COLOR_GRAY2RGB);

    worm_img = (worm_img*intensity_rescale).astype(np.uint8)
            
    worm_rgb = cv2.cvtColor(worm_img, cv2.COLOR_GRAY2RGB);
    if skeleton.size==0 or np.all(np.isnan(skeleton)):
        worm_rgb[:,:,1][worm_mask!=0] = 204
        worm_rgb[:,:,2][worm_mask!=0] = 102
    else:
        pts = np.round(cnt_side1).astype(np.int32)
        cv2.polylines(worm_rgb, [pts], False, colorpalette[1], thickness=1, lineType = 8)
        pts = np.round(cnt_side2).astype(np.int32)
        cv2.polylines(worm_rgb, [pts], False, colorpalette[2], thickness=1, lineType = 8)
        
        pts = np.round(skeleton).astype(np.int32)
        cv2.polylines(worm_rgb, [pts], False, colorpalette[0], thickness=1, lineType = 8)
        
        #mark the head
        cv2.circle(worm_rgb, tuple(pts[0]), 2, (225,225,225), thickness=-1, lineType = 8)
        cv2.circle(worm_rgb, tuple(pts[0]), 3, (0,0,0), thickness=1, lineType = 8)
    
    return worm_rgb


def movies2Skeletons(masked_image_file, skeletons_file, trajectories_file, \
create_single_movies = False, roi_size = 128, resampling_N = 49, min_mask_area = 50):    
    
    #get trajectories, threshold and indexes from the first part of the tracker
    trajectories_df, worms_frame_range, tot_rows = \
    getSmoothTrajectories(trajectories_file, displacement_smooth_win = 101, min_displacement = 0, threshold_smooth_win = 501)
    #trajectories_df = trajectories_df.query('frame_number>43275')
    
    #pointer to the compressed videos
    mask_fid = h5py.File(masked_image_file, 'r');
    mask_dataset = mask_fid["/mask"]
    
    ske_file_id = h5py.File(skeletons_file, "w");
    
    data_strS = ['skeleton', 'contour_side1', 'contour_side2']
    
    for data_str in data_strS:
        length_str = data_str + '_length'
        ske_file_id.create_dataset("/" + length_str, (tot_rows,))

        ske_file_id.create_dataset("/"+data_str, (tot_rows, resampling_N, 2), 
                                    dtype = "double", maxshape = (tot_rows, resampling_N,2), 
                                    chunks = (1, resampling_N,2),
                                    compression="gzip", 
                                    compression_opts=4,
                                    shuffle=True);
                    
    ske_file_id.create_dataset("/contour_width", (tot_rows, resampling_N), 
                                    dtype = "double", maxshape = (tot_rows, resampling_N), 
                                    chunks = (1, resampling_N),
                                    compression="gzip", 
                                    compression_opts=4,
                                    shuffle=True);
    
    #save the data from trajectories_df
    for str_field in ['coord_x','coord_y','threshold']:
        ske_file_id.create_dataset("/" + str_field, data = trajectories_df[str_field].values)
    
    for str_field in ['frame_number','plate_worm_id','worm_index_joined']:
        ske_file_id.create_dataset("/" + str_field, data = trajectories_df[str_field].values.astype(np.int32))
    
    ske_file_id.create_dataset("/skeleton_id", data = trajectories_df.index.astype(np.int32))
    
    #dictionary to store previous skeletons
    prev_skeleton = {}
    
    #timer
    progressTime = timeCounterStr('Processing data.');
    for frame, frame_data in trajectories_df.groupby('frame_number'):
        #if frame >100: break
        
        img = mask_dataset[frame,:,:]
        for segworm_id, row_data in frame_data.iterrows():
            
            worm_img, roi_corner = getWormROI(img, row_data['coord_x'], row_data['coord_y'], roi_size)
            worm_mask = getWormMask(worm_img, row_data['threshold'])
            
            worm_index = row_data['worm_index_joined']
            if not worm_index in prev_skeleton:
                prev_skeleton[worm_index] = np.zeros(0)
            
            skeleton, ske_len, cnt_side1, cnt_side1_len, cnt_side2, cnt_side2_len, cnt_widths = \
            getSkeleton(worm_mask, prev_skeleton[worm_index], resampling_N, min_mask_area)
                            
            if skeleton.size>0:
                prev_skeleton[worm_index] = skeleton.copy()
                
                #save segwrom_results
                ske_file_id['/skeleton_length'][segworm_id] = ske_len 
                ske_file_id['/contour_side1_length'][segworm_id] = cnt_side1_len
                ske_file_id['/contour_side2_length'][segworm_id] = cnt_side2_len

                ske_file_id['/contour_width'][segworm_id, :] = cnt_widths                
                #convert into the main image coordinates
                ske_file_id['/skeleton'][segworm_id, :, :] = skeleton + roi_corner
                ske_file_id['/contour_side1'][segworm_id, :, :] = cnt_side1 + roi_corner
                ske_file_id['/contour_side2'][segworm_id, :, :] = cnt_side2 + roi_corner
                        
        if frame % 500 == 0:
            progress_str = progressTime.getStr(frame)
            print(base_name + ' ' + progress_str);

    
    ske_file_id.close()
    mask_fid.close()        

def writeIndividualMovies(masked_image_file, skeletons_file, video_save_dir, roi_size = 128, fps=25):    
    
    if os.path.exists(video_save_dir):
        shutil.rmtree(video_save_dir)
    os.makedirs(video_save_dir)

    #pointer to the compressed videos
    mask_fid = h5py.File(masked_image_file, 'r');
    mask_dataset = mask_fid["/mask"]
    
    #pointer to file with the skeletons
    skeletons_fid = h5py.File(skeletons_file, 'r');
    
    #data to extract the ROI
    skeletons_df = pd.DataFrame({'worm_index':skeletons_fid['/worm_index_joined'][:], \
        'frame_number':skeletons_fid['/frame_number'][:], \
        'coord_x':skeletons_fid['/coord_x'][:], 'coord_y':skeletons_fid['/coord_y'][:],
        'threshold':skeletons_fid['/threshold'][:]})

    #get first and last frame for each worm
    worms_frame_range = skeletons_df.groupby('worm_index').agg({'frame_number': [min, max]})['frame_number']
    
    video_list = {}
    progressTime = timeCounterStr('Creating videos.');
    for frame, frame_data in skeletons_df.groupby('frame_number'):
        img = mask_dataset[frame,:,:]
        for segworm_id, row_data in frame_data.iterrows():
            worm_index = row_data['worm_index']
            worm_img, roi_corner = getWormROI(img, row_data['coord_x'], row_data['coord_y'], roi_size)
            
            skeleton = skeletons_fid['/skeleton'][segworm_id,:,:]-roi_corner
            cnt_side1 = skeletons_fid['/contour_side1'][segworm_id,:,:]-roi_corner
            cnt_side2 = skeletons_fid['/contour_side2'][segworm_id,:,:]-roi_corner
            
            if not np.all(np.isnan(skeleton)):
                worm_mask = getWormMask(worm_img, row_data['threshold'])
            else:
                worm_mask = np.zeros(0)
            
            if (worms_frame_range['min'][worm_index] == frame) or (not worm_index in video_list):
                movie_save_name = video_save_dir + ('worm_%i.avi' % worm_index)
                video_list[worm_index] = cv2.VideoWriter(movie_save_name, \
                cv2.VideoWriter_fourcc('M','J','P','G'), fps, (roi_size, roi_size), isColor=True)
            
            worm_rgb = drawWormContour(worm_img, worm_mask, skeleton, cnt_side1, cnt_side2)
            assert (worm_rgb.shape[0] == worm_img.shape[0]) and (worm_rgb.shape[1] == worm_img.shape[1]) 
            video_list[worm_index].write(worm_rgb)
        
            if (worms_frame_range['max'][worm_index] == frame):
                video_list[worm_index].release();
                video_list.pop(worm_index, None)
            
        if frame % 500 == 0:
            progress_str = progressTime.getStr(frame)
            print(base_name + ' ' + progress_str);

#%%
if __name__ == '__main__':  
    base_name = 'Capture_Ch3_12052015_194303'
    root_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150512/'    

    #root_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150511/'
    #base_name = 'Capture_Ch1_11052015_195105'
    
    masked_image_file = root_dir + '/Compressed/' + base_name + '.hdf5'
    trajectories_file = root_dir + '/Trajectories/' + base_name + '_trajectories.hdf5'
    skeletons_file = root_dir + '/Trajectories/' + base_name + '_skeletons.hdf5'
    intensities_file = root_dir + '/Trajectories/' + base_name + '_intensities.hdf5'
    
    movies2Skeletons(masked_image_file, skeletons_file, trajectories_file, \
    create_single_movies = False, roi_size = 128, resampling_N = 49, min_mask_area = 50)

    correctHeadTail(skeletons_file, max_gap_allowed = 10, \
    window_std = 25, segment4angle = 5, min_block_size = 250)

    video_save_dir = root_dir + '/Worm_Movies_corrected/' + base_name + os.sep
    writeIndividualMovies(masked_image_file, skeletons_file, video_save_dir, roi_size = 128, fps=25)

                