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
import h5py
import tables
import time
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
    
    segworm_df = pd.DataFrame({\
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
    
        segworm_df.loc[segworm_id,'worm_index_joined'] = worm_index
        segworm_df.loc[segworm_id,'coord_x'] = xnew
        segworm_df.loc[segworm_id,'coord_y'] = ynew
        segworm_df.loc[segworm_id,'frame_number'] = np.arange(first_frame, last_frame+1, dtype=np.int32)
        segworm_df.loc[segworm_id,'threshold'] = threshnew
        segworm_df.loc[segworm_id,'plate_worm_id'] = plate_worm_id
        
    assert tot_rows == tot_rows_ini
    
    return segworm_df, worms_frame_range, tot_rows

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





def trace_calls(frame, event, arg):
    if event != 'call':
        return
    co = frame.f_code
    func_name = co.co_name
    if func_name == 'write':
        # Ignore write() calls from print statements
        return
    func_line_no = frame.f_lineno
    func_filename = co.co_filename
    #func_filename = os.path.split(func_filename)[1]
    
    caller = frame.f_back
    caller_line_no = caller.f_lineno 
    caller_filename = caller.f_code.co_filename
    if not 'segworm_python' in caller_filename:
        return
    if 'anaconda' in func_filename:
        return
    #caller_filename = os.path.split(caller_filename)[1]
    
    print 'Call to %s on line %s of %s from line %s of %s' % \
        (func_name, func_line_no, func_filename,
         caller_line_no, caller_filename)
    return
#import pdb
if __name__ == '__main__':
#%%    
    root_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150512/'    
    base_name = 'Capture_Ch3_12052015_194303'

    #root_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150511/'
    #base_name = 'Capture_Ch1_11052015_195105'
    
    
    masked_image_file = root_dir + '/Compressed/' + base_name + '.hdf5'
    trajectories_file = root_dir + '/Trajectories/' + base_name + '_trajectories.hdf5'
    skeletons_file = root_dir + '/Trajectories/' + base_name + '_segworm.hdf5'
    video_save_dir = root_dir + '/Worm_Movies_new/' + base_name + os.sep
#%%  
    roi_size = 128
    min_mask_area = 50
    resampling_N = 50
    width_resampling = 7;
    
    is_draw_contour = True
    
    if is_draw_contour:
        if os.path.exists(video_save_dir):
            shutil.rmtree(video_save_dir)
        os.makedirs(video_save_dir)
    
    #pointer to the compressed videos
    mask_fid = h5py.File(masked_image_file, 'r');
    mask_dataset = mask_fid["/mask"]
    
    #get trajectories, threshold and indexes from the first part of the tracker
    segworm_df, worms_frame_range, tot_rows = \
    getSmoothTrajectories(trajectories_file, displacement_smooth_win = 101, min_displacement = 0, threshold_smooth_win = 501)
    
    
    #segworm_df = segworm_df.query('frame_number>43275')
    
    ske_file_id = h5py.File(skeletons_file, "w");
    
    data_strS = ['skeleton', 'contour_side1', 'contour_side2']
    worm_data = {}
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
    ske_file_id.create_dataset("/straighten_worm_intensity", (tot_rows, resampling_N, width_resampling), 
                                    dtype = "double", maxshape = (tot_rows, resampling_N, width_resampling), 
                                    chunks = (1, resampling_N, width_resampling),
                                    compression="gzip", 
                                    compression_opts=4,
                                    shuffle=True);
    
    #save the data from segworm_df
    for str_field in ['coord_x','coord_y','threshold']:
        ske_file_id.create_dataset("/" + str_field, data = segworm_df[str_field].values)
    
    for str_field in ['frame_number','plate_worm_id','worm_index_joined']:
        ske_file_id.create_dataset("/" + str_field, data = segworm_df[str_field].values.astype(np.int32))
    
    ske_file_id.create_dataset("/segworm_id", data = segworm_df.index.astype(np.int32))
        
    #dictionary to store previous skeletons
    prev_skeleton = {}

    video_list = {}
    
    #timer
    progressTime = timeCounterStr('Processing data.');
    for frame, frame_data in segworm_df.groupby('frame_number'):
        #if frame >500: break
        
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
                
                #this function is quite slow due to a 2D interpolation (RectBivariateSpline), 
                #but it might be useful for me in a further analysis of the image textures
                straighten_worm = getStraightenWormInt(worm_img, skeleton, cnt_widths)
                ske_file_id['/straighten_worm_intensity'][segworm_id, :, :]  = straighten_worm.T
                
                #save segwrom_results
                ske_file_id['/skeleton_length'][segworm_id] = ske_len 
                ske_file_id['/contour_side1_length'][segworm_id] = cnt_side1_len
                ske_file_id['/contour_side2_length'][segworm_id] = cnt_side2_len

                ske_file_id['/contour_width'][segworm_id, :] = cnt_widths                
                #convert into the main image coordinates
                ske_file_id['/skeleton'][segworm_id, :, :] = skeleton + roi_corner
                ske_file_id['/contour_side1'][segworm_id, :, :] = cnt_side1 + roi_corner
                ske_file_id['/contour_side2'][segworm_id, :, :] = cnt_side2 + roi_corner
            
            #add frame to worm video
            if is_draw_contour:
                if (worms_frame_range[worm_index][0] == frame) or (not worm_index in video_list):
                    movie_save_name = video_save_dir + ('worm_%i.avi' % worm_index)
                    #gray pixels if no contour is drawn
                    video_list[worm_index] = cv2.VideoWriter(movie_save_name, \
                    cv2.cv.FOURCC('M','J','P','G'), 25, (roi_size, roi_size), isColor=True)
                
                
                worm_rgb = drawWormContour(worm_img, worm_mask, skeleton, cnt_side1, cnt_side2)
                assert (worm_rgb.shape[0] == worm_img.shape[0]) and (worm_rgb.shape[1] == worm_img.shape[1]) 
                video_list[worm_index].write(worm_rgb)
                
            
                if (worms_frame_range[worm_index][1] == frame):
                    video_list[worm_index].release();
                    video_list.pop(worm_index, None)

        #if (frame-1 % 10000) == 0:
        #    #reopen hdf5 to avoid a buffer overflow  https://github.com/h5py/h5py/issues/480
        #    ske_file_id.close()
        #    ske_file_id = h5py.File(skeletons_file, "r+");
            
            
        if frame % 500 == 0:
            progress_str = progressTime.getStr(frame)
            print(base_name + ' ' + progress_str);
        

    
    ske_file_id.close()
    mask_fid.close()        