# -*- coding: utf-8 -*-
"""
Created on Sat May 16 12:18:53 2015

@author: ajaver
"""
import pandas as pd
import numpy as np
import h5py
import cv2

import sys
sys.path.append('../videoCompression/')
from parallelProcHelper import timeCounterStr


masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Compressed/Capture_Ch1_11052015_195105.hdf5'
trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Trajectories/Capture_Ch1_11052015_195105_trajectories.hdf5'

def drawTrajectoriesVideo(masked_image_file, trajectories_file, max_track_draw = 100, n_frames_jumped = 25, movie_scale = 0.25):
    base_name = os.path.split(masked_image_file)[-1][:-5]
    
    #get id of trajectories with calculated skeletons
    table_fid = pd.HDFStore(trajectories_file, 'r');
    df = table_fid['/plate_worms'];
    df =  df[df['worm_index_joined'] > 0]
    #df = df[df[str_index]==4423] 
    
    good_index = df[df['segworm_id']>=0]['worm_index_joined'].unique();
    
    df = df[df['worm_index_joined'].isin(good_index)];
    table_fid.close()
    
    max_frame = df[['worm_index_joined', 'frame_number']].groupby('worm_index_joined').aggregate(np.max);
    last_frame = max_frame['frame_number'].max()
    #change into a dictionary for speed
    max_frame = max_frame.to_dict()['frame_number']
    
    
    mask_fid = h5py.File(masked_image_file, "r");
    I_worms = mask_fid["/mask"]
    
    track_list = {}
    track_colour = {};
    
    movie_save_name = trajectories_file[:-5] + '.avi';
    im_size = tuple([int(x*movie_scale) for x in I_worms.shape[1:]])
    video_writer = cv2.VideoWriter(movie_save_name, cv2.cv.FOURCC('M','J','P','G'), 25, \
                    im_size, isColor=True)
    
    progressTime = timeCounterStr('Writing trajectories videos.');
    for frame in range(0, last_frame, n_frames_jumped):
        frame_data = df[df['frame_number'] == frame]
        #clear track that has already ended
        for worm_ind in track_list.keys():
            if frame > max_frame[worm_ind]:
                track_list.pop(worm_ind)
                track_colour.pop(worm_ind)
        
        for  row_ind, worm_row in frame_data.iterrows():
            worm_ind = int(worm_row['worm_index_joined'])
            
            if not worm_ind in track_list:
                track_list[worm_ind] = [];
                track_colour[worm_ind] = np.random.randint(100, 255, 3)
                
            track_list[worm_ind].append((worm_row['coord_x'], worm_row['coord_y']))
            if len(track_list[worm_ind]) > max_track_draw:
                track_list[worm_ind].pop(0)
        
        img = I_worms[frame,:,:]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB);
    #%%    
        img = cv2.resize(img, (0,0), fx=movie_scale, fy=movie_scale);
        for worm_ind in track_list:
            pts = np.array(zip(track_list[worm_ind]))*movie_scale
            cv2.polylines(img, [pts.astype(np.int32)], False, track_colour[worm_ind], thickness=2, lineType = 8) 
        video_writer.write(img)
    #%%    
        if frame % 1000 == 0:
            progress_str = progressTime.getStr(frame)
            print(base_name + ' ' + progress_str);
            
    video_writer.release()