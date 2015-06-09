# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 00:31:30 2015

@author: ajaver
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:16:47 2015

@author: ajaver
"""
import os
import subprocess as sp
import h5py
import pandas as pd
import cv2
import datetime
import time
#import statsmodels.api as sm
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import numpy as np

import sys
sys.path.append('../videoCompression/')
from parallelProcHelper import sendQueueOrPrint

class writeVideoffmpeg:
    def __init__(self, file_name, width = 100, height = 100, pix_fmt = 'gray'):
        command = [ 'ffmpeg',
        '-y', # (optional) overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', '%ix%i' % (width,height), # size of one frame
        '-pix_fmt', pix_fmt,
        '-r', '25', # frames per second
        '-i', '-', # The imput comes from a pipe
        '-an', # Tells FFMPEG not to expect any audio
        '-vcodec', 'mjpeg',
        '-threads', '0',
        '-qscale:v', '0',
        file_name]
        
        devnull = open(os.devnull, 'w')  #use devnull to avoid printing the ffmpeg command output in the screen
        self.pipe = sp.Popen(command, stdin=sp.PIPE, stderr=devnull)
        
    def write(self, image):
        self.pipe.stdin.write(image.tostring())
        #self.pipe.stdin.write(image)
    
    def release(self):
        self.pipe.terminate()

class timeCounterStr_ver2:
    def __init__(self, task_str, tot_rows):
        self.initial_time = time.time();
        self.current_row = 0;
        self.task_str = task_str;
        self.rps_time = float('nan');
        self.tot_rows = float(tot_rows);
        
    def getStr(self, n_row_added):
        #calculate the progress and put it in a string
        time_str = str(datetime.timedelta(seconds = round(time.time()-self.initial_time)))
        rps = (n_row_added)/(time.time()-self.rps_time)
        self.current_row += n_row_added
        percentage = self.current_row/self.tot_rows*100
        progress_str = '%s %2.1f%%, total time = %s, rows per second = %2.1f; '\
            % (self.task_str, percentage, time_str, rps)
        self.rps_time = time.time()
        return progress_str;

if __name__ == '__main__':
#    masked_movies_dir = sys.argv[1]
#    trajectories_dir = sys.argv[2]
#    base_name = sys.argv[3]
#    main_video_save_dir = sys.argv[4]
    
    masked_movies_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150323/'
    trajectories_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/'
    base_name = 'CaptureTest_90pc_Ch1_02022015_141431'
    main_video_save_dir = r'/Users/ajaver/Desktop/Gecko_compressed/20150323/Worm_Movies/'
    
    masked_image_file = masked_movies_dir + base_name + '.hdf5'
    trajectories_file = trajectories_dir + base_name + '_trajectories.hdf5'
    segworm_file = trajectories_dir + base_name + '_segworm.hdf5'
    video_save_dir = main_video_save_dir + base_name + os.sep    
#    video_save_dir_gray = main_video_save_dir + base_name + '_gray' + os.sep

     #create movies of individual worms
    
#    getIndividualWormVideos(masked_image_file, trajectories_file, \
#    segworm_file, video_save_dir, is_draw_contour = True, max_frame_number = -1,\
#    base_name = base_name)



#def getIndividualWormVideos(masked_image_file, trajectories_file, \
#segworm_file, video_save_dir, max_frame_number = -1, \
#smooth_window_size = 101, roi_size = 128, movie_scale = 1, \
#is_draw_contour = False, status_queue = '', base_name = '', \
#colorpalette = [(27, 158, 119), (217, 95, 2), (231, 41, 138)]):
    
    max_frame_number = -1
    smooth_window_size = 101
    roi_size = 128
    movie_scale = 1
    is_draw_contour = True 
    status_queue = ''
    colorpalette = [(27, 158, 119), (217, 95, 2), (231, 41, 138)]
    #Colormap from colorbrewer Dark2 5 - [0, 1, 3]    
    if not os.path.exists(masked_image_file) or \
    not os.path.exists(segworm_file) or \
    not os.path.exists(trajectories_file): 
        print('Individual Worm Videos Failed. Some or the files were not found. Nothing to do here.')
        pass;
        
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir)
#%%
    #get id of trajectories with calculated skeletons
    table_fid = pd.HDFStore(trajectories_file, 'r');
    df = table_fid['/plate_worms'];
    df =  df[df['worm_index_joined'] > 0]
     
    good_index = df[df['segworm_id']>=0]['worm_index_joined'].unique();
    df = df[df.worm_index_joined.isin(good_index)];
    table_fid.close()
    
    if len(df) == 0: #no valid trajectories identified
        pass;
    
    #get the last frame to be included in the video
    last_frame = df['frame_number'].max();
    if max_frame_number <0 or max_frame_number > last_frame:
        max_frame_number = last_frame;
        
    #print max_frame_number
#%%    
    #smooth trajectories (reduce jiggling from the CM to obtain a nicer video)
    smoothed_CM = {};
    for worm_index in good_index:
        dat = df[df['worm_index_joined']==worm_index][['coord_x', 'coord_y', 'frame_number']]
        x = np.array(dat['coord_x']);
        y = np.array(dat['coord_y']);
        t = np.array(dat['frame_number']);
        
        
        first_frame = np.min(t);
        last_frame = np.max(t);
        tnew = np.arange(first_frame, last_frame+1);
        if len(tnew) <= smooth_window_size:
            continue
        
        fx = interp1d(t, x)
        xnew = savgol_filter(fx(tnew), smooth_window_size, 3);
        fy = interp1d(t, y)
        ynew = savgol_filter(fy(tnew), smooth_window_size, 3);
        
        smoothed_CM[worm_index] = {}
        smoothed_CM[worm_index]['coord_x'] = xnew
        smoothed_CM[worm_index]['coord_y'] = ynew
        smoothed_CM[worm_index]['first_frame'] = first_frame
        smoothed_CM[worm_index]['last_frame'] = last_frame
    
    #open the file with the masked images
    mask_fid = h5py.File(masked_image_file, 'r');
    mask_dataset = mask_fid["/mask"]
    
    #open the file with the skeleton data
    if is_draw_contour:
        segworm_fid = h5py.File(segworm_file, 'r')
    
    progressTime = timeCounterStr_ver2('Writing individual worm videos.',  len(df));
    for worm_index in smoothed_CM:
        movie_save_name = video_save_dir + ('worm_%i.avi' % worm_index)
        pix_fmt = 'rgb24' if is_draw_contour else 'gray';
        #video_writer = writeVideoffmpeg(movie_save_name, width = roi_size*movie_scale, \
        #        height = roi_size*movie_scale, pix_fmt = pix_fmt)
        video_writer = cv2.VideoWriter(movie_save_name, cv2.cv.FOURCC('M','J','P','G'), 25, (roi_size*movie_scale, roi_size*movie_scale), isColor=is_draw_contour)
        
        worm_dat = df[df['worm_index_joined']==worm_index]
        
        for frame in range(smoothed_CM[worm_index]['first_frame'], smoothed_CM[worm_index]['last_frame']+1):
            #obtain bounding box from the trajectories
            ind = int(frame-smoothed_CM[worm_index]['first_frame'])
            range_x = np.round(smoothed_CM[worm_index]['coord_x'][ind]).astype(np.int) + [-roi_size/2, roi_size/2]
            range_y = np.round(smoothed_CM[worm_index]['coord_y'][ind]).astype(np.int) + [-roi_size/2, roi_size/2]
            
            if range_x[0]<0: range_x -= range_x[0]
            if range_y[0]<0: range_y -= range_y[0]
            
            if range_x[1]>mask_dataset.shape[2]: range_x += mask_dataset.shape[2]-range_x[1]-1
            if range_y[1]>mask_dataset.shape[1]: range_y += mask_dataset.shape[1]-range_y[1]-1
            
            worm_img = mask_dataset[frame, range_y[0]:range_y[1], range_x[0]:range_x[1]]
        
            if not is_draw_contour:
                video_writer.write(worm_img)
            else:
                worm_row = worm_dat[worm_dat['frame_number']==frame]
                threshold = worm_row['threshold'].values
                segworm_id = worm_row['segworm_id'].values

                worm_rgb= cv2.cvtColor(worm_img, cv2.COLOR_GRAY2RGB);
                if (len(threshold) == 1) and (segworm_id < 0):
                    worm_mask = ((worm_img<threshold)&(worm_img!=0)).astype(np.uint8)        
                    worm_mask = cv2.morphologyEx(worm_mask, cv2.MORPH_CLOSE,np.ones((3,3)))
                    worm_rgb[:,:,1][worm_mask!=0] = 150

                worm_rgb = cv2.resize(worm_rgb, (0,0), fx=movie_scale, fy=movie_scale);
                
                if (len(segworm_id)==1) and (segworm_id >= 0):
                    for ii, key in enumerate(['contour_ventral', 'skeleton', 'contour_dorsal']):
                        dat = np.squeeze(segworm_fid['/segworm_results/' + key][segworm_id,:,:]);
    
                        xx = (dat[1,:]-range_x[0])*movie_scale;
                        yy = (dat[0,:]-range_y[0])*movie_scale;
                        pts = np.round(np.vstack((xx,yy))).astype(np.int32).T
                        cv2.polylines(worm_rgb, [pts], False, colorpalette[ii], thickness=1, lineType = 8)
    
                    cv2.circle(worm_rgb, tuple(pts[0]), 2, (225,225,225), thickness=-1, lineType = 8)
                    cv2.circle(worm_rgb, tuple(pts[0]), 3, (0,0,0), thickness=1, lineType = 8)
                #write frame to video
                video_writer.write(worm_rgb)
    
        #close video
        video_writer.release()    
     
        progress_str = progressTime.getStr( len(dat))
        sendQueueOrPrint(status_queue, progress_str, base_name);
            
    mask_fid.close()
    if is_draw_contour:
        segworm_fid.close()

