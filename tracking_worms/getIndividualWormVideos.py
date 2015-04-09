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
#import statsmodels.api as sm
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import numpy as np

import sys
sys.path.append('../videoCompression/')
from parallelProcHelper import sendQueueOrPrint, timeCounterStr

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


def getIndividualWormVideos(masked_image_file, trajectories_file, \
segworm_file, video_save_dir, max_frame_number = -1, \
smooth_window_size = 101, roi_size = 128, movie_scale = 1, \
is_draw_contour = False, status_queue = '', base_name = '', \
colorpalette = [(27, 158, 119), (217, 95, 2), (231, 41, 138)]):
    #Colormap from colorbrewer Dark2 5 - [0, 1, 3]    
    if not os.path.exists(video_save_dir):
        os.mkdir(video_save_dir)
#%%
    #get id of trajectories with calculated skeletons
    table_fid = pd.HDFStore(trajectories_file, 'r');
    df = table_fid['/plate_worms'];
    df =  df[df['worm_index_joined'] > 0]
    good_index = df[df['segworm_id']>=0]['worm_index_joined'].unique();
    df = df[df.worm_index_joined.isin(good_index)];
    table_fid.close()
    
    if len(df) == 0: #no valid trajectories identified
        return;
    
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
    
    #hear a saved the id of the videos being recorded
    video_list = {};
    
    progressTime = timeCounterStr('Writing individual worm videos.');
    
    for frame in range(0,max_frame_number):
        
        img = mask_dataset[frame,:,:]
        
        index2search = []
        for worm_index in smoothed_CM:
            if (frame >= smoothed_CM[worm_index]['first_frame']) and \
            (frame <= smoothed_CM[worm_index]['last_frame']):
                index2search.append(worm_index)
        
        wormsInFrame = df[df['frame_number']==frame]
        for worm_index in index2search:
            
            if frame == smoothed_CM[worm_index]['first_frame']:
                #intialize figure and movie recorder
                video_list[worm_index] = {};
                movie_save_name = video_save_dir + ('worm_%i.avi' % worm_index)
                
                #gray pixels if no contour is drawn
                pix_fmt = 'rgb24' if is_draw_contour else 'gray';
                
                video_list[worm_index]['writer'] = \
                writeVideoffmpeg(movie_save_name, width = roi_size*movie_scale, \
                height = roi_size*movie_scale, pix_fmt = pix_fmt)
        
            #obtain bounding box from the trajectories
            ind = int(frame-smoothed_CM[worm_index]['first_frame'])
            range_x = np.round(smoothed_CM[worm_index]['coord_x'][ind]) + [-roi_size/2, roi_size/2]
            range_y = np.round(smoothed_CM[worm_index]['coord_y'][ind]) + [-roi_size/2, roi_size/2]
            
            if range_x[0]<0: range_x -= range_x[0]
            if range_y[0]<0: range_y -= range_y[0]
            
            if range_x[1]>img.shape[1]: range_x += img.shape[1]-range_x[1]-1
            if range_y[1]>img.shape[0]: range_y += img.shape[0]-range_y[1]-1
            
            worm_img =  img[range_y[0]:range_y[1], range_x[0]:range_x[1]]
            
            if not is_draw_contour:
                video_list[worm_index]['writer'].write(worm_img)
            else:
                worm = wormsInFrame[wormsInFrame['worm_index_joined'] == worm_index]
                worm = worm.head(1); #in case duplicated values...                
                threshold = worm['threshold'].values
                segworm_id = worm['segworm_id'].values
            
                worm_rgb= cv2.cvtColor(worm_img, cv2.COLOR_GRAY2RGB);
            
                if (len(threshold) == 1) and (segworm_id < 0):
                    worm_mask = ((worm_img<threshold)&(worm_img!=0)).astype(np.uint8)        
                    worm_mask = cv2.morphologyEx(worm_mask, cv2.MORPH_CLOSE,np.ones((3,3)))
                    worm_rgb[:,:,1][worm_mask!=0] = 150
                else:
                    pass
#                    worm_rgb[0:2,0:2,0] = 255            
#                    worm_rgb[0:2,0:2,1] = 0
#                    worm_rgb[0:2,0:2,2] = 0

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
                
                
                video_list[worm_index]['writer'].write(worm_rgb)
    
        if frame == smoothed_CM[worm_index]['last_frame']:
            video_list[worm_index]['writer'].release()    
     
            
        if frame%25 == 0:
            progress_str = progressTime.getStr(frame)
            sendQueueOrPrint(status_queue, progress_str, base_name);
            
    for worm_index in video_list:
        video_list[worm_index]['writer'].release()  
        
    mask_fid.close()
    if is_draw_contour:
        segworm_fid.close()


if __name__ == '__main__':
#    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Capture_Ch1_23032015_111907.hdf5'
#    trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/trajectories/Capture_Ch1_23032015_111907.hdf5'
#    segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/trajectories/Capture_Ch1_23032015_111907_segworm.hdf5'
#    video_save_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150323/trajectories/worms_Ch1_kezhia/'
    
    masked_image_file = r'/Users/ajaver/Desktop/sygenta/Compressed/data_20150114/compound_a_repeat_2_fri_5th_dec.hdf5'
    trajectories_file = r'/Users/ajaver/Desktop/sygenta/Trajectories/data_20150114/compound_a_repeat_2_fri_5th_dec_trajectories.hdf5'
    segworm_file = r'/Users/ajaver/Desktop/sygenta/Trajectories/data_20150114/compound_a_repeat_2_fri_5th_dec_segworm.hdf5'
    video_save_dir = r'/Users/ajaver/Desktop/sygenta/Worm_Movies/data_20150114/compound_a_repeat_2_fri_5th_dec/'

    
    getIndividualWormVideos(masked_image_file, trajectories_file, \
    segworm_file, video_save_dir, is_draw_contour = True)
    
    #video_save_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150323/trajectories/worms_Ch1a/'
    
    #getIndividualWormVideos(masked_image_file, trajectories_file, \
    #segworm_file, video_save_dir, is_draw_contour = True, max_frame_number = 1000)