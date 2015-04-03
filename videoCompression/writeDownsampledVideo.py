# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:23:09 2015

@author: ajaver
"""

import os
import h5py
import time
import datetime
import subprocess as sp
from parallelProcHelper import sendQueueOrPrint

def writeDownsampledVideo(masked_image_file, base_name = '', status_queue = '', save_video_file = -1, 
                          final_frame_size = 0, n_frames_jumped = 25):
    '''
    Write a downsampled video for visualization purposes from the processed hdf5 file. 
    The downsampling is both in scale (final_frame_size), and in time (n_frames_jumped)
    This function requires ffmpeg
    '''
                     
    #if no save_video_file is given, the save name will be derived from masked_image_file                         
    if save_video_file == -1:
        save_video_file = os.path.splitext(masked_image_file)[0] +  '_downsampled.avi';
    
    #open the hdf5 with masked data
    mask_fid = h5py.File(masked_image_file, "r");
    I_worms = mask_fid["/mask"]
    
    if final_frame_size == 0:
        final_frame_size = (I_worms.shape[-1]/4, I_worms.shape[-2]/4)
    
    #parameters to minimize the video using ffmpeg
    command = [ 'ffmpeg',
            '-y', # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-s', '%ix%i' % (I_worms.shape[-1], I_worms.shape[-2]), # size of one frame
            '-pix_fmt', 'gray',
            '-r', '25', # frames per second
            '-i', '-', # The imput comes from a pipe
            '-an', # Tells FFMPEG not to expect any audio
            '-vcodec', 'mjpeg',
            '-vf', 'scale=%i:%i' % final_frame_size,
            '-threads', '0',
            '-qscale:v', '0',
            save_video_file]
    
    
    devnull = open(os.devnull, 'w')  #use devnull to avoid printing the ffmpeg command output in the screen
    pipe = sp.Popen(command, stdin=sp.PIPE, stderr=devnull)
    
    #total number of frames.
    tot_frames = float(I_worms.shape[0])
    initial_time = fps_time = time.time()
    last_frame = 0;
    
    
    for frame_number in range(0,I_worms.shape[0],n_frames_jumped):
        pipe.stdin.write(I_worms[frame_number,:,:].tostring()) #write frame 
        
        #calculate progress
        if frame_number%1000 == 0:
            time_str = str(datetime.timedelta(seconds=round(time.time()-initial_time)))
            fps = (frame_number-last_frame+1)/(time.time()-fps_time)
            progress_str = 'Downsampling video. Total time = %s, fps = %2.1f; %3.2f%% '\
                % (time_str, fps, frame_number/tot_frames*100)
            
            sendQueueOrPrint(status_queue, progress_str, base_name);
            fps_time = time.time()
            last_frame = frame_number;

    #close files
    pipe.terminate()
    mask_fid.close()
    sendQueueOrPrint(status_queue, 'Downsampled video done.', base_name);