# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:57:54 2016

@author: ajaver
"""
import numpy as np
import json
import tables

import os
import subprocess as sp
from collections import OrderedDict

import cv2
import matplotlib.pylab as plt
import numpy as np

import sys
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking')
from MWTracker.compressVideos.readVideoffmpeg import readVideoffmpeg

def dict2recarray(dat):
    '''convert into recarray (pytables friendly)'''
    dtype = [(kk, dat[kk].dtype) for kk in dat]
    N = len(dat[dtype[0][0]])

    recarray = np.recarray(N, dtype)
    for kk in dat: recarray[kk] = dat[kk]
    
    return recarray

def ffprobeReadMetadata(json_file):
    #%%
    with open(json_file) as fid:
        dat = json.load(fid)
    
    if not dat: raise BufferError(buff_err)
    
    #use the first frame as reference
    frame_fields = list(dat['frames'][0].keys())
    
    #consider data where the best_effort_timestamp was not calculated
    valid_frames = [x for x in dat['frames'] if all(ff in x for ff in frame_fields)]
    
    video_metadata = OrderedDict()
    for field in frame_fields:
        video_metadata[field] = [frame[field] for frame in valid_frames]

        try: #if possible convert the data into float
            video_metadata[field] = [float(dd) for dd in video_metadata[field]]
        except ValueError:
            #pytables does not support unicode strings (python3)
            video_metadata[field] = [bytes(dd, 'utf-8') for dd in video_metadata[field]]
        video_metadata[field] = np.asarray(video_metadata[field])
    
    video_metadata = dict2recarray(video_metadata)
#%%
    return video_metadata

def getValidTimestamp(best_effort_timestamp, best_effort_timestamp_time):
    
    #delta from the best effort indexes
    xx = np.diff(best_effort_timestamp);
    good_N = xx != 0    
    delta_x = np.median(xx[good_N])
    
    #delta from the best effort times
    xx_t = np.diff(best_effort_timestamp_time)
    good_Nt = xx_t != 0
    delta_t = np.median(xx_t[good_Nt])
    
    #test that the zero delta from the index and time are the same
    assert np.all(good_N == good_Nt)
    
    #check that the normalization factors make sense
    xx_N = np.round(xx/delta_x).astype(np.int)
    assert np.all(xx_N == np.round(xx_t/delta_t).astype(np.int))
    
    
    #get the indexes with a valid frame
    timestamp = np.arange(1,len(xx_N)+1) #add one to consider compensate for the np.diff
    timestamp = timestamp[good_N] + xx_N[good_N]-1
    timestamp = np.hstack((0,timestamp))
    
    
    timestamp_time = timestamp*delta_t
    
    return timestamp, timestamp_time

if __name__ == '__main__':
    #video_file = '/Users/ajaver/Desktop/Videos/test_timeframe/Test_Bertie.mjpg'
    video_file = '/Users/ajaver/Desktop/Videos/test_timeframe/bad.avi'
    json_file = video_file.rsplit('.')[0] + '.json'
    json_cmd = 'ffprobe -v error -show_frames -print_format json %s > %s' % (video_file, json_file)

    print(json_cmd)
    dat = ffprobeReadMetadata(json_file)
    for rec in dat.dtype.names:
        if dat[rec].dtype == '<f8':
            continue                       
            xx = np.diff(dat[rec])
            if not np.all(xx[0] == xx): 
                plt.figure()
                plt.plot(xx)
                plt.title(rec)
                
                print(rec, np.mean(np.abs(xx)))
        else:
            assert np.all(dat['media_type'][0] == dat['media_type'])
    #%%
    op_vid_frame_pos = []
    op_vid_time_pos = []
    
    #vid_ff = readVideoffmpeg(video_file);
    vid_op = cv2.VideoCapture(video_file);
    ii = 0
    while 1:
        op_vid_frame_pos.append(int(vid_op.get(cv2.CAP_PROP_POS_FRAMES)))
        op_vid_time_pos.append(vid_op.get(cv2.CAP_PROP_POS_MSEC))
        
        #ret_ff, image_ff = vid_ff.read() #get video frame, stop program when no frame is retrive (end of file)
        ret_op, image_op = vid_op.read()
        
        if ret_op == 0 :#and ret_ff == 0:
            break
                
        image_op = cv2.cvtColor(image_op, cv2.COLOR_RGB2GRAY)  
        
        
        print(ii)
        ii += 1
        
        #assert np.all(image_ff == image_op)
    
    #print(len(vid_ff.vid_frame_pos), len(vid_ff.vid_time_pos))
    
    #assert np.array(vid_ff.vid_frame_pos) == np.array(op_vid_frame_pos)
    best_effort_timestamp = dat['best_effort_timestamp']
    best_effort_timestamp_time = dat['best_effort_timestamp_time']

    timestamp, timestamp_time = getValidTimestamp(best_effort_timestamp, best_effort_timestamp_time)


#%%
#xx_N = xx_N[xx_N != 0]
   