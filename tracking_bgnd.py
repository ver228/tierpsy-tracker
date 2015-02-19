# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 16:08:07 2015

@author: ajaver
"""

import sqlite3 as sql
import matplotlib.pylab as plt

import os
import errno
import cv2
import numpy as np
import h5py

fileName = '/Volumes/ajaver$/DinoLite/Videos/Exp5-20150116/A002 - 20150116_140923.wmv';
saveDir = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/';
saveFile = "A002 - 20150116_140923H.hdf5"


BUFF_SIZE = 30;
BUFF_DELTA = 1200;
BGND_ORD = 0.7 #percentage in the ordered buffer that correspond to the background (set 0.5 for the median)
BGND_IND = round(BGND_ORD*BUFF_SIZE)-1;

INITIAL_FRAME =  25e6;
TOT_CHUNKS = 50;
#make the save directory if it didn't exist before
if not os.path.exists(saveDir):
    try:
        os.makedirs(saveDir)
    #throw an exeption if the directory didn't exist
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(saveDir):
            pass

#create a buffer ring to calculate the background
class imRingBuffer:
    def __init__(self, height, width, buff_size, data_type = np.uint8):
        self.buffer = np.zeros((buff_size, height, width), np.uint8)
        self.index = 0;
        self.buff_size = buff_size;
        
    def add(self, new_image):
        if self.index >= self.buff_size:
            self.index = 0;
        
        self.buffer[self.index,:,:] = new_image;
        self.index += 1;


vid = cv2.VideoCapture(fileName)
im_width = vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
im_height = vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
im_buffer = imRingBuffer(height = im_height, width = im_width, buff_size = BUFF_SIZE);


tot_frames = vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)#this normally does not work very good


#Initialize the opening of the hdf5 file, and delete the dataset bgnd there was some data previously
fidh5 = h5py.File(saveDir + saveFile, "w")
bgnd_set = fidh5.create_dataset("/bgnd", (0, im_height, im_width), 
                                dtype = "u1", 
                                chunks = (1, im_height, im_width), 
                                maxshape = (None, im_height, im_width), 
                                compression="gzip");
bgnd_set.attrs['buffer_size'] = BUFF_SIZE
bgnd_set.attrs['delta_btw_bgnd'] = BUFF_DELTA
bgnd_set.attrs['bgnd_buffer_order'] = BGND_ORD
bgnd_set.attrs['initial_frame'] = INITIAL_FRAME
bgnd_set.attrs['video_source_file'] = fileName

tot_bgnd = 0;
N_chunks = 0;
while N_chunks < TOT_CHUNKS:
    vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, INITIAL_FRAME + N_chunks*BUFF_DELTA);
    
    retval, image = vid.read()
    if not retval:
        break;    
    image = cv2.cvtColor(image, cv2.cv.CV_RGB2GRAY).astype(np.double);
    im_buffer.add(image);
    
    #for kk in range(BUFF_DELTA-1):   
    #    retval, image = vid.read()
    
    if N_chunks >=BUFF_SIZE:
        # calculate background only once the buffer had been filled
        sorted_buff = np.sort(im_buffer.buffer,0);
        bgnd_set.resize(tot_bgnd+1, axis=0); 
        bgnd_set[tot_bgnd,:,:] = sorted_buff[BGND_IND,:,:];
        tot_bgnd += 1
    
    N_chunks += 1;
    print N_chunks;
#%%


vid.release()
fidh5.close()


#for bgnd in allBgnd:
#    plt.figure();
#    fig = plt.imshow(bgnd);
#    fig.set_cmap('gray');
#    fig.set_interpolation('none');    
#%%
#vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES);
#retval, image = vid.read()
#
#plt.imshow(image)

#    
#    
#        
#
#DBName = 'A001_results.db';
#conn = sql.connect(saveDir + DBName)
#
#conn.close()
