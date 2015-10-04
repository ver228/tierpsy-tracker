# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 00:32:57 2015

@author: ajaver
"""

import tables
import matplotlib.pylab as plt
import numpy as np

filename = '/Users/ajaver/Desktop/CSTCTest_Ch1_02102015_122155.hdf5'

#with tables.FILE(filename, 'r') as ff:
ff =  tables.File(filename, 'r')
masks = ff.get_node('/mask') 

full_img = masks[0,:,:]



curr_img = masks[100,:,:]
mask_bw = curr_img == 0
curr_img[mask_bw] = full_img[mask_bw]



#dat = masks[:, 0:20, :]
#dd = np.median(dat, (1,2))
#plt.plot(dd, '.')

class readVideoHDF5:
    '''
    Read video frame using ffmpeg. Assumes 8bits gray video.
    Requires that ffmpeg is installed in the computer.
    This class is an alternative of the captureframe of opencv since:
    -> it can be a pain to compile opencv with ffmpeg compability. 
    -> this funciton is a bit faster (less overhead), but only works with gecko's mjpeg 
    '''
    def __init__(self, fileName, full_img_period = np.inf):
        self.fid = tables.File(fileName, 'r')
        self.dataset = self.fid.get_node('/mask') 
        
        self.tot_frames = self.dataset.shape[0]
        
        self.width = self.dataset.shape[1]
        self.height = self.dataset.shape[2]
        self.tot_pix = self.height*self.width
        
        #initialize pointer for frames
        self.curr_frame = -1
        
        #how often we get a full frame
        self.full_img_period = full_img_period
        
        
    def read(self):
        self.curr_frame += 1
        if self.curr_frame % self.full_img_period:
            self.full_img = self.dataset[self.curr_frame, :, :]
        
        if self.curr_frame < self.tot_frames:
            image = self.dataset[self.curr_frame, :, :]
            mask_bw = image == 0
            image[mask_bw] = self.full_img[mask_bw]
            return (1, image)
        else:
            return (0, [])
        
    
    def release(self):
        #close the buffer
        self.fid.close()