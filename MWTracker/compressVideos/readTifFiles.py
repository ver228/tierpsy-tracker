# -*- coding: utf-8 -*-
"""
Created on Tue 18 July 2016

@author: ljschumacher
"""
import cv2
import numpy as np
import os
from sys import exit
import glob

# http://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html
RETURN_AS_IT_IS = -1
RETURN_UINT8_GRAY = 0
RETURN_RGB = 1
    
class readTifFiles:
    """ Reads a single tif image, to be expanded for reading videos/stacks"""

    
    def __init__(self, dir_name, imread_flag=RETURN_UINT8_GRAY):
        self.imread_flag = imread_flag
        self.dir_name = dir_name
        if not os.path.exists(self.dir_name):
            raise FileNotFoundError('Error: Directory (%s) does not exist.' % self.dir_name)
            
        # make a list of the files belonging to this tif series
        self.files = glob.glob(os.path.join(self.dir_name, '*.tif'))
        # extract string list of filenumbers from list of files

        file_num_str = [os.path.split(x)[1].split(
            '_X')[1].split('.tif')[0] for x in self.files]
        # numerically sort list of file numbers
        self.dat_order = sorted([int(x) for x in file_num_str])
        # check in the indexes in the file order are continuous. The ordered
        # index should go 1, 2, 3, ...
        # This will throw and error if it is not the case
        assert all(np.diff(self.dat_order) == 1)


        # read the first image to determine width and height
        image = cv2.imread(self.files[0], self.imread_flag)
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.num_frames = len(self.dat_order)
        # initialize pointer for frames
        self.curr_frame = -1

    def read(self):
        self.curr_frame += 1
        if self.curr_frame < self.num_frames:
            filename = self.files[self.dat_order[self.curr_frame]]
            image = cv2.imread(filename, self.imread_flag)
            return (1, image)
        else:
            return(0, [])

    def release(self):
        pass
