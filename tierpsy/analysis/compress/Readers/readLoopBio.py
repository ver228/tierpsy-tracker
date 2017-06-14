#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 16:56:55 2016

@author: ajaver
"""

class readLoopBio():
    def __init__(self, video_file):
        import imgstore

        self.vid = imgstore.new_for_filename(video_file)
        
        self.first_frame = self.vid.frame_min
        self.frame_max = self.vid.frame_max
        
        img, (frame_number, frame_timestamp) = self.vid.get_next_image()
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.dtype = img.dtype
        
        self.vid.close()
        self.vid = imgstore.new_for_filename(video_file)
        self.frames_read = []

    def read(self):
        if not self.frames_read or self.frames_read[-1][0] < self.frame_max:
            img, (frame_number, frame_timestamp) = self.vid.get_next_image()
            self.frames_read.append((frame_number, frame_timestamp))
            return 1, img
        else:
            return 0, None
    
    def release(self):
        return self.vid.close()
        