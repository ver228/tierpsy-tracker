# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:03:37 2015

@author: ajaver
"""
import cv2
import glob
import sys
import numpy as np
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking/')

from MWTracker.compressVideos.readVideoffmpeg import readVideoffmpeg

main_dir = '/Volumes/behavgenom$/Silvana/18Nov15/'

video_files = glob.glob(main_dir + '*.mjpg')

tot_vid = len(video_files)

bad_img = np.zeros((2048,2048), dtype = np.uint8)


for ivid, video_file in enumerate(video_files):
    print('%i of %i' % (ivid+1, tot_vid))
    
    image_file = video_file.replace('.mjpg', '.jpg')
    
    try:    
        vid = readVideoffmpeg(video_file);
        ret, image = vid.read()
        vid.release()
    except:
        image = bad_img
        print(video_file)
    cv2.imwrite(image_file, image)
    #print(image.shape)
    