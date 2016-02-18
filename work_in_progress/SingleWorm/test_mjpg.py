# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 16:07:25 2016

@author: ajaver
"""


import cv2
import glob

main_dir = "/Volumes/D/Bertie/13_1/"

for file in glob.glob(main_dir + '*.mjpg'):
    vid = cv2.VideoCapture(video_file);
    ret, image = vid.read()
    print(image.shape)

 