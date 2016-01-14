# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:20:13 2016

@author: ajaver
"""

import cv2

video_file = "/Users/ajaver/Desktop/Videos/Check_Align_samples/Videos/npr-13 (tm1504)V on food L_2010_01_25__11_56_02___4___2.avi"

vid = cv2.VideoCapture(video_file);


data = []
ret = 1
while ret !=0:
    a = vid.get(cv2.CAP_PROP_POS_MSEC)
    b = vid.get(cv2.CAP_PROP_POS_AVI_RATIO)
    c = vid.get(cv2.CAP_PROP_FPS)
    d = vid.get(cv2.CAP_PROP_FOURCC)
    e = vid.get(cv2.CAP_PROP_POS_FRAMES)
    
    ret, image = vid.read()
    print(e)
    data.append((a,b,c,d,e))

import matplotlib.pylab as plt