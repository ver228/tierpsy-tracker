# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 15:36:44 2015

@author: ajaver
"""
import h5py
import matplotlib.pylab as plt
import cv2
#import subprocess
#import numpy as np


h5File = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/A002 - 20150116_140923D.hdf5';

fidh5 = h5py.File(h5File, "r")
bgnd = fidh5["/bgnd"];

fourcc = cv2.cv.CV_FOURCC(*'X264')
out = cv2.VideoWriter('test2.avi', -1, 25, (960,1280))#bgnd.shape[1:3]);

for frame in range(bgnd.shape[0]):
    #I = cv2.cvtColor(bgnd[frame,:,:], cv2.cv.CV_GRAY2RGB)
    #out.write(I);
    I = bgnd[frame,:,:]
    #cv2.imshow("live",I);
    out.write(I);
    print frame

plt.figure()
plt.imshow(bgnd[0,:,:])

plt.figure()
plt.imshow(bgnd[-1,:,:])


out.release()
fidh5.close()