# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import time
from MWTracker.compressVideos.readTifFiles import readTifFiles
#from MWTracker.compressVideos.compressVideo import normalizeImage

import numexpr as ne

#from numba import jit
#@jit
def normalizeImage(img):
    # normalise image intensities if the data type is other
    # than uint8
    imax = img.max()
    imin = img.min()
    factor = 255/(imax-imin)
    
#    imgN = (img-imax)*factor
#    imgN = imgN.astype(np.uint8)

    return img, (imin, imax) 

directory_name = r'E:\\28.6.16 recording 8\\recording 8.1\\8.1 TIFF\\'


reader = readTifFiles(directory_name)

tic = time.time()
for n in range(200):
    ret, img = reader.read()
    
    alpha = np.min(img)
    beta = np.max(img)
    
    img_N = normalizeImage(img)
#    img = img.astype(np.float32)
#    img_N =  np.zeros_like(img)
#    cv2.normalize(img,img_N, beta, alpha)
    if n % 100 == 0:
        print(n, time.time() - tic)
        tic = time.time()
        