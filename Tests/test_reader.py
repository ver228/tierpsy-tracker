# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import time

from MWTracker.compressVideos.readTifFiles import readTifFiles, RETURN_AS_IT_IS, RETURN_UINT8_GRAY
#from MWTracker.compressVideos.compressVideo import normalizeImage

# from normalizeImage import normalizeImage
import numexpr as ne

#from numba import jit
#@jit
def normalizeImage_ne(img):
    # normalise image intensities if the data type is other
    # than uint8
    imax = img.max()
    imin = img.min()
    factor = 255/(imax-imin)
    
    imgN = ne.evaluate('(img-imin)*factor')
    imgN = imgN.astype(np.uint8)

    return imgN, (imin, imax) 

def normalizeImage_python(img):
    # normalise image intensities if the data type is other
    # than uint8
    imax = img.max()
    imin = img.min()
    factor = 255/(imax-imin)
    
    #imgN = ne.evaluate('(img-imin)*factor')
    imgN = (img-imin)*factor
    imgN = imgN.astype(np.uint8)

    return imgN, (imin, imax) 

directory_name = r'E:\\28.6.16 recording 8\\recording 8.1\\8.1 TIFF\\'
# directory_name = '/Users/ajaver/Desktop/Videos/tiffs/'


tic = time.time()
reader = readTifFiles(directory_name, RETURN_AS_IT_IS)
for n in range(100):
    ret, img = reader.read()
print('Read only', time.time() - tic)

tic = time.time()
reader = readTifFiles(directory_name, RETURN_UINT8_GRAY)
for n in range(100):
    ret, img = reader.read()
print('Read norm', time.time() - tic)

tic = time.time()
reader = readTifFiles(directory_name, RETURN_AS_IT_IS)
for n in range(100):
    ret, img = reader.read()
    img_N = normalizeImage_python(img)
print('python', time.time() - tic)

tic = time.time()
reader = readTifFiles(directory_name, RETURN_AS_IT_IS)
for n in range(100):
    ret, img = reader.read()
    img_N = normalizeImage_ne(img)
print('numexpr', time.time() - tic)

# tic = time.time()
# reader = readTifFiles(directory_name)
# for n in range(100):
#     ret, img = reader.read()
#     img_N = normalizeImage(img)
# print('cython', time.time() - tic)
