# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 16:08:07 2015

@author: ajaver
"""
import h5py
import matplotlib.pylab as plt
import numpy as np
from math import floor
import time
import cv2
from skimage.measure import label, regionprops

from skimage.filter import threshold_otsu
from scipy.ndimage.filters import minimum_filter, median_filter
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage as nd


def triangle_th(hist):
    # adapted from m-file in MATLAB central form:
    #     Dr B. Panneton, June, 2010
    #     Agriculture and Agri-Food Canada
    #     St-Jean-sur-Richelieu, Qc, Canad
    #     bernard.panneton@agr.gc.ca

    #   Find maximum of histogram and its location along the x axis
    xmax = np.argmax(hist)

    # find first and last nonzero index
    ind = np.nonzero(hist)[0]
    fnz = ind[0]
    lnz = ind[-1]

    #   Pick side as side with longer tail. Assume one tail is longer.
    if lnz - xmax > xmax - fnz:
        hist = hist[::-1]
        a = hist.size - lnz
        b = hist.size - xmax + 1
        isflip = True
    else:
        isflip = False
        a = fnz
        b = xmax

    #   Compute parameters of the straight line from first non-zero to peak
    #   To simplify, shift x axis by a (bin number axis)
    m = hist[xmax] / (b - a)

    #   Compute distances
    x1 = np.arange((b - a))
    y1 = hist[x1 + a]

    beta = y1 + x1 / m
    x2 = beta / (m + 1 / m)
    y2 = m * x2
    L = ((y2 - y1)**2 + (x2 - x1)**2)**0.5

    level = a + np.argmax(L)
    if isflip:
        level = hist.size - level
    return level

maskFile = '/Volumes/ajaver$/GeckoVideo/Compressed/CaptureTest_90pc_Ch1_16022015_174636.hdf5'

mask_fid = h5py.File(maskFile, 'r')

#full_dataset = mask_fid["/full_data"]
mask_dataset = mask_fid["/mask"]

# for ii, kk in enumerate(range(0, mask_dataset.shape[0], full_dataset.attrs['save_interval'])):
#    if ii%3 == 0:
#        f, (ax1, ax2) = plt.subplots(1, 2)
#        ax1.imshow(full_dataset[ii,:,:], cmap = 'gray', interpolation = 'none' )
#        ax2.imshow(mask_dataset[kk,:,:], cmap = 'gray', interpolation = 'none' )

#%%
# plt.imshow()
#plt.imshow(mask_dataset[kk,:,:], cmap = 'gray', interpolation = 'none' )

image = mask_dataset[300000, :, :]
# mask = cv2.adaptiveThreshold(image, 1,
# cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,61,15)
L, L_num = label(image != 0, return_num=True)

props = regionprops(L)


for ii in range(len(props)):
    bb = props[ii].bbox
    Icrop = image[bb[0]:bb[2], bb[1]:bb[3]]

    hist = cv2.calcHist([Icrop], [0], None, [256], [0, 256]).T[0]
    hist[0] = 0
    level = triangle_th(hist)
    mask = cv2.threshold(Icrop, level, 1, cv2.THRESH_BINARY_INV)[1]
    mask[Icrop == 0] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    plt.figure()
    # plt.imshow(Icrop)
    plt.imshow(Icrop * mask, interpolation='none', cmap='gray')
