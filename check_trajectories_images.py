# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 11:24:34 2015

@author: ajaver
"""

import h5py 
import tables
import matplotlib.pylab as plt



#maskFile = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch2_18022015_230213.hdf5';
#featuresFile = '/Users/ajaver/Desktop/Gecko_compressed/aFeatures_CaptureTest_90pc_Ch2_18022015_230213.hdf5';

maskFile = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch4_16022015_174636.hdf5';
featuresFile = '/Users/ajaver/Desktop/Gecko_compressed/bFeatures_CaptureTest_90pc_Ch4_16022015_174636.hdf5';

mask_fid = h5py.File(maskFile, 'r');
mask_dataset = mask_fid["/mask"]

feature_fid = tables.open_file(featuresFile, mode = 'r', title = '')
feature_table = feature_fid.get_node('/plate_worms')

ini_frame = 0
first_frame_indexes = [row['worm_index'] for row in feature_table.where('frame_number==%i' % ini_frame)]


all_data = []
for ind in first_frame_indexes:
    data = []
    for row in feature_table.where('worm_index==%i'% ind):
        data.append(row[:])
    all_data.append(data)

track_size = [len(x) for x in all_data]
print track_size

image = mask_dataset[ini_frame,:,:]
#%%
plt.figure()
plt.imshow(image, interpolation = 'none', cmap = 'gray')
for data_coord in all_data:
    data_coord = zip(*data_coord)
    #plt.figure()
    plt.plot(data_coord[2], data_coord[3], '-')
#%%
#plt.figure()
#plt.imshow(image, interpolation = 'none', cmap = 'gray')
for data_coord in all_data:
    data_coord = zip(*data_coord)
    plt.figure()
    plt.plot(data_coord[1], data_coord[4], '-')
    
#%%
import cv2
import numpy as np
def triangle_th(hist):
    #adapted from m-file in MATLAB central form: 
    #     Dr B. Panneton, June, 2010
    #     Agriculture and Agri-Food Canada
    #     St-Jean-sur-Richelieu, Qc, Canad
    #     bernard.panneton@agr.gc.ca
    
    #   Find maximum of histogram and its location along the x axis
    xmax = np.argmax(hist)

    #find first and last nonzero index
    ind = np.nonzero(hist)[0]
    fnz = ind[0];
    lnz = ind[-1];
    
    #   Pick side as side with longer tail. Assume one tail is longer.
    if lnz-xmax > xmax-fnz:
        hist = hist[::-1]
        a = hist.size - lnz;
        b = hist.size - xmax +1;
        isflip = True
    else:
        isflip = False;
        a = fnz;
        b = xmax;
    
    #   Compute parameters of the straight line from first non-zero to peak
    #   To simplify, shift x axis by a (bin number axis)
    m = hist[xmax]/(b-a);
    #   Compute distances
    x1 = np.arange((b-a));
    y1 = hist[x1+a];
    
    beta=y1+x1/m;
    x2=beta/(m+1/m);
    y2=m*x2;
    L= ((y2-y1)**2+(x2-x1)**2)**0.5;
    
    level = a + np.argmax(L)
    if isflip:
        level = hist.size - level
    return level
#%%
all_coord = []
for frame in range(3208, 3217):    
    #image = mask_dataset[frame,1750:2000,350:700]
    #Icrop = mask_dataset[frame,1750:2000,350:700]
    #Icrop = mask_dataset[frame,500:800,1300:1600]
    
    Icrop = mask_dataset[frame,1650:1800,1200:1350]
    
    hist = cv2.calcHist([Icrop],[0],None,[256],[0,256]).T[0]
    hist[0] = 0;
    level = triangle_th(hist)-10
    mask = cv2.threshold(Icrop, level, 1, cv2.THRESH_BINARY_INV)[1]
    mask[Icrop == 0] = 0;
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    #mask = cv2.erode(mask, np.ones((2,2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
    
    
    plt.figure()
    plt.imshow(Icrop, interpolation = 'none', cmap = 'gray')
    #plt.imshow(Icrop, interpolation = 'none', cmap = 'gray')
    
    
    [worm_contours, hierarchy]= cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area_max = 0;
    centroid = [];
    for worm_cnt in worm_contours:
        area = cv2.contourArea(worm_cnt)
        (CMx,CMy),(MA,ma),angle = cv2.minAreaRect(worm_cnt)
        plt.plot(CMx, CMy, '.r');
        
        if area > area_max:
            area_max = area;
            centroid = (CMx, CMy)
            
            
            rect = cv2.minAreaRect(worm_cnt)
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(Icrop,[box],0,(0,0,255),1)
    all_coord.append(centroid)
all_coord = np.array(all_coord)
            #plt.imshow(Icrop, interpolation = 'none', cmap = 'gray')
#%%
    
plt.figure()
plt.imshow(Icrop, interpolation = 'none', cmap = 'gray')            
plt.plot(all_coord[:,0], all_coord[:,1])

all_coord = np.array(all_coord)

delX = np.diff(all_coord[:,0])
delY = np.diff(all_coord[:,1])
R = np.sqrt(delX**2 + delY**2)
print R
#%%
plt.figure()
plt.plot(data_coord[2][3208:],data_coord[3][3208:], 'o')
plt.plot(all_coord[:,0]+1200, all_coord[:,1]+1650, '.g')
#    plt.figure()
#    plt.imshow(Icrop, interpolation = 'none', cmap = 'gray')
#    image = mask_dataset[frame,:,:]
#    plt.figure()
#    plt.imshow(image, interpolation = 'none', cmap = 'gray')
#    plt.ylim([1750,2000])
#    plt.xlim([350, 700])
#    

mask_fid.close()
feature_fid.close()
