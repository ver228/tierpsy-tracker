# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 16:08:07 2015

@author: ajaver
"""
import h5py
import matplotlib.pylab as plt
import numpy as np
import tables
from math import sqrt
import time
import cv2

from sklearn.utils.linear_assignment_ import linear_assignment #hungarian algorithm
from scipy.spatial.distance import cdist
from collections import defaultdict


MIN_AREA = 20
MIN_LENGHT = 5
MAX_ALLOWED_DIST = 20;

def list_duplicates(seq):
    # get index of replicated elements. It only returns replicates not the first occurence.
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    
    badIndex = []
    for locs in [locs for key, locs in tally.items() 
                            if len(locs)>1]:
        badIndex += locs[1:]
    return badIndex


class plate_worms(tables.IsDescription):
#class for the pytables 
    worm_index = tables.Int32Col(pos=0)
    frame_number = tables.Int32Col(pos=1)
    #label_image = tables.Int32Col(pos=2)
    coord_x = tables.Float32Col(pos=2)
    coord_y = tables.Float32Col(pos=3) 
    area = tables.Float32Col(pos=4) 
    perimeter = tables.Float32Col(pos=5) 
    major_axis = tables.Float32Col(pos=6) 
    minor_axis = tables.Float32Col(pos=7) 
    eccentricity = tables.Float32Col(pos=8) 
    compactness = tables.Float32Col(pos=9) 
    orientation = tables.Float32Col(pos=10) 
    solidity = tables.Float32Col(pos=11) 
    intensity_mean = tables.Float32Col(pos=12)
    intensity_std = tables.Float32Col(pos=13)
    speed = tables.Float32Col(pos=14)
    
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

#maskFile = '/Volumes/ajaver$/GeckoVideo/Compressed/CaptureTest_90pc_Ch2_16022015_174636.hdf5';
#featuresFile = '/Volumes/ajaver$/GeckoVideo/Trajectories/Features_CaptureTest_90pc_Ch2_16022015_174636.hdf5';

#maskFile = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch2_18022015_230213.hdf5';
#featuresFile = '/Users/ajaver/Desktop/Gecko_compressed/Features_CaptureTest_90pc_Ch2_18022015_230213.hdf5';

maskFile = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch4_16022015_174636.hdf5';
featuresFile = '/Users/ajaver/Desktop/Gecko_compressed/Features_CaptureTest_90pc_Ch4_16022015_174636.hdf5';


mask_fid = h5py.File(maskFile, 'r');
mask_dataset = mask_fid["/mask"]

feature_fid = tables.open_file(featuresFile, mode = 'w', title = '')
feature_table = feature_fid.create_table('/', "plate_worms", plate_worms,"Worm feature List")


tic = time.time()

coord_prev = np.empty([0]);
indexListPrev = np.empty([0]);
totWorms = 0;

for frame_number in range(mask_dataset.shape[0]):# range(100000):
    feature_list = [];
        
    image = mask_dataset[frame_number,:,:]
    [contours, hierarchy]= cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        bbox_seg = cv2.boundingRect(contour) 
        if bbox_seg[1] < MIN_LENGHT or bbox_seg[3] < MIN_LENGHT:
            continue #box too small to be a worm
        Icrop = image[bbox_seg[1]:(bbox_seg[1]+bbox_seg[3]),bbox_seg[0]:(bbox_seg[0]+bbox_seg[2])];
        
        
        hist = cv2.calcHist([Icrop],[0],None,[256],[0,256]).T[0]
        hist[0] = 0;
        level = triangle_th(hist)
        mask = cv2.threshold(Icrop, level, 1, cv2.THRESH_BINARY_INV)[1]
        mask[Icrop == 0] = 0;
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))

        [worm_contours, hierarchy]= cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        for ii_worm, worm_cnt in enumerate(worm_contours):
            area = float(cv2.contourArea(worm_cnt))
            if area < MIN_AREA:
                continue #area too small to be a worm
            (CMx,CMy),(MA,ma),angle = cv2.minAreaRect(worm_cnt)
            if ma > MA: dd = MA; ma = MA; ma = dd;  
            
            eccentricity = sqrt(1-ma**2/MA**2)
            hull = cv2.convexHull(worm_cnt)
            solidity = area/cv2.contourArea(hull);
            perimeter = float(cv2.arcLength(worm_cnt,True))
            compactness = perimeter**2/area
            
            maskCrop = np.zeros(Icrop.shape, dtype = np.uint8);
            cv2.drawContours(maskCrop, worm_contours, ii_worm, 255, -1)
            intensity_mean, intensity_std = cv2.meanStdDev(Icrop, mask = maskCrop)
            
            feature_list.append((frame_number, CMx + bbox_seg[0], 
                                 CMy + bbox_seg[1], area, perimeter, MA, ma, 
                             eccentricity, compactness, angle, solidity, 
                             intensity_mean[0,0], intensity_std[0,0]))
        
    #link trajectories
    feature_list = zip(*feature_list)
    coord = np.array(feature_list[1:3]).T
    if coord_prev.size!=0:
        costMatrix = cdist(coord_prev, coord);
        assigment = linear_assignment(costMatrix)
        
        indexList = np.zeros(coord.shape[0]);
        speed = np.zeros(coord.shape[0])

        for row, column in assigment: #ll = 1:numel(indexList)
            #print row, column, costMatrix[row,column]
            if costMatrix[row,column] < MAX_ALLOWED_DIST:
                if indexList[column] == 0:
                    indexList[column] = indexListPrev[row];
                    speed[column] = costMatrix[row][column];
                else:
                    print "!!!"
        
        unmatched = indexList==0
        vv = np.arange(np.sum(unmatched)) + totWorms + 1
        if vv.size>0:
            totWorms = vv[-1]
            indexList[unmatched] = vv
#           elif column > coord.shape[0]:
#                totWorms = totWorms +1;
#                indexList[column] = totWorms;
#        
        
#        for rep_ind in list_duplicates(indexList):
#            totWorms = totWorms +1; #assign new worm_index to joined trajectories
#            indexList[rep_ind] = totWorms;
        
    else:
        indexList = totWorms + np.arange(1,coord.shape[0]+1);
        totWorms = indexList[-1]
        speed = totWorms*[None]
    #print indexList
#    if coord_prev.size!=0:
#        plt.figure()
#        plt.imshow(image, interpolation = 'none', cmap = 'gray')
#        plt.plot(coord_prev[:,0],coord_prev[:,1], '.g')
#        plt.plot(coord[:,0],coord[:,1], '.b')
#            
        
    dum = coord_prev
    coord_prev = coord;
    indexListPrev = indexList;
    feature_list = zip(*([tuple(indexList)] + feature_list + [tuple(speed)]))
    
    feature_table.append(feature_list)
    if frame_number%25 == 0:
        toc = time.time()
        print frame_number, toc-tic
        tic = toc

plt.figure()
plt.imshow(image, interpolation = 'none', cmap = 'gray')
plt.plot(feature_table.cols.coord_x, feature_table.cols.coord_y, '.r')
     
feature_table.flush()
feature_table.cols.frame_number.create_csindex()
feature_table.cols.worm_index.create_csindex()
feature_fid.close()
#    print feature_table
    #feature_fid.close()
           
        
    #plt.figure()
    #plt.imshow(image, interpolation = 'none', cmap = 'gray')
    #plt.plot(coord_y, coord_x, 'rx')