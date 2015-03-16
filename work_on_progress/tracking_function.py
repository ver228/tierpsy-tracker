# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:07:19 2015

@author: ajaver
"""
import h5py
import matplotlib.pylab as plt
import numpy as np
import tables
from math import sqrt
import time
import cv2
import os
from sklearn.utils.linear_assignment_ import linear_assignment #hungarian algorithm
from scipy.spatial.distance import cdist
from skimage import filter


class plate_worms(tables.IsDescription):
#class for the pytables 
    worm_index = tables.Int32Col(pos=0)
    worm_index_joined = tables.Int32Col(pos=1)
    
    frame_number = tables.Int32Col(pos=2)
    coord_x = tables.Float32Col(pos=3)
    coord_y = tables.Float32Col(pos=4) 
    area = tables.Float32Col(pos=5) 
    perimeter = tables.Float32Col(pos=6) 
    major_axis = tables.Float32Col(pos=7) 
    minor_axis = tables.Float32Col(pos=8) 
    eccentricity = tables.Float32Col(pos=9) 
    compactness = tables.Float32Col(pos=10) 
    orientation = tables.Float32Col(pos=11) 
    solidity = tables.Float32Col(pos=12) 
    intensity_mean = tables.Float32Col(pos=13)
    intensity_std = tables.Float32Col(pos=14)
    
    threshold = tables.Int32Col(pos=15)
    bounding_box_xmin = tables.Int32Col(pos=16)
    bounding_box_xmax = tables.Int32Col(pos=17)
    bounding_box_ymin = tables.Int32Col(pos=18)
    bounding_box_ymax = tables.Int32Col(pos=19)
    
def triangle_th(hist):
    '''
    useful function to threshold the worms in a ROI
    adapted from m-file in MATLAB central form: Dr B. Panneton, June, 2010
    '''
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


#masked_image_file = '/Volumes/behavgenom$/GeckoVideo/Compressed/20150220/CaptureTest_90pc_Ch3_20022015_183607.hdf5'
#trajectories_file = '/Volumes/behavgenom$/GeckoVideo/Trajectories/20150220/aCaptureTest_90pc_Ch3_20022015_183607.hdf5'
masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch2_18022015_230213.hdf5'
trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Trajectory_CaptureTest_90pc_Ch2_18022015_230213.hdf5'

#masked_image_file, trajectories_file, 
initial_frame = 0
last_frame = 10
min_area = 20
min_length = 5
max_allowed_dist = 20
area_ratio_lim = (0.67, 1.5)
buffer_size = 25

#open hdf5 to read the images
mask_fid = h5py.File(masked_image_file, 'r');
mask_dataset = mask_fid["/mask"]
print mask_dataset.shape[0]

#open pytables to save the coordinates
feature_fid = tables.open_file(trajectories_file, mode = 'w', title = '')
feature_table = feature_fid.create_table('/', "plate_worms", plate_worms,"Worm feature List")

if last_frame <= 0:
    last_frame = mask_dataset.shape[0]

#initialized variabes
tot_worms = 0;
buff_last_coord = np.empty([0]);
buff_last_index = np.empty([0]);
buff_last_area = np.empty([0]);

for frame_number in range(initial_frame, last_frame, buffer_size):
    tic = time.time()
    
    #load image buffer
    #image_buffer = mask_dataset[frame_number:(frame_number+buffer_size),:,:]
    image_buffer = mask_dataset[frame_number:(frame_number+buffer_size),:,:]#mask_dataset[frame_number:(frame_number+buffer_size),1250:1550, 1700:1900]#,900:1200,400:600]
    
    #select pixels as connected regions that were selected as worms at least once in the masks
    main_mask = np.any(image_buffer, axis=0)
    main_mask[0:15, 0:479] = False; #remove the timestamp region in the upper corner
    main_mask = main_mask.astype(np.uint8) #change from bool to uint since same datatype is required in opencv
    [ROIs, hierarchy]= cv2.findContours(main_mask.copy(), \
    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    buff_feature_table = []
    buff_last = []

    #examinate each region of interest        
    for ROI_ind, ROI_cnt in enumerate(ROIs):
        ROI_bbox = cv2.boundingRect(ROI_cnt) 
        #boudning box too small to be a worm
        if ROI_bbox[1] < min_length or ROI_bbox[3] < min_length:
            continue 
        
        #select ROI for all buffer slides.
        ROI_buffer = image_buffer[:,ROI_bbox[1]:(ROI_bbox[1]+ROI_bbox[3]),ROI_bbox[0]:(ROI_bbox[0]+ROI_bbox[2])].copy();            
        #ROI_buffer_avg = np.zeros(ROI_buffer.shape, dtype = ROI_buffer.dtype)
        #for ii in range(ROI_buffer.shape[0]):
        #    ROI_buffer_avg[ii,:,:] = cv2.medianBlur(ROI_buffer[ii,:,:], 3)
            #ROI_buffer_avg[ii,:,:] = cv2.bilateralFilter(ROI_buffer[ii,:,:],-1, 3, 3)
        #calculate threshold using the nonzero pixels. 
        #Using the buffer instead of a single image, improves the threshold calculation, since better statistics are recoverd
        pix_valid = ROI_buffer[ROI_buffer!=0]
        
        otsu_thresh = filter.threshold_otsu(pix_valid)        
        
        pix_hist = np.bincount(pix_valid)
        int_max = np.argmax(pix_hist)
        if otsu_thresh < int_max:
            thresh = np.argmin(pix_hist[otsu_thresh:int_max]) + otsu_thresh;
        else:
            thresh = otsu_thresh
        
        
        #try:
        #    thresh = np.where((np.diff(pix_hist[:int_max])<0))[0][-1]
        #except:
        #    thresh = np.where(pix_hist[:(int_max-1)]==0)[0][-1] #if not valid pixel, just return the neareast zero value, from the maximum
           
        #thresh = triangle_th(pix_hist)
        
        
        #plt.figure()
        #plt.plot(pix_hist)
        if buff_last_coord.size!=0:
            #select data from previous trajectories only within the contour bounding box.
            #used to link with the previous chunks (buffers)
            good = (buff_last_coord[:,0] > ROI_bbox[0]) & \
            (buff_last_coord[:,1] > ROI_bbox[1]) & \
            (buff_last_coord[:,0] < ROI_bbox[0]+ROI_bbox[2]) & \
            (buff_last_coord[:,1] < ROI_bbox[1]+ROI_bbox[3])
            
            coord_prev = buff_last_coord[good,:];
            area_prev = buff_last_area[good]; 
            index_list_prev = buff_last_index[good];
            
        else:
            #if it is the first buffer, reinitiailize all the variables
            coord_prev = np.empty([0]);
            area_prev = np.empty([0]); 
            index_list_prev = np.empty([0]);
        
        buff_tot = image_buffer.shape[0]
        for buff_ind in range(buff_tot):
            #calculate figures for each image in the buffer
            ROI_image = ROI_buffer[buff_ind,:,:]
            
            #get binary image, and clean it using morphological closing
            ROI_mask = ((ROI_image<thresh) & (ROI_image != 0)).astype(np.uint8)
            ROI_mask = cv2.morphologyEx(ROI_mask, cv2.MORPH_CLOSE,np.ones((3,3)))
            #ROI_mask = cv2.erode(ROI_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)), iterations=2)

            
            #get worms, assuming each contour in the ROI is a worm
            [ROI_worms, hierarchy]= cv2.findContours(ROI_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)           
            mask_feature_list = [];
            
            for worm_ind, worm_cnt in enumerate(ROI_worms):
                
                #obtain freatures for each worm
                
                #ignore contours from holes
                if hierarchy[0][worm_ind][3] != -1:
                    continue
            
                area = float(cv2.contourArea(worm_cnt))
                if area < min_area:
                    continue #area too small to be a worm

                worm_bbox = cv2.boundingRect(worm_cnt) 
                
                #find use the best rotated bounding box, the fitEllipse function produces bad results quite often
                (CMx,CMy),(MA,ma),angle = cv2.minAreaRect(worm_cnt)
                if ma > MA: dd = MA; MA = ma; ma = dd;  
                
                eccentricity = sqrt(1-ma**2/MA**2)
                hull = cv2.convexHull(worm_cnt) #for the solidity
                solidity = area/cv2.contourArea(hull);
                perimeter = float(cv2.arcLength(worm_cnt,True))
                compactness = perimeter**2/area
                
                #calculate the mean intensity of the worm
                worm_mask = np.zeros(ROI_image.shape, dtype = np.uint8);
                cv2.drawContours(worm_mask, ROI_worms, worm_ind, 255, -1)
                intensity_mean, intensity_std = cv2.meanStdDev(ROI_image, mask = worm_mask)
                
                if buff_ind == -1:
                    bin_border = cv2.morphologyEx(worm_mask.astype(np.uint8), cv2.MORPH_GRADIENT,np.ones((2,2)))
                    img_worm = ROI_image.copy()
                    img_worm[bin_border!=0] = 255;
                    img_worm[0] = 0;
                    
                    plt.figure()
                    plt.imshow(img_worm, cmap =  'gray', interpolation = 'none')
                    
                    
                
                #worm_mask CAN BE USED TO CALCULATE THE SKELETON AT THIS POINT
                #with open('/Users/ajaver/Desktop/Gecko_compressed/image_dums/B%i_C%i_W%i.txt'%(buff_ind, ROI_ind, worm_ind), 'w') as f:
                #    np.savetxt(f, worm_mask, delimiter = ',')
                
                #append worm features. Use frame_number+1, to avoid 0 index.
                mask_feature_list.append((frame_number+ buff_ind + 1, 
                                          CMx + ROI_bbox[0], CMy + ROI_bbox[1], 
                                          area, perimeter, MA, ma, 
                                          eccentricity, compactness, angle, solidity, 
                                          intensity_mean[0,0], intensity_std[0,0], thresh,
                                          ROI_bbox[0] + worm_bbox[0], ROI_bbox[0] + worm_bbox[0] + worm_bbox[2],
                                          ROI_bbox[1] + worm_bbox[1], ROI_bbox[1] + worm_bbox[1] + worm_bbox[3]))
            
            if len(mask_feature_list)>0:
                mask_feature_list = zip(*mask_feature_list)
                coord = np.array(mask_feature_list[1:3]).T
                area = np.array(mask_feature_list[3]).T.astype(np.float)
                if coord_prev.size!=0:
                    costMatrix = cdist(coord_prev, coord); #calculate the cost matrix
                    #costMatrix[costMatrix>MA] = 1e10 #eliminate things that are farther 
                    assigment = linear_assignment(costMatrix) #use the hungarian algorithm
                    
                    index_list = np.zeros(coord.shape[0], dtype=np.int);
                    
                    #Final assigment. Only allow assigments within a maximum allowed distance, and an area ratio
                    for row, column in assigment:
                        if costMatrix[row,column] < max_allowed_dist:
                            area_ratio = area[column]/area_prev[row];
                            
                            if area_ratio>area_ratio_lim[0] and area_ratio<area_ratio_lim[1]:
                                index_list[column] = index_list_prev[row];
                                
                    #add a new index if no assigment was found
                    unmatched = index_list==0
                    vv = np.arange(1,np.sum(unmatched)+1) + tot_worms
                    if vv.size>0:
                        tot_worms = vv[-1]
                        index_list[unmatched] = vv
                else:
                    #initialize worm indexes
                    n_new_worms = len(mask_feature_list[0])
                    index_list = tot_worms + np.arange(1, n_new_worms+1);
                    tot_worms = index_list[-1]
                    #speed = n_new_worms*[None]
                    
                #append the new feature list to the pytable
                mask_feature_list = zip(*([tuple(index_list), 
                                           tuple(len(index_list)*[0])] + 
                                           mask_feature_list))
                buff_feature_table += mask_feature_list
                
                if buff_ind == buff_tot-1 : 
                    #add only features if it is the last frame in the list
                    buff_last += mask_feature_list
            else:
                #consider the case where not valid coordinates where found
                coord = np.empty([0]);
                area = np.empty([0]); 
                index_list = []
            
            #save the features for the linkage to the next frame in the buffer
            coord_prev = coord;
            area_prev = area;
            index_list_prev = index_list;
    
    #save the features for the linkage to the next buffer
    buff_last = zip(*buff_last)
    buff_last_coord = np.array(buff_last[3:5]).T
    buff_last_index = np.array(buff_last[0])
    buff_last_area = np.array(buff_last[5])
    
    #append data to pytables
    feature_table.append(buff_feature_table)
    
    if frame_number % 1000 == 0:
        feature_fid.flush()
    #print progress
    print frame_number, time.time() - tic

#close files and create indexes
feature_table.flush()
feature_table.cols.frame_number.create_csindex()
feature_table.cols.worm_index.create_csindex()
feature_table.flush()
feature_fid.close()