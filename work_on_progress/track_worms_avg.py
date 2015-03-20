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
import os
from skimage.filter import threshold_otsu


from sklearn.utils.linear_assignment_ import linear_assignment #hungarian algorithm
from scipy.spatial.distance import cdist

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
    
    segworm_id = tables.Int32Col(pos=20);
    
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

#@profile
def getTrajectories(masked_image_file, trajectories_file, initial_frame = 0, last_frame = -1, \
min_area = 20, min_length = 5, max_allowed_dist = 20, \
area_ratio_lim = (0.5, 2), buffer_size = 25):
    '''
    #read images from 'masked_image_file', and save the linked trajectories and their features into 'trajectories_file'
    #use the first 'total_frames' number of frames, if it is equal -1, use all the frames in 'masked_image_file'
    min_area -- min area of the segmented worm
    min_length -- min size of the bounding box in the ROI of the compressed image
    max_allowed_dist -- maximum allowed distance between to consecutive trajectories
    area_ratio_lim -- allowed range between the area ratio of consecutive frames 
    ''' 
    #open hdf5 to read the images
    mask_fid = h5py.File(masked_image_file, 'r');
    mask_dataset = mask_fid["/mask"]
    print mask_dataset.shape[0]


    SEGWORM_ID_DEFAULT = -1; #default value for the column segworm_id
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
        image_buffer = mask_dataset[frame_number:(frame_number+buffer_size),:, :]
        
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
            ROI_buffer = image_buffer[:,ROI_bbox[1]:(ROI_bbox[1]+ROI_bbox[3]),ROI_bbox[0]:(ROI_bbox[0]+ROI_bbox[2])];            
            #calculate threshold using the nonzero pixels. 
            #Using the buffer instead of a single image, improves the threshold calculation, since better statistics are recoverd
            pix_valid = ROI_buffer[ROI_buffer!=0]
            #thresh,_ = cv2.threshold(pix_valid,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #pix_hist = np.bincount(pix_valid)
            #int_max = np.argmax(pix_hist)
            #try:
            #    thresh = np.where((np.diff(pix_hist[:int_max])<0))[0][-1]
            #except:
            #    thresh = np.where(pix_hist[:(int_max-1)]==0)[0][-1] #if not valid pixel, just return the neareast zero value, from the maximum
            
            #thresh = triangle_th(pix_hist)
            
            otsu_thresh = threshold_otsu(pix_valid)        
        
            pix_hist = np.bincount(pix_valid)
            int_max = np.argmax(pix_hist)
            if otsu_thresh < int_max:
                cumhist = np.cumsum(pix_hist);    
                xx_t = np.arange((otsu_thresh-2),(otsu_thresh+3));
                yy_t = cumhist[xx_t]
                pp = np.polyfit(xx_t, yy_t, 1)
                xx = np.arange(otsu_thresh, cumhist.size)
                yy = np.polyval(pp, xx)
                try:
                    thresh = np.where((cumhist[xx]-yy)/yy>0.02)[0][0] + otsu_thresh
                except:
                    thresh = np.argmin(pix_hist[otsu_thresh:int_max]) + otsu_thresh;
            else:
                thresh = otsu_thresh
            
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
            
            buff_tot = image_buffer.shape[0] #consider the case where there are not enough images left to fill the buffer
            
            for buff_ind in range(buff_tot):
                #calculate figures for each image in the buffer
                ROI_image = ROI_buffer[buff_ind,:,:]
                
                #get the border of the ROI mask, this will be used to filter for valid worms
                ROI_valid = (ROI_image != 0)
                ROI_border_ind, _ =  cv2.findContours(ROI_valid.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  
                
                if len(ROI_border_ind) <= 1: #consider the case where there is more than one contour
                    valid_ind = 0;
                else:
                    ROI_area = [cv2.contourArea(x) for x in ROI_border_ind]
                    valid_ind = np.argmax(ROI_area);
                
                if len(ROI_border_ind)==1 and ROI_border_ind[0].shape[0] > 1: 
                    ROI_border_ind = np.squeeze(ROI_border_ind[valid_ind])
                    ROI_border_ind = (ROI_border_ind[:,1],ROI_border_ind[:,0])
                else:
                    continue
            
                #get binary image, and clean it using morphological closing
                ROI_mask = ((ROI_image<thresh) & ROI_valid).astype(np.uint8)
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
                    
                    #check if the worm touches the ROI contour, if it does it is likely to be garbage
                    worm_mask = np.zeros(ROI_image.shape, dtype = np.uint8);
                    cv2.drawContours(worm_mask, ROI_worms, worm_ind, 255, -1)
                    if np.any(worm_mask[ROI_border_ind]):
                        continue

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
                    intensity_mean, intensity_std = cv2.meanStdDev(ROI_image, mask = worm_mask)
                    
                    
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
                                              ROI_bbox[1] + worm_bbox[1], ROI_bbox[1] + worm_bbox[1] + worm_bbox[3],
                                              SEGWORM_ID_DEFAULT)) 
                
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
                                               tuple(len(index_list)*[-1])] + 
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
        buff_last_index = np.array(buff_last[0:1]).T
        buff_last_area = np.array(buff_last[5:6]).T
        
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


def joinTrajectories(trajectories_file, min_track_size = 50, \
max_time_gap = 100, area_ratio_lim = (0.67, 1.5)):
    '''
    area_ratio_lim -- allowed range between the area ratio of consecutive frames 
    min_track_size -- minimum tracksize accepted
    max_time_gap -- time gap between joined trajectories
    '''
    feature_fid = tables.open_file(trajectories_file, mode = 'r+')
    feature_table = feature_fid.get_node('/plate_worms')
    
    #calculate the track size, and select only tracks with at least min_track_size length
    track_size = np.bincount(feature_table.cols.worm_index)    
    indexes = np.arange(track_size.size);
    indexes = indexes[track_size>=min_track_size]
    
    #select the first and the last points of a each trajectory
    last_frames = [];
    first_frames = [];
    for ii in indexes:
        min_frame = 1e32;
        max_frame = 0;
        
        for dd in feature_table.where('worm_index == %i'% ii):
            if dd['frame_number'] < min_frame:
                min_frame = dd['frame_number']
                min_row = (dd['worm_index'], dd['frame_number'], dd['coord_x'], dd['coord_y'], dd['area'], dd['major_axis'])
            
            if dd['frame_number'] > max_frame:
                max_frame = dd['frame_number']
                max_row = (dd['worm_index'], dd['frame_number'], dd['coord_x'], dd['coord_y'], dd['area'], dd['major_axis'])
        last_frames.append(max_row)
        first_frames.append(min_row)
    
    #use data as a recarray (less confusing)
    frame_dtype = np.dtype([('worm_index', int), ('frame_number', int), 
                            ('coord_x', float), ('coord_y',float), ('area',float), 
                            ('major_axis', float)])
    last_frames = np.array(last_frames, dtype = frame_dtype)
    first_frames = np.array(first_frames, dtype = frame_dtype)
    
    
    #find pairs of trajectories that could be joined
    join_frames = [];
    for kk in range(last_frames.shape[0]):
        
        possible_rows = first_frames[ np.bitwise_and( \
        first_frames['frame_number'] > last_frames['frame_number'][kk], 
        first_frames['frame_number'] < last_frames['frame_number'][kk] + max_time_gap)]
        
        if possible_rows.size > 0:
            areaR = last_frames['area'][kk]/possible_rows['area'];
            
            good = np.bitwise_and(areaR>area_ratio_lim[0], areaR<area_ratio_lim[1])
            possible_rows = possible_rows[good]
            
            R = np.sqrt( (possible_rows['coord_x']  - last_frames['coord_x'][kk]) ** 2 + \
            (possible_rows['coord_y']  - last_frames['coord_y'][kk]) ** 2)
            if R.shape[0] == 0:
                continue
            
            indmin = np.argmin(R)
            if R[indmin] <= last_frames['major_axis'][kk]: #only join trajectories that move at most one worm body
                join_frames.append((possible_rows['worm_index'][indmin],last_frames['worm_index'][kk]))
    
    relations_dict = dict(join_frames)

    
    for ii in indexes:
        ind = ii;
        while ind in relations_dict:
            ind = relations_dict[ind]

        for row in feature_table.where('worm_index == %i'% ii):
            row['worm_index_joined'] = ind;
            row.update()
    
    feature_fid.flush()
    feature_fid.close()

if __name__ == '__main__':
    #masked_image_file = '/Volumes/ajaver$/GeckoVideo/Compressed/CaptureTest_90pc_Ch2_16022015_174636.hdf5';
    #trajectories_file = '/Volumes/ajaver$/GeckoVideo/Trajectories/Features_CaptureTest_90pc_Ch2_16022015_174636.hdf5';
    
#    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch2_18022015_230213.hdf5';
#    trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Features_CaptureTest_90pc_Ch2_18022015_230213.hdf5';
#    
#    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch4_16022015_174636.hdf5';
#    trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Features_CaptureTest_90pc_Ch4_16022015_174636.hdf5';

#    masked_image_dir = '/Volumes/behavgenom$/GeckoVideo/Compressed/20150223/';
#    baseName = 'CaptureTest_90pc_Ch1_23022015_192449';
#    masked_image_file = masked_image_dir + baseName + '.hdf5';
#    
#    trajectories_dir = '/Volumes/behavgenom$/GeckoVideo/Trajectories_test/20150223/';
#    trajectories_file = trajectories_dir + 'Trajectory_' + baseName + '.hdf5';
#    if not os.path.exists(trajectories_dir):
#        os.mkdir(trajectories_dir)

#    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch2_18022015_230213.hdf5'
#    trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Trajectory_CaptureTest_90pc_Ch2_18022015_230213.hdf5'
#    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch3_21022015_210020.hdf5'
#    trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Trajectory_CaptureTest_90pc_Ch3_21022015_210020.hdf5'
#    masked_image_file = '/Volumes/behavgenom$/GeckoVideo/Compressed/20150220/CaptureTest_90pc_Ch3_20022015_183607.hdf5'
#    trajectories_file = '/Volumes/behavgenom$/GeckoVideo/Trajectories/20150220/CaptureTest_90pc_Ch3_20022015_183607.hdf5'

#    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/prueba/CaptureTest_90pc_Ch1_02022015_141431.hdf5'
#    trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/prueba/trajectories/CaptureTest_90pc_Ch1_02022015_141431.hdf5'
   
    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch2_18022015_230213.hdf5'
    trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Trajectory_CaptureTest_90pc_Ch2_18022015_230213.hdf5'

#    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/prueba/CaptureTest_90pc_Ch1_02022015_141431.hdf5'
#    trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/prueba/trajectories/CaptureTest_90pc_Ch1_02022015_141431.hdf5'

    getTrajectories(masked_image_file, trajectories_file, last_frame = -1)
    joinTrajectories(trajectories_file)
#############

#%%    

    import matplotlib.pylab as plt
    import numpy as np
    import tables

    #plot top 20 largest trajectories
    feature_fid = tables.open_file(trajectories_file, mode = 'r')
    feature_table = feature_fid.get_node('/plate_worms')
    dum = np.array(feature_table.cols.worm_index_joined);
    dum[dum<0]=0
    track_size = np.bincount(dum)
    track_size[0] = 0
    indexes = np.argsort(track_size)[::-1]
    
    fig = plt.figure()
    for ii in indexes[0:20]:
        if ii == 0:
            break;
        coord = [(row['coord_x'], row['coord_y'], row['frame_number']) \
        for row in feature_table.where('worm_index_joined == %i'% ii)]

        coord = np.array(coord).T
        if coord.size!=0:
            plt.plot(coord[0,:], coord[1,:], '-');
    plt.xlim([0,2048])
    plt.ylim([0,2048])
    feature_fid.close()