# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:33:34 2015

@author: ajaver
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 16:08:07 2015

@author: ajaver
"""

import numpy as np
import pandas as pd
import tables
import pandas as pd
from math import sqrt
import cv2
from skimage.filters import threshold_otsu
import os, sys

from sklearn.utils.linear_assignment_ import linear_assignment #hungarian algorithm
from scipy.spatial.distance import cdist

from ..helperFunctions.timeCounterStr import timeCounterStr
from ..compressVideos.extractMetaData import storeMetaData, getTimestamp

table_filters = tables.Filters(complevel=5, complib='zlib', shuffle=True)
    
class plate_worms(tables.IsDescription):
#class for the pytables 
    worm_index = tables.Int32Col(pos=0)
    worm_index_joined = tables.Int32Col(pos=1)
    
    frame_number = tables.Int32Col(pos=2)
    coord_x = tables.Float32Col(pos=3)
    coord_y = tables.Float32Col(pos=4) 
    area = tables.Float32Col(pos=5) 
    perimeter = tables.Float32Col(pos=6) 
    box_length = tables.Float32Col(pos=7) 
    box_width = tables.Float32Col(pos=8) 
    quirkiness = tables.Float32Col(pos=9) 
    compactness = tables.Float32Col(pos=10) 
    box_orientation = tables.Float32Col(pos=11) 
    solidity = tables.Float32Col(pos=12) 
    intensity_mean = tables.Float32Col(pos=13)
    intensity_std = tables.Float32Col(pos=14)
    
    threshold = tables.Int32Col(pos=15)
    bounding_box_xmin = tables.Int32Col(pos=16)
    bounding_box_xmax = tables.Int32Col(pos=17)
    bounding_box_ymin = tables.Int32Col(pos=18)
    bounding_box_ymax = tables.Int32Col(pos=19)
    
    segworm_id = tables.Int32Col(pos=20); #deprecated, probably it would be good to remove it in the future

    hu0 = tables.Float32Col(pos=21)
    hu1 = tables.Float32Col(pos=22)
    hu2 = tables.Float32Col(pos=23)
    hu3 = tables.Float32Col(pos=24)
    hu4 = tables.Float32Col(pos=25)
    hu5 = tables.Float32Col(pos=26)
    hu6 = tables.Float32Col(pos=27)

def getWormThreshold(pix_valid):
    #calculate otsu_threshold as lower limit. Otsu understimate the threshold.
    try:
        otsu_thresh = threshold_otsu(pix_valid)
    except:
       return np.nan
    
    #calculate the histogram
    pix_hist = np.bincount(pix_valid)
    
    #the higher limit is the most frequent value in the distribution (background)
    largest_peak = np.argmax(pix_hist)
    if otsu_thresh < largest_peak and otsu_thresh+2 < len(pix_hist)-1:
        #smooth the histogram to find a better threshold
        pix_hist = np.convolve(pix_hist, np.ones(3), 'same')
        cumhist = np.cumsum(pix_hist);
        
        xx = np.arange(otsu_thresh, cumhist.size)
        try:
            #the threshold is calculated as the pixel level where there would be 
            #larger increase in the object area.
            hist_ratio = pix_hist[xx]/cumhist[xx]
            thresh = np.where((hist_ratio[3:]-hist_ratio[:-3])>0)[0][0] + otsu_thresh
        except IndexError:
            thresh = np.argmin(pix_hist[otsu_thresh:largest_peak]) + otsu_thresh;
    else:
        #if otsu is larger than the maximum peak keep otsu threshold
        thresh = otsu_thresh
        
    return thresh

def getWormContours(ROI_image, thresh, strel_size = (5,5)):
    #get the border of the ROI mask, this will be used to filter for valid worms
    ROI_valid = (ROI_image != 0).astype(np.uint8)
    _, ROI_border_ind, _ =  cv2.findContours(ROI_valid, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  
    
    if len(ROI_border_ind) <= 1: 
        valid_ind = 0;
    else:
        #consider the case where there is more than one contour
        #i.e. the is a neiboring ROI in the square, just keep the largest area
        ROI_area = [cv2.contourArea(x) for x in ROI_border_ind]
        valid_ind = np.argmax(ROI_area);
        ROI_valid = np.zeros_like(ROI_valid);
        ROI_valid = cv2.drawContours(ROI_valid, ROI_border_ind, valid_ind, 1, -1)
        ROI_image = ROI_image*ROI_valid
    
    #the indexes of the maskborder
    #if len(ROI_border_ind)==1 and ROI_border_ind[0].shape[0] > 1: 
    #ROI_border_ind = np.squeeze(ROI_border_ind[valid_ind])
    #ROI_border_ind = (ROI_border_ind[:,1],ROI_border_ind[:,0])
    #else:
    #    continue

    #a median filter sharps edges
    #ROI_image = cv2.medianBlur(ROI_image, 3);
    #get binary image, and clean it using morphological closing
    ROI_mask = ((ROI_image<thresh) & (ROI_image != 0)).astype(np.uint8)
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, strel_size)
    ROI_mask = cv2.morphologyEx(ROI_mask, cv2.MORPH_CLOSE, strel)

    #ROI_mask = cv2.morphologyEx(ROI_mask, cv2.MORPH_CLOSE,np.ones((5,5)))
    #ROI_mask = cv2.erode(ROI_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)), iterations=2)
    
    #get worms, assuming each contour in the ROI is a worm
    [_, ROI_worms, hierarchy]= cv2.findContours(ROI_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)           
    #if is_single_worm: ROI_worms, hierarchy = filterLargestArea(ROI_worms, hierarchy)

    return ROI_worms, hierarchy

def getWormFeatures(worm_cnt, ROI_image, ROI_bbox, current_frame, thresh, min_area):
    SEGWORM_ID_DEFAULT = -1; #default value for the column segworm_id

    area = float(cv2.contourArea(worm_cnt))
    if area < min_area:
        return None #area too small to be a worm
    
    worm_bbox = cv2.boundingRect(worm_cnt) 
    
    #find use the best rotated bounding box, the fitEllipse function produces bad results quite often
    #this method is better to obtain an estimate of the worm length than eccentricity
    (CMx,CMy),(L, W),angle = cv2.minAreaRect(worm_cnt)
    if W > L: L,W = W,L;   #switch if width is larger than length
    quirkiness = sqrt(1-W**2/L**2)

    hull = cv2.convexHull(worm_cnt) #for the solidity
    solidity = area/cv2.contourArea(hull);
    perimeter = float(cv2.arcLength(worm_cnt,True))
    compactness = area/(4*np.pi*perimeter**2)
    
    #calculate the mean intensity of the worm
    worm_mask = np.zeros(ROI_image.shape, dtype = np.uint8);
    cv2.drawContours(worm_mask, [worm_cnt], 0, 255, -1)
    intensity_mean, intensity_std = cv2.meanStdDev(ROI_image, mask = worm_mask)
    
    #calculate hu moments, they are scale and rotation invariant
    hu_moments = cv2.HuMoments(cv2.moments(worm_cnt))
    
    #save everything into the the proper output format
    mask_feat = (current_frame, 
          CMx + ROI_bbox[0], CMy + ROI_bbox[1], 
          area, perimeter, L, W, 
          quirkiness, compactness, angle, solidity, 
          intensity_mean[0,0], intensity_std[0,0], thresh,
          ROI_bbox[0] + worm_bbox[0], ROI_bbox[0] + worm_bbox[0] + worm_bbox[2],
          ROI_bbox[1] + worm_bbox[1], ROI_bbox[1] + worm_bbox[1] + worm_bbox[3],
          SEGWORM_ID_DEFAULT, *hu_moments)

    return mask_feat

def joinConsecutiveFrames(index_list_prev, coord, coord_prev, area, area_prev, tot_worms, max_allowed_dist, area_ratio_lim):
    #TODO probably it is better to convert the whole getWormTrajectories function into a class for clearity 
    if coord_prev.size != 0:
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
        index_list = tot_worms + np.arange(1, len(area) + 1);
        tot_worms = index_list[-1]
        #speed = n_new_worms*[None]

    return index_list, tot_worms

def getWormTrajectories(masked_image_file, trajectories_file, initial_frame = 0, last_frame = -1, \
min_area = 25, min_length = 5, max_allowed_dist = 20, \
area_ratio_lim = (0.5, 2), buffer_size = 25, threshold_factor = 1., strel_size = (5,5)):
    '''
    #read images from 'masked_image_file', and save the linked trajectories and their features into 'trajectories_file'
    #use the first 'total_frames' number of frames, if it is equal -1, use all the frames in 'masked_image_file'
    min_area -- min area of the segmented worm
    min_length -- min size of the bounding box in the ROI of the compressed image
    max_allowed_dist -- maximum allowed distance between to consecutive trajectories
    area_ratio_lim -- allowed range between the area ratio of consecutive frames 
    threshold_factor -- The calculated threshold will be multiplied by this factor. Desperate attempt to solve for the swimming case. 
    ''' 
    

    #check that the mask file is correct
    if not os.path.exists(masked_image_file):
        raise Exception('HDF5 Masked Image file does not exists.')
    
    with tables.File(masked_image_file, 'r') as mask_fid:
        mask_dataset = mask_fid.get_node("/mask")
        if not mask_dataset._v_attrs['has_finished'] >= 1:
            raise Exception('HDF5 Masked Image was not finished correctly.')
        if mask_dataset.shape[0] == 0:
            raise Exception('Empty set in masked image file. Nothing to do here.')
            
    #intialize variables
    base_name = masked_image_file.rpartition('.')[0].rpartition(os.sep)[-1]
    progress_str = '####'

    
    #read timestamps from the masked_image_file 
    timestamp, timestamp_time = getTimestamp(masked_image_file)

    with tables.File(masked_image_file, 'r') as mask_fid, \
    tables.open_file(trajectories_file, mode = 'w') as feature_fid:
        mask_dataset = mask_fid.get_node("/mask")
        if mask_dataset.shape[0] > timestamp.size:
             #pad with nan the extra space
             N = mask_dataset.shape[0] - timestamp.size
             timestamp = np.hstack((timestamp, np.full(N, np.nan)))
             timestamp_time = np.hstack((timestamp_time, np.full(N, np.nan)))
             assert mask_dataset.shape[0] == timestamp.size

        #initialize
        feature_fid.create_group('/', 'timestamp')
        feature_fid.create_carray('/timestamp', 'raw', obj = np.asarray(timestamp))
        feature_fid.create_carray('/timestamp', 'time', obj = np.asarray(timestamp_time))

        
        feature_table = feature_fid.create_table('/', "plate_worms", 
                                                 plate_worms, "Worm feature List",
                                                 filters = table_filters)
        
        #flag used to determine if the function finished correctly
        feature_table._v_attrs['has_finished'] = 0
        
        if last_frame <= 0:
            last_frame = mask_dataset.shape[0]
        
        #initialized variables
        tot_worms = 0;
        buff_last_coord = np.empty([0]);
        buff_last_index = np.empty([0]);
        buff_last_area = np.empty([0]);
        
        progressTime = timeCounterStr('Calculating trajectories.');
        for frame_number in range(initial_frame, last_frame, buffer_size):
            
            #load image buffer
            image_buffer = mask_dataset[frame_number:(frame_number+buffer_size),:, :]
            #select pixels as connected regions that were selected as worms at least once in the masks
            main_mask = np.any(image_buffer, axis=0)
            
            #TODO addd the option to do not remove this information in videos without the timestamp
            main_mask[0:15, 0:479] = False; #remove the timestamp region in the upper corner
            
            main_mask = main_mask.astype(np.uint8) #change from bool to uint since same datatype is required in opencv
            [_, ROIs, hierarchy]= cv2.findContours(main_mask, \
            cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            #if is_single_worm: ROIs, hierarchy = filterLargestArea(ROIs, hierarchy)
                
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
                ROI_buffer_med = np.zeros_like(ROI_buffer)
                for ii in range(ROI_buffer.shape[0]): 
                    ROI_buffer_med[ii] = cv2.medianBlur(ROI_buffer[ii], 3);
                    #make a close operation to make the worm and the background smoother

                #calculate threshold using the nonzero pixels. 
                #Using the buffer instead of a single image, improves the threshold calculation, since better statistics are recoverd
                pix_valid = ROI_buffer_med[ROI_buffer_med!=0]
                if pix_valid.size==0: 
                    continue
                #caculate threshold
                thresh = getWormThreshold(pix_valid)
                thresh *= threshold_factor
                                             
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
                    coord_prev, area_prev,index_list_prev = (np.empty([0]),)*3
    
                buff_tot = image_buffer.shape[0] #consider the case where there are not enough images left to fill the buffer
                
                for buff_ind in range(buff_tot):
                    #calculate figures for each image in the buffer
                    
                    #get the contour of possible worms
                    ROI_worms, hierarchy = getWormContours(ROI_buffer_med[buff_ind,:,:], thresh, strel_size)

                    mask_feature_list = [];
                    
                    current_frame = frame_number + buff_ind
                    for worm_ind, worm_cnt in enumerate(ROI_worms):
                        #ignore contours from holes
                        if hierarchy[0][worm_ind][3] != -1:
                            continue
                        
                        #obtain freatures for each worm
                        mask_feat = getWormFeatures(worm_cnt, ROI_buffer[buff_ind,:,:], ROI_bbox, current_frame, thresh, min_area)
    
                        #append worm features.
                        if mask_feat is not None:
                            mask_feature_list.append(mask_feat) 

                    
                    if len(mask_feature_list)>0:
                        mask_feature_list = list(zip(*mask_feature_list))
                        
                        coord = np.array(mask_feature_list[1:3]).T
                        area = np.array(mask_feature_list[3]).T.astype(np.float)
                        
                        index_list, tot_worms = joinConsecutiveFrames(index_list_prev, coord, coord_prev, area, area_prev, tot_worms, max_allowed_dist, area_ratio_lim)
                        #if is_single_worm: index_list, tot_worms = [1], 1

                        #append the new feature list to the pytable
                        mask_feature_list = list(zip(*([tuple(index_list), 
                                                   tuple(len(index_list)*[-1])] + 
                                                   mask_feature_list)))
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
            buff_last = list(zip(*buff_last))
            buff_last_coord = np.array(buff_last[3:5]).T
            buff_last_index = np.array(buff_last[0:1]).T
            buff_last_area = np.array(buff_last[5:6]).T
            
            #append data to pytables
            if buff_feature_table:
                feature_table.append(buff_feature_table)
            
            if frame_number % 1000 == 0:
                feature_fid.flush()
                
                
            if frame_number % 500 == 0:
                #calculate the progress and put it in a string
                progress_str = progressTime.getStr(frame_number)
                print(base_name + ' ' + progress_str);
                sys.stdout.flush()
        #flush any remaining and create indexes
        feature_table.flush()
        feature_table.cols.frame_number.create_csindex() #make searches faster
        feature_table.cols.worm_index.create_csindex()
        feature_table.flush()
    
        #flag used to determine if the function finished correctly
        feature_table._v_attrs['has_finished'] = 1
    
    
    #if is_single_worm: correctIndSigleWorm(trajectories_file)

    print(base_name + ' ' + progress_str);
    sys.stdout.flush()

def correctTrajectories(trajectories_file, is_single_worm, join_traj_param):
    if is_single_worm: 
        correctSingleWormCase(trajectories_file)
    else:
        joinTrajectories(trajectories_file, **join_traj_param)

    with tables.File(trajectories_file, "r+") as traj_fid:
        traj_fid.get_node('/plate_worms')._v_attrs['has_finished'] = 2
        traj_fid.flush()

def _validRowsByArea(plate_worms):
    #here I am assuming that most of the time the largest area in the frame is a worm. Therefore a very large area is likely to be
    #noise
    groupsbyframe = plate_worms.groupby('frame_number')
    maxAreaPerFrame = groupsbyframe.agg({'area':'max'})
    med_area = np.median(maxAreaPerFrame)
    mad_area = np.median(np.abs(maxAreaPerFrame-med_area))
    min_area = med_area - mad_area*6
    max_area = med_area + mad_area*6

    groupByIndex = plate_worms.groupby('worm_index')

    median_area_by_index = groupByIndex.agg({'area':np.median})

    good = ((median_area_by_index>min_area) & (median_area_by_index<max_area)).values
    valid_ind = median_area_by_index[good].index;

    plate_worms_f = plate_worms[plate_worms['worm_index'].isin(valid_ind)]

    #median location, it is likely the worm spend more time here since the stage movements tries to get it in the centre of the frame
    CMx_med = plate_worms_f['coord_x'].median()
    CMy_med = plate_worms_f['coord_y'].median();
    L_med = plate_worms_f['box_length'].median();

    #let's use a threshold of movement of at most a quarter of the worm size, otherwise we discard frame.
    L_th = L_med/4

    #now if there are still a lot of valid blobs we decide by choising the closest blob
    valid_rows = []
    tot_frames = plate_worms['frame_number'].max() + 1

    def get_valid_indexes(frame_number, prev_row):    
        try:
            current_group_f = groupbyframe_f.get_group(frame_number)
        except KeyError:
            #there are not valid index in the current group
            prev_row = -1
            return prev_row
        
        #pick the closest blob if there are more than one blob to pick
        if not isinstance(prev_row, int):
            delX = current_group_f['coord_x'] - prev_row['coord_x']
            delY = current_group_f['coord_y'] - prev_row['coord_y']
        else:
            delX = current_group_f['coord_x'] - CMx_med
            delY = current_group_f['coord_y'] - CMy_med
        
        R = np.sqrt(delX*delX + delY*delY)
        good_ind = np.argmin(R)
        if R[good_ind] < L_th:
            prev_row = current_group_f.loc[good_ind]
            valid_rows.append(good_ind)
        else:
            prev_row = -1

        
        return prev_row

    #group by frame
    groupbyframe_f = plate_worms_f.groupby('frame_number')

    prev_row = -1
    first_frame = tot_frames
    for frame_number in range(tot_frames):
        prev_row = get_valid_indexes(frame_number, prev_row) 
        if not isinstance(prev_row, int) and first_frame > frame_number:
            first_frame = frame_number

    #if the first_frame is larger than zero it means that it might have lost some data in from the beggining
    #let's try to search again from opposite direction
    if frame_number > 0:
        prev_row = plate_worms_f.loc[np.min(valid_rows)]
        for frame_number in range(frame_number, -1, -1):
            prev_row = get_valid_indexes(frame_number, prev_row) 


    #valid_rows = list(set(valid_rows))

    return valid_rows

def correctSingleWormCase(trajectories_file):
    '''
    Only keep the object with the largest area when cosider the case of individual worms.
    '''
    with pd.HDFStore(trajectories_file, 'r') as traj_fid:
        plate_worms = traj_fid['/plate_worms']
    
    #emtpy table nothing to do here
    if len(plate_worms) == 0: return

    valid_rows = _validRowsByArea(plate_worms)
    
    plate_worms['worm_index_joined'] = np.array(-1, dtype=np.int32) #np.array(1, dtype=np.int32)
    plate_worms.loc[valid_rows, 'worm_index_joined'] = 1

    with tables.File(trajectories_file, "r+") as traj_fid:
        table_filters = tables.Filters(complevel=5, complib='zlib', shuffle=True, fletcher32=True)
        newT = traj_fid.create_table('/', 'plate_worms_t', 
                                        obj = plate_worms.to_records(index=False), 
                                        filters=table_filters)
        newT._v_attrs['has_finished'] = 2
        traj_fid.remove_node('/', 'plate_worms')
        newT.rename('plate_worms')


def _findNextTraj(df, area_ratio_lim, min_track_size, max_time_gap):
    '''
    area_ratio_lim -- allowed range between the area ratio of consecutive frames 
    min_track_size -- minimum tracksize accepted
    max_time_gap -- time gap between joined trajectories
    '''

    df = df[['worm_index', 'frame_number', 'coord_x', 'coord_y', 'area', 'box_length']]
     #select the first and last frame_number for each separate trajectory
    tracks_data = df[['worm_index', 'frame_number']]
    tracks_data = tracks_data.groupby('worm_index')
    tracks_data = tracks_data.aggregate({'frame_number': [np.argmin, np.argmax, 'count']})

    #filter data only to include trajectories larger than min_track_size
    tracks_data = tracks_data[tracks_data['frame_number']['count']>=min_track_size]
    valid_indexes = tracks_data.index

    #select the corresponding first and last rows of each trajectory
    first_rows = df.ix[tracks_data['frame_number']['argmin'].values]
    last_rows = df.ix[tracks_data['frame_number']['argmax'].values]
    #let's use the particle id as index instead of the row number
    last_rows.index = tracks_data['frame_number'].index 
    first_rows.index = tracks_data['frame_number'].index

    #%% look for trajectories that could be join together in a small time gap
    join_frames = []
    for curr_index in valid_indexes:
        #the possible connected trajectories must have started after the end of the current trajectories, 
        #within a timegap given by max_time_gap
        possible_rows = first_rows[ \
                (first_rows['frame_number'] > last_rows['frame_number'][curr_index]) &
                (first_rows['frame_number'] < last_rows['frame_number'][curr_index] + max_time_gap)]
        
        #the area change must be smaller than the one given by area_ratio_lim
        #it is better to use the last point change of area because we are considered changes near that occur near time
        areaR = last_rows['area'][curr_index]/possible_rows['area'];
        possible_rows = possible_rows[(areaR > area_ratio_lim[0]) & (areaR < area_ratio_lim[1])]
        
        #not valid rows left
        if len(possible_rows) == 0: continue
            
        R = np.sqrt( (possible_rows['coord_x']  - last_rows['coord_x'][curr_index]) ** 2 + \
                    (possible_rows['coord_y']  - last_rows['coord_x'][curr_index]) ** 2)
        
        indmin = np.argmin(R)
        #only join trajectories that move at most one worm body
        if R[indmin] <= last_rows['box_length'][curr_index]:
            #print(curr_index, indmin)
            join_frames.append((indmin, curr_index))

    relations_dict = dict(join_frames)

    return relations_dict, valid_indexes

def _joinDict2Index(worm_index, relations_dict, valid_indexes):
    worm_index_joined = np.full_like(worm_index, -1)
        
    for ind in valid_indexes:
        #seach in the dictionary for the first index in the joined trajectory group
        ind_joined = ind;
        while ind_joined in relations_dict:
            ind_joined = relations_dict[ind_joined]
        
        #replace the previous index for the root index
        worm_index_joined[worm_index == ind] = ind_joined
    
    return worm_index_joined


def joinTrajectories(trajectories_file, min_track_size = 50, \
max_time_gap = 100, area_ratio_lim = (0.67, 1.5)):
    '''
    area_ratio_lim -- allowed range between the area ratio of consecutive frames 
    min_track_size -- minimum tracksize accepted
    max_time_gap -- time gap between joined trajectories
    '''
    
    #check the previous step finished correctly
    with tables.open_file(trajectories_file, mode = 'r') as fid:
        traj_table = fid.get_node('/plate_worms')
        assert traj_table._v_attrs['has_finished'] >= 1

    #%% get the first and last rows for each trajectory. Pandas is easier of manipulate than tables.
    with pd.HDFStore(trajectories_file, 'r') as fid:
        df = fid['plate_worms'][['worm_index', 'frame_number', 'coord_x', 'coord_y', 'area', 'box_length']]

    relations_dict,valid_indexes = _findNextTraj(df, area_ratio_lim, min_track_size, max_time_gap)
    
    #%%
    #update worm_index_joined field 
    with tables.open_file(trajectories_file, mode = 'r+') as fid:
        plate_worms = fid.get_node('/plate_worms')
        
        #read the worm_index column, this is the index order that have to be conserved in the worm_index_joined column
        worm_index = plate_worms.col('worm_index')
        worm_index_joined = _joinDict2Index(worm_index, relations_dict, valid_indexes)
        
        #add the result the column worm_index_joined
        plate_worms.modify_column(colname = 'worm_index_joined', column = worm_index_joined)
        
        #flag the join data as finished
        plate_worms._v_attrs['has_finished'] = 2
        fid.flush()


#DEPRECATED
def joinTrajectories_old(trajectories_file, min_track_size = 50, \
max_time_gap = 100, area_ratio_lim = (0.67, 1.5)):
    '''
    area_ratio_lim -- allowed range between the area ratio of consecutive frames 
    min_track_size -- minimum tracksize accepted
    max_time_gap -- time gap between joined trajectories
    '''
    
    
    #use data as a recarray (less confusing)
    frame_dtype = np.dtype([('worm_index', int), ('frame_number', int), 
                        ('coord_x', float), ('coord_y',float), ('area',float), 
                        ('box_length', float)])
    with tables.open_file(trajectories_file, mode = 'r+') as feature_fid:
        feature_table = feature_fid.get_node('/plate_worms')
        
        if 'has_finished' in dir(feature_table._v_attrs):
            assert feature_table._v_attrs['has_finished'] == 1

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
            max_row = (); #empty tuple
            min_row = ();

            #get the first and last row for each index
            for dd in feature_table.where('worm_index == %i'% ii):
                if dd['frame_number'] < min_frame:
                    min_frame = dd['frame_number']
                    min_row = (dd['worm_index'], dd['frame_number'], dd['coord_x'], dd['coord_y'], dd['area'], dd['box_length'])
                
                if dd['frame_number'] > max_frame:
                    max_frame = dd['frame_number']
                    max_row = (dd['worm_index'], dd['frame_number'], dd['coord_x'], dd['coord_y'], dd['area'], dd['box_length'])
            
            if len(min_row) == 0 or len(max_row) == 0: 
                continue
            last_frames.append(max_row)
            first_frames.append(min_row)
        
        last_frames = np.array(last_frames, dtype = frame_dtype)
        first_frames = np.array(first_frames, dtype = frame_dtype)
        
        
        #find pairs of trajectories that could be joined
        join_frames = [];
        for kk in range(last_frames.shape[0]):
            
            possible_rows = first_frames[ \
            (first_frames['frame_number'] > last_frames['frame_number'][kk]) &
            (first_frames['frame_number'] < last_frames['frame_number'][kk] + max_time_gap)]
            
            if possible_rows.size > 0:
                areaR = last_frames['area'][kk]/possible_rows['area'];
                
                good = (areaR>area_ratio_lim[0]) & (areaR<area_ratio_lim[1])
                possible_rows = possible_rows[good]
                
                R = np.sqrt( (possible_rows['coord_x']  - last_frames['coord_x'][kk]) ** 2 + \
                (possible_rows['coord_y']  - last_frames['coord_y'][kk]) ** 2)
                if R.shape[0] == 0:
                    continue
                
                indmin = np.argmin(R)
                #only join trajectories that move at most one worm body
                if R[indmin] <= last_frames['box_length'][kk]:
                    join_frames.append((possible_rows['worm_index'][indmin],last_frames['worm_index'][kk]))
        
        relations_dict = dict(join_frames)
    
        
        for ii in indexes:
            ind = ii;
            while ind in relations_dict:
                ind = relations_dict[ind]
    
            for row in feature_table.where('worm_index == %i'% ii):
                row['worm_index_joined'] = ind;
                row.update()
        
        feature_table._v_attrs['has_finished'] = 2
        feature_fid.flush()

#DEPRECATED
def plotLongTrajectories(trajectories_file, plot_file = '', \
number_trajectories = 20, plot_limits = (2048,2048), index_str = 'worm_index_joined'):
    #DEPRECATED
    if not plot_file:
        plot_file = trajectories_file.rsplit('.')[0] + '.pdf'
    
    with pd.HDFStore(trajectories_file, 'r') as table_fid:
        df = table_fid['/plate_worms'].query('%s > 0' % index_str)
        
        tracks_size = df[index_str].value_counts()
        
        number_trajectories = number_trajectories if len(tracks_size)>=number_trajectories else len(tracks_size);
        
        for ii in range(number_trajectories):
            coord = df[df[index_str] == tracks_size.index[ii]][['coord_x', 'coord_y', 'frame_number']]
            if len(coord)!=0:
                plt.plot(coord['coord_x'], coord['coord_y'], '-');
        
        plt.xlim([0,plot_limits[0]])
        plt.ylim([0,plot_limits[1]])
        plt.axis('equal')
        plt.savefig(plot_file, bbox_inches='tight')
        
        plt.close()

#%%    

if __name__ == '__main__':
    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/Anne_Strains_20150619/Capture_Ch2_19062015_170506.hdf5'
    trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Results/Anne_Strains_20150619/Capture_Ch2_19062015_170506_trajectories.hdf5'
        
    getWormTrajectories(masked_image_file, trajectories_file, last_frame = -1,\
    initial_frame = 0)
    joinTrajectories(trajectories_file)
    
    from getDrawTrajectories import drawTrajectoriesVideo
    drawTrajectoriesVideo(masked_image_file, trajectories_file)
    
    
    plotLongTrajectories(trajectories_file)
    