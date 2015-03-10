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

from sklearn.utils.linear_assignment_ import linear_assignment #hungarian algorithm
from scipy.spatial.distance import cdist

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
    worm_index_joined = tables.Int32Col(pos=15)
    
def triangle_th(hist):
    #useful function to threshold the worms in a ROI
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

def getTrajectories(masked_image_file, trajectories_file, initial_frame = 0, last_frame = -1, \
MIN_AREA = 20, MIN_LENGHT = 5, MAX_ALLOWED_DIST = 20, \
AREA_RATIO_LIM = (0.67, 1.5), BUFFER_SIZE = 25):
    pass
    #read images from 'masked_image_file', and save the linked trajectories and their features into 'trajectories_file'
    #use the first 'total_frames' number of frames, if it is equal -1, use all the frames in 'masked_image_file'
    #MIN_AREA : min area of the segmented worm
    #MIN_LENGHT : min size of the bounding box in the ROI of the compressed image
    #MAX_ALLOWED_DIST : maximum allowed distance between to consecutive trajectories
    #AREA_RATIO_LIM : allowed range between the area ratio of consecutive frames 
        
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
    
    for frame_number in range(initial_frame, last_frame, BUFFER_SIZE):
        tic = time.time()
        
        #select pixels as connected regions that were selected as worms at least once in the masks
        image_buffer = mask_dataset[frame_number:(frame_number+BUFFER_SIZE),:,:]
        main_mask = np.any(image_buffer, axis=0)
        main_mask[0:15, 0:479] = False;
        main_mask = main_mask.astype(np.uint8)
        [contours, hierarchy]= cv2.findContours(main_mask.copy(), \
        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        buff_feature_table = []
        buff_last = []
        
        for ii_contour, contour in enumerate(contours):
            bbox_seg = cv2.boundingRect(contour) 
            if bbox_seg[1] < MIN_LENGHT or bbox_seg[3] < MIN_LENGHT:
                continue #box too small to be a worm
            
            Icrop = image_buffer[:,bbox_seg[1]:(bbox_seg[1]+bbox_seg[3]),bbox_seg[0]:(bbox_seg[0]+bbox_seg[2])];            
            pix_valid = Icrop[Icrop!=0]
            pix_hist = np.bincount(pix_valid)
            thresh = triangle_th(pix_hist)
            
            
            if buff_last_coord.size!=0:
                #select data only within the contour bounding box
                
                good = (buff_last_coord[:,0] > bbox_seg[0]) & \
                (buff_last_coord[:,1] > bbox_seg[1]) & \
                (buff_last_coord[:,0] < bbox_seg[0]+bbox_seg[2]) & \
                (buff_last_coord[:,1] < bbox_seg[1]+bbox_seg[3])
                
                coord_prev = buff_last_coord[good,:];
                area_prev = buff_last_area[good]; 
                index_list_prev = buff_last_index[good];
                
            else:
                coord_prev = np.empty([0]);
                area_prev = np.empty([0]); 
                index_list_prev = np.empty([0]);
            
            buff_tot = min(BUFFER_SIZE, image_buffer.shape[0])
            for buff_ind in range(buff_tot):
                im_worm = Icrop[buff_ind,:,:]
                
                worm_mask = ((im_worm<thresh*0.95) & (im_worm != 0)).astype(np.uint8)
                worm_mask = cv2.morphologyEx(worm_mask, cv2.MORPH_CLOSE,np.ones((3,3)))
                
                [worm_contours, hierarchy]= cv2.findContours(worm_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)           
                
                mask_feature_list = [];
                for ii_worm, worm_cnt in enumerate(worm_contours):
                    area = float(cv2.contourArea(worm_cnt))
                    if area < MIN_AREA:
                        continue #area too small to be a worm
      
                    #find use the best rotated bounding box, the fitEllipse function produces bad results quite often
                    (CMx,CMy),(MA,ma),angle = cv2.minAreaRect(worm_cnt)
                    if ma > MA: dd = MA; MA = ma; ma = dd;  
                    
                    eccentricity = sqrt(1-ma**2/MA**2)
                    hull = cv2.convexHull(worm_cnt) #for the solidity
                    solidity = area/cv2.contourArea(hull);
                    perimeter = float(cv2.arcLength(worm_cnt,True))
                    compactness = perimeter**2/area
                    
                    #calculate the mean intensity of the worm
                    
                    single_mask = np.zeros(im_worm.shape, dtype = np.uint8);
                    cv2.drawContours(single_mask, worm_contours, ii_worm, 255, -1)
                    intensity_mean, intensity_std = cv2.meanStdDev(im_worm, mask = single_mask)
                    
                    with open('/Users/ajaver/Desktop/Gecko_compressed/image_dums/B%i_C%i_W%i.txt'%(buff_ind, ii_contour, ii_worm), 'w') as f:
                        np.savetxt(f, single_mask, delimiter = ',')
                    
                    #append worm features
                    #use frame_number+1, to avoid 0 index
                    mask_feature_list.append((frame_number+ buff_ind + 1, CMx + bbox_seg[0], 
                                         CMy + bbox_seg[1], area, perimeter, MA, ma, 
                                     eccentricity, compactness, angle, solidity, 
                                     intensity_mean[0,0], intensity_std[0,0]))
                
                if len(mask_feature_list)>0:
                    mask_feature_list = zip(*mask_feature_list)
                    coord = np.array(mask_feature_list[1:3]).T
                    area = np.array(mask_feature_list[3]).T.astype(np.float)
                    if coord_prev.size!=0:
                        costMatrix = cdist(coord_prev, coord); #calculate the cost matrix
                        #costMatrix[costMatrix>MA] = 1e10 #eliminate things that are farther 
                        assigment = linear_assignment(costMatrix) #use the hungarian algorithm
                        
                        index_list = np.zeros(coord.shape[0], dtype=np.int);
                        speed = np.zeros(coord.shape[0])
                        
                        #Final assigment. Only allow assigments within a maximum allowed distance, and an area ratio
                        for row, column in assigment:
                            if costMatrix[row,column] < MAX_ALLOWED_DIST:
                                area_ratio = area[column]/area_prev[row];
                                
                                if area_ratio>AREA_RATIO_LIM[0] and area_ratio<AREA_RATIO_LIM[1]:
                                    index_list[column] = index_list_prev[row];
                                    speed[column] = costMatrix[row][column];
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
                        speed = n_new_worms*[None]
                        
                    #append the new feature list to the pytable
                    mask_feature_list = zip(*([tuple(index_list)] + mask_feature_list + \
                    [ tuple(speed), tuple(len(index_list)*[0]) ] ))
                    buff_feature_table += mask_feature_list
                    if buff_ind == buff_tot-1 : buff_last += mask_feature_list
                else:
                    #consider the case where not valid coordinates where found
                    coord = np.empty([0]);
                    area = np.empty([0]); 
                    index_list = []
                
                coord_prev = coord;
                area_prev = area;
                index_list_prev = index_list;
        
        buff_last = zip(*buff_last)
        buff_last_coord = np.array(buff_last[2:4]).T
        buff_last_index = np.array(buff_last[0])
        buff_last_area = np.array(buff_last[4])
        
        #append data to pytables
        feature_table.append(buff_feature_table)
        
        #print tot_worms, len(buff_feature_table)
        print frame_number, time.time() - tic
    
    #close files and create indexes
    feature_table.flush()
    feature_table.cols.frame_number.create_csindex()
    feature_table.cols.worm_index.create_csindex()
    feature_fid.close()
        
#    index = np.array(zip(*buff_feature_table)[0])
#    x = np.array(zip(*buff_feature_table)[2])
#    y = np.array(zip(*buff_feature_table)[3])
#    plt.figure()
#    plt.imshow(image_buffer[-1,:,:], cmap = 'gray', interpolation = 'none')
#    for ii in np.unique(index):
#        good = index==ii
#        if np.sum(good)>100:
#            plt.plot(x[good],y[good])    


def joinTrajectories(trajectories_file, MIN_TRACK_SIZE = 50, \
MAX_TIME_GAP = 25, AREA_RATIO_LIM = (0.67, 1.5)):
    #AREA_RATIO_LIM : allowed range between the area ratio of consecutive frames 
    #MIN_TRACK_SIZE : minimum tracksize accepted
    #MAX_TIME_GAP : time gap between joined trajectories

    feature_fid = tables.open_file(trajectories_file, mode = 'r+')
    feature_table = feature_fid.get_node('/plate_worms')
    
    #calculate the track size, and select only tracks with at least MIN_TRACK_SIZE length
    track_size = np.bincount(feature_table.cols.worm_index)    
    indexes = np.arange(track_size.size);
    indexes = indexes[track_size>=MIN_TRACK_SIZE]
    
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
        first_frames['frame_number'] < last_frames['frame_number'][kk] + MAX_TIME_GAP)]
        
        if possible_rows.size > 0:
            areaR = last_frames['area'][kk]/possible_rows['area'];
            
            good = np.bitwise_and(areaR>AREA_RATIO_LIM[0], areaR<AREA_RATIO_LIM[1])
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
#    
    
    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch4_16022015_174636.hdf5'
    trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Trajectory_CaptureTest_90pc_Ch4_16022015_174636.hdf5'

    getTrajectories(masked_image_file, trajectories_file, last_frame = 25)
    joinTrajectories(trajectories_file)
#############


#%%    
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
        coord = [(row['coord_x'], row['coord_y'], row['frame_number']) \
        for row in feature_table.where('worm_index_joined == %i'% ii)]

        coord = np.array(coord).T
        if coord.size!=0:
            plt.plot(coord[0,:], coord[1,:], '-');
        
    feature_fid.close()