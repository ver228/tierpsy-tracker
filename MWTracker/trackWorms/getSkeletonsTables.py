# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:21:39 2015

@author: ajaver
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:39:41 2015

@author: ajaver
"""
import pandas as pd
import tables #h5py gives an error when I tried to do a large amount of write operations (~1e6)
import os, sys
import shutil
import cv2
import numpy as np

from scipy.ndimage.filters import median_filter
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from ..helperFunctions.timeCounterStr import timeCounterStr
from .segWormPython.mainSegworm import getSkeleton

def getSmoothTrajectories(trajectories_file, roi_size = -1, displacement_smooth_win = 101, 
    min_displacement = 0, threshold_smooth_win = 501):
    '''
    Smooth trajectories and thresholds created by getWormTrajectories. 
    If min_displacement is specified there is the option to filter immobile particles, typically spurious.
    '''
    #read that frame an select trajectories that were considered valid by join_trajectories
    with pd.HDFStore(trajectories_file, 'r') as table_fid:
        df = table_fid['/plate_worms'][['worm_index_joined', 'frame_number', \
        'coord_x', 'coord_y','threshold', 'bounding_box_xmax', 'bounding_box_xmin',\
        'bounding_box_ymax' , 'bounding_box_ymin']]
        
        df =  df[df['worm_index_joined'] > 0]
    
    tracks_data = df.groupby('worm_index_joined').aggregate(['max', 'min', 'count'])
    
    #filter for trajectories that move too little (static objects)
    if min_displacement > 0:
        delX = tracks_data['coord_x']['max'] - tracks_data['coord_x']['min']
        delY = tracks_data['coord_y']['max'] - tracks_data['coord_y']['min']
        
        good_index = tracks_data[(delX>min_displacement) & (delY>min_displacement)].index
        df = df[df.worm_index_joined.isin(good_index)]
    
    #get the total length of the tracks, this is more accurate than using count since parts of the track could have got lost for a few frames
    track_lenghts = (tracks_data['frame_number']['max'] - tracks_data['frame_number']['min']+1)
    tot_rows_ini = track_lenghts[track_lenghts>displacement_smooth_win].sum()
    del track_lenghts


    #add the possibility to have variable size ROI
    if roi_size <= 0:
        bb_x = df['bounding_box_xmax']-df['bounding_box_xmin']+1;
        bb_y = df['bounding_box_ymax']-df['bounding_box_ymin']+1;
        worm_lim = pd.concat([bb_x, bb_y], axis=1).max(axis=1)
        
        df_bb = pd.DataFrame({'worm_index_joined':df['worm_index_joined'], 'roi_range': worm_lim})
        #roi_size = df_bb.groupby('worm_index').agg([max , functools.partial(np.percentile, q=0.98)])
        roi_range = df_bb.groupby('worm_index_joined').agg(max) + 10
        roi_range = dict(roi_range['roi_range'])
    else:
        roi_range = {ii:roi_size for ii in np.unique(df['worm_index_joined'])}

    #initialize output data as a numpy recarray (pytables friendly format)
    trajectories_df = np.recarray(tot_rows_ini, dtype = [('frame_number', np.int32), \
    ('worm_index_joined', np.int32), \
    ('plate_worm_id', np.int32), ('skeleton_id', np.int32), \
    ('coord_x', np.float32), ('coord_y', np.float32), ('threshold', np.float32), 
    ('has_skeleton', np.uint8), ('roi_size', np.float32)])

    #store the maximum and minimum frame of each worm
    worms_frame_range = {}
    
    #smooth trajectories (reduce giggling from the CM to obtain a nicer video)
    #interpolate for possible missing frames in the trajectories
    tot_rows = 0;    
    for worm_index, worm_data in df.groupby('worm_index_joined'):
        x = worm_data['coord_x'].values
        y = worm_data['coord_y'].values
        t = worm_data['frame_number'].values
        thresh = worm_data['threshold'].values
        
        first_frame = np.min(t);
        last_frame = np.max(t);
        worms_frame_range[worm_index] = (first_frame, last_frame)
        
        tnew = np.arange(first_frame, last_frame+1);
        if len(tnew) <= displacement_smooth_win:
            continue
        
        #iterpolate missing points in the trajectory and smooth data using the savitzky golay filter
        fx = interp1d(t, x)
        fy = interp1d(t, y)
        xnew = fx(tnew)
        ynew = fy(tnew)

        if displacement_smooth_win > 3:
            xnew = savgol_filter(xnew, displacement_smooth_win, 3);
            ynew = savgol_filter(ynew, displacement_smooth_win, 3);
        
        #smooth the threshold (the worm intensity shouldn't change abruptly along the trajectory)
        fthresh = interp1d(t, thresh)
        threshnew = median_filter(fthresh(tnew), threshold_smooth_win);
        
        #skeleton_id useful to organize the data in the other tables (skeletons, contours, etc)
        new_total = tot_rows + xnew.size
        skeleton_id = np.arange(tot_rows, new_total, dtype = np.int32);
        tot_rows = new_total
        
        #store the indexes in the original plate_worms table
        plate_worm_id = np.empty(xnew.size, dtype = np.int32)
        plate_worm_id.fill(-1)
        plate_worm_id[t - first_frame] =  worm_data.index
    
        trajectories_df['worm_index_joined'][skeleton_id] = worm_index
        trajectories_df['coord_x'][skeleton_id] = xnew
        trajectories_df['coord_y'][skeleton_id] = ynew
        trajectories_df['frame_number'][skeleton_id] = np.arange(first_frame, last_frame+1, dtype=np.int32)
        trajectories_df['threshold'][skeleton_id] = threshnew
        trajectories_df['plate_worm_id'][skeleton_id] = plate_worm_id
        trajectories_df['skeleton_id'][skeleton_id] = skeleton_id
        trajectories_df['has_skeleton'][skeleton_id] = False
        trajectories_df['roi_size'][skeleton_id] = roi_range[worm_index]
        
    assert tot_rows == tot_rows_ini
    
    return trajectories_df, worms_frame_range, tot_rows

def getWormROI(img, CMx, CMy, roi_size = 128):
    '''
    Extract a square Region Of Interest (ROI)
    img - 2D numpy array containing the data to be extracted
    CMx, CMy - coordinates of the center of the ROI
    roi_size - side size in pixels of the ROI
    '''

    if np.isnan(CMx) or np.isnan(CMy):
        return np.zeros(0, dtype=np.uint8), np.array([np.nan]*2)

    roi_center = int(roi_size)//2
    roi_range = np.array([-roi_center, roi_center])

    #obtain bounding box from the trajectories
    range_x = round(CMx) + roi_range
    range_y = round(CMy) + roi_range
    
    if range_x[0]<0: range_x -= range_x[0]
    if range_y[0]<0: range_y -= range_y[0]
    
    if range_x[1]>img.shape[1]: range_x += img.shape[1]-range_x[1]-1
    if range_y[1]>img.shape[0]: range_y += img.shape[0]-range_y[1]-1
    worm_img = img[range_y[0]:range_y[1], range_x[0]:range_x[1]]
    
    roi_corner = np.array([range_x[0]-1, range_y[0]-1])
    
    return worm_img, roi_corner

def getWormMask(worm_img, threshold):
    '''
    Calculate worm mask using an specific threshold.
    '''

    if np.any(worm_img.shape<3):
        return np.zeros_like(worm_img)
    
    #make the worm more uniform. This is important to get smoother contours.
    worm_img = cv2.medianBlur(worm_img, 3);
    
    #compute the thresholded mask
    worm_mask = ((worm_img < threshold) & (worm_img!=0)).astype(np.uint8)
    
    #smooth mask by morphological closing
    worm_mask = cv2.morphologyEx(worm_mask, cv2.MORPH_CLOSE,np.ones((3,3)))

    return worm_mask


def trajectories2Skeletons(masked_image_file, skeletons_file, trajectories_file, \
create_single_movies = False, resampling_N = 49, min_mask_area = 50, smoothed_traj_param = {}):    
    
    #extract the base name from the masked_image_file. This is used in the progress status.
    base_name = masked_image_file.rpartition('.')[0].rpartition(os.sep)[-1]

    #pytables filters.
    table_filters = tables.Filters(complevel=5, complib='zlib', shuffle=True, fletcher32=True)
    
    #get trajectories, threshold and indexes from the first part of the tracker.
    #Note that data is sorted by worm index. This speed up access for access individual worm data.
    trajectories_df, worms_frame_range, tot_rows = \
    getSmoothTrajectories(trajectories_file, **smoothed_traj_param)
    
    
    #pytables saving format is more convenient...
    with tables.File(skeletons_file, "w") as ske_file_id:
        ske_file_id.create_table('/', 'trajectories_data', obj = trajectories_df, filters=table_filters)
    
    
    #...but it is easier to process data with pandas
    with pd.HDFStore(skeletons_file, 'r') as ske_file_id:
        trajectories_df = ske_file_id['/trajectories_data']
    
    #open skeleton file for append and #the compressed videos as read
    with tables.File(skeletons_file, "r+") as ske_file_id, \
    tables.File(masked_image_file, 'r') as mask_fid:
        
        mask_dataset = mask_fid.get_node("/mask")
        
        skel_arrays = {}
        
        data_strS = ['skeleton', 'contour_side1', 'contour_side2']        
        
        #initialize compressed arrays to save the data. Note that the data will be sorted according to trajectories_df
        for data_str in data_strS:
            length_str = data_str + '_length'
            
            skel_arrays[length_str] = ske_file_id.create_carray("/", length_str, \
                        tables.Float32Atom(dflt=np.nan), (tot_rows,), filters=table_filters)
    
            skel_arrays[data_str] = ske_file_id.create_carray("/", data_str, \
                                        tables.Float32Atom(dflt=np.nan), \
                                       (tot_rows, resampling_N, 2), filters=table_filters, \
                                        chunkshape = (1, resampling_N,2));
                        
        skel_arrays['contour_width'] = ske_file_id.create_carray('/', "contour_width", \
                                        tables.Float32Atom(dflt=np.nan), \
                                        (tot_rows, resampling_N), filters=table_filters, \
                                        chunkshape = (1, resampling_N));

        skel_arrays['contour_area'] = ske_file_id.create_carray('/', "contour_area", \
                                        tables.Float32Atom(dflt = np.nan), \
                                        (tot_rows,), filters = table_filters);



        #flags to mark if a frame was skeletonized
        has_skeleton = ske_file_id.get_node('/trajectories_data').cols.has_skeleton

        #flag to mark if this function finished succesfully
        skel_arrays['skeleton']._v_attrs['has_finished'] = 0;

        #dictionary to store previous skeletons
        prev_skeleton = {}
        
        #timer
        progressTime = timeCounterStr('Calculation skeletons.');
        for frame, frame_data in trajectories_df.groupby('frame_number'):
            
            img = mask_dataset[frame,:,:]
            for skeleton_id, row_data in frame_data.iterrows():
                
                worm_img, roi_corner = getWormROI(img, row_data['coord_x'], row_data['coord_y'], row_data['roi_size'])
                worm_mask = getWormMask(worm_img, row_data['threshold'])
                
                worm_index = row_data['worm_index_joined']
                if not worm_index in prev_skeleton:
                    prev_skeleton[worm_index] = np.zeros(0)
                
                #get skeletons
                
                skeleton, ske_len, cnt_side1, cnt_side1_len, cnt_side2, cnt_side2_len, cnt_widths, cnt_area = \
                getSkeleton(worm_mask, prev_skeleton[worm_index], resampling_N, min_mask_area)
                                
                if skeleton.size>0:
                    prev_skeleton[worm_index] = skeleton.copy()
                    
                    #save segwrom_results
                    skel_arrays['skeleton_length'][skeleton_id] = ske_len 
                    skel_arrays['contour_side1_length'][skeleton_id] = cnt_side1_len
                    skel_arrays['contour_side2_length'][skeleton_id] = cnt_side2_len
    
                    skel_arrays['contour_width'][skeleton_id, :] = cnt_widths                
                    skel_arrays['contour_area'][skeleton_id] = cnt_area                
                                        

                    #convert into the main image coordinates
                    skel_arrays['skeleton'][skeleton_id, :, :] = skeleton + roi_corner
                    skel_arrays['contour_side1'][skeleton_id, :, :] = cnt_side1 + roi_corner
                    skel_arrays['contour_side2'][skeleton_id, :, :] = cnt_side2 + roi_corner
                    
                    has_skeleton[skeleton_id] = True
            if frame % 500 == 0:
                progress_str = progressTime.getStr(frame)
                print(base_name + ' ' + progress_str);
                sys.stdout.flush()
        #Mark a succesful termination
        skel_arrays['skeleton']._v_attrs['has_finished'] = 1;


#drawWormContour and writeIndividualMovies are used to create individual worm movies.
def drawWormContour(worm_img, worm_mask, skeleton, cnt_side1, cnt_side2, \
colorpalette = [(119, 158,27 ), (2, 95, 217), (138, 41, 231)]):
    
    '''
    Draw the worm contour and skeleton. If the contour is not valid, draw the thresholded mask.
    '''
    
    assert worm_img.dtype == np.uint8
    

    max_int = np.max(worm_img)
    
    #the image is likely to be all zeros
    if max_int==0:
        return cv2.cvtColor(worm_img, cv2.COLOR_GRAY2RGB);

    #rescale the intensity range for visualization purposes.
    intensity_rescale = 255./min(1.1*max_int,255.);
    worm_img = (worm_img*intensity_rescale).astype(np.uint8)
            
    worm_rgb = cv2.cvtColor(worm_img, cv2.COLOR_GRAY2RGB);
    if skeleton.size==0 or np.all(np.isnan(skeleton)):
        worm_rgb[:,:,1][worm_mask!=0] = 204
        worm_rgb[:,:,2][worm_mask!=0] = 102
    else:
        pts = np.round(cnt_side1).astype(np.int32)
        cv2.polylines(worm_rgb, [pts], False, colorpalette[1], thickness=1, lineType = 8)
        pts = np.round(cnt_side2).astype(np.int32)
        cv2.polylines(worm_rgb, [pts], False, colorpalette[2], thickness=1, lineType = 8)
        
        pts = np.round(skeleton).astype(np.int32)
        cv2.polylines(worm_rgb, [pts], False, colorpalette[0], thickness=1, lineType = 8)
        
        #mark the head
        cv2.circle(worm_rgb, tuple(pts[0]), 2, (225,225,225), thickness=-1, lineType = 8)
        cv2.circle(worm_rgb, tuple(pts[0]), 3, (0,0,0), thickness=1, lineType = 8)
    
    return worm_rgb


def writeIndividualMovies(masked_image_file, skeletons_file, video_save_dir, 
                          fps=25, bad_seg_thresh = 0.5, save_bad_worms = False):    
    
    '''
        Create individual worms videos.
        
        masked_image_file - hdf5 with the masked videos.
        skeleton_file - file with skeletons and trajectory data previously created by trajectories2Skeletons
        video_save_dir - directory where individual videos are saved
        roi_size - region of interest size.
        fps - frames per second in the individual video.
        bad_seg_thresh - min the fraction of skeletonized frames in the whole trajectory, allowed before being rejected (bad_worm).
        save_bad_worms - (bool flag) if True videos from bad worms are created.
    '''
    
    #extract the base name from the masked_image_file. This is used in the progress status.
    base_name = masked_image_file.rpartition('.')[0].rpartition(os.sep)[-1]
    
    #remove previous data if exists
    if os.path.exists(video_save_dir):
        shutil.rmtree(video_save_dir)
    os.makedirs(video_save_dir)
    if save_bad_worms:
        bad_videos_dir = video_save_dir + 'bad_worms' + os.sep;
        os.mkdir(bad_videos_dir)
    
    #data to extract the ROI
    with pd.HDFStore(skeletons_file, 'r') as ske_file_id:
        trajectories_df = ske_file_id['/trajectories_data']
        skeleton_fracc = trajectories_df[['worm_index_joined', 'has_skeleton']].groupby('worm_index_joined').agg('mean')
        skeleton_fracc = skeleton_fracc['has_skeleton']
        valid_worm_index = skeleton_fracc[skeleton_fracc>=bad_seg_thresh].index
        if not save_bad_worms:
            #remove the bad worms, we do not care about them
            trajectories_df = trajectories_df[trajectories_df['worm_index_joined'].isin(valid_worm_index)]

    with tables.File(skeletons_file, "r") as ske_file_id, tables.File(masked_image_file, 'r') as mask_fid:
        #pointers to masked images dataset
        mask_dataset = mask_fid.get_node("/mask")

        #pointers to skeletons, and contours
        skel_array = ske_file_id.get_node('/skeleton')
        cnt1_array = ske_file_id.get_node('/contour_side1')
        cnt2_array = ske_file_id.get_node('/contour_side2')

        #get first and last frame for each worm
        worms_frame_range = trajectories_df.groupby('worm_index_joined').agg({'frame_number': [min, max]})['frame_number']
        
        video_list = {}
        progressTime = timeCounterStr('Creating individual worm videos.');
        for frame, frame_data in trajectories_df.groupby('frame_number'):
            assert isinstance(frame, (np.int64, int))

            img = mask_dataset[frame,:,:]
            for skeleton_id, row_data in frame_data.iterrows():
                worm_index = row_data['worm_index_joined']
                assert not np.isnan(row_data['coord_x']) and not np.isnan(row_data['coord_y'])

                worm_img, roi_corner = getWormROI(img, row_data['coord_x'], row_data['coord_y'], row_data['roi_size'])

                skeleton = skel_array[skeleton_id,:,:]-roi_corner
                cnt_side1 = cnt1_array[skeleton_id,:,:]-roi_corner
                cnt_side2 = cnt2_array[skeleton_id,:,:]-roi_corner
                
                if not row_data['has_skeleton']:
                    #if it does not have an skeleton get a the thresholded mask
                    worm_mask = getWormMask(worm_img, row_data['threshold'])
                else:
                    worm_mask = np.zeros(0)
                
                if (worms_frame_range['min'][worm_index] == frame) or (not worm_index in video_list):
                    #if it is the first frame of a worm trajectory initialize a VideoWriter object
                    
                    movie_save_name = video_save_dir + ('worm_%i.avi' % worm_index)
                    if save_bad_worms and not worm_index in valid_worm_index:
                        #if we want to save the 'bad worms', save them in a different directory
                        movie_save_name = bad_videos_dir + ('worm_%i.avi' % worm_index)
                    else:
                        movie_save_name = video_save_dir + ('worm_%i.avi' % worm_index)
                    
                    video_list[worm_index] = cv2.VideoWriter(movie_save_name, \
                    cv2.VideoWriter_fourcc('M','J','P','G'), fps, (row_data['roi_size'], row_data['roi_size']), isColor=True)
                
                #draw contour/mask
                worm_rgb = drawWormContour(worm_img, worm_mask, skeleton, cnt_side1, cnt_side2)
                assert (worm_rgb.shape[0] == worm_img.shape[0]) and (worm_rgb.shape[1] == worm_img.shape[1]) 

                #write video frame
                video_list[worm_index].write(worm_rgb)
            
                #release VideoWriter object if it is the last frame of the trajectory
                if (worms_frame_range['max'][worm_index] == frame):
                    video_list[worm_index].release();
                    video_list.pop(worm_index, None)
            
            #update progress bar
            if frame % 500 == 0:
                progress_str = progressTime.getStr(frame)
                print(base_name + ' ' + progress_str);
                sys.stdout.flush()
#%%
if __name__ == '__main__':  
    #masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/Masked_videos/20150512/Capture_Ch3_12052015_194303.hdf5'
    #save_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150512/'    
    
    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/Masked_videos/20150511/Capture_Ch5_11052015_195105.hdf5'
    save_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150511/'    
    
    base_name = masked_image_file.rpartition(os.sep)[-1].rpartition('.')[0]
    trajectories_file = save_dir + base_name + '_trajectories.hdf5'    
    skeletons_file = save_dir + base_name + '_skeletons.hdf5'
    video_save_dir = save_dir + base_name + os.sep

    assert os.path.exists(masked_image_file)
    assert os.path.exists(trajectories_file)
    
    #trajectories2Skeletons(masked_image_file, skeletons_file, trajectories_file, \
    #create_single_movies = False, roi_size = 128, resampling_N = 49, min_mask_area = 50)
    #from checkHeadOrientation import correctHeadTail
    #correctHeadTail(skeletons_file, max_gap_allowed = 10, \
    #window_std = 25, segment4angle = 5, min_block_size = 250)

    #writeIndividualMovies(masked_image_file, skeletons_file, video_save_dir, roi_size = 128, fps=25, save_bad_worms=True)