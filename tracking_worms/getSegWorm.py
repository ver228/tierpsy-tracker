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
import h5py
import tables
import time
from StringIO import StringIO
import os
import sys
sys.path.append('../videoCompression/')
from parallelProcHelper import sendQueueOrPrint

def getValidTrajectories(trajectories_file, \
    min_displacement = 20, thresh_smooth_window = 1501, save_csv_name = ''):
    #read that frame an select trajectories that were considered valid by join_trajectories
    table_fid = pd.HDFStore(trajectories_file, 'r')
    df = table_fid['/plate_worms']
    df =  df[df['worm_index_joined'] > 0]
    #df =  df[df['worm_index_joined'] == 8]
    
    tracks_data = df[['worm_index_joined', 'frame_number', 'coord_x', 'coord_y']].groupby('worm_index_joined').aggregate(['max', 'min', 'count'])
    
    #filter for trajectories that move too little (static objects)
     
    delX = tracks_data['coord_x']['max'] - tracks_data['coord_x']['min']
    delY = tracks_data['coord_y']['max'] - tracks_data['coord_y']['min']
    
    good_index = tracks_data[(delX>min_displacement) & (delY>min_displacement)].index
    df = df[df.worm_index_joined.isin(good_index)]
    
    from scipy.ndimage.filters import median_filter
    for dat_group in df.groupby('worm_index_joined'):
        dat = dat_group[1][['threshold', 'frame_number']].sort('frame_number')
        df.loc[dat.index,'threshold'] = median_filter(dat['threshold'],thresh_smooth_window)
        
    table_fid.close()
    
    if save_csv_name:
        df[['worm_index_joined', 'frame_number', 'coord_x', 'coord_y', 'threshold']].to_csv(save_csv_name, index = True, header = True);
    
    return df;

def getSegWorm(masked_image_file, trajectories_file, segworm_file, \
status_queue = '', base_name = '', min_displacement = 20, thresh_smooth_window = 1501):
    sendQueueOrPrint(status_queue, 'Obtaining valid trajectories...', base_name);
    df = getValidTrajectories(trajectories_file, min_displacement, thresh_smooth_window);
    if len(df)==0:
        print "Segworm: no valid trajectories, nothing to do here"
        return

    sendQueueOrPrint(status_queue, 'Initializing MATLAB...', base_name);
    eng = matlab.engine.start_matlab()
    #eng.addpath(eng.genpath('/Users/ajaver/GitHub_repositories/SegWorm/Only_segWorm'));
    eng.addpath(eng.genpath('/Users/ajaver/GitHub_repositories/Multiworm_Tracking/OnlySegWorm/'));
    eng.warning('off', 'normWorms:VulvaContourTooShort')
    eng.warning('off', 'normWorms:NonVulvaContourTooShort')
    
    #export data into a matlab format
    data = {'plate_worms_id':matlab.double(list(df.index)), 
    'frame_number':matlab.double(list(df['frame_number'])), 
    'worm_index_joined':matlab.double(list(df['worm_index_joined'])),
    'coord_x':matlab.double(list(df['coord_x'])),
    'coord_y':matlab.double(list(df['coord_y'])),
    'threshold':matlab.double(list(df['threshold']))};
    
    fun_output = StringIO()
    #calculate segworm data using the MATLAB engine
    future = eng.movie2segwormfun(data, masked_image_file, segworm_file, \
    nargout = 0, async = True)#, stdout = fun_output)
    sendQueueOrPrint(status_queue, 'Segworm started.', base_name);
    
    while not future.done():
        time.sleep(1.0)
#        progress_str = fun_output.getvalue()
#        #print 'a', progress_str
#        if progress_str:
#            print 'a'
#            sendQueueOrPrint(status_queue, progress_str, base_name);
#        
    #update pytables with the correct index for segworm_file
    results_fid = tables.open_file(trajectories_file, 'r+')
    tracking_table = results_fid.get_node('/plate_worms')
    
    segworm_fid = h5py.File(segworm_file, 'r')
    
    plate_worms_id = segworm_fid['/segworm_results/plate_worms_id']
    
    for ii in range(plate_worms_id.size):
        tracking_table.cols.segworm_id[int(plate_worms_id[ii])] = ii;
    
    results_fid.flush()
    results_fid.close()
    segworm_fid.close()
    
    fun_output.close() #close matlab output

    #try quit matlab engine, sometimes it can have problems to be close...
    try:
        eng.quit()
    except:
        pass;
    del eng
    
    sendQueueOrPrint(status_queue, 'Skeletonization completed.', base_name);
    
def getSegWorm_noMATLABeng(masked_image_file, trajectories_file, segworm_file, \
status_queue = '', base_name = '', min_displacement = 20, 
thresh_smooth_window = 1501):

    sendQueueOrPrint(status_queue, 'Obtaining valid trajectories...', base_name);
    csv_tmp_dir = os.path.split(trajectories_file)[0] + os.sep
    csv_tmp = os.path.abspath(csv_tmp_dir + 'tmp_' + base_name + '.csv')
    
    df = getValidTrajectories(trajectories_file, min_displacement, thresh_smooth_window, save_csv_name = csv_tmp);
    if len(df)==0:
        print "Segworm: no valid trajectories, nothing to do here"
        return
    else:
        del(df) #no longer needed
        
    #calculate segworm data using the MATLAB engine
#%%
    matlab_path = (os.sep).join([os.path.dirname(os.path.abspath(__file__)), '..', 'OnlySegWorm']) + os.sep;
    cmd = """matlab -nojvm -nodisplay -nosplash -r "addpath(genpath('%s')); movie2segworm_csv('%s', '%s', '%s'); exit();" </dev/null """  \
    % (matlab_path, csv_tmp, masked_image_file, segworm_file);

    print(cmd)   
    os.system(cmd) #excecute MATLAB program
    
    if os.path.exists(csv_tmp):
        print("Segworm MATLAB subprocess was not completed succesfully.")
        raise 
    
#%%     
    #update pytables with the correct index for segworm_file
    results_fid = tables.open_file(trajectories_file, 'r+')
    tracking_table = results_fid.get_node('/plate_worms')
    
    segworm_fid = h5py.File(segworm_file, 'r')
    
    plate_worms_id = segworm_fid['/segworm_results/plate_worms_id']
    
    tracking_table.cols.segworm_id[:] = [-1]*tracking_table.nrows #make sure all that there were not previous values in the segworm_id
    for ii in range(plate_worms_id.size):
        tracking_table.cols.segworm_id[int(plate_worms_id[ii])] = ii;
    
    results_fid.flush()
    results_fid.close()
    segworm_fid.close()
    
    
    print(base_name + ' Skeletonization completed.');    
#%%  
    
import cv2
import numpy as np

if __name__ == '__main__':
    masked_movies_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Compressed/'
    trajectories_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Trajectories/'
    base_name = 'Capture_Ch1_11052015_195105'


    masked_image_file = masked_movies_dir + base_name + '.hdf5'
    trajectories_file = trajectories_dir + base_name + '_trajectories.hdf5'
    segworm_file = trajectories_dir + base_name + '_segworm.hdf5'
    
    roi_size = 128
    min_mask_area = 50
    
    roi_center = roi_size/2
    roi_window = [-roi_center, roi_center]

    mask_fid = h5py.File(masked_image_file, 'r');
    mask_dataset = mask_fid["/mask"]    
    
    df = getValidTrajectories(trajectories_file)
    df = df[['worm_index_joined', 'frame_number', 'coord_x', 'coord_y', 'threshold']]
    df = df.sort(['worm_index_joined', 'frame_number'])
    
    df['segword_id'] = np.arange(len(df))
    #df = df.sort('frame_number')
    #%%
    for frame, frame_data in df.groupby('frame_number'):
        print frame
        img = mask_dataset[frame,:,:]
        for plate_worms_id, row_data in frame_data.iterrows():
            #obtain bounding box from the trajectories
            
            worm_CM = np.round([row_data['coord_x'], row_data['coord_y']])
            range_x = worm_CM[0] + roi_window
            range_y = worm_CM[1] + roi_window
            
            if range_x[0]<0: range_x -= range_x[0]
            if range_y[0]<0: range_y -= range_y[0]
            
            if range_x[1]>img.shape[1]: range_x += img.shape[1]-range_x[1]-1
            if range_y[1]>img.shape[0]: range_y += img.shape[0]-range_y[1]-1
            
            worm_img =  img[range_y[0]:range_y[1], range_x[0]:range_x[1]]
            worm_mask = ((worm_img < row_data['threshold']) & (worm_img!=0)).astype(np.uint8)        
            worm_mask = cv2.morphologyEx(worm_mask, cv2.MORPH_CLOSE,np.ones((5,5)))




