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
import matlab.engine
import time
from StringIO import StringIO

import sys
sys.path.append('../videoCompression/')
from parallelProcHelper import sendQueueOrPrint

def getValidTrajectories(trajectories_file, \
    min_displacement = 20, thresh_smooth_window = 1501, save_csv_name = ''):
    #read that frame an select trajectories that were considered valid by join_trajectories
    table_fid = pd.HDFStore(trajectories_file, 'r')
    df = table_fid['/plate_worms']
    df =  df[df['worm_index_joined'] > 0]
    
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
    
if __name__ == '__main__':
#    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Capture_Ch4_23032015_111907.hdf5'
#    trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/trajectories/Capture_Ch4_23032015_111907.hdf5'
#    segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/trajectories/Capture_Ch4_23032015_111907_segworm.hdf5'

#    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/prueba/CaptureTest_90pc_Ch1_02022015_141431.hdf5'
#    trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/prueba/trajectories/CaptureTest_90pc_Ch1_02022015_141431.hdf5'
#    segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/prueba/trajectories/CaptureTest_90pc_Ch1_02022015_141431_segworm.hdf5'
#    save_csv_name = '/Users/ajaver/Desktop/Gecko_compressed/prueba/trajectories/CaptureTest_90pc_Ch1_02022015_141431.csv';
    #df = getValidTrajectories(trajectories_file, save_csv_name = save_csv_name)
    #getSegWorm(masked_image_file, trajectories_file, segworm_file)

    trajectories_file = r'/Users/ajaver/Desktop/sygenta/Trajectories/data_20150114/control_9_fri_12th_dec_2_trajectories.hdf5'
    save_csv_name = '/Users/ajaver/Desktop/sygenta/Trajectories/control_9_fri_12th_dec_2.csv'
    df = getValidTrajectories(trajectories_file, save_csv_name = save_csv_name)