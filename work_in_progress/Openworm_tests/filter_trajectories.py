# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 20:38:01 2015

@author: ajaver
"""
import os
import pandas as pd
import matplotlib.pylab as plt
 
masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/Masked_videos/20150511/Capture_Ch5_11052015_195105.hdf5'
save_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150511/'    
    
base_name = masked_image_file.rpartition(os.sep)[-1].rpartition('.')[0]
trajectories_file = save_dir + base_name + '_trajectories.hdf5'    

#def getSmoothTrajectories(trajectories_file, displacement_smooth_win = 101, min_displacement = 0, threshold_smooth_win = 501):
if __name__ == '__main__':
    
    min_displacement = 0
    #read that frame an select trajectories that were considered valid by join_trajectories
    table_fid = pd.HDFStore(trajectories_file, 'r')
    df = table_fid['/plate_worms']#[['worm_index_joined', 'frame_number', 'coord_x', 'coord_y','threshold']]
    df =  df[df['worm_index_joined'] > 0]
    table_fid.close()
    
    tracks_data = df.groupby('worm_index_joined').aggregate(['max', 'min', 'count', 'median'])
    #if min_displacement > 0:
    #filter for trajectories that move too little (static objects)     
    delX = tracks_data['coord_x']['max'] - tracks_data['coord_x']['min']
    delY = tracks_data['coord_y']['max'] - tracks_data['coord_y']['min']
    
    good_index = tracks_data[(delX>min_displacement) & (delY>min_displacement)].index
