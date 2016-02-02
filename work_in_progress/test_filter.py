# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 01:37:44 2016

@author: ajaver
"""
import pandas as pd

trajectories_file = '/Users/ajaver/Desktop/Videos/Camille/Results/CSTCTest_Ch2_30102015_212048_trajectories.hdf5'

with pd.HDFStore(trajectories_file, 'r') as table_fid:
    df = table_fid['/plate_worms'][['worm_index_joined', 'frame_number', \
    'coord_x', 'coord_y','threshold', 'bounding_box_xmax', 'bounding_box_xmin',\
    'bounding_box_ymax' , 'bounding_box_ymin', 'area']]
    
    df =  df[df['worm_index_joined'] > 0]


min_track_size = 2
tracks_data = df.groupby('worm_index_joined').aggregate(['max', 'min', 'count'])
track_lenghts = (tracks_data['frame_number']['max'] - tracks_data['frame_number']['min']+1)
tot_rows_ini = track_lenghts[track_lenghts>min_track_size].sum()
    