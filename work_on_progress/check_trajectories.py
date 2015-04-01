# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 09:41:48 2015

@author: ajaver
"""

import pandas as pd
#import matplotlib.pylab as plt
import numpy as np
import h5py
import cv2
import time
import tables
import matplotlib.pylab as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Capture_Ch4_23032015_111907.hdf5'
trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/trajectories/Capture_Ch4_23032015_111907.hdf5'

#read that frame an select trajectories that were considered valid by join_trajectories
table_fid = pd.HDFStore(trajectories_file, 'r')
df_I = table_fid['/plate_worms']
df_I =  df_I[df_I['worm_index_joined'] > 0]

tracks_data = df_I[['worm_index_joined', 'frame_number', 'coord_x', 'coord_y']].groupby('worm_index_joined').aggregate(['max', 'min', 'count'])

#filter for trajectories that move too little (static objects)
MIN_DISPLACEMENT = 20;
ROI_SIZE = 130;

delX = tracks_data['coord_x']['max'] - tracks_data['coord_x']['min']
delY = tracks_data['coord_y']['max'] - tracks_data['coord_y']['min']

good_index_I = tracks_data[(delX>MIN_DISPLACEMENT) & (delY>MIN_DISPLACEMENT)].index
df_I = df_I[df_I.worm_index_joined.isin(good_index_I)]
table_fid.close()


table_fid = pd.HDFStore(trajectories_file, 'r');
df = table_fid['/plate_worms'];
df =  df[df['worm_index_joined'] > 0]
good_index = df[df['segworm_id']>=0]['worm_index_joined'].unique();
df = df[df.worm_index_joined.isin(good_index)];
table_fid.close()


WINDOW_SIZE = 101;
smoothed_CM = {};
for worm_index in good_index:
    dat = df[df['worm_index_joined']==worm_index][['coord_x', 'coord_y', 'frame_number']]
    x = np.array(dat['coord_x']);
    y = np.array(dat['coord_y']);
    t = np.array(dat['frame_number']);
    
    
    first_frame = np.min(t);
    last_frame = np.max(t);
    tnew = np.arange(first_frame, last_frame+1);
    if len(tnew) <= WINDOW_SIZE:
        continue
    
    fx = interp1d(t, x)
    xnew = savgol_filter(fx(tnew), 101, 3);
    fy = interp1d(t, y)
    ynew = savgol_filter(fy(tnew), 101, 3);
    
    smoothed_CM[worm_index] = {}
    smoothed_CM[worm_index]['coord_x'] = xnew
    smoothed_CM[worm_index]['coord_y'] = ynew
    smoothed_CM[worm_index]['first_frame'] = first_frame
    smoothed_CM[worm_index]['last_frame'] = last_frame