# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 19:41:36 2015

@author: ajaver
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

#trajectories_file = r'/Volumes/behavgenom$/syngenta/Trajectories/data_20150114/compound_a_repeat_2_fri_5th_dec_trajectories.hdf5'
trajectories_file = r'/Users/ajaver/Desktop/sygenta/Trajectories/data_20150114/compound_a_repeat_2_fri_5th_dec_trajectories.hdf5'

table_fid = pd.HDFStore(trajectories_file, 'r')
df = table_fid['/plate_worms']
df =  df[df['worm_index_joined'] > 0]

tracks_data = df[['worm_index_joined', 'frame_number', 'coord_x', 'coord_y']].groupby('worm_index_joined').aggregate(['max', 'min', 'count'])

#filter for trajectories that move too little (static objects)
 
delX = tracks_data['coord_x']['max'] - tracks_data['coord_x']['min']
delY = tracks_data['coord_y']['max'] - tracks_data['coord_y']['min']
delT = tracks_data['frame_number']['max'] - tracks_data['frame_number']['min']

plt.figure()
for ind in delT.index:
    good = df['worm_index_joined']==ind;
    plt.plot(df.loc[good,'frame_number'], df.loc[good, 'threshold'], '.-')
#plt.xlim(0,500)
#plt.ylim(0,800)
#plt.axis('equal')
table_fid.close()


smooth_window_size = 101
good_index = df[df['segworm_id']>=0]['worm_index_joined'].unique();

smoothed_CM = {};
for worm_index in good_index:
    dat = df[df['worm_index_joined']==worm_index][['coord_x', 'coord_y', 'frame_number']]
    x = np.array(dat['coord_x']);
    y = np.array(dat['coord_y']);
    t = np.array(dat['frame_number']);
    
    
    first_frame = np.min(t);
    last_frame = np.max(t);
    tnew = np.arange(first_frame, last_frame+1);
    if len(tnew) <= smooth_window_size:
        continue
    
    fx = interp1d(t, x)
    xnew = savgol_filter(fx(tnew), smooth_window_size, 3);
    fy = interp1d(t, y)
    ynew = savgol_filter(fy(tnew), smooth_window_size, 3);
    
    smoothed_CM[worm_index] = {}
    smoothed_CM[worm_index]['coord_x'] = xnew
    smoothed_CM[worm_index]['coord_y'] = ynew
    smoothed_CM[worm_index]['first_frame'] = first_frame
    smoothed_CM[worm_index]['last_frame'] = last_frame