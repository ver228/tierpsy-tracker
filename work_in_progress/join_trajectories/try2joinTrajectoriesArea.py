# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:46:36 2016

@author: ajaver
"""
import pandas as pd
import os
import tables
import numpy as np
import matplotlib.pylab as plt
from collections import OrderedDict

import cv2
from scipy.signal import savgol_filter
from scipy.signal import medfilt

import sys
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking')
from MWTracker.helperFunctions.timeCounterStr import timeCounterStr

from MWTracker.trackWorms.getSkeletonsTables import getWormROI, getWormMask, binaryMask2Contour

def getAreaIntersecRatio(connect_dict, node1_cnts, node2_cnts):    

    intersect_ratio = {}
    for current_ind in connect_dict:
        current_cnt = node1_cnts[current_ind]
        
        bot = np.min(current_cnt, axis=0);
        top = np.max(current_cnt, axis=0);
        
        for pii in connect_dict[current_ind]:
            bot_p = np.min(node2_cnts[pii],axis=0);
            top_p = np.max(node2_cnts[pii],axis=0);
            
            bot = np.min((bot, bot_p), axis=0)
            top = np.max((top, top_p), axis=0)
        
        roi_size = top-bot + (1,1)
        roi_size = roi_size[::-1]
    
        mask_curr = np.zeros(roi_size, np.int32)
        worm_cnt = [(current_cnt-bot).astype(np.int32)];
        cv2.drawContours(mask_curr, worm_cnt, 0, 1, -1)
        area_curr = np.sum(mask_curr)    
        
        for pii in connect_dict[current_ind]:
            mask_possible = np.zeros(roi_size, np.int32)
            worm_cnt = [(node2_cnts[pii]-bot).astype(np.int32)];
            cv2.drawContours(mask_possible, worm_cnt, 0, 1, -1)
            
            area_intersect = np.sum(mask_curr & mask_possible)
        
            intersect_ratio[(current_ind, pii)] = area_intersect/area_curr
        
    return intersect_ratio

if __name__ == '__main__':
    #base directory
    #masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch5_17112015_205616.hdf5'
    #masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch3_17112015_205616.hdf5'
    masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch1_18112015_075624.hdf5'
    #masked_image_file = '/Users/ajaver/Desktop/Videos/04-03-11/MaskedVideos/575 JU440 swimming_2011_03_04__13_16_37__8.hdf5'    
    #masked_image_file = '/Users/ajaver/Desktop/Videos/04-03-11/MaskedVideos/575 JU440 on food Rz_2011_03_04__12_55_53__7.hdf5'    
    
    skeletons_file = masked_image_file.replace('MaskedVideos', 'Results')[:-5] + '_skeletons.hdf5'
    intensities_file = skeletons_file.replace('_skeletons', '_intensities')
    
    #get the trajectories table
    with pd.HDFStore(skeletons_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    #%% 
    #xx_cnt = trajectories_data['cnt_coord_x']
    #xx = trajectories_data['coord_x']
    #good = ~np.isnan(xx_cnt)
    #plt.plot(xx[good]-xx_cnt[good], '.')
    
    #trajectories_data = trajectories_data[good]
    #%%
    
    valid_data = OrderedDict()
    
    grouped_trajectories = trajectories_data.groupby('worm_index_joined')
    
    tot_worms = len(grouped_trajectories)
    base_name = skeletons_file.rpartition('.')[0].rpartition(os.sep)[-1].rpartition('_')[0]
    progress_timer = timeCounterStr('');
    
    win_area = 10    
    
    valid_data = np.recarray(tot_worms, [('worm_index',np.int), ('t0',np.int), ('tf',np.int), \
    ('x0',np.float), ('xf',np.float), ('y0',np.float), ('yf',np.float), \
    ('a0',np.float), ('af',np.float),  ('th0',np.float), ('thf',np.float), \
    ('roi_size',np.int)])
    
    fields_needed = ['coord_x', 'coord_y', 'roi_size', 'frame_number', 'threshold']
    for index_n, (worm_index, trajectories_worm) in enumerate(grouped_trajectories):
        print(worm_index)        
        row0 = trajectories_worm['frame_number'].argmin();
        rowf = trajectories_worm['frame_number'].argmax();
        
        worm_areas = trajectories_worm['cnt_area'].values;
        
        a0 = np.median(worm_areas[:win_area])
        
        af = np.median(worm_areas[-win_area:])
        
        dd = trajectories_worm.loc[[row0, rowf], fields_needed]
        
        t0, tf = dd['frame_number'].values
                
        roi_size = dd['roi_size'].values[0]

        x0, xf = dd['coord_x'].values
        y0, yf = dd['coord_y'].values
        
        th0, th1 = dd['threshold'].values     
        
        valid_data[index_n] = (worm_index, t0, tf, x0,xf, y0,yf, a0, af, th0, th1, roi_size)
        #if worm_index == 50: break;
    valid_data = pd.DataFrame(valid_data, index=valid_data['worm_index'])
    #%%
    grouped_t0 = valid_data.groupby('t0')
    grouped_tf = valid_data.groupby('tf')
    
    uT0 = np.unique(valid_data['t0'])
    uTf = np.unique(valid_data['tf'])
    
    initial_cnt = OrderedDict()
    final_cnt = OrderedDict()
    
    with tables.File(masked_image_file, 'r') as fid:
        mask_group = fid.get_node('/mask')

        for frame_number in np.unique(np.concatenate((uT0,uTf))):
            img = mask_group[frame_number]
            
            
            if frame_number in uT0:
                dd = grouped_t0.get_group(frame_number)
                for ff, row in dd.iterrows():
                    worm_img, roi_corner = getWormROI(img, row['x0'], row['y0'], row['roi_size'])
                    worm_mask = getWormMask(worm_img, row['th0'])
                    worm_cnt, _ = binaryMask2Contour(worm_mask)
                    if worm_cnt.size > 0:
                        worm_cnt += roi_corner                    
                    initial_cnt[int(row['worm_index'])] = worm_cnt
            if frame_number in uTf:
                dd = grouped_tf.get_group(frame_number) 
                for ff, row in dd.iterrows():
                    worm_img, roi_corner = getWormROI(img, row['xf'], row['yf'], row['roi_size'])
                    worm_mask = getWormMask(worm_img, row['thf'])
                    worm_cnt, _ = binaryMask2Contour(worm_mask)
                    if worm_cnt.size > 0:
                        worm_cnt += roi_corner    
                    final_cnt[int(row['worm_index'])] = worm_cnt
                    
            print(frame_number)  
    #%%    
    connect_before = OrderedDict()
    connect_after = OrderedDict()
    
    max_gap = 25;
    for worm_index in valid_data.index:
        curr_data = valid_data.loc[worm_index]
        other_data = valid_data[valid_data.index != worm_index].copy()
        
        other_data['gap'] = curr_data['t0'] - other_data['tf']
        before_data = other_data.query('gap > 0 & gap <=%i' %max_gap).copy()
    
        other_data['gap'] = other_data['t0'] - curr_data['tf']
        after_data =  other_data.query('gap > 0 & gap <=%i' %max_gap).copy()
        
        Rlim = curr_data['roi_size']**2
        
        delXb = curr_data['x0'] - before_data['xf']
        delYb = curr_data['y0'] - before_data['yf']
        before_data['R2'] = delXb*delXb + delYb*delYb
        before_data = before_data[before_data['R2']<=Rlim]
        
        #before_data['AR'] =  curr_data['a0']/before_data['af']
        before_data = before_data[(curr_data['a0']!=0) & (before_data['af']!=0)]
        
        delXa = curr_data['xf'] - after_data['x0']
        delYa = curr_data['yf'] - after_data['y0']
        after_data['R2'] = delXa*delXa + delYa*delYa
        after_data = after_data[after_data['R2']<=Rlim]
        
        #after_data['AR'] =  curr_data['af']/after_data['a0']
        after_data = after_data[(curr_data['af']!=0) & (after_data['a0']!=0)]
        
        assert worm_index == curr_data['worm_index']
        if len(before_data) > 0:        
            connect_before[worm_index] = list(before_data.index.values)
        
        if len(after_data) > 0:        
            connect_after[worm_index] = list(after_data.index.values)
#%%
    #after_ratio = getAreaIntersecRatio(connect_after, initial_cnt, final_cnt)
    #before_ratio = getAreaIntersecRatio(connect_before, final_cnt, initial_cnt)
#%%
    for ii in connect_after:
        print(ii, initial_cnt[ii].shape)
        