# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:25:41 2015

@author: ajaver
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 22:15:48 2015

@author: ajaver
"""

import pandas as pd
import numpy as np
import tables
from scipy.interpolate import RectBivariateSpline
import sys
import os

import cv2
import matplotlib.pylab as plt

sys.path.append('../trackWorms/')
from WormClass import WormClass

def getAnglesDelta(dx,dy):
    angles = np.arctan2(dx,dy)
    dAngles = np.diff(angles)
        
    
    positiveJumps = np.where(dAngles > np.pi)[0] + 1; #%+1 to cancel shift of diff
    negativeJumps = np.where(dAngles <-np.pi)[0] + 1;
        
    #% subtract 2pi from remainging data after positive jumps
    for jump in positiveJumps:
        angles[jump:] = angles[jump:] - 2*np.pi;
        
    #% add 2pi to remaining data after negative jumps
    for jump in negativeJumps:
        angles[jump:] = angles[jump:] + 2*np.pi;
    
    #% rotate skeleton angles so that mean orientation is zero
    meanAngle = np.mean(angles);
    angles = angles - meanAngle;

    return angles, meanAngle

def calculateHeadTailAng(skeletons, segment4angle, good):
    angles_head = np.empty(skeletons.shape[0])
    angles_head.fill(np.nan)
    angles_tail = angles_head.copy()
    
    dx = skeletons[good,segment4angle,0] - skeletons[good,0,0]
    dy = skeletons[good,segment4angle,1] - skeletons[good,0,1]
    
    angles_head[good], _ = getAnglesDelta(dx,dy)
    
    dx = skeletons[good,-segment4angle-1,0] - skeletons[good,-1,0]
    dy = skeletons[good,-segment4angle-1,1] - skeletons[good,-1,1]
    
    angles_tail[good], _ = getAnglesDelta(dx,dy)
    return angles_head, angles_tail

base_name = 'Capture_Ch3_12052015_194303'
mask_dir = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/20150512/'
results_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150512/'    

#base_name = 'Capture_Ch1_11052015_195105'
#mask_dir = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/20150511/'
#results_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150511/'    

masked_image_file = mask_dir + base_name + '.hdf5'
trajectories_file = results_dir + base_name + '_trajectories.hdf5'
skeletons_file = results_dir + base_name + '_skeletons.hdf5'

#%%
base_name = masked_image_file.rpartition('.')[0].rpartition(os.sep)[-1]


worm_id = 209

max_gap_allowed = 10
min_block_size=250
window_std = 25

with pd.HDFStore(skeletons_file, 'r') as ske_file_id:
    #data to extract the ROI
    trajectories_df = ske_file_id['/trajectories_data']
    trajectories_df = trajectories_df.query('worm_index_joined==%i' % worm_id)
    
    #get the first and last frame of each worm_index
    indexes_data = trajectories_df[['worm_index_joined', 'skeleton_id']]
    rows_indexes = indexes_data.groupby('worm_index_joined').agg([min, max])['skeleton_id']
    del indexes_data



#def getIntensitiesMap(masked_image_file, skeletons_file, intensities_file, roi_size = 128):
with tables.File(masked_image_file, 'r')  as mask_fid, \
     tables.File(skeletons_file, 'r') as ske_file_id:
    for worm_index, row_range in rows_indexes.iterrows():
            
        worm_data = WormClass(skeletons_file, worm_index, \
                        rows_range = (row_range['min'],row_range['max']))
        angles_head, angles_tail = calculateHeadTailAng(worm_data.skeleton, 5, ~np.isnan(worm_data.skeleton[:,0,0]))
        
        ts = pd.DataFrame({'head_angle':angles_head, 'tail_angle':angles_tail})
    #%%
    roll_std = pd.rolling_std(ts, window = window_std, min_periods = window_std-max_gap_allowed);
    
    plt.figure()
    plt.tick_params(labelsize=15)
    plt.plot(ts['head_angle'], 'g', label='head', linewidth = 1.5)
    plt.plot(ts['tail_angle'], 'm', label='tail', linewidth = 1.5)
    plt.xlim((0, 4500))
    plt.xlabel('Time Frame', fontsize=20)
    plt.ylabel('Angle (Rad)', fontsize=20)
    plt.legend()
    plt.savefig('head_tail.png', dpi = 300)
    #%%
    plt.figure()
    plt.tick_params(labelsize=15)
    
    plt.axis(fontsize=20)
    plt.plot(roll_std['head_angle'], 'g', linewidth = 1.5)
    plt.plot(roll_std['tail_angle'], 'm', linewidth = 1.5)
    plt.xlim((0, 4500))
    
    plt.xlabel('Time Frame', fontsize=20)
    plt.ylabel('Moving SD Angle (Rad)', fontsize=20)
    plt.legend()
    plt.savefig('head_tail_std.png', dpi = 300)