# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:39:34 2016

@author: ajaver
"""

import os
import h5py
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

from MWTracker.trackWorms.getWormTrajectories import _validRowsByArea, _findNextTraj, _joinDict2Index, correctSingleWormCase

file_mask = '/Users/ajaver/Desktop/Videos/single_worm/agar_1/MaskedVideos/gpa-11 (pk349)II on food L_2010_02_25__11_24_39___8___6.hdf5'

file_traj = file_mask.replace('MaskedVideos', 'Results').replace('.hdf5', '_trajectories.hdf5')
assert(os.path.exists(file_mask))
assert(os.path.exists(file_traj))

file_skel = file_mask.replace('MaskedVideos', 'Results').replace('.hdf5', '_skeletons.hdf5')
assert(os.path.exists(file_skel))
with pd.HDFStore(file_skel, 'r') as fid:
    trajectories_data = fid['/trajectories_data']

with pd.HDFStore(file_traj, 'r') as fid:
    plate_worms = fid['/plate_worms']


groupsbyframe = plate_worms.groupby('frame_number')
maxAreaPerFrame = groupsbyframe.agg({'area':'max'})
med_area = np.median(maxAreaPerFrame)
mad_area = np.median(np.abs(maxAreaPerFrame-med_area))
min_area = med_area - mad_area*6
max_area = med_area + mad_area*6

groupByIndex = plate_worms.groupby('worm_index')

median_area_by_index = groupByIndex.agg({'area':np.median})

good = ((median_area_by_index>min_area) & (median_area_by_index<max_area)).values
valid_ind = median_area_by_index[good].index;

plate_worms_f = plate_worms[plate_worms['worm_index'].isin(valid_ind)]

filtered_areas = plate_worms_f['area'].values
med_area = np.median(filtered_areas)
mad_area = np.median(np.abs(filtered_areas-med_area))
min_area = med_area - 3*mad_area
max_area = med_area + 3*mad_area




#track_sizes = plate_worms_f['worm_index'].value_counts();
#groupByIndex_f = plate_worms_f.groupby('worm_index');
#for rr in range(10):
#    ind = track_sizes.index[rr]
#    
#    traj_dat = groupByIndex_f.get_group(ind);
#    plt.plot(traj_dat['frame_number'], traj_dat['coord_x'], '-')



#relations_dict, valid_indexes = _findNextTraj(plate_worms_f, area_ratio_lim= (0.67, 1.5), min_track_size=1, max_time_gap=30)
#worm_index_joined = joinDict2Index(plate_worms['worm_index'].values, relations_dict, valid_indexes)



#groupsbyframe_f = plate_worms_f.groupby('frame_number')

#groupsbyframe_f

valid_rows = _validRowsByArea(plate_worms)
#%%
plt.figure()
plt.plot(plate_worms_f['frame_number'], plate_worms_f['coord_x'], '.')
xx = plate_worms.loc[valid_rows, 'coord_x'].values
tt = plate_worms.loc[valid_rows, 'frame_number'].values
plt.plot(tt, xx, 'r')

df = plate_worms[plate_worms['worm_index_joined']==1]
plt.figure()
plt.plot(df['frame_number'], df['coord_x'], 'b')
plt.plot(trajectories_data['frame_number'], trajectories_data['coord_x'], 'r')


#%%
