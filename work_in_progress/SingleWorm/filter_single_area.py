# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:19:16 2015

@author: ajaver
"""

import pandas as pd
import tables
import numpy as np
import matplotlib.pylab as plt

import sys
import glob
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking/')
from MWTracker.trackWorms.getSkeletonsTables import getSmoothTrajectories


trajectories_file = '/Users/ajaver/Desktop/Videos/copied_from_pc207-13/Results/135 CB4852 on food L_2011_03_09__15_51_36___1___8_trajectories.hdf5'
skel_file = '/Users/ajaver/Desktop/Videos/copied_from_pc207-13/Results/135 CB4852 on food L_2011_03_09__15_51_36___1___8_skeletons.hdf5'
#trajectories_file = '/Users/ajaver/Tmp/Results/135 CB4852 on food L_2011_03_09__15_51_36___1___8_trajectories.hdf5'

if __name__ == '__main__':
    main_dir = '/Users/ajaver/Desktop/Videos/copied_from_pc207-13/Results/'
    for trajectories_file in glob.glob(main_dir + '*_trajectories.hdf5'):
        with pd.HDFStore(trajectories_file, 'r') as traj_fid:
            plate_worms = traj_fid['/plate_worms']
    
        trajectories_df, worms_frame_range, tot_rows = \
        getSmoothTrajectories(trajectories_file, roi_size = -1, displacement_smooth_win = 101, 
        min_displacement = 0, threshold_smooth_win = 501)
        
        x = plate_worms['coord_x'].values
        y = plate_worms['coord_y'].values
        t = plate_worms['frame_number'].values
        thresh = plate_worms['threshold'].values
        area = plate_worms['area'].values
        
        first_frame = np.min(t);
        last_frame = np.max(t);
        tnew = np.arange(first_frame, last_frame+1);
        
        plt.plot(t, area)
        
        t[np.argsort(area)]
    
    #with pd.HDFStore(skel_file, 'r') as skel_fid:
    #    trajectories_data = skel_fid['/trajectories_data']
    
    
#    tot_frames = plate_worms['frame_number'].max() + 1
#    
#    groupsbyframe = plate_worms[['frame_number', 'area']].groupby('frame_number')
#    
#    valid_rows = np.full(tot_frames, np.nan)
#    
#    for ii, frame_data in groupsbyframe:
#        valid_rows[ii] = frame_data['area'].argmax()
#    
#    valid_rows = valid_rows[~np.isnan(valid_rows)]
#    
#    plate_worms = plate_worms.ix[valid_rows]
#    plate_worms['worm_index'] = np.array(1, dtype=np.int32)
#    
#    with tables.File(trajectories_file, "r+") as traj_fid:
#        table_filters = tables.Filters(complevel=5, complib='zlib', shuffle=True, fletcher32=True)
#        newT = traj_fid.create_table('/', 'plate_worms_t', 
#                                        obj = plate_worms.to_records(index=False), 
#                                        filters=table_filters)
#        traj_fid.remove_node('/', 'plate_worms')
#        newT.rename('plate_worms')
##assert(tot_frames, plate_worms['frame_number'].size)
#plate_worms['area'].plot()