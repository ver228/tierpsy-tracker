#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:00:17 2017

@author: ajaver
"""
import numpy as np
import pandas as pd
import tables
import glob

from tierpsy.helper.misc import TimeCounter, print_flush, get_base_name, save_modified_table

def _get_signed_velocity(xx,yy, skel_coords):
    velocity = np.vstack((np.gradient(xx), np.gradient(yy))).T
    #is this head coord vs tail coord? 
    orientation = skel_coords[:,0,:]-skel_coords[:,-1,:]
    
    speed = np.sqrt(np.sum(velocity**2, 1));
    signed_speed = np.sign(np.sum(velocity*orientation, 1))*speed;
    return signed_speed, velocity


DEBUG = False

if __name__ == '__main__':
    filenames = glob.glob("/data2/shared/data/twoColour/Results/*/*/*52.1g_X1_skeletons.hdf5")

    for skeletons_file in filenames:
        base_name = get_base_name(skeletons_file)
        progress_prefix =  base_name + ' Calculating skeletons.'
        
        with pd.HDFStore(skeletons_file, 'r') as ske_file_id:
            trajectories_data = ske_file_id['/trajectories_data']
            blob_features = ske_file_id['/blob_features']
        
        #I want to update blob_features
        blob_features['signed_speed'] = np.float32(np.nan)
        blob_features['velocity_x'] = np.float32(np.nan)
        blob_features['velocity_y'] = np.float32(np.nan)

        progress_timer = TimeCounter('')
        with tables.File(skeletons_file, 'r') as fid:
            skeletons = fid.get_node('/skeleton')
            grouped_by_index = trajectories_data.groupby('worm_index_joined')
            tot_worms = len(grouped_by_index)
            for ii, (worm_index, worm_data) in enumerate(grouped_by_index):
                feats = blob_features.loc[worm_data.index]
                skel_coords = skeletons[worm_data.index]
                xx = feats['coord_x']
                yy = feats['coord_y']
                
                signed_speed, velocity = _get_signed_velocity(xx, yy, skel_coords)
                #import pdb
                #pdb.set_trace()
                blob_features.loc[worm_data.index, 'signed_speed'] = signed_speed
                blob_features.loc[worm_data.index, 'velocity_x'] = velocity[:,0]
                blob_features.loc[worm_data.index, 'velocity_y'] = velocity[:,1]
                
                if ii % 100 == 0:
                    dd = " Calculating signed speed. Worm %i of %i." % (ii + 1, tot_worms)
                    dd = base_name + dd + ' Total time:' + progress_timer.get_time_str()
                    print_flush(dd)
        
        save_modified_table(skeletons_file, blob_features, 'blob_features')
