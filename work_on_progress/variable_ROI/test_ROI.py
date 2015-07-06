# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 13:10:32 2015

@author: ajaver
"""

import pandas as pd
import numpy as np
import functools

trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150511/Capture_Ch1_11052015_195105_trajectories.hdf5'
skeletons_file = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150511/Capture_Ch1_11052015_195105_skeletons.hdf5'

table_fid = pd.HDFStore(trajectories_file, 'r')
ske_fid = pd.HDFStore(skeletons_file, 'r')


df_ske = ske_fid['/trajectories_data']['worm_index_joined']
valid_indexes = np.unique(df_ske.index)

df = table_fid['/plate_worms']
df = df[df['worm_index_joined'].isin(valid_indexes)]


bb_x = df['bounding_box_xmax']-df['bounding_box_xmin']+1;
bb_y = df['bounding_box_ymax']-df['bounding_box_ymin']+1;
roi_range = pd.concat([bb_x, bb_y], axis=1).max(axis=1)




df_bb = pd.DataFrame({'worm_index':df['worm_index'], 'roi_range': roi_range})
#roi_size = df_bb.groupby('worm_index').agg([max , functools.partial(np.percentile, q=0.95)])
roi_range = df_bb.groupby('worm_index').agg(max) + 5



