# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:51:21 2016

@author: ajaver
"""

import pandas as pd
import h5py
import numpy as np
import matplotlib.pylab as plt

import sys
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking')
from MWTracker.featuresAnalysis.obtainFeaturesHelper import WLAB, smoothCurve, calWormAngles

masked_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch1_18112015_075624.hdf5'
skel_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch1_18112015_075624_skeletons.hdf5'
intensity_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch1_18112015_075624_intensities.hdf5'

min_num_skel = 100

#%%
with pd.HDFStore(skel_file, 'r') as fid:
    trajectories_data = fid['/trajectories_data']

good = trajectories_data['auto_label'] == WLAB['GOOD_SKE'];
trajectories_data = trajectories_data[good]

N = trajectories_data.groupby('worm_index_joined').agg({'has_skeleton':np.nansum})
N = N[N>min_num_skel].dropna()

good = trajectories_data['worm_index_joined'].isin(N.index)
trajectories_data = trajectories_data.loc[good]
#%%

import table
table_filters = tables.Filters(complevel=5, complib='zlib', shuffle=True, fletcher32=True)

skeleton_id = trajectories_data['skeleton_id'].values
tot_skel = len(skeleton_id)
skel_smoothed = fid.create_carray('/', 'skeletons_smoothed', \
                                    tables.Float32Atom(dflt = np.nan), (tot_skel, 49, 2), filters = table_filters)
skel_angles = fid.create_carray('/', 'skeletons_angles', \
                                    tables.Float32Atom(dflt = np.nan), (tot_skel, 49), filters = table_filters)
intensity_maps = fid.create_carray('/', 'intensity_maps', \
                                    tables.Float32Atom(dflt = np.nan), (tot_skel, 49, 10), filters = table_filters)

for ii, skel_id in enumerate(skeleton_id):
    skeleton = 
    skel_smooth = smoothCurve(skeleton, window = 5, pol_degree = 3)
    calWormAngles(skeleton[:,0], skeleton[:,1])
    skel_smoothed[ii] = skel_smooth
    
    