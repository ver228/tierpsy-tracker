# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:02:51 2016

@author: ajaver
"""
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

main_dir = '/Users/ajaver/Desktop/Videos/single_worm/agar_2/MaskedVideos/'
files = glob.glob(os.path.join(main_dir, '*.hdf5' ))
files = sorted(files)


good_files = []
bad_files = []
no_stage_vec = []
for mask_id, masked_image_file in enumerate(files):
    skeletons_file = masked_image_file.replace('MaskedVideos', 'Results')[:-5] + '_skeletons.hdf5'
    feat_file = masked_image_file.replace('MaskedVideos', 'Features')[:-5] + '_features.mat'

    with pd.HDFStore(skeletons_file, 'r') as fid:
        trajectories_data = fid['trajectories_data']
        
        N_frames = trajectories_data['frame_number'].max();
        N_skel = trajectories_data['has_skeleton'].sum()
        N_good = trajectories_data['is_good_skel'].sum()
        N_out = (trajectories_data['skel_outliers_flag']>0).sum()
        assert N_skel-N_good == N_out
        print('')
        print(mask_id, ')', os.path.split(masked_image_file)[1])
        print(N_frames, N_skel, N_good, N_out)
        