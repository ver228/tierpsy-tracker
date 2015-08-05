# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:16:24 2015

@author: ajaver
"""


import os

import pandas as pd



MWTracker_dir = '/Users/ajaver/Documents/GitHub/Multiworm_Tracking'
import sys
sys.path.append(MWTracker_dir)

from MWTracker import config_param #add the directory path for the validation movement

from MWTracker.FeaturesAnalysis.obtainFeatures_N import getWormFeaturesLab

if __name__ == "__main__":
    
#    base_name = 'Capture_Ch3_12052015_194303'
#    mask_dir = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/20150512/'
#    results_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150512/'    

    for ii in [4]:
        skeletons_file = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150521_1115/Capture_Ch%i_21052015_111806_skeletons.hdf5' % ii
        print(skeletons_file)
        base_name = skeletons_file.rpartition('_skeletons')[0].rpartition(os.sep)[-1]
        
        results_dir = skeletons_file.rpartition(os.sep)[0] + os.sep
        
        skeletons_file = results_dir + base_name + '_skeletons.hdf5'
        features_file = results_dir + base_name + '_features.hdf5'
        
        assert os.path.exists(skeletons_file)
        
        with pd.HDFStore(skeletons_file, 'r') as ske_file_id:
            trajectories_data = ske_file_id['/trajectories_data']
        
        #select only data labelled as worms
        trajectories_data = trajectories_data[trajectories_data['worm_label']==1]
        worm_indexes = trajectories_data['worm_index_N'].unique()

        getWormFeaturesLab(skeletons_file, features_file, worm_indexes)

