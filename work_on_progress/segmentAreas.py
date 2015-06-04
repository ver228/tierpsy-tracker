# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:32:50 2015

@author: ajaver
"""
import sys
sys.path.append('../../movement_validation')

#from movement_validation.pre_features import WormParsing

from obtain_features import WormFromTable

if __name__ == "__main__":
    root_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150511/'
    base_name = 'Capture_Ch1_11052015_195105'
    
    masked_image_file = root_dir + '/Compressed/' + base_name + '.hdf5'
    trajectories_file = root_dir + '/Trajectories/' + base_name + '_trajectories.hdf5'
    skeleton_file = root_dir + '/Trajectories/' + base_name + '_skeletons.hdf5'
    
    worm_index = 773
    rows_range = (0,0)
    #assert rows_range[0] <= rows_range[1]
    
    file_name = skeleton_file;
    worm = WormFromTable()
    worm.fromFile(file_name, worm_index)
    #worm.changeAxis()
    
    
    #vi = VideoInfo(masked_image_file, 25)    
    # Generate the OpenWorm movement validation repo version of the features
    #openworm_features = WormFeatures(worm, vi)