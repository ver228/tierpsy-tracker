# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 01:56:06 2015

@author: ajaver
"""
import os
import sys
import tables

from ..trackWorms.getWormTrajectories import getWormTrajectories, joinTrajectories
from ..trackWorms.getDrawTrajectories import drawTrajectoriesVideo
from ..trackWorms.getSkeletonsTables import trajectories2Skeletons, writeIndividualMovies
from ..trackWorms.checkHeadOrientation import correctHeadTail

from ..FeaturesAnalysis.obtainFeatures import getWormFeatures

from .. import config_param as param

def getTrajectoriesWorker(masked_image_file, results_dir, overwrite = False):
    
    #check if the file with the masked images exists
    assert os.path.exists(masked_image_file)
        
    if results_dir[-1] != os.sep:
        results_dir += os.sep
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
        except:
            pass
    
    #construct file names
    base_name = masked_image_file.rpartition('.')[0].rpartition(os.sep)[-1]
    
    trajectories_file = results_dir + base_name + '_trajectories.hdf5'
    trajectories_video = results_dir + base_name + '_trajectories.avi'
    skeletons_file = results_dir + base_name + '_skeletons.hdf5'
    features_file = results_dir + base_name + '_features.hdf5'
    video_save_dir = results_dir + base_name + os.sep
    
    
    start_point = 1e6
    #determine for where to start
    if overwrite:
        start_point = 0;
    elif not os.path.exists(trajectories_file):
        start_point = 0;
    else:
        with tables.open_file(trajectories_file, mode = 'r') as feature_fid:
             feature_table = feature_fid.get_node('/plate_worms')
             if 'has_finished' in dir(feature_table._v_attrs):
                 if feature_table._v_attrs['has_finished'] == 0:
                     start_point = 0;
                 elif feature_table._v_attrs['has_finished'] == 1:
                     start_point = 1;
    
    if start_point > 2 or not os.path.exists(trajectories_video):
        start_point = 2;
        
    if start_point > 3 or not os.path.exists(skeletons_file):
        start_point = 3;
    else:
        with tables.File(skeletons_file, "r") as ske_file_id:
            skeleton_table = ske_file_id.get_node('/skeleton')
            if 'has_finished' in dir(skeleton_table._v_attrs):
                if skeleton_table._v_attrs['has_finished'] == 0:
                     start_point = 3;
                elif skeleton_table._v_attrs['has_finished'] == 1:
                     start_point = 4;
    
    if start_point > 5 or not os.path.exists(features_file):
        start_point = 5;
    
    if start_point > 6 or not os.path.exists(video_save_dir):
        start_point = 6;
    
    
    #get trajectory data
    if start_point <= 0:
        getWormTrajectories(masked_image_file, trajectories_file, **param.get_trajectories_param)
    if start_point <= 1:        
        joinTrajectories(trajectories_file, **param.join_traj_param)
    if start_point <= 2:
        drawTrajectoriesVideo(masked_image_file, trajectories_file)
    
    #get skeletons data    
    if start_point <= 3:
        trajectories2Skeletons(masked_image_file, skeletons_file, trajectories_file, **param.get_skeletons_param)
    if start_point <= 4:
        correctHeadTail(skeletons_file, **param.head_tail_param)
    
    if start_point <= 5:
        #extract features
        getWormFeatures(skeletons_file, features_file, **param.features_param)

    if start_point <= 6:
        #create movies of individual worms
        writeIndividualMovies(masked_image_file, skeletons_file, video_save_dir, **param.ind_mov_param)           
    
    print(base_name + ' Finished')
    

if __name__ == '__main__':
    masked_image_file = sys.argv[1]
    results_dir = sys.argv[2]
    
    #masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/20150511/Capture_Ch1_11052015_195105.hdf5'
    #results_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150511/' 
    
    getTrajectoriesWorker(masked_image_file, results_dir, over_write_previous = False)
    
    