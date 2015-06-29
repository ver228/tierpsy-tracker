# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 01:56:06 2015

@author: ajaver
"""
import os
import sys

sys.path.append('../trackWorms/')
from getWormTrajectories import getWormTrajectories, joinTrajectories
from getDrawTrajectories import drawTrajectoriesVideo
from getSkeletonsTables import trajectories2Skeletons, writeIndividualMovies
from checkHeadOrientation import correctHeadTail

sys.path.append('../../movement_validation')
sys.path.append('../work_on_progress/Features_analysis/')
from obtain_features import getWormFeatures

import config_param as param

def getTrajectoriesWorker(masked_image_file, results_dir, over_write_previous = False):
    
    #check if the file with the masked images exists
    assert os.exists(masked_image_file)
        
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
    skeletons_file = results_dir + base_name + '_skeletons.hdf5'
    features_file = results_dir + base_name + '_features.hdf5'
    video_save_dir = results_dir + base_name + os.sep
    
    doTrajectories = False
    doSkeletons = False
    doFeatures = False
    doVideos = False

    if over_write_previous or not os.path.exists(trajectories_file):
        doTrajectories = True
        doSkeletons = True
        doFeatures = True
        doVideos = True
    elif not os.path.exists(skeletons_file):
        doSkeletons = True
        doFeatures = True
        doVideos = True
    else:
        if not os.path.exists(features_file):
            doFeatures = True
        if not os.path.exists(video_save_dir):
            doVideos = True
    
    if doTrajectories:
        if os.path.exists(trajectories_file):
            os.remove(trajectories_file)
        getWormTrajectories(masked_image_file, trajectories_file, **param.get_trajectories_param)
        joinTrajectories(trajectories_file, **param.join_traj_param)
        drawTrajectoriesVideo(masked_image_file, trajectories_file)

    if doSkeletons:
        #get skeletons data
        if os.path.exists(skeletons_file):
            os.remove(skeletons_file);

        trajectories2Skeletons(masked_image_file, skeletons_file, trajectories_file, **param.get_skeletons_param)
    
        correctHeadTail(skeletons_file, **param.head_tail_param)
    if doVideos:
        #create movies of individual worms
        writeIndividualMovies(masked_image_file, skeletons_file, video_save_dir, **param.ind_mov_param)           
        
    if doFeatures:
        #extract features
        getWormFeatures(skeletons_file, features_file, **param.features_param)



    print(base_name + ' Finished')
    

if __name__ == '__main__':
    masked_image_file = sys.argv[1]
    results_dir = sys.argv[2]
    
    #masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/20150511/Capture_Ch1_11052015_195105.hdf5'
    #results_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150511/' 
    
    getTrajectoriesWorker(masked_image_file, results_dir, over_write_previous = False)
    
    