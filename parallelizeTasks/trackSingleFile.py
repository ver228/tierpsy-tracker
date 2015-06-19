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

def getTrajectoriesWorker(masked_image_file, results_dir, over_write_previous = False):
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
    
    if os.path.exists(trajectories_file):
        os.remove(trajectories_file)
    
    doTrajectories = False
    doSkeletons = False
    doVideos = False

    if over_write_previous or not os.path.exists(trajectories_file):
        doTrajectories = True
        doSkeletons = True
        doVideos = True
    elif not os.path.exists(skeletons_file):
        doSkeletons = True
        doVideos = True
    elif not os.path.exists(video_save_dir):
        doVideos = True
    
    if doTrajectories:
        getWormTrajectories(masked_image_file, trajectories_file, last_frame = -1)
        joinTrajectories(trajectories_file)
        drawTrajectoriesVideo(masked_image_file, trajectories_file)

    if doSkeletons:
        #get skeletons data
        if os.path.exists(skeletons_file):
            os.remove(skeletons_file);

        trajectories2Skeletons(masked_image_file, skeletons_file, trajectories_file, \
                           create_single_movies = False, roi_size = 128, resampling_N = 49, min_mask_area = 50)
    
        correctHeadTail(skeletons_file, max_gap_allowed = 10, \
                        window_std = 25, segment4angle = 5, min_block_size = 250)

        #extract features
        getWormFeatures(skeletons_file, features_file, bad_seg_thresh = 0.5, video_fps = 25)

    if doVideos:
        #create movies of individual worms
        writeIndividualMovies(masked_image_file, skeletons_file, video_save_dir, \
                              roi_size = 128, fps=25)

    print(base_name + ' Finished')
    

if __name__ == '__main__':
    masked_image_file = sys.argv[1]
    results_dir = sys.argv[2]
    
    getTrajectoriesWorker(masked_image_file, results_dir, over_write_previous = False)
    
    