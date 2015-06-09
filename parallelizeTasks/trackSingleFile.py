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


def getTrajectoriesWorker(masked_image_file, results_dir, resume_from_previous = False):
    
    if results_dir[-1] != os.sep:
        results_dir += os.sep
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    #construct file names
    base_name = masked_image_file.rpartition('.')[0].rpartition(os.sep)[-1]
    
    trajectories_file = results_dir + base_name + '_trajectories.hdf5'
    skeletons_file = results_dir + base_name + '_skeletons.hdf5'
    video_save_dir = results_dir + base_name + os.sep
    
    if os.path.exists(trajectories_file):
        os.remove(trajectories_file)
        
    getWormTrajectories(masked_image_file, trajectories_file, last_frame = -1)
    joinTrajectories(trajectories_file)
    drawTrajectoriesVideo(masked_image_file, trajectories_file)

    #get skeletons data
    if os.path.exists(skeletons_file):
        os.remove(skeletons_file);

    trajectories2Skeletons(masked_image_file, skeletons_file, trajectories_file, \
    create_single_movies = False, roi_size = 128, resampling_N = 49, min_mask_area = 50)
    
    correctHeadTail(skeletons_file, max_gap_allowed = 10, \
    window_std = 25, segment4angle = 5, min_block_size = 250)
    
    #create movies of individual worms
    writeIndividualMovies(masked_image_file, skeletons_file, video_save_dir, \
                          roi_size = 128, fps=25)

    print(base_name + ' Finished')
    

if __name__ == '__main__':
    masked_image_file = sys.argv[1]
    results_dir = sys.argv[2]
    
    getTrajectoriesWorker(masked_image_file, results_dir)
    
    