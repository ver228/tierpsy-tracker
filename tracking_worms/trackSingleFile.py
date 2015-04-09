# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 01:56:06 2015

@author: ajaver
"""
import os

from getWormTrajectories import getWormTrajectories, joinTrajectories, plotLongTrajectories
from getSegWorm import getSegWorm
from getIndividualWormVideos import getIndividualWormVideos

import sys
import time
import h5py
sys.path.append('../videoCompression/')
from parallelProcHelper import sendQueueOrPrint

def getTrajectoriesWorker(masked_movies_dir, trajectories_dir, main_video_save_dir, \
base_name, status_queue= ''):
    #construct file names
    masked_image_file = masked_movies_dir + base_name + '.hdf5'
    trajectories_file = trajectories_dir + base_name + '_trajectories.hdf5'
    trajectories_plot_file = trajectories_dir + base_name + '_trajectories.pdf'
    segworm_file = trajectories_dir + base_name + '_segworm.hdf5'
    video_save_dir = main_video_save_dir + base_name + os.sep

    if not os.path.exists(video_save_dir):
        os.mkdir(video_save_dir)
    
    try:
        #track individual worms
        getWormTrajectories(masked_image_file, trajectories_file, last_frame = -1,\
        base_name=base_name, status_queue=status_queue)
        joinTrajectories(trajectories_file)
        
        mask_fid = h5py.File(masked_image_file, "r");
        plot_limits = mask_fid['/mask'].shape[1:]
        mask_fid.close()
        plotLongTrajectories(trajectories_file, trajectories_plot_file, plot_limits=plot_limits)
    except:
        sendQueueOrPrint(status_queue, 'Tracking failed', base_name)
        raise'Tracking failed'
    
    n_trials = 0;
    while n_trials<5:
        try:
            #obtain skeletons
            getSegWorm(masked_image_file, trajectories_file, segworm_file,\
            base_name = base_name, status_queue=status_queue, \
            min_displacement = 2, thresh_smooth_window = 1501)
            n_trials = 5;
        except:
            sendQueueOrPrint(status_queue, 'Segworm failed', base_name)
            n_trials +=1;
            time.sleep(30)
            if n_trials == 5:
                raise 'Segworm failed'
        
    try:
        #create movies of individual worms
        getIndividualWormVideos(masked_image_file, trajectories_file, \
        segworm_file, video_save_dir, is_draw_contour = True, max_frame_number = -1,\
        base_name = base_name, status_queue=status_queue)
    except:
        sendQueueOrPrint(status_queue, 'Create individual worm videos failed.', base_name)
        raise 'Create individual worm videos failed.'

if __name__ == '__main__':
#python trackSingleFile.py "/Users/ajaver/Desktop/Gecko_compressed/20150323/" "/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/" "/Users/ajaver/Desktop/Gecko_compressed/20150323/Worm_Movies/" "Capture_Ch4_23032015_111907"
#    masked_movies_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150323/'
#    trajectories_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/'
#    main_video_save_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Worm_Movies/'
    
    masked_movies_dir = sys.argv[1]
    trajectories_dir = sys.argv[2]
    main_video_save_dir = sys.argv[3]
    base_name = sys.argv[4]
    getTrajectoriesWorker(masked_movies_dir, trajectories_dir, main_video_save_dir, base_name)

    
    