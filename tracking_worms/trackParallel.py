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
sys.path.append('../videoCompression/')
from parallelProcHelper import parallelizeTask, sendQueueOrPrint

def getTrajectoriesWorker(masked_movies_dir, trajectories_dir, main_video_save_dir, base_name='', status_queue=''):
    #construct file names
    masked_image_file = masked_movies_dir + base_name + '.hdf5'
    trajectories_file = trajectories_dir + base_name + '_trajectories.hdf5'
    segworm_file = trajectories_dir + base_name + '_segworm.hdf5'
    video_save_dir = main_video_save_dir + base_name + os.sep
    
    if not os.path.exists(video_save_dir):
        os.mkdir(video_save_dir)
    
    try:
        #track individual worms
        getWormTrajectories(masked_image_file, trajectories_file, last_frame = -1,\
        base_name=base_name, status_queue=status_queue)
        joinTrajectories(trajectories_file)
    except:
        sendQueueOrPrint(status_queue, 'Tracking failed', base_name)
        raise
#    
    try:
        #obtain skeletons
        getSegWorm(masked_image_file, trajectories_file, segworm_file,\
        base_name = base_name, status_queue=status_queue)
    except:
        sendQueueOrPrint(status_queue, 'Segworm failed', base_name)
        raise
        
    try:
        #create movies of individual worms
        getIndividualWormVideos(masked_image_file, trajectories_file, \
        segworm_file, video_save_dir, is_draw_contour = False, max_frame_number = -1,\
        base_name = base_name, status_queue=status_queue)
    except:
        sendQueueOrPrint(status_queue, 'Create individual worm videos failed.', base_name)
    raise

if __name__ == '__main__':
    masked_movies_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150323/'
    trajectories_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/'
    main_video_save_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Worm_Movies/'
    
    if not os.path.exists(trajectories_dir):
        os.mkdir(trajectories_dir)
    if not os.path.exists(main_video_save_dir):
        os.mkdir(main_video_save_dir)
    
    #get a list 
    file_list = os.listdir(masked_movies_dir);
    base_name_list = [os.path.splitext(x)[0] for x in file_list if ('.hdf5' in x)]#    #start the parallizeTask object, obtain the queue where the progress status is stored
    
#    base_name = base_name_list[0]    
#    getTrajectoriesWorker(masked_movies_dir, trajectories_dir, main_video_save_dir, base_name)

    task = parallelizeTask(6);
    
    #get a list of arguments for each child
    workers_arg = {};
    for base_name in base_name_list[1:2]:
        workers_arg[base_name] = (masked_movies_dir, trajectories_dir, main_video_save_dir, base_name, task.status_queue)
    
    task.start(getTrajectoriesWorker, workers_arg)
    
    #plot the top 20th trajectories
    #It is not possible to plot using the multiprocessing module
    for base_name in base_name_list:
        trajectories_plot_file = trajectories_dir + base_name + '_trajectories.pdf'
        trajectories_file = trajectories_dir + base_name + '_trajectories.hdf5'
        plotLongTrajectories(trajectories_file, trajectories_plot_file)
    
    