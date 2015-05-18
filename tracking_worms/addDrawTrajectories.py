# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:12:17 2015

@author: ajaver
"""

import glob
from getWormTrajectories import drawTrajectoriesVideo

root_dir = '/Volumes/behavgenom$/GeckoVideo/Trajectories/'

file_list = glob.glob(root_dir + '*/*_trajectories.hdf5')[::-1]

for trajectories_file in file_list:    
    masked_image_file = trajectories_file.replace('Trajectories', 'Compressed').replace('_trajectories', '')
    drawTrajectoriesVideo(masked_image_file, trajectories_file)
    
    
    