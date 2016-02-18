# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:50:39 2016

@author: ajaver
"""

import sys
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking/')
sys.path.append('/Users/ajaver/Documents/GitHub/movement_validation')


from MWTracker.helperFunctions.tracker_param import tracker_param

from git import Repo
import open_worm_analysis_toolbox
import MWTracker


commit_hash = getGitCommitHash()

param = tracker_param()

output_file = masked_image_file
#args = (masked_image_file, trajectories_file)


argkws = param.join_traj_param
func = getWormTrajectories

function_name = func.__name__


#func(*args, **argkws)

