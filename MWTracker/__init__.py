# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:29:01 2015

@author: ajaver
"""

import os
import sys

curr_script_dir = os.path.dirname(os.path.realpath(__file__))
movement_validation_dir = os.path.join(curr_script_dir, '..', '..', 'open-worm-analysis-toolbox')
movement_validation_dir = os.path.abspath(movement_validation_dir)
if not os.path.exists(movement_validation_dir):
    raise FileNotFoundError('%s does not exists. Introduce a valid path in the variable movement_validation_dir of the file config_param.py. \
    	The path must point to the openWorm movement_validation repository.' % movement_validation_dir)

    
sys.path.append(movement_validation_dir)

try:
    import open_worm_analysis_toolbox
except ImportError:
    raise ImportError('No module open_worm_analysis_toolbox in %s\nTake a look into MWTracker __init__ path' % movement_validation_dir)