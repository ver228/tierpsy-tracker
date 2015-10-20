# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:29:01 2015

@author: ajaver
"""

import os
import sys



# absolute path for the movement validation repository
movement_validation_dir = ''

#add the movement validation directory
if not movement_validation_dir:
	curr_script_dir = os.path.dirname(os.path.realpath(__file__))
	movement_validation_dir = os.path.join(curr_script_dir, '..', '..', 'movement_validation')#'/Users/ajaver/Documents/GitHub/movement_validation'
	#movement_validation_dir = os.path.expanduser('~') + '/Documents/GitHub/movement_validation'

if not os.path.exists(movement_validation_dir):
    print('%s does not exists. Introduce a valid path in the variable movement_validation_dir of the file config_param.py. \
    	The path must point to the openWorm movement_validation repository.' % movement_validation_dir)
    raise
    
sys.path.append(movement_validation_dir)
        