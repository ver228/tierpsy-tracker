# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:35:11 2015

@author: ajaver
"""

import glob
import os
import time
from collections import OrderedDict

import sys
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking/work_in_progress/GUI_test/ControlConsole/')
from start_console import runMultiCMD

script_abs_path = 'feat_median.py'
if __name__ == '__main__':
    #main_dir = '/Volumes/behavgenom$/GeckoVideo/Results/Avelino_17112015_2100/'
    main_dir = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/'
    assert(os.path.exists(main_dir))
    feat_files = glob.glob(main_dir + '*_features.hdf5')
    feat_files = sorted(feat_files)
    
    cmd_list = []
    for feat_file in feat_files:

        cmd_list += [['python3', script_abs_path, feat_file]]
        print(' '.join(cmd_list[-1]))
    
    runMultiCMD(cmd_list, max_num_process = 6, refresh_time = 10)
        
    
    


