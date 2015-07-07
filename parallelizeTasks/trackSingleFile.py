# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:36:06 2015

@author: ajaver
"""

import sys
sys.path.append('..')

from MWTracker.helperFunctions.getTrajectoriesWorker import getTrajectoriesWorker

if __name__ == '__main__':
    masked_image_file = sys.argv[1]
    results_dir = sys.argv[2]
    
    getTrajectoriesWorker(masked_image_file, results_dir, overwrite = False)

