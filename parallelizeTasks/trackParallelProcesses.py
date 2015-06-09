# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:46:25 2015

@author: ajaver
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 01:56:06 2015

@author: ajaver
"""
import os
import sys

sys.path.append('../helperFunctions/')
from parallelProcHelper import runMultiSubproc

def get_tracking_cmd(masked_movie_file, results_dir):
    cmd =  ' '.join(["python3 trackSingleFile.py", masked_movie_file, results_dir, '</dev/null'])
    return cmd

if __name__ == '__main__':
    
    masked_movies_dir = '/Volumes/behavgenom$/GeckoVideo/MaskedVideos/20150519/'
    results_dir = '/Volumes/behavgenom$/GeckoVideo/Results/20150519/'
    
    max_num_process = 6;
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    #get a file list 
    masked_movies_list = os.listdir(masked_movies_dir);
    #filter for files with .hdf5 and add the full path
    args_list = [(masked_movies_dir + x, results_dir) \
    for x in masked_movies_list if ('.hdf5' in x)]
    
    runMultiSubproc(get_tracking_cmd, max_num_process = 6)
    

    
