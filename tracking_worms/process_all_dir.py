# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 20:52:52 2015

@author: ajaver
"""

import os

with open('/Users/ajaver/list_dir') as f:
    dir_list = f.readlines();

for fdir in dir_list:
    fdir = fdir[:-1]
    print('python trackParallelProcesses.py ' + fdir)
    os.system('python trackParallelProcesses.py ' + fdir)
