# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:00:52 2015

@author: ajaver
"""

import os
import tables

dir2check = '/Volumes/D/hdf5_bad_20151003_2000/'

for file_name in os.listdir(dir2check):
    full_name = dir2check + file_name
    
    #if os.path.exists(full_name):    
    #    print(full_name)
    
    try:
        with tables.File(full_name, 'r') as fid:
            dat_shape = fid.get_node('/mask').shape
            print(dat_shape)
    
    except OSError:
        print('Read Error:', full_name)
    
    except tables.HDF5ExtError:
        print('Error:', full_name)