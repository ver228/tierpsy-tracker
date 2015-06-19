# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:13:47 2015

@author: ajaver
"""

import numpy as np

def featureStat(func, data, name, is_signed, is_motion, motion_mode = np.zeros(0)):
    stats = {}
    
    motion_types = {'all':np.nan};
    if is_motion:
        assert motion_mode.size == data.size
        motion_types['Foward'] = 1;
        motion_types['Paused'] = 0;
        motion_types['Backward'] = -1;
    
    for key in motion_types:
        if 'all':
            valid = ~np.isnan(data)
            sub_name = name
        else:
            valid = motion_mode == motion_types[key]
            sub_name = name + '_' + key
            
            
        stats[sub_name] = func(data[valid]);
        if is_signed:
            stats[sub_name + '_Abs'] = func(np.abs(data[valid]))
            stats[sub_name + '_Neg'] = func(data[data>0 & valid])
            stats[sub_name + '_Pos'] = func(data[data<0 & valid])
        