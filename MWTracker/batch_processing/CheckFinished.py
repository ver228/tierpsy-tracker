# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 00:24:17 2016

@author: ajaver
"""
import os
from functools import partial

import tables


def _isValidFlag(field, flag_value):
    return (field._v_attrs['has_finished'] >= flag_value)

def _isValidProvenance(field, point_name):
    return (point_name in field)

def _checkFlagsFun(fname, field_name, test_value, test_func=_isValidFlag, extra_files=[]):
    accepted_errors = (tables.exceptions.HDF5ExtError, 
        tables.exceptions.NoSuchNodeError, KeyError,IOError)
    try:
        with tables.open_file(fname, mode='r') as fid:
            field = fid.get_node(field_name)
            has_finished = test_func(field, test_value)

            #check that all the extra files do exist
            has_finished = has_finished and all(os.path.exists(x) for x in extra_files)
            
            return has_finished

    except accepted_errors:
            return False



class CheckFinished(object):
    def __init__(self, output_files):
        
        #check that the correct provenance point is stored in the corresponding file
        self._provenance_funcs = {}
        for point in output_files:
            provenance_file = output_files[point][0]
            extra_files = output_files[point][1:]
            
            self._provenance_funcs[point] = partial(_checkFlagsFun, 
                provenance_file,  
                '/provenance_tracking', 
                point, 
                _isValidProvenance,
                extra_files)
        
        #I plan to check succesful processing using only provenance. I keep this for backwards compatibility.
        outf = lambda x : output_files[x][0]
        self._flag_funcs = {
            'compress': partial(_checkFlagsFun, outf('compress'), '/mask', 1),
            'compress_add_data': partial(_checkFlagsFun, outf('compress'), '/mask', 2),
            'traj_create': partial(_checkFlagsFun, outf('traj_create'), '/plate_worms', 1),
            'traj_join': partial(_checkFlagsFun, outf('traj_join'), '/plate_worms', 2),
            'ske_create': partial(_checkFlagsFun, outf('ske_create'), '/skeleton', 1),
            'ske_filt': partial(_checkFlagsFun, outf('ske_filt'), '/skeleton', 2),
            'ske_orient': partial(_checkFlagsFun, outf('ske_orient'), '/skeleton', 3),
            'int_profile': partial(_checkFlagsFun, outf('int_profile'), '/straighten_worm_intensity_median', 1),
            'int_ske_orient': partial(_checkFlagsFun, outf('int_ske_orient'), '/skeleton', 4),
            'feat_create': partial(_checkFlagsFun, outf('feat_create'), '/features_means', 1),
            'feat_manual_create': partial(_checkFlagsFun, outf('feat_manual_create'), '/features_means', 1),
        }
    
    def getUnfinishedPoints(self, checkpoints2process):
        unfinished_points = checkpoints2process[:]
        for point in checkpoints2process:
            if self.get(point):
                unfinished_points.pop(0)
            else:
                break
        return unfinished_points
    
    def get(self, point):
        has_finished = self._provenance_funcs[point]()
        
        #we test flags for backwards compatibility
        if not has_finished and point in self._flag_funcs:
            has_finished = self._flag_funcs[point]()
        
        return has_finished