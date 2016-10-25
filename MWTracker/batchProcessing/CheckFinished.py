# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 00:24:17 2016

@author: ajaver
"""
import tables
from functools import partial

def _isValidFlag(field, flag_value):
    return (field._v_attrs['has_finished'] >= flag_value)

def _isValidProvenance(field, point_name):
    return (point_name in field)

def _checkFlagsFun(fname, field_name, test_value, test_func=_isValidFlag):
    accepted_errors = (tables.exceptions.HDF5ExtError, 
        tables.exceptions.NoSuchNodeError, KeyError,IOError)
    try:
        with tables.open_file(fname, mode='r') as fid:
            field = fid.get_node(field_name)
            
            return test_func(field, test_value)
    except accepted_errors:
            return False

class CheckFinished(object):
    def __init__(self, provenance_files):
        outf = provenance_files
        
        #check that the correct provenance point is stored in the corresponding file
        self._provenance_funcs = {point: partial(_checkFlagsFun, 
                                            outf[point], 
                                            '/provenance_tracking', 
                                            point, 
                                            _isValidProvenance) for point in provenance_files.keys()}
        
        #I plan to check succesful processing using only provenance. I keep this for backwards compatibility.
        self._flag_funcs = {
            'COMPRESS': partial(_checkFlagsFun, outf['COMPRESS'], '/mask', 1),
            'COMPRESS_ADD_DATA': partial(_checkFlagsFun, outf['COMPRESS'], '/mask', 2),
            'TRAJ_CREATE': partial(_checkFlagsFun, outf['TRAJ_CREATE'], '/plate_worms', 1),
            'TRAJ_JOIN': partial(_checkFlagsFun, outf['TRAJ_JOIN'], '/plate_worms', 2),
            'SKE_CREATE': partial(_checkFlagsFun, outf['SKE_CREATE'], '/skeleton', 1),
            'SKE_FILT': partial(_checkFlagsFun, outf['SKE_FILT'], '/skeleton', 2),
            'SKE_ORIENT': partial(_checkFlagsFun, outf['SKE_ORIENT'], '/skeleton', 3),
            'INT_PROFILE': partial(_checkFlagsFun, outf['INT_PROFILE'], '/straighten_worm_intensity_median', 1),
            'INT_SKE_ORIENT': partial(_checkFlagsFun, outf['INT_SKE_ORIENT'], '/skeleton', 4),
            'FEAT_CREATE': partial(_checkFlagsFun, outf['FEAT_CREATE'], '/features_means', 1),
            'FEAT_MANUAL_CREATE': partial(_checkFlagsFun, outf['FEAT_MANUAL_CREATE'], '/features_means', 1),
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