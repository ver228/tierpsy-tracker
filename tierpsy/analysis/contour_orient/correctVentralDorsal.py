import json
import os
import numpy as np
import tables
import warnings

from tierpsy.analysis.ske_filt.getFilteredSkels import _h_calAreaSignedArray
from tierpsy.helper.params import read_ventral_side

VALID_CNT = ['clockwise', 'anticlockwise', 'unknown']

def _add_ventral_side(skeletons_file, ventral_side=''):
    #I am giving priority to a contour stored in experiments_info, rather than one read by the json file.
    #currently i am only using the experiments_info in the re-analysis of the old schafer database
    ventral_side_f = read_ventral_side(skeletons_file)
    if ventral_side_f in VALID_CNT:
        ventral_side = ventral_side_f

    #add ventral side if given
    if ventral_side in VALID_CNT:
        with tables.File(skeletons_file, 'r+') as fid:
            fid.get_node('/trajectories_data').attrs['ventral_side'] = ventral_side
    return ventral_side

def _switch_cnt(skeletons_file):
    with tables.File(skeletons_file, 'r+') as fid:
        # since here we are changing all the contours, let's just change
        # the name of the datasets
        side1 = fid.get_node('/contour_side1')
        side2 = fid.get_node('/contour_side2')

        side1.rename('contour_side1_bkp')
        side2.rename('contour_side1')
        side1.rename('contour_side2')

def isBadVentralOrient(skeletons_file, ventral_side=''):
    ventral_side = _add_ventral_side(skeletons_file, ventral_side) 
    if not ventral_side in VALID_CNT:
        return True

    elif ventral_side == 'unknown':
        is_bad =  False
    
    elif ventral_side in ['clockwise', 'anticlockwise']:
        with tables.File(skeletons_file, 'r') as fid:
            has_skeletons = fid.get_node('/trajectories_data').col('has_skeleton')

            # let's use the first valid skeleton, it seems like a waste to use all the other skeletons.
            # I checked earlier to make sure the have the same orientation.

            valid_ind = np.where(has_skeletons)[0]
            if valid_ind.size == 0:
                #no valid skeletons, nothing to do here.
                is_bad = True
            else:
                cnt_side1 = fid.get_node('/contour_side1')[valid_ind[0], :, :]
                cnt_side2 = fid.get_node('/contour_side2')[valid_ind[0], :, :]
                A_sign = _h_calAreaSignedArray(cnt_side1, cnt_side2)
                
                # if not (np.all(A_sign > 0) or np.all(A_sign < 0)):
                #    raise ValueError('There is a problem. All the contours should have the same orientation.')
                if ventral_side == 'clockwise':
                    is_bad = A_sign[0] < 0
                elif ventral_side == 'anticlockwise':
                    is_bad = A_sign[0] > 0
                else:
                    raise ValueError

        if is_bad:
            _switch_cnt(skeletons_file)
            is_bad = False


    return is_bad

def isGoodVentralOrient(skeletons_file, ventral_side=''):
    return not isBadVentralOrient(skeletons_file, ventral_side=ventral_side)

        
