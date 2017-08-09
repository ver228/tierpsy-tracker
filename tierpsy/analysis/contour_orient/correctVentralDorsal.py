import json
import os
import numpy as np
import tables
import warnings

from tierpsy.analysis.ske_filt.getFilteredSkels import _h_calAreaSignedArray
from tierpsy.helper.params import read_ventral_side

VALID_CNT = ['clockwise', 'anticlockwise', 'unknown']

def _read_or_pass(skeletons_file, ventral_side):
    #decide if to read or pass the value ventral_side. I give preference to read from file.
    if skeletons_file:
        #if it is not supplied try to read it from the file
        ventral_side_f = read_ventral_side(skeletons_file)

    if ventral_side_f in VALID_CNT:
        
        return ventral_side_f
    else:
        return ventral_side

def is_valid_cnt_info(skeletons_file='', ventral_side=''):
    ventral_side = _read_or_pass(skeletons_file, ventral_side)
    
    is_valid = ventral_side in VALID_CNT
    # if not is_valid:
    #     base_name = os.path.basename(skeletons_file).replace('_skeletons.hdf5', '')
    #     print('{} Not valid ventral_side:({}) in /experiments_info'.format(base_name, exp_info['ventral_side']))
    
    # only clockwise and anticlockwise are valid contour orientations
    return is_valid


def isBadVentralOrient(skeletons_file):
    ventral_side = read_ventral_side(skeletons_file)
    if ventral_side == 'unknown':
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
    else:
        is_bad = True

    return is_bad

def isGoodVentralOrient(skeletons_file):
    return not isBadVentralOrient(skeletons_file)
    
def _add_ventral_side(skeletons_file, ventral_side):
    ventral_side = _read_or_pass(skeletons_file, ventral_side)
    if ventral_side in VALID_CNT:
        with tables.File(skeletons_file, 'r+') as fid:
            fid.get_node('/trajectories_data').attrs['ventral_side'] = ventral_side
    return ventral_side

def switchCntSingleWorm(skeletons_file, ventral_side=''):
    ventral_side = _add_ventral_side(skeletons_file, ventral_side)
    if isBadVentralOrient(skeletons_file):
        with tables.File(skeletons_file, 'r+') as fid:
            # since here we are changing all the contours, let's just change
            # the name of the datasets
            side1 = fid.get_node('/contour_side1')
            side2 = fid.get_node('/contour_side2')

            side1.rename('contour_side1_bkp')
            side2.rename('contour_side1')
            side1.rename('contour_side2')
