import json
import os
import numpy as np
import tables
import warnings

from tierpsy.analysis.ske_filt.getFilteredSkels import _h_calAreaSignedArray

VALID_CNT = ['clockwise', 'anticlockwise', 'unknown']

def read_ventral_side(skeletons_file, ventral_orientation=''):
    if ventral_orientation in VALID_CNT:
        return ventral_orientation
    if ventral_orientation == 'read_basename':
        bn = os.path.basename(skeletons_file)
        if '_R_' in bn:
            return 'anticlockwise'
        elif '_L_' in bn:
            return 'clockwise'
        else:
            return 'unknown'

    #default try to read attribute
    try:
        with tables.File(skeletons_file, 'r') as fid:
            exp_info_b = fid.get_node('/experiment_info').read()
            exp_info = json.loads(exp_info_b.decode("utf-8"))
            return exp_info['ventral_side']
    except:
        return ""

def hasExpCntInfo(skeletons_file, ventral_orientation=''):
    # i'm reading this data twice (one more in switchCntSingleWorm), but I think this is cleaner
    # from a function organization point of view.
    ventral_side = read_ventral_side(skeletons_file, ventral_orientation)    

    is_valid = ventral_side in VALID_CNT
    # if not is_valid:
    #     base_name = os.path.basename(skeletons_file).replace('_skeletons.hdf5', '')
    #     print('{} Not valid ventral_side:({}) in /experiments_info'.format(base_name, exp_info['ventral_side']))
    
    # only clockwise and anticlockwise are valid contour orientations
    return is_valid

def isBadVentralOrient(skeletons_file, ventral_orientation=''):
    ventral_side = read_ventral_side(skeletons_file, ventral_orientation)
    if not ventral_side in VALID_CNT:
        # msg = '{}: "{}" is not a valid value for ventral side orientation. '.format(skeletons_file, exp_info['ventral_side'])
        # msg += 'Only "clockwise" or "anticlockwise" are accepted values'
        # warnings.warn(msg)
        return True
    
    if ventral_side == 'unknown':
        return False

    with tables.File(skeletons_file, 'r') as fid:
        

        has_skeletons = fid.get_node('/trajectories_data').col('has_skeleton')

        # let's use the first valid skeleton, it seems like a waste to use all the other skeletons.
        # I checked earlier to make sure the have the same orientation.

        valid_ind = np.where(has_skeletons)[0]
        if valid_ind.size == 0:
            return

        cnt_side1 = fid.get_node('/contour_side1')[valid_ind[0], :, :]
        cnt_side2 = fid.get_node('/contour_side2')[valid_ind[0], :, :]
        A_sign = _h_calAreaSignedArray(cnt_side1, cnt_side2)

        # if not (np.all(A_sign > 0) or np.all(A_sign < 0)):
        #    raise ValueError('There is a problem. All the contours should have the same orientation.')

        return (ventral_side == 'clockwise' and A_sign[0] < 0) or \
            (ventral_side == 'anticlockwise' and A_sign[0] > 0)

def isGoodVentralOrient(skeletons_file, ventral_orientation=''):
    #save as isBadVentral but opposite, and fault tolerant
    try:
        return not isBadVentralOrient(skeletons_file, ventral_orientation)
    except:
        return False

def switchCntSingleWorm(skeletons_file, ventral_orientation=''):

    if isBadVentralOrient(skeletons_file):
        with tables.File(skeletons_file, 'r+') as fid:
            # since here we are changing all the contours, let's just change
            # the name of the datasets
            side1 = fid.get_node('/contour_side1')
            side2 = fid.get_node('/contour_side2')

            side1.rename('contour_side1_bkp')
            side2.rename('contour_side1')
            side1.rename('contour_side2')
