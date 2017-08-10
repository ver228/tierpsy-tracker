from .correctVentralDorsal import ventral_orient_wrapper

# from functools import partial
#from .correctVentralDorsal import switchCntSingleWorm, is_valid_cnt_info, isGoodVentralOrient, switchCntSingleWorm
# def is_valid_cnt_info(skeletons_file='', ventral_side=''):
#     ventral_side = _read_or_pass(skeletons_file, ventral_side)
    
#     is_valid = ventral_side in VALID_CNT
#     # if not is_valid:
#     #     base_name = os.path.basename(skeletons_file).replace('_skeletons.hdf5', '')
#     #     print('{} Not valid ventral_side:({}) in /experiments_info'.format(base_name, exp_info['ventral_side']))
    
#     # only clockwise and anticlockwise are valid contour orientations
#     return is_valid
# def args_(fn, param):
#     #arguments used by AnalysisPoints.py
#     return {
#     'func': switchCntSingleWorm,
#     'argkws': {'skeletons_file': fn['skeletons'], 'ventral_side':param.p_dict['ventral_side']},
#     'input_files' : [fn['skeletons']],
#     'output_files': [fn['skeletons']],
#     'requirements' : ['SKE_CREATE',
#                       ('is_valid_cnt_info', partial(is_valid_cnt_info, fn['skeletons'], param.p_dict['ventral_side']))],
#     }
