from functools import partial

from .correctVentralDorsal import switchCntSingleWorm, is_valid_cnt_info, isGoodVentralOrient

def args_(fn, param):
    #arguments used by AnalysisPoints.py
    return {
    'func': switchCntSingleWorm,
    'argkws': {'skeletons_file': fn['skeletons'], 'ventral_side':param.p_dict['ventral_side']},
    'input_files' : [fn['skeletons']],
    'output_files': [fn['skeletons']],
    'requirements' : ['SKE_CREATE',
                      ('is_valid_cnt_info', partial(is_valid_cnt_info, fn['skeletons'], param.p_dict['ventral_side']))],
    }