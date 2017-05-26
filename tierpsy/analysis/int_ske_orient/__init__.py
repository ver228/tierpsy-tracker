from ..ske_orient import _get_head_tail_param
from .correctHeadTailIntensity import correctHeadTailIntensity

def args_(fn, param):
    p = param.p_dict
    argkws_d = {
        'skeletons_file': fn['skeletons'], 
        'intensities_file': fn['intensities'],
        'smooth_W': -1,
        'gap_size': p['int_max_gap_allowed_block'],
        'min_block_size': -1,
        'local_avg_win': -1,
        'min_frac_in': 0.85,
        'head_tail_param': _get_head_tail_param(p),
        'head_tail_int_method': p['head_tail_int_method']
    }
    #arguments used by AnalysisPoints.py
    return {
        'func': correctHeadTailIntensity,
        'argkws': argkws_d,
        'input_files' : [fn['skeletons'], fn['intensities']],
        'output_files': [fn['skeletons']],
        'requirements' : ['INT_PROFILE'],
        }
