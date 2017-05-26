from .checkHeadOrientation import correctHeadTail

def _get_head_tail_param(p):
    return {
        'max_gap_allowed': p['max_gap_allowed_block'], 
        'window_std': -1,
        'segment4angle': p['ht_orient_segment'],
        'min_block_size': -1
        }

def args_(fn, param):
    argkws_d = _get_head_tail_param(param.p_dict)
    argkws_d['skeletons_file'] = fn['skeletons']

    #arguments used by AnalysisPoints.py
    return {
        'func': correctHeadTail,
        'argkws': argkws_d,
        'input_files' : [fn['skeletons']],
        'output_files': [fn['skeletons']],
        'requirements' : ['SKE_CREATE'],
    }