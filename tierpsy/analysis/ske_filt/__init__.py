from tierpsy.helper.params import get_prefix_params
from .getFilteredSkels import getFilteredSkels


def _get_feat_filt_param(p):
    feat_filt_param = get_prefix_params(p, 'filt_')
    feat_filt_param['min_num_skel'] = -1
    return feat_filt_param


def args_(fn, param):
    argkws_d = _get_feat_filt_param(param.p_dict)
    argkws_d['skeletons_file']= fn['skeletons']

    #arguments used by AnalysisPoints.py
    return {
     	'func': getFilteredSkels,
        'argkws': argkws_d,
        'input_files' : [fn['skeletons']],
        'output_files': [fn['skeletons']],
        'requirements' : ['SKE_CREATE'],
    }