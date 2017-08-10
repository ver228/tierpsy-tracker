from .obtainFeatures import getWormFeaturesFilt
from ..ske_filt import get_feat_filt_param

def _get_feats_param(p):
  return {
    'feat_filt_param': get_feat_filt_param(p),
    'split_traj_time' : p['split_traj_time'],
    'is_single_worm': p['analysis_type'] == 'SINGLE_WORM_SHAFER'
    }



def args_(fn, param):
  # getWormFeatures
  main_func = getWormFeaturesFilt
  requirements = ['SKE_CREATE']
  if param.p_dict['analysis_type'] == 'SINGLE_WORM_SHAFER':
    from functools import partial
    from ..contour_orient import ventral_orient_wrapper
    from ..stage_aligment import isGoodStageAligment 

    requirements += ['STAGE_ALIGMENT', ('is_valid_alignment', partial(isGoodStageAligment, fn['skeletons']))]
    main_func = partial(ventral_orient_wrapper, main_func, fn['skeletons'], param.p_dict['ventral_side'])

  #arguments used by AnalysisPoints.py
  return {
        'func': main_func,
        'argkws': {'skeletons_file': fn['skeletons'], 'features_file': fn['features'],
                   **_get_feats_param(param.p_dict),
                   'use_skel_filter': True, 'use_manual_join': False
                   },
        'input_files' : [fn['skeletons']],
        'output_files': [fn['features']],
        'requirements' : requirements,
    }