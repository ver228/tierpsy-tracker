from .obtainFeatures import getWormFeaturesFilt
from ..ske_filt import get_feat_filt_param

def _get_feats_param(param):
  p = param.p_dict
  return {
    'feat_filt_param': get_feat_filt_param(p),
    'split_traj_time' : p['split_traj_time'],
    'is_single_worm': param.is_WT2
    }



def args_(fn, param):
  # getWormFeatures
  main_func = getWormFeaturesFilt
  requirements = ['SKE_CREATE']
  if param.is_WT2:
    from functools import partial
    from ..contour_orient import ventral_orient_wrapper
    from ..stage_aligment import isGoodStageAligment 

    requirements += ['STAGE_ALIGMENT', ('is_valid_alignment', partial(isGoodStageAligment, fn['skeletons']))]
    main_func = partial(ventral_orient_wrapper, main_func, fn['skeletons'], param.p_dict['ventral_side'])
    main_func.__name__ = getWormFeaturesFilt.__name__ # I use the name for provenance tracking

  #arguments used by AnalysisPoints.py
  return {
        'func': main_func,
        'argkws': {'skeletons_file': fn['skeletons'], 'features_file': fn['features'],
                   **_get_feats_param(param),
                   'use_skel_filter': True, 'use_manual_join': False
                   },
        'input_files' : [fn['skeletons']],
        'output_files': [fn['features']],
        'requirements' : requirements,
    }