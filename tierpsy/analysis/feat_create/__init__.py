from .obtainFeatures import getWormFeaturesFilt

def args_(fn, param):
  requirements = ['SKE_CREATE']
  if param.p_dict['analysis_type'] == 'SINGLE_WORM_SHAFER':
    from functools import partial
    from ..contour_orient import isGoodVentralOrient
    requirements += ['CONTOUR_ORIENT', ('is_valid_contour', partial(isGoodVentralOrient, fn['skeletons']))]
    
    from ..stage_aligment import isGoodStageAligment 
    requirements += ['STAGE_ALIGMENT', ('is_valid_alignment', partial(isGoodStageAligment, fn['skeletons']))]

  #arguments used by AnalysisPoints.py
  return {
        'func': getWormFeaturesFilt,
        'argkws': {'skeletons_file': fn['skeletons'], 'features_file': fn['features'],
                   **param.feats_param,
                   'use_skel_filter': True, 'use_manual_join': False
                   },
        'input_files' : [fn['skeletons']],
        'output_files': [fn['features']],
        'requirements' : requirements,
    }