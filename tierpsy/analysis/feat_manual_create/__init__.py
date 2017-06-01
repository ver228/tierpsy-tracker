from functools import partial
from ..feat_create.obtainFeatures import getWormFeaturesFilt, hasManualJoin
from ..feat_create import _get_feats_param

def args_(fn, param):
  argkws_d ={
      'skeletons_file': fn['skeletons'], 
      'features_file': fn['feat_manual'],
      **_get_feats_param(param.p_dict),
      'use_skel_filter':  True, 
      'use_manual_join': True,
    }
    
  #arguments used by AnalysisPoints.py
  return {
          'func': getWormFeaturesFilt,
          'argkws': argkws_d,
          'input_files' : [fn['skeletons']],
          'output_files': [fn['feat_manual']],
          'requirements' : ['SKE_CREATE',
                            ('has_manual_joined_traj', partial(hasManualJoin, fn['skeletons']))],
      }