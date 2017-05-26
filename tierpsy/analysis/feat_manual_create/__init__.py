from functools import partial
from ..feat_create.obtainFeatures import getWormFeaturesFilt, hasManualJoin

def args_(fn, param):
  #arguments used by AnalysisPoints.py
  return {
          'func': getWormFeaturesFilt,
          'argkws': {'skeletons_file': fn['skeletons'], 'features_file': fn['feat_manual'],
                     **param.feats_param,
                     'use_skel_filter':  True, 'use_manual_join': True,
                     },
          'input_files' : [fn['skeletons']],
          'output_files': [fn['feat_manual']],
          'requirements' : ['SKE_CREATE',
                            ('has_manual_joined_traj', partial(hasManualJoin, fn['skeletons']))],
      }