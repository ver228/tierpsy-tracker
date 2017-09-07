from .obtain_tierpsy_features import get_tierpsy_features

def args_(fn, param):
  # getWormFeatures
  main_func = get_tierpsy_features
  requirements = ['SKE_CREATE']
  
  is_WT2 = param.p_dict['analysis_type'] == 'WT2'
  #arguments used by AnalysisPoints.py
  return {
        'func': main_func,
        'argkws': {'skeletons_file': fn['skeletons'], 
                  'features_file': fn['featuresN'],
                  'is_WT2' : is_WT2
                  },
        'input_files' : [fn['skeletons']],
        'output_files': [fn['featuresN']],
        'requirements' : requirements
    }