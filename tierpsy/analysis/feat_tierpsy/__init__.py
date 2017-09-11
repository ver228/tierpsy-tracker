from .get_tierpsy_features import get_tierpsy_features

def args_(fn, param):
  # getWormFeatures
  main_func = get_tierpsy_features
  requirements = ['FEAT_INIT']
  
  #arguments used by AnalysisPoints.py
  return {
        'func': main_func,
        'argkws': {
                  'features_file': fn['featuresN'],
                  'velocity_delta_time' : 1/3,
                  'curvature_window' : 7
                  },
        'input_files' : [fn['featuresN']],
        'output_files': [fn['featuresN']],
        'requirements' : requirements
    }

    