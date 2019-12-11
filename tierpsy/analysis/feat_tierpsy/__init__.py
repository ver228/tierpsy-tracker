from .get_tierpsy_features import get_tierpsy_features

def args_(fn, param):
  # getWormFeatures
  main_func = get_tierpsy_features
  requirements = ['FEAT_INIT']
  
  
  # FOV splitting
  fovsplitter_param_f = ['MWP_total_n_wells', 'MWP_whichsideup', 'MWP_well_shape']
  if not all(k in param.p_dict for k in fovsplitter_param_f): # both total wells and which side up have to be in param
    fovsplitter_param = {}
  else:
    fovsplitter_param = {x.replace('MWP_',''):param.p_dict[x] for x in fovsplitter_param_f}   
    
  if fovsplitter_param['total_n_wells']<0:
    fovsplitter_param = {}
  
  #arguments used by AnalysisPoints.py
  return {
        'func': main_func,
        'argkws': {
                  'features_file': fn['featuresN'],
                  'derivate_delta_time': param.p_dict['feat_derivate_delta_time'],
                  'fovsplitter_param': fovsplitter_param
                  },
        'input_files' : [fn['featuresN']],
        'output_files': [fn['featuresN']],
        'requirements' : requirements
    }
    