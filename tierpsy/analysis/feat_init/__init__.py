from .smooth_skeletons_table import smooth_skeletons_table

def args_(fn, param):
  # getWormFeatures
  main_func = smooth_skeletons_table
  requirements = ['SKE_CREATE']
  
  is_WT2 = param.p_dict['analysis_type'] == 'WT2'
  #arguments used by AnalysisPoints.py
  return {
        'func': main_func,
        'argkws': {'skeletons_file': fn['skeletons'], 
                  'features_file': fn['featuresN'],
                  'is_WT2' : is_WT2,
                  'skel_smooth_window' : param.p_dict['feat_skel_smooth_window'],
                  'coords_smooth_window_s' : param.p_dict['feat_coords_smooth_window_s'],
                  'gap_to_interp_s' : param.p_dict['feat_gap_to_interp_s']
                  },
        'input_files' : [fn['skeletons']],
        'output_files': [fn['featuresN']],
        'requirements' : requirements
    }

