import os
from tierpsy import AUX_FILES_DIR
from .processTrajectoryData import processTrajectoryData

def _correct_filter_model_name(filter_model_name):
    if filter_model_name:
        if not os.path.exists(filter_model_name):
            #try to look for the file in the AUX_FILES_DIR
            filter_model_name = os.path.join(AUX_FILES_DIR, filter_model_name)
        assert  os.path.exists(filter_model_name)

    return filter_model_name

def args_(fn, param):
    p = param.p_dict
    # getSmoothTrajectories
    smoothed_traj_param = {
        'min_track_size': 0,
        'displacement_smooth_win': -1,
        'threshold_smooth_win': -1,
        'roi_size': p['roi_size']
        }
        
    argkws_d = {
        'skeletons_file': fn['skeletons'],
        'masked_image_file':fn['masked_image'],
        'trajectories_file': fn['skeletons'],
        'smoothed_traj_param': smoothed_traj_param,
        'filter_model_name' : _correct_filter_model_name(p['filter_model_name'])
        }
    #arguments used by AnalysisPoints.py
    return {
        'func': processTrajectoryData,
        'argkws': argkws_d,
        'input_files' : [fn['masked_image'], fn['skeletons']],
        'output_files': [fn['skeletons']],
        'requirements' : ['TRAJ_JOIN'],
    }