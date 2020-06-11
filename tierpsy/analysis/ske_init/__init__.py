import os
from tierpsy.helper.params.models_path import get_model_filter_worms
from .processTrajectoryData import processTrajectoryData


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
        'masked_image_file': fn['masked_image'],
        'trajectories_file': fn['skeletons'],
        'smoothed_traj_param': smoothed_traj_param,
        'filter_model_name': get_model_filter_worms(p)
        }
    # print(f"Using model: {argkws_d['filter_model_name']}")
    # arguments used by AnalysisPoints.py
    return {
        'func': processTrajectoryData,
        'argkws': argkws_d,
        'input_files': [fn['masked_image'], fn['skeletons']],
        'output_files': [fn['skeletons']],
        'requirements': ['TRAJ_JOIN'],
    }
