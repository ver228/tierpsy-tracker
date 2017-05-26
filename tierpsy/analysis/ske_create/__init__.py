from .getSkeletonsTables import trajectories2Skeletons
from tierpsy.helper.params.tracker_param import get_prefix_params

def args_(fn, param):

    p = param.p_dict
    if p['analysis_type'] == 'ZEBRAFISH':
        skel_args = get_prefix_params(p, 'zf_')
    else:
        skel_args = {'num_segments' : p['w_num_segments'],
                     'head_angle_thresh' : p['w_head_angle_thresh']}

    # trajectories2Skeletons
    argkws_d = {
        'skeletons_file': fn['skeletons'], 
        'masked_image_file': fn['masked_image'],
        'resampling_N': p['resampling_N'],
        'worm_midbody': (0.33, 0.67),
        'min_blob_area': p['traj_min_area'],
        'strel_size': p['strel_size'],
        'analysis_type': p['analysis_type'],
        'skel_args' : skel_args
        }

    #arguments used by AnalysisPoints.py
    return {
        'func': trajectories2Skeletons,
        'argkws': argkws_d,
        'input_files' : [fn['masked_image']],
        'output_files': [fn['skeletons']],
        'requirements' : ['SKE_INIT'],
    }