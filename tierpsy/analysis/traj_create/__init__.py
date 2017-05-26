from .getBlobTrajectories import getBlobsTable

def args_(fn, param):

    p = param.p_dict
    trajectories_param_f = ['traj_min_area', 'traj_min_box_width',
        'worm_bw_thresh_factor', 'strel_size', 'analysis_type', 'thresh_block_size',
        'n_cores_used']
    trajectories_param = {x.replace('traj_', ''):p[x] for x in trajectories_param_f}
    trajectories_param['buffer_size'] = p['compression_buff']
    
    argkws_d = {'masked_image_file': fn['masked_image'], 
                'trajectories_file': fn['skeletons'],
                **trajectories_param
                }
                
    #arguments used by AnalysisPoints.py
    return {
        'func': getBlobsTable,
        'argkws': argkws_d,
        'input_files' : [fn['masked_image']],
        'output_files': [fn['skeletons']],
        'requirements' : ['COMPRESS'],
    }