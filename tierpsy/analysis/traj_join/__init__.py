from .joinBlobsTrajectories import joinBlobsTrajectories

def args_(fn, param):
	p = param.p_dict
	#arguments for joinBlobsTrajectories
	argkws_d = {
		'trajectories_file': fn['skeletons'],
		'max_allowed_dist': p['traj_max_allowed_dist'],
        'min_track_size': 0, 
        'max_time_gap': 0, 
        'area_ratio_lim': p['traj_area_ratio_lim'],
        'analysis_type': p['analysis_type']
		}

	#arguments used by AnalysisPoints.py
	return {
        'func': joinBlobsTrajectories,
        'argkws': argkws_d,
        'input_files' : [fn['skeletons']],
        'output_files': [fn['skeletons']],
        'requirements' : ['TRAJ_CREATE'],
    	}