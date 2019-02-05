from .joinBlobsTrajectories import joinBlobsTrajectories

def args_(fn, param):
	p = param.p_dict
	#arguments for joinBlobsTrajectories
	argkws_d = {
		'trajectories_file': fn['skeletons'],
		'max_allowed_dist': p['traj_max_allowed_dist'],
        'min_track_size': 0,
        'max_frames_gap': p['traj_max_frames_gap'], 
        'area_ratio_lim': p['traj_area_ratio_lim'],
        'is_one_worm': param.is_one_worm,
        'is_WT2' : param.is_WT2
		}

	#arguments used by AnalysisPoints.py
	return {
        'func': joinBlobsTrajectories,
        'argkws': argkws_d,
        'input_files' : [fn['skeletons']],
        'output_files': [fn['skeletons']],
        'requirements' : ['TRAJ_CREATE'],
    	}