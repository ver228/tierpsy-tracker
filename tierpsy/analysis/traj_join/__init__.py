from .joinBlobsTrajectories import joinBlobsTrajectories

def args_(fn, param):
	#arguments used by AnalysisPoints.py
	return {
        'func': joinBlobsTrajectories,
        'argkws': {**{'trajectories_file': fn['skeletons']},
                    **param.join_traj_param},
        'input_files' : [fn['skeletons']],
        'output_files': [fn['skeletons']],
        'requirements' : ['TRAJ_CREATE'],
    }