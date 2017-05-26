from .getBlobTrajectories import getBlobsTable

def args_(fn, param):
	#arguments used by AnalysisPoints.py
	return {
        'func': getBlobsTable,
        'argkws': {**{'masked_image_file': fn['masked_image'], 'trajectories_file': fn['skeletons']},
                   **param.trajectories_param},
        'input_files' : [fn['masked_image']],
        'output_files': [fn['skeletons']],
        'requirements' : ['COMPRESS'],
    }