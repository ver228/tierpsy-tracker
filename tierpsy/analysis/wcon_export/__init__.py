from .exportWCON import exportWCON

def args_(fn, param):
	#arguments used by AnalysisPoints.py
	return {
        'func': exportWCON,
        'argkws': {'features_file': fn['features']},
        'input_files' : [fn['features']],
        'output_files': [fn['features'], fn['wcon']],
        'requirements' : ['FEAT_CREATE'],
    }