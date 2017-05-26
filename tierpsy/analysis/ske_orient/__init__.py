from .checkHeadOrientation import correctHeadTail

def args_(fn, param):
	#arguments used by AnalysisPoints.py
	return {
        'func': correctHeadTail,
        'argkws': {**{'skeletons_file': fn['skeletons']}, **param.head_tail_param},
        'input_files' : [fn['skeletons']],
        'output_files': [fn['skeletons']],
        'requirements' : ['SKE_CREATE'],
    }