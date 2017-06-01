from .alignStageMotion import alignStageMotion, isGoodStageAligment

def args_(fn, param):
	#arguments used by AnalysisPoints.py
	return {
        'func': alignStageMotion,
        'argkws': {'masked_image_file': fn['masked_image'], 'skeletons_file': fn['skeletons']},
        'input_files' : [fn['skeletons'], fn['masked_image']],
        'output_files': [fn['skeletons']],
        'requirements' : ['COMPRESS_ADD_DATA', 'SKE_CREATE'],
    }