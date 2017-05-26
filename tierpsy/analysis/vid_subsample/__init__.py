from .createSampleVideo import createSampleVideo

def args_(fn, param):
	#arguments used by AnalysisPoints.py
	return {
        'func':createSampleVideo,
        'argkws' : {**{'masked_image_file': fn['masked_image'], 'sample_video_name':fn['subsample']}, 
                    **param.subsample_vid_param},
        'input_files' : [fn['masked_image']],
        'output_files': [fn['masked_image'], fn['subsample']],
        'requirements' : ['COMPRESS'],
    }