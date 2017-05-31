from .createSampleVideo import createSampleVideo

def args_(fn, param):
	
    #input arguments for createSampleVideo
	argkws_d = {
		'masked_image_file': fn['masked_image'], 
		'sample_video_name':fn['subsample'],
        'time_factor' : 8, 
        'size_factor' : 5, 
        'dflt_fps' : param.p_dict['expected_fps']
        }
	
    #arguments used by AnalysisPoints.py
	return {
        'func':createSampleVideo,
        'argkws' : argkws_d,
        'input_files' : [fn['masked_image']],
        'output_files': [fn['masked_image'], fn['subsample']],
        'requirements' : ['COMPRESS'],
    	}