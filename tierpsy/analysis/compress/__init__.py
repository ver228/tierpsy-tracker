from functools import partial

from .processVideo import processVideo, isGoodVideo


def args_(fn, param):

    requirements = [('can_read_video', partial(isGoodVideo, fn['original_video']))]
    if param.p_dict['analysis_type'] == 'SINGLE_WORM_SHAFER':
        from ..compress_add_data import storeAdditionalDataSW, hasAdditionalFiles
        #if a shaffer single worm video does not have the additional files (info.xml log.csv) do not even execute the compression 
        requirements += [('has_additional_files', partial(hasAdditionalFiles, fn['original_video']))]

    #arguments used by AnalysisPoints.py
    return {
        'func':processVideo,
        'argkws' : {'video_file': fn['original_video'], 
                    'masked_image_file' : fn['masked_image'],
                    'compress_vid_param' : param.compress_vid_param},
        'input_files' : [fn['original_video']],
        'output_files': [fn['masked_image']],
        'requirements' : requirements,
    }