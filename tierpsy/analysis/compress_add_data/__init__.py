from functools import partial
from .getAdditionalData import storeAdditionalDataSW, hasAdditionalFiles

def args_(fn, param):
    requirements = [
                'COMPRESS',
                ('has_additional_files', partial(hasAdditionalFiles, fn['original_video']))
                ]

    #arguments used by AnalysisPoints.py
    return {
        'func':storeAdditionalDataSW,
        'argkws' : {'video_file': fn['original_video'], 'masked_image_file': fn['masked_image']},
        'input_files' : [fn['original_video'], fn['masked_image']],
        'output_files': [fn['masked_image']],
        'requirements' : requirements
    }