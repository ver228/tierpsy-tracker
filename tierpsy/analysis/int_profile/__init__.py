from .getIntensityProfile import getIntensityProfile

def args_(fn, param):
    requirements = ['SKE_CREATE']
    if param.p_dict['analysis_type'] == 'SINGLE_WORM_SHAFER':
        from ..contour_orient import isGoodVentralOrient
        from functools import partial
        requirements += ['CONTOUR_ORIENT', ('is_valid_contour', partial(isGoodVentralOrient, fn['skeletons']))]
        
    #arguments used by AnalysisPoints.py
    return {
        'func': getIntensityProfile,
        'argkws': {**{'masked_image_file': fn['masked_image'], 'skeletons_file': fn['skeletons'],
                      'intensities_file': fn['intensities']}, **param.int_profile_param},
        'input_files' : [fn['skeletons'],fn['masked_image']],
        'output_files': [fn['intensities'], fn['skeletons']],
        'requirements' : ['SKE_CREATE'],
    }