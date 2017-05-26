from .correctHeadTailIntensity import correctHeadTailIntensity

def args_(fn, param):
    #arguments used by AnalysisPoints.py
    return {
        'func': correctHeadTailIntensity,
        'argkws': {**{'skeletons_file': fn['skeletons'], 'intensities_file': fn['intensities']},
                   **param.head_tail_int_param},
        'input_files' : [fn['skeletons'], fn['intensities']],
        'output_files': [fn['skeletons']],
        'requirements' : ['INT_PROFILE'],
    }