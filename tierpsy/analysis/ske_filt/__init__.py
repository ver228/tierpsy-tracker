from .getFilteredSkels import getFilteredSkels

def args_(fn, param):
    #arguments used by AnalysisPoints.py
    return {
        'func': getFilteredSkels,
        'argkws': {**{'skeletons_file': fn['skeletons']}, **param.feat_filt_param},
        'input_files' : [fn['skeletons']],
        'output_files': [fn['skeletons']],
        'requirements' : ['SKE_CREATE'],
    }