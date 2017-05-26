from .getBlobsFeats import getBlobsFeats

def args_(fn, param):
    #arguments used by AnalysisPoints.py
    return {
        'func': getBlobsFeats,
        'argkws': {**{'skeletons_file': fn['skeletons'], 
                    'masked_image_file': fn['masked_image']}, 
                    **param.blob_feats_param},
        'input_files' : [fn['skeletons'], fn['masked_image']],
        'output_files': [fn['skeletons']],
        'requirements' : ['SKE_INIT'],
    }