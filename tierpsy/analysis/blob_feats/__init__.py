from .getBlobsFeats import getBlobsFeats

def args_(fn, param):
    # arguments for getBlobsTable
    argkws_d = {'skeletons_file':fn['skeletons'], 
                'masked_image_file':fn['masked_image'],
                'strel_size' : param.p_dict['strel_size']
                }

    #arguments used by AnalysisPoints.py
    return {
        'func': getBlobsFeats,
        'argkws': argkws_d,
        'input_files' : [fn['skeletons'], fn['masked_image']],
        'output_files': [fn['skeletons']],
        'requirements' : ['SKE_INIT'],
    }

    getBlobsFeats(skeletons_file, masked_image_file, strel_size)