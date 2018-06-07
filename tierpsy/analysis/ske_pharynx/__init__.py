from .orient_pharinx import orient_pharinx

def args_(fn, param):
    return {
        'func': orient_pharinx,
        'argkws': {'masked_file': fn['masked_image'], 'skeletons_file': fn['skeletons']},
        'input_files' : [fn['skeletons'], fn['masked_image']],
        'output_files': [fn['skeletons']],
        'requirements' : ['SKE_CREATE'],
    }