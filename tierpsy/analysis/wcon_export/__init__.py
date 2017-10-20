from functools import partial
from .exportWCON import exportWCON

def _h_check_zip_integrity(fname):
    try:
        with zipfile.ZipFile(fname) as zf:
            ret = zf.testzip()
    except:
        ret = -3

    if ret is None:
        return True
    else:
        return False
#('is_valid_zip', partial(_h_check_zip_integrity, fn['wcon']))
                        
def args_(fn, param):
    #arguments used by AnalysisPoints.py
    return {
        'func': exportWCON,
        'argkws': {'features_file': fn['features']},
        'input_files' : [fn['features']],
        'output_files': [fn['features'], fn['wcon']],
        'requirements' : ['FEAT_CREATE']
    }