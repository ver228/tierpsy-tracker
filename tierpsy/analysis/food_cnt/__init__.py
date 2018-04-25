from .getFoodContour import getFoodContour

def args_(fn, param):
  requirements = ['SKE_INIT']
  
  argkws = dict(
          cnt_method = 'NN',
          solidity_th=0.98,
          batch_size = 100000,
          _is_debug = False

  )
  #arguments used by AnalysisPoints.py
  return {
        'func': getFoodContour,
        'argkws': {
                    'mask_file': fn['masked_image'], 
                    'skeletons_file': fn['skeletons']
                   },
        'input_files' : [fn['masked_image']],
        'output_files': [fn['skeletons'], fn['masked_image']],
        'requirements' : requirements
    }
