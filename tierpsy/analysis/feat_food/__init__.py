from .getFoodFeatures import getFoodFeatures

def args_(fn, param):
  requirements = ['SKE_CREATE']
  
  #arguments used by AnalysisPoints.py
  return {
        'func': getFoodFeatures,
        'argkws': {
                    'mask_file': fn['masked_image'], 
                    'skeletons_file': fn['skeletons'],
                    'features_file': fn['featuresN']
                   },
        'input_files' : [fn['masked_image'], fn['skeletons']],
        'output_files': [fn['featuresN'], fn['skeletons']],
        'requirements' : requirements
    }


# cnt_method = 'NN',
# solidity_th=0.98,
# batch_size = 100000,