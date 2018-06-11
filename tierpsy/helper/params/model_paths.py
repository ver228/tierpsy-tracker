from tierpsy import AUX_FILES_DIR
import os
import warning


DFLT_MODEL_FILTER_WORMS = os.path.join(AUX_FILES_DIR, 'model_isworm_20170407_184845.h5')
if not os.path.exists(DFLT_MODEL_FILTER_WORMS):
    warning.warn('The default model file to filter spurious particles was not found. This step will not be done.')
    DFLT_MODEL_FILTER_WORMS = ''

DFLT_MODEL_FOOD_CONTOUR = os.path.join(AUX_FILES_DIR, 'unet_RMSprop-5-04999-0.3997.h5')
if not os.path.exists(DFLT_MODEL_FOOD_CONTOUR):
    warning.warn('The default model to obtain the food contour was not found. I would try a less accurate algorithm to calculate the contour.')
    DFLT_MODEL_FOOD_CONTOUR = ''

