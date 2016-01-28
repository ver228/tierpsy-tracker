# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 12:23:12 2016

@author: ajaver
"""

import sys
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking/')
sys.path.append('/Users/ajaver/Documents/GitHub/movement_validation')
from MWTracker.featuresAnalysis.obtainFeaturesHelper import WormFromTable, calWormAnglesAll, smoothCurvesAll
from open_worm_analysis_toolbox import WormFeaturesDos
from open_worm_analysis_toolbox.features.feature_processing_options import FeatureProcessingOptions

import pandas as pd
import numpy as np
import matplotlib.pylab as plt


#%%
#skeletons_file = '/Users/ajaver/Desktop/Videos/Check_Align_samples/Results/npr-2 (ok419)IV on food R_2010_01_25__15_29_03___4___10_skeletons.hdf5'
skeletons_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch1_17112015_205616_skeletons.hdf5'

worm_index = 1
worm = WormFromTable(skeletons_file, worm_index, is_openworm = True)#, smooth_window = 5)


#%%
fpo = FeatureProcessingOptions()
worm_features = WormFeaturesDos(worm, fpo)

fpo.locomotion.velocity_tip_diff = 0.5
fpo.locomotion.velocity_body_diff = 1
worm_features_vel = WormFeaturesDos(worm, fpo)

fpo.locomotion.velocity_tip_diff = 1
fpo.locomotion.velocity_body_diff = 2
worm_features_vel2 = WormFeaturesDos(worm, fpo)


plt.figure()
plt.plot(worm_features.features['locomotion.velocity.midbody.speed'].value)

plt.figure()
plt.plot(worm_features_vel.features['locomotion.velocity.midbody.speed'].value)

plt.figure()
plt.plot(worm_features_vel2.features['locomotion.velocity.midbody.speed'].value)

#%%

