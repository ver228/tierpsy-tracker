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

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

#skeletons_file = '/Users/ajaver/Desktop/Videos/Check_Align_samples/Results/npr-2 (ok419)IV on food R_2010_01_25__15_29_03___4___10_skeletons.hdf5'
skeletons_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch5_17112015_205616_skeletons.hdf5'

worm_index = 1
worm_smooth = WormFromTable(skeletons_file, worm_index, smooth_window = 5, is_openworm = True)
worm = WormFromTable(skeletons_file, worm_index, is_openworm = True)

#%%
frame_number = 1
plt.figure()
plt.plot(worm.skeleton[:,0,frame_number], worm.skeleton[:,1,frame_number])
plt.plot(worm_smooth.skeleton[:,0,frame_number], worm_smooth.skeleton[:,1,frame_number])
#plt.axis('equal')
#%%
frame_number = 1
plt.figure()
plt.plot(worm.angles[:, frame_number])
plt.plot(worm_smooth.angles[:, frame_number])
#%%
worm_features_smooth = WormFeaturesDos(worm_smooth)
worm_features = WormFeaturesDos(worm)

#%%
for feat_name in ['posture.amplitude_max', 'posture.primary_wavelength', 'posture.secondary_wavelength',
                  'posture.kinks', 'posture.bends.midbody.mean', 'posture.eigen_projection0',
                  'posture.eigen_projection1', 'posture.eigen_projection2', 
                  'posture.eigen_projection3', 'posture.eigen_projection4',
                  'posture.eigen_projection5']:
    plt.figure()
    plt.plot(worm_features.features[feat_name].value, worm_features_smooth.features[feat_name].value, '.')
    plt.title(feat_name)    
    plt.xlabel('normal')
    plt.ylabel('smoothed')
#%%
plt.figure()
plt.plot(worm_smooth.skeleton[25,0, :],worm_smooth.skeleton[25,1,:])
plt.axis('equal')