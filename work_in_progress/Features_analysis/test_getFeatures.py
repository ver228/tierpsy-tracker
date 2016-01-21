# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:20:17 2016

@author: ajaver
"""
import sys
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking/')
sys.path.append('/Users/ajaver/Documents/GitHub/movement_validation/')


skeletons_file = '/Users/ajaver/Desktop/Videos/Check_Align_samples/Results/npr-2 (ok419)IV on food R_2010_01_25__15_29_03___4___10_skeletons.hdf5'

from MWTracker.featuresAnalysis.obtainFeaturesHelper import getValidIndexes
from MWTracker.featuresAnalysis.obtainFeatures import getWormFeatures

from open_worm_analysis_toolbox.features.worm_features import WormFeaturesDos
from MWTracker.featuresAnalysis.obtainFeaturesHelper import wormStatsClass, WormFromTable

worm = WormFromTable()
worm.fromFile(skeletons_file, 1, fps = 25, isOpenWorm = True)
worm_features = WormFeaturesDos(worm)
wStats = wormStatsClass()

#%%
stats = wStats.getWormStats(worm_features)
 
#%%
#getWormFeatures(skeletons_file, 'test.hdf5')
