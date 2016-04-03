# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 17:31:45 2016

@author: ajaver
"""
import matplotlib.pylab as plt
from MWTracker.featuresAnalysis.obtainFeaturesHelper import getValidIndexes, calWormArea
from MWTracker.featuresAnalysis.getFilteredFeats import readFeat2Check, getMahalanobisRobust

from sklearn.covariance import EllipticEnvelope, EmpiricalCovariance, MinCovDet

skeletons_file = '/Users/ajaver/Tmp/Results/nas207-1/experimentBackup/from pc207-18/!worm_videos/copied from pc207-15/Andre/24-02-11/371 JU402 swimming_2011_02_24__12_35_14___2___2_skeletons.hdf5'

min_num_skel = 120
bad_seg_thresh = 0.8
min_dist = 0

good_traj_index , good_skel_row = getValidIndexes(skeletons_file, \
        min_num_skel = min_num_skel, bad_seg_thresh = bad_seg_thresh, min_dist = min_dist)

datFeats = readFeat2Check(skeletons_file)

robust_cov = MinCovDet().fit(datFeats[1][good_skel_row])

#%%
dat = datFeats[1][good_skel_row]
for kk in dat.shape[1]:
    x = dat[:, kk]
    
    plt.plot(x)