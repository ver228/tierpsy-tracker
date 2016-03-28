# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 19:10:00 2016

@author: ajaver
"""
import h5py
import numpy as np
import matplotlib.pylab as plt
from MWTracker.featuresAnalysis.getFilteredFeats import nodes2Array

from sklearn.covariance import EllipticEnvelope, EmpiricalCovariance, MinCovDet

file_mask = '/Users/ajaver/Desktop/Videos/single_worm/agar_2/MaskedVideos/798 JU258 on food L_2011_03_22__16_26_58___1___12.hdf5'

skeletons_file = file_mask.replace('MaskedVideos', 'Results').replace('.hdf5', '_skeletons.hdf5')
trajectories_file = file_mask.replace('MaskedVideos', 'Results').replace('.hdf5', '_trajectories.hdf5')

#feats2read = ['width_head_base', 'width_head_tip', 'width_hips', 'width_midbody', 
#'width_neck', 'width_tail_base', 'width_tail_tip', 'contour_area']
feats = {}

with h5py.File(skeletons_file, 'r') as fid:
    #skeletons = fid['/skeleton']
    cnt_widths = fid['/contour_width'][:]
    cnt_areas = fid['/contour_area'][:]
    skel_length = fid['/skeleton_length'][:]
#%%
X4fito = nodes2Array(skeletons_file)
#
#for kk in range(X4fito.shape[1]):
#    plt.figure()
#    plt.plot(X4fito[:,kk], '.')
    #plt.hist(X4fit[:,kk], 100)
#%%
med_width = np.nanmedian(cnt_widths,axis=0);

#dd = cnt_widths-med_width;
#
#for kk in range(49):
#    plt.figure()
#    plt.hist(dd[~np.isnan(dd[:,kk]), kk], 100)


#%%
bad = np.any(np.isnan(X4fito),axis=1);
X4fit = X4fito[~bad,:]
#%%
emp_cov = EmpiricalCovariance().fit(X4fit)
robust_cov = MinCovDet().fit(X4fit)

#%%
#mahal_emp_cov = emp_cov.mahalanobis(X4fit)
emp_mahal = emp_cov.mahalanobis(X4fit - np.mean(X4fit, 0))**(1/3)

robust_mahal = robust_cov.mahalanobis(X4fit - robust_cov.location_) ** (0.33)
#%%
clf = EllipticEnvelope().fit(X4fit)
#%%
Z = clf.decision_function(X4fito)
Z1 = clf.decision_function(X4fito, raw_values=True)

plt.figure()
plt.plot(Z)
#plt.plot(Z1)

#%%
clf = EllipticEnvelope(contamination=1e-6).fit(X4fit)

Z = clf.decision_function(X4fito)

plt.plot(Z)