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

#file_mask = '/Users/ajaver/Desktop/Videos/single_worm/agar_2/MaskedVideos/798 JU258 on food L_2011_03_22__16_26_58___1___12.hdf5'
#file_mask = '/Users/ajaver/Desktop/Videos/single_worm/agar_2/MaskedVideos/egl-18 (ok290)IV on food R_2010_07_20__11_26_31___8___3.hdf5'
#file_mask = '/Users/ajaver/Desktop/Videos/single_worm/agar_2/MaskedVideos/C11D2.2 (ok1565)IV on food L_2011_08_24__10_24_57___7___2.hdf5'
#file_mask = '/Users/ajaver/Desktop/Videos/single_worm/agar_2/MaskedVideos/C24G7.1 (ok1822) on food L_2010_03_30__12_07_16___4___3.hdf5'
file_mask = '/Users/ajaver/Desktop/Videos/single_worm/agar_2/MaskedVideos/acr-6 (ok3117)I on food L_2010_02_23__13_11_55___1___13.hdf5'

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
    width_head_tip = fid['/width_head_tip'][:]
    width_head_base = fid['/width_head_base'][:]
    width_neck = fid['/width_neck'][:]
    width_midbody = fid['/width_midbody'][:]
    width_hips = fid['/width_hips'][:]
    width_tail_base = fid['/width_tail_base'][:]
    width_tail_tip = fid['/width_tail_tip'][:]
#%%
widths = np.stack([width_head_base, width_head_tip, width_hips, width_midbody, width_neck, width_tail_base, width_tail_tip])


#13769, 18968
#%%
from scipy.spatial.distance import mahalanobis
from scipy.stats import chisquare

from scipy.stats import chi2
#lim = chi2.ppf(0.5, 47);

#%%

def getMahalanobisFromMedian(dat, critical_alpha = 0.01):

    '''Calculate the Mahalanobis distance from the sample vector.'''
    good = np.any(~np.isnan(dat), axis=1);

    V = np.cov(dat[good].T)
    VI = np.linalg.inv(V)
    
    vec_med = np.median(dat[good], axis=0)
    
    mahalanobis_dist = np.zeros(dat.shape[0])
    for ii in range(dat.shape[0]):
        mahalanobis_dist[ii] = mahalanobis(dat[ii], vec_med, VI) 
        #np.sqrt(np.dot(np.dot(delW[ii], VI),delW[ii].T))

    #critial distance of the maholanobis distance using the chi-square distirbution
    #https://en.wikiversity.org/wiki/Mahalanobis%27_distance
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
    maha_lim = chi2.ppf(1-critical_alpha, dat.shape[1])
    outliers = mahalanobis_dist>maha_lim
    
    return mahalanobis_dist, outliers, maha_lim


def getMahalanobisRobust(dat, critical_alpha = 0.01):

    '''Calculate the Mahalanobis distance from the sample vector.'''
    good = np.any(~np.isnan(dat), axis=1);
    robust_cov = MinCovDet().fit(dat[good])
    
    #critial distance of the maholanobis distance using the chi-square distirbution
    #https://en.wikiversity.org/wiki/Mahalanobis%27_distance
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
    mahalanobis_dist = np.sqrt(robust_cov.mahalanobis(dat))
    maha_lim = chi2.ppf(1-critical_alpha, dat.shape[1])
    
    #http://imaging.mrc-cbu.cam.ac.uk/statswiki/FAQ/RegressionOutliers
    N = np.sum(good)
    p = dat.shape[1]
    #maha_lim = (2*(p+1)/N - 1/N)*(N-1)
    
    #padj = (0.25 - 0.0018*p)/np.sqrt(N)*p
    #maha_lim = chi2.ppf(1-critical_alpha, padj)
    
    outliers = mahalanobis_dist>maha_lim
    
    
#    clf = EllipticEnvelope().fit(dat[good])
#    maha_lim=-1
#    mahalanobis_dist = clf.decision_function(dat).ravel()
#    outliers = mahalanobis_dist<maha_lim
    
    
    
    return mahalanobis_dist, outliers, maha_lim
#%%
#The log(x+1e-1) transform skew the distribution to the right, so the lower values have a higher change to 
#be outliers.I do this because a with close to zero is typically an oulier, and this strongly penalises these distributions.
def logTransform(x): 
    return np.log(x+1)#+1e-1);

head_widths = cnt_widths[:, 1:7]
tail_widths = cnt_widths[:, -7:-1]

neck_widths = cnt_widths[:, 10:15]
waist_widths = cnt_widths[:, -15:-10]

head_widthsL = logTransform(head_widths)
tail_widthsL = logTransform(tail_widths)

neck_widthsL = logTransform(neck_widths)
waist_widthsL = logTransform(waist_widths)

med = np.nanmedian(width_midbody)
mad = np.nanmedian(np.abs(med-width_midbody))
width_midbody_N = (width_midbody-med)/mad

#skel_length/width_tail_base, skel_length/width_head_base
worm_morph = np.stack((skel_length, cnt_areas, skel_length/width_midbody)).T#, cnt_areas/(width_midbody*skel_length))).T #cnt_areas/width_midbody)).T

worm_morph_N = worm_morph.copy()
for kk in range(worm_morph.shape[1]):
    x = worm_morph[:,kk]
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(med-x))
    worm_morph_N[:,kk]=(x-med)/mad

widths_HT = np.hstack((head_widths, tail_widths))
widths_HT_L = np.hstack((head_widthsL, tail_widthsL))


#widths_HT = np.stack((np.median(cnt_widths[:, 1:5], axis=1), np.median(cnt_widths[:, 5:8], axis=1),
#np.median(cnt_widths[:, -8:-5], axis=1), np.median(cnt_widths[:, -5:-1], axis=1))).T

#widths_HT_L = logTransform(widths_HT)
#outliers = np.zeros(skel_length.shape[0], np.bool)
#for dat in [head_widths, head_widthsL, tail_widths, tail_widthsL, worm_morph]:
#    maha, out_d, lim_d = getMahalanobisFromMedian(dat)
#
#    outliers = outliers | out_d

outliers_rob = np.zeros(skel_length.shape[0], np.bool)
outlier_flag = np.zeros(skel_length.shape[0], np.int64)
for out_ind, dat in enumerate([worm_morph, head_widths, head_widthsL, tail_widths, tail_widthsL]):#, neck_widthsL, waist_widthsL]):
    maha, out_d, lim_d = getMahalanobisRobust(dat)

    outliers_rob = outliers_rob | out_d
    
    outlier_flag += (out_d)*(2**out_ind)
#%%
    plt.figure()
    plt.plot(maha,'.')
    plt.plot(plt.xlim(), (lim_d,lim_d), 'r:')

#%%
#print(np.where(outliers)[0])
print('R', np.where(outliers_rob)[0], np.where(outliers_rob)[0].shape)


#4315
#%%
#plt.figure()
#
#for dd in [skel_length/width_midbody, cnt_areas, cnt_areas/(width_midbody*skel_length)]:
#    dd = dd[~np.isnan(dd)]
#    med = np.nanmedian(dd)
#    mad = np.nanmedian(np.abs(med-dd))
#    dd = (dd-med)/mad
#
#    [count, bins] = np.histogram(dd,100);
#    plt.plot(bins[:-1], count)

#%%
plt.figure()
plt.plot(cnt_widths[19410:19420].T)
#plt.plot(cnt_widths[21060:21070].T)
#%%
#bot, top = 21060, 21070
##bot, top = 19410, 19420
#
w_med = np.nanmedian(cnt_widths, axis=0)


#%%
np.where(np.any((cnt_widths[:,5:-5]/w_med[5:-5])>2, axis=1))
#dd = cnt_widths[bot:top,1:-1]/w_med[1:-1]
#
#plt.figure()
#plt.plot(dd.T)
#%%
#dat = cnt_widths[good, -5:-1]
#datL = np.log(dat+1)
#counts, bins = np.histogram(datL, 100)
#plt.plot(bins[:-1], counts)

#counts, bins = np.histogram(datL, 100)
#plt.plot(bins[:-1], counts)

#%%
##%%
##
##for kk in range(49):
##    plt.figure()
##    plt.hist(dd[~np.isnan(dd[:,kk]), kk], 100)
#
#
##%%
#bad = np.any(np.isnan(X4fito),axis=1);
#X4fit = X4fito[~bad,:]
##%%
#emp_cov = EmpiricalCovariance().fit(X4fit)
#robust_cov = MinCovDet().fit(X4fit)
#
##%%
##mahal_emp_cov = emp_cov.mahalanobis(X4fit)
#emp_mahal = emp_cov.mahalanobis(X4fit - np.mean(X4fit, 0))**(1/3)
#
#robust_mahal = robust_cov.mahalanobis(X4fit - robust_cov.location_) ** (0.33)
##%%
#clf = EllipticEnvelope().fit(X4fit)
##%%
#Z = clf.decision_function(X4fito)
#Z1 = clf.decision_function(X4fito, raw_values=True)
#
#plt.figure()
#plt.plot(Z)
##plt.plot(Z1)
#
##%%
#clf = EllipticEnvelope(contamination=1e-6).fit(X4fit)
#
#Z = clf.decision_function(X4fito)
#
#plt.plot(Z)