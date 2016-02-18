# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 18:18:36 2016

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
file_name = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch5_17112015_205616_features.hdf5'
#file_name = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch1_17112015_205616_features.hdf5'
skel_file_name = file_name.replace('features', 'skeletons')

with pd.HDFStore(file_name, 'r') as fid:
    #print(fid['/features_timeseries'].columns)
    features_table = fid['/features_timeseries']
#%% filter points with valid skeletons
valid_skeletons = ~np.isnan(features_table['length'])
skel_per_frame = features_table[valid_skeletons]['frame_number'].value_counts().sort_index()

traj_size = features_table['worm_index'].value_counts()

#plt.figure()
#skel_per_frame.plot()
#plt.figure()
#traj_size.hist(bins=100)
#%%
largest_traj = traj_size.argmax()

feat_ind = features_table[features_table['worm_index']==largest_traj]
feat_ind['skel_indexes'] = feat_ind.index
feat_ind.index = feat_ind['frame_number'].values

#%%
nw = WormFromTable()
nw.fromFile(skel_file_name, largest_traj)

ind = 20000
#%%
plt.figure()
n_points = nw.skeleton.shape[1]
for ii in [1, 3, 5]:
    ang,_ = calWormAnglesAll(nw.skeleton, segment_size=ii)
    
    #xx = np.linspace(0, 49, 49-ii)
    plt.plot(ang[ind], '.-')
#%%
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from MWTracker.featuresAnalysis.obtainFeaturesHelper import calWormAngles
#%%
xx = nw.skeleton[ind, :, 0]
yy = nw.skeleton[ind, :, 1]

ang, _ = calWormAngles(xx,yy, 1)

plt.figure()
plt.plot(ang)
#%%
for window in [5, 7, 9]:
    pol_degree = 3
    xx_s1 = savgol_filter(xx, window, pol_degree)
    yy_s1 = savgol_filter(yy, window, pol_degree)
    ang_s1, _ = calWormAngles(xx_s1,yy_s1, 1)
    plt.plot(ang_s1)

#%%
skel_smooth = smoothCurvesAll(nw.skeleton.copy(), window = 5)
#%%
plt.figure()
n_points = skel_smooth.shape[1]
for ii in [1, 3, 5]:
    ang,_ = calWormAnglesAll(skel_smooth, segment_size=ii)
    
    #xx = np.linspace(0, 49, 49-ii)
    plt.plot(ang[ind], '.-')
#%%

plt.figure()
plt.plot(nw.skeleton[ind,:, 0], nw.skeleton[ind,:,1], '-o')
plt.plot(skel_smooth[ind,:, 0], skel_smooth[ind,:,1], '-o')


#%%
nw_smooth = WormFromTable()
nw_smooth.fromFile(skel_file_name, largest_traj, smooth_skeletons = True)
#import copy
#nw_smooth = copy.deepcopy(nw)

#%%
nw.changeAxis()
worm_features = WormFeaturesDos(nw)
#
##%%
nw_smooth.changeAxis()
worm_features_smooth = WormFeaturesDos(nw_smooth)
#%%
for feat_name in ['posture.amplitude_max', 'posture.primary_wavelength', 
                  'posture.kinks', 'posture.bends.midbody.mean', 'posture.eigen_projection0']:
    #plt.plot(worm_features.features[feat_name].value, '.')
    #plt.plot(worm_features_smooth.features[feat_name].value, '.')
    
    plt.figure()
    plt.plot(worm_features.features[feat_name].value, worm_features_smooth.features[feat_name].value, '.')
    plt.title(feat_name)    
    plt.xlabel('normal')
    plt.ylabel('smoothed')
#%%
plt.figure()
plt.plot(nw.skeleton[:, 0, ind], nw.skeleton[:, 1, ind], '-o')
plt.plot(skel_smooth[ind,:, 0], skel_smooth[ind,:,1], '-o')
#%%
#
#

##%%
#plt.figure()
#plt.plot(nw.skeleton[:,0, 0], nw.skeleton[:,1,0], '-o')
#plt.plot(nw_smooth.skeleton[:,0, 0], nw_smooth.skeleton[:,1, 0], '-o')
#plt.axis('equal')
##%%
#plt.figure()
#plt.plot(nw.angles[:,0])
#plt.plot(nw_smooth.angles[:,0])
#
##%%
##feat_name = 'posture.eigen_projection1'
#feat_name = 'posture.kinks'
#plt.figure()
#plt.plot(worm_features.features[feat_name].value, worm_features_smooth.features[feat_name].value, '.')
##%%
#


#%%
#curves = nw_smooth.skeleton
#print(curves.shape)
#for ii in range(curves.shape[0]):
#    if not np.any(np.isnan(curves[ii])):
#        print(ii)
#        pass
    #print(ii, np.any(np.isnan(curves[ii])))
#%%
    #
        #print(ii)
        #curves[ii] = smoothCurve(curves[ii], window = window, pol_degree = pol_degree)
    
#plt.figure()
#plt.plot(worm_features.nw.angles[:,0])

#%%
#%%

#%%
#for feat_name in ['length']:#['area', 'area_length_ratio', 'length', 'width_length_ratio']:#['length', 'midbody_width', 'head_width', 'tail_width']:
#    plt.figure()
#    
#    feat_ind[feat_name].plot(style='.', c =(0.75,0.75,0.75))
#    pd.rolling_median(feat_ind[feat_name].dropna(), 10).plot(c = 'k')
#    plt.xlabel('frame number')
#    plt.ylabel(feat_name)
#
##%%
#for feat_name in feat_ind.columns:#['area', 'area_length_ratio', 'length', 'width_length_ratio']:#['length', 'midbody_width', 'head_width', 'tail_width']:
#    if not 'bend' in feat_name: continue 
#    plt.figure()
#    
#    feat_ind[feat_name].plot(style='.', c =(0.75,0.75,0.75))
#    pd.rolling_median(feat_ind[feat_name].dropna(), 10).plot(c = 'k')
#    plt.xlabel('frame number')
#    plt.ylabel(feat_name)
#%%

