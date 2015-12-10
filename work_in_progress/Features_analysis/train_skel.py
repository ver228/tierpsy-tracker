# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:46:03 2015

@author: ajaver
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def get_traj_file(traj_file, skel_file):
    #%%
    with pd.HDFStore(traj_file, 'r') as fid:
        plate_worms = fid['/plate_worms']

    #trajectories data is already filtered and contains the flag is an skeleton was calculated succesfully or not
    with pd.HDFStore(skel_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    
    #I am getting negative indexes. I do not know where they come from, I need to check this...
    trajectories_data = trajectories_data[trajectories_data['plate_worm_id']>=0]    
    
    has_skeleton = trajectories_data['has_skeleton']
    has_skeleton.index = trajectories_data['plate_worm_id'].values
    
    del trajectories_data
    
    plate_worms['has_skeleton'] = 0
    plate_worms['has_skeleton'] = has_skeleton
    del has_skeleton
    
    
    plate_worms.dropna(how = 'any', inplace=True)
    return plate_worms
traj_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch1_18112015_005619_trajectories.hdf5'
skel_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch1_18112015_005619_skeletons.hdf5'

plate_worms = get_traj_file(traj_file, skel_file)


#NOTES: 
# 'area', 'perimeter', 'box_length', 'box_width' are in pixel, a better generalization will be to use microns
# other parameters that could be used are: 'intensity_mean', 'intensity_std', 'threshold'. 
# But they will depend on the imaging conditions
col_pred = ['area', 'perimeter', 'box_length', 'box_width', 'quirkiness',
       'compactness', 'solidity', 'hu0', 'hu1',
       'hu2', 'hu3', 'hu4', 'hu5', 'hu6']

X = plate_worms[col_pred]
y = plate_worms['has_skeleton']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

max_feat = int(round(np.sqrt(X_train.shape[1])))
clf = RandomForestClassifier(n_estimators = 20, max_features = max_feat, n_jobs=8)
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

conf_mat = classification_report(y_pred, y_test)
print(conf_mat)

print([x for y, x in sorted(zip(clf.feature_importances_, col_pred))])


traj_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch1_18112015_075624_trajectories.hdf5'
skel_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch1_18112015_075624_skeletons.hdf5'


plate_worms = get_traj_file(traj_file, skel_file)
X_true = plate_worms[col_pred]
y_true = plate_worms['has_skeleton']
y_pred = clf.predict(X_true)

dd =  classification_report(y_pred, y_true)
print(dd)
