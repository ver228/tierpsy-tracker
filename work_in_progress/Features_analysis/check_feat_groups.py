# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 00:35:34 2016

@author: ajaver
"""

import sys
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking/')
sys.path.append('/Users/ajaver/Documents/GitHub/movement_validation/')

from open_worm_analysis_toolbox.features.worm_features import get_feature_processing_specs, WormFeatures, WormFeaturesDos
from MWTracker.featuresAnalysis.obtainFeaturesHelper import wormStatsClass, WormFromTable
from open_worm_analysis_toolbox.statistics import specifications


def getFieldData(worm_features, name):
    data = worm_features
    for field in name.split('.'):
        data = getattr(data, field)
    return data
    

f_specs = get_feature_processing_specs()

#for spec in f_specs:
#    print(spec.class_name, spec.name)
    

skeletons_file = '/Users/ajaver/Desktop/Videos/Check_Align_samples/Results/npr-2 (ok419)IV on food R_2010_01_25__15_29_03___4___10_skeletons.hdf5'
worm = WormFromTable()
worm.fromFile(skeletons_file, 1, fps = 25, isOpenWorm = True)

worm_features = WormFeaturesDos(worm)

#%%
for spec in f_specs:
    for ff in worm_features.features:
        aa = worm_features.features[ff].value
        print(ff, type(aa))
        #print(worm_features.features[ff], type)
#%%
import csv
with open('Features Specifications.csv') as fid:
    reader = csv.reader(fid)
    data = [line for line in reader]
    
    header = data[0]
    data.pop(0)
    
    data = {key:col for key, col in zip(header, zip(*data))}

with open('features_names.csv', 'w') as fid:
    motion_mode = worm_features._temp_features['locomotion.motion_mode'].value
    
    csvwriter = csv.writer(fid)
    csvwriter.writerow(('feat_name_table', 'feat_name_obj', 'is_time_series', 'is_signed', 'units'))
    
    for ii in range(len(data['Feature Name'])):
        feat_name = data['name'][ii]
        feat_struct = data['Feature Name'][ii]
        is_signed = data['is_signed'][ii]
        is_time_series = data['is_time_series'][ii]
        
        units = data['units'][ii]
        
        tmp = worm_features.features[feat_struct].value
        N = 0 if tmp is None else 1 if isinstance(tmp, (float, int)) else tmp.size
        
        feat_type = 'motion' if N == motion_mode.size else 'event'
        
        feat_name = feat_name.split(' (')[0].replace(' ', '_').replace('.', '').replace('-', '_')
        if '/' in feat_name: feat_name = feat_name.replace('/', '_') + '_ratio'
        feat_name = feat_name.lower()    
        
        csvwriter.writerow((feat_name, feat_struct, is_time_series, is_signed, units))
#%%
import pandas as pd
df = pd.read_csv('features_names.csv', index_col=0)


for ii, row in df.iterrows():
    print(ii, row['units'])
#%%
from MWTracker.featuresAnalysis.obtainFeaturesHelper import wormStatsClass
wStats = wormStatsClass()
wStats.getWormStats(worm_features)
#%%
from MWTracker.featuresAnalysis.obtainFeaturesHelper
getWormFeatures(skeletons_file, 'test.hdf5')


#'locomotion.motion_mode'

#for key in worm_features._temp_features:
#    tmp = worm_features._temp_features[key].value 
#    N = 0 if tmp is None else 1 if isinstance(tmp, (float, int)) else tmp.shape   
#    print(name, N)
