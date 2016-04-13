# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:54:06 2016

@author: ajaver
"""
import os
import csv
from datetime import datetime
import numpy as np
import pandas as pd

import tables

feat_dir = '/Volumes/behavgenom$/GeckoVideo/Curro/Results/exp2'

db_data = {'camera_focus':[], 'file_name':[], 'channel':[], 'exp_type':[], 'fogginess':[], 'position':[]}
with open('exp1_db.csv') as fid:
    csvreader = csv.DictReader(fid)
    for row in csvreader:
        for k in row:
            db_data[k].append(row[k])

db_data['channel'] = list(map(int, db_data['channel']))
db_data['fogginess'] = list(map(int, db_data['fogginess']))
db_data['camera_focus'] = list(map(float, db_data['camera_focus']))

dates = ['_'.join(x.rpartition('.')[0].split('_')[-2:]) for x in db_data['file_name']]
db_data['dates'] = [np.datetime64(datetime.strptime(x, '%d%m%Y_%H%M%S').isoformat()) for x in dates]

##strain_eq = ['daf-2(e1370)', 'daf-16(mu86)', 'N2']
#strains = set(x.partition(' ')[0] for x in  db_data['exp_type'])
#assert len(strains) == 3
#
#db_data['strain'] = [strain for strain in strains for exp_type in  db_data['exp_type'] if exp_type.startswith(strain)]
#assert len(db_data['exp_type']) == len(db_data['strain'])
#
#
#db_data['cisplatin'] = [c for c in con_eq for exp_type in  db_data['exp_type'] if c in exp_type]
#assert len(db_data['exp_type']) == len(db_data['cisplatin'])

#%%
db_data_np = {x:np.array(db_data[x]) for x in db_data}

dtypes = [(x, db_data_np[x].dtype) for x in db_data_np]
dat_tup = list(zip(*(db_data_np[x[0]] for x in dtypes)))

db_data_pd = pd.DataFrame(np.array(dat_tup, dtype=dtypes))

db_data_pd['strain'] = db_data_pd['exp_type'].apply(lambda x: x.split()[0])

def getCon(dd):
    for c in ['control', '60ug/ml', '100ug/ml']: 
        if c in dd: return c
db_data_pd['cisplatin'] = db_data_pd['exp_type'].apply(getCon)


first_time = db_data_pd.groupby('position').agg({'dates':'min'})
first_time.columns = ['first_date']
db_data_pd = db_data_pd.merge(first_time, left_on='position', right_index=True)
hours = (db_data_pd['dates']-db_data_pd['first_date'])
db_data_pd['hours'] = hours.apply(lambda x : round(x/ np.timedelta64(1, 'h'),1))
#%%
from MWTracker.featuresAnalysis.obtainFeaturesHelper import _featureStat, wormStatsClass

p = wormStatsClass()
p.features_info = p.features_info[p.features_info['is_time_series']==1];
p.builtFeatAvgNames()
good_feats = p.feat_avg_names
for bad in ['worm_index', 'n_frames', 'n_valid_skel']:
    good_feats.remove(bad)

#%%

#feat_std = pd.DataFrame(columns=good_feats, index=db_data_pd.index)
#worms_per_frame = pd.DataFrame(columns=good_feats, index=db_data_pd.index)
feat_avg = pd.DataFrame(columns=good_feats, index=db_data_pd.index)


#%%

val_lim = {'length':(90, 150), 'area':(400, 1200), 'midbody_width':(6, 12)}

all_features_ori = {}
all_features = {}

for row_ind, fname in db_data_pd['file_name'].iteritems():
    feat_file = os.path.join(feat_dir, fname.replace('.hdf5', '_features.hdf5'))
    skel_file = os.path.join(feat_dir, fname.replace('.hdf5', '_skeletons.hdf5'))
    if os.path.exists(feat_file):
        print(row_ind, fname)
        with pd.HDFStore(feat_file, 'r') as fid:
            features_ori = fid['/features_timeseries']
            #eliminate data from dropped frames or no skeletons
            features_ori = features_ori[features_ori['timestamp']!=-1]
            all_features_ori[row_ind] = features_ori
            
            features = features_ori
            for feat in val_lim:
                features = features[features[feat]>val_lim[feat][0]]
                features = features[features[feat]<val_lim[feat][1]]
            
            motion_modes = features['motion_modes'].values
            features = features.drop(['worm_index', 'timestamp', 'motion_modes'], 1)
            
            all_features[row_ind] = features
            
            for feat_name in features.columns:
                data = features[feat_name]
                is_signed = p.features_info.loc['length', 'is_signed'];
                stats = _featureStat(np.nanmean, data, feat_name, is_signed, True, motion_modes)
                for stat in stats:
                    feat_avg.loc[row_ind, stat] = stats[stat]
            
            #feat_med.loc[row_ind, :] = features[good_feats].apply(np.nanmedian);
            
            #feat_avg.loc[row_ind, :] = features[good_feats].apply(np.nanmean);
            #feat_std.loc[row_ind, :] = features[good_feats].apply(np.nanstd);

            #good_skels = ~features[good_feats].isnull();
            #good_skels['timestamp'] = features['timestamp']
            #wpf = good_skels.groupby('timestamp').agg('sum')
            #worms_per_frame.loc[row_ind, :] = wpf[good_feats].apply(np.median)
            
#%%

index_per_group = {}
for ind_tuple, group in db_data_pd.groupby(['strain', 'cisplatin', 'hours']):
    good = group['fogginess']<=1
    index_per_group[ind_tuple] = group[good].index
#%%

#control_ind = ('N2', '100ug/ml', 23.5)
control_ind = ('N2', 'control', 23.5)
group_indexes = index_per_group[control_ind]

import matplotlib.pylab as plt
#plt.figure()

for gind in group_indexes:
    features = all_features[gind]
    print(len(features))
    dat = features['midbody_speed'].dropna()
    
    counts, bins = np.histogram(dat, 100)
    counts = counts/np.sum(counts)/(bins[1]-bins[0])
    plt.plot(bins[:-1], counts, 'b')
    plt.xlim((-120, 100))
    plt.ylim((0, 0.12))
#%%
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as smm

def getPValues(control_avg, curr_avg):

    p_values = pd.Series(index=curr_avg.columns)
    for feat in curr_avg.columns:
        x = control_avg[feat].values.astype(np.float)
        y = curr_avg[feat].values.astype(np.float)
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        #if np.all(np.isnan(x)) or np.all(np.isnan(y)):
        #    continue
        _, p_value = ttest_ind(x,y, equal_var=False)
        #_, p_value = ranksums(x,y)
        
        #p_value positive if N2 is larger than the strain
        p_values[feat] = p_value

    p_values = p_values.dropna()
    #correct for false discovery rate using 2-stage Benjamini-Krieger-Yekutieli
    reject, pvals_corrected, alphacSidak, alphacBonf = \
    smm.multipletests(p_values.values, method = 'fdr_tsbky')
    
    p_values_corr = pd.Series(pvals_corrected, index=p_values.index)
    return p_values, p_values_corr
    
#%%

control_ind = ('N2', 'control', 0);
control_avg = feat_avg.loc[index_per_group[control_ind], :]

curr_ind = ('N2', '100ug/ml', 23.5);
curr_avg = feat_avg.loc[index_per_group[curr_ind], :]

p_values, p_values_corr = getPValues(control_avg, curr_avg)
