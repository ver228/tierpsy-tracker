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

feat_dir = '/Volumes/behavgenom$/GeckoVideo/Curro/Results_old/exp2'

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

good_feats = ['length', 'head_width', 'midbody_width', 'tail_width', 'area', 'area_length_ratio',
       'width_length_ratio', 'head_bend_mean', 'neck_bend_mean',
       'midbody_bend_mean', 'hips_bend_mean', 'tail_bend_mean', 'head_bend_sd',
       'neck_bend_sd', 'midbody_bend_sd', 'hips_bend_sd', 'tail_bend_sd',
       'max_amplitude', 'amplitude_ratio', 'primary_wavelength',
       'secondary_wavelength', 'track_length', 'eccentricity', 'bend_count',
       'tail_to_head_orientation', 'head_orientation', 'tail_orientation',
       'eigen_projection_1', 'eigen_projection_2', 'eigen_projection_3',
       'eigen_projection_4', 'eigen_projection_5', 'eigen_projection_6',
       'head_tip_speed', 'head_speed', 'midbody_speed', 'tail_speed',
       'tail_tip_speed', 'head_tip_motion_direction', 'head_motion_direction',
       'midbody_motion_direction', 'tail_motion_direction',
       'tail_tip_motion_direction', 'foraging_amplitude', 'foraging_speed',
       'head_crawling_amplitude', 'midbody_crawling_amplitude',
       'tail_crawling_amplitude', 'head_crawling_frequency',
       'midbody_crawling_frequency', 'tail_crawling_frequency', 'path_range',
       'path_curvature']
#%%
for fname in db_data_pd['file_name']:
    feat_file = os.path.join(feat_dir, fname.replace('.hdf5', '_features.hdf5'))
    if os.path.exists(feat_file):
        with pd.HDFStore(feat_file, 'r') as fid:
            features = fid['/features_timeseries']
            
            feat_avg = features[good_feats].apply(np.nanmean);
            
            good_skels = features[good_feats].isnull();
            good_skels['timestamp'] = features['timestamp']
            worms_per_frame = good_skels.groupby('timestamp').agg('sum')
            #worm_per_frame = features.groupby('timestamp').agg(lambda x:np.sum(np.isnan(x)))
            
            break