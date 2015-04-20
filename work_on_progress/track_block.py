# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 00:21:11 2015

@author: ajaver
"""
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_trajectories.hdf5';
segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_segworm.hdf5';
contrastmap_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_cmap.hdf5';

MAX_DELT = 5;

table_fid = pd.HDFStore(trajectories_file, 'r');
df = table_fid['/plate_worms'];
df =  df[df['worm_index_joined'] > 0]
df = df[df['segworm_id']>=0];
table_fid.close()

track_counts = df['worm_index_joined'].value_counts()

ind = track_counts.index[0];

worm = df[(df['worm_index_joined']==ind)]
delTs = np.diff(worm['frame_number']);
block_ind = np.zeros(len(worm));

block_ind[0] = 1;
for ii, delT in enumerate(delTs):
    if delT <= MAX_DELT:
        block_ind[ii+1] = block_ind[ii];
    else:
        block_ind[ii+1] = block_ind[ii]+1;

segworm_id = worm['segworm_id'].values

#plt.figure()
#plt.plot(block_ind)

import h5py
contrastmap_fid = h5py.File(contrastmap_file, 'r');

jumpF = 10;
prev_dat = {};
prev_dat['pos'] = { 'H': contrastmap_fid['/worm_H_pos'][segworm_id[0]],
'T':contrastmap_fid['/worm_T_pos'][segworm_id[0]]};
prev_dat['neg'] = { 'H': contrastmap_fid['/worm_H_neg'][segworm_id[0]],
'T':contrastmap_fid['/worm_T_neg'][segworm_id[0]]};

tot = len(segworm_id)//jumpF;
diff_dat = {};
for ff in ['HH','TT','TH','HT',]:
    for fp in ['pos', 'neg']:
        diff_dat[ff + '_' + fp] = np.zeros(tot);

for kk in range(jumpF, len(segworm_id), jumpF):
    next_dat = {};
    next_dat['pos'] = { 'H': contrastmap_fid['/worm_H_pos'][segworm_id[kk]],
     'T':contrastmap_fid['/worm_T_pos'][segworm_id[kk]]};
    next_dat['neg'] = { 'H': contrastmap_fid['/worm_H_neg'][segworm_id[kk]],
     'T':contrastmap_fid['/worm_T_neg'][segworm_id[kk]]};

    for f1 in ['H', 'T']:
        for f2 in ['H', 'T']:
            for fp in ['pos', 'neg']:
                key = f1 + f2 + '_' + fp
                diff_dat[key][(kk//jumpF)-1] = np.sum(np.abs(prev_dat[fp][f1]-next_dat[fp][f2]));
    
#%%
plt.figure()
plt.plot(diff_dat['HH_pos'])
plt.plot(diff_dat['HT_pos'])

plt.figure()
plt.plot(diff_dat['TT_pos'])
plt.plot(diff_dat['TH_pos'])



#for kk in range(860,870):
#    f, ax = plt.subplots(4,1)
#    
#    ax[0].imshow(contrastmap_fid['/worm_H_pos'][segworm_id[kk]])
#    ax[1].imshow(contrastmap_fid['/worm_T_pos'][segworm_id[kk]])
#    ax[2].imshow(contrastmap_fid['/worm_H_neg'][segworm_id[kk]])
#    ax[3].imshow(contrastmap_fid['/worm_T_neg'][segworm_id[kk]])
