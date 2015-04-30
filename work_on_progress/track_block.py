# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 00:21:11 2015

@author: ajaver
"""
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import h5py

from image_difference import image_difference

trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/prueba/CaptureTest_90pc_Ch1_02022015_141431_trajectories.hdf5';
segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/prueba/CaptureTest_90pc_Ch1_02022015_141431_segworm.hdf5';
contrastmap_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_cmap.hdf5';

MAX_DELT = 5;

table_fid = pd.HDFStore(trajectories_file, 'r');
df = table_fid['/plate_worms'];
df =  df[df['worm_index_joined'] > 0]
df = df[df['segworm_id']>=0];
table_fid.close()

track_counts = df['worm_index_joined'].value_counts()

ind = track_counts.index[2];

worm = df[(df['worm_index_joined']==ind)]
delTs = np.diff(worm['frame_number']);
block_ind = np.zeros(len(worm), dtype = np.int);

block_ind[0] = 1;
for ii, delT in enumerate(delTs):
    if delT <= MAX_DELT:
        block_ind[ii+1] = block_ind[ii];
    else:
        block_ind[ii+1] = block_ind[ii]+1;

worm_segworm_id = worm['segworm_id'].values

contrastmap_fid = h5py.File(contrastmap_file, 'r');
#
#%%

#ii = 2;
#
#prev_block = worm_segworm_id[block_ind==ii-1]
#next_block = worm_segworm_id[block_ind==ii]
#
#
#totp = prev_block.size//10;
#totn = next_block.size//10;
#diff_avg_HH = np.zeros((totp, totn));
#diff_avg_HT = np.zeros((totp, totn));
#
#jumpD = 10;
#for kn in range(totn):
#    print kn, totn
#    for kp in range(totp):
#        diff_avg_HH[kp, kn] = image_difference(contrastmap_fid['/worm_H_neg'][prev_block[kp*jumpD],:,:], 
#                contrastmap_fid['/worm_H_neg'][next_block[kn*jumpD],:,:])
#        diff_avg_HT[kp, kn] = image_difference(contrastmap_fid['/worm_H_neg'][prev_block[kp*jumpD],:,:], 
#                contrastmap_fid['/worm_T_neg'][next_block[kn*jumpD],:,:])
#                
#plt.figure()
#plt.plot(np.min(diff_avg_HT, axis = 0)), plt.plot(np.min(diff_avg_HH, axis = 0))
#%%


#%%
jumpD = 5
delD = 50;   
for ii in range(2, block_ind[-1]+1):
    prev_block = worm_segworm_id[block_ind==ii-1]
    if prev_block.size > delD:
        prev_block = prev_block[-delD*jumpD::jumpD]
    else:
        prev_block = prev_block[-delD:]
    
    
    next_block = worm_segworm_id[block_ind==ii]
    if next_block.size > delD:
        next_block = next_block[:delD*jumpD:jumpD]
    else:
        next_block = next_block[:delD]
    
    
    
    delD_prev = np.min((delD, prev_block.size));
    delD_next = np.min((delD, next_block.size));
    
    diff_avg_HH = np.zeros((delD_prev, delD_next));
    diff_avg_HT = np.zeros((delD_prev, delD_next));
    diff_avg_DD = np.zeros((delD_prev, delD_next));
    diff_avg_DV = np.zeros((delD_prev, delD_next));
    
    for kn in range(diff_avg_HH.shape[1]):
        print kn, delD
        for kp in range(diff_avg_HH.shape[0]):
  
            diff_avg_HH[kp, kn] = image_difference(contrastmap_fid['/worm_H_neg'][prev_block[kp],:,:], 
                    contrastmap_fid['/worm_H_neg'][next_block[kn],:,:])
            diff_avg_HT[kp, kn] = image_difference(contrastmap_fid['/worm_H_neg'][prev_block[kp],:,:], 
                    contrastmap_fid['/worm_T_neg'][next_block[kn],:,:])
            
            diff_avg_DD[kp, kn] = image_difference(contrastmap_fid['/worm_D_neg'][prev_block[kp],:,:], 
                    contrastmap_fid['/worm_D_neg'][next_block[kn],:,:])
            diff_avg_DV[kp, kn] = image_difference(contrastmap_fid['/worm_D_neg'][prev_block[kp],:,:], 
                    contrastmap_fid['/worm_V_neg'][next_block[kn],:,:])
#            diff_avg_HH[kp, kn] = image_difference(contrastmap_fid['/worm_H_pos'][prev_block[kp],:,:], 
#                    contrastmap_fid['/worm_H_pos'][next_block[kn],:,:])
#            diff_avg_HT[kp, kn] = image_difference(contrastmap_fid['/worm_H_pos'][prev_block[kp],:,:], 
#                    contrastmap_fid['/worm_T_pos'][next_block[kn],:,:])
#            
#            diff_avg_DD[kp, kn] = image_difference(contrastmap_fid['/worm_D_pos'][prev_block[kp],:,:], 
#                    contrastmap_fid['/worm_D_pos'][next_block[kn],:,:])
#            diff_avg_DV[kp, kn] = image_difference(contrastmap_fid['/worm_D_pos'][prev_block[kp],:,:], 
#                    contrastmap_fid['/worm_V_pos'][next_block[kn],:,:])
#    
#  
    plt.figure()
    plt.plot(np.min(diff_avg_HT, axis = 0)), plt.plot(np.min(diff_avg_HH, axis = 0))    

    plt.figure()
    plt.plot(np.min(diff_avg_DV, axis = 0), 'r'), plt.plot(np.min(diff_avg_DD, axis = 0), 'b')    
        

#image_difference(


#%%
#tot = len(segworm_id)//jumpF;
#min_diff_dat = {};
#for ff in ['HH','TT','TH','HT',]:
#    for fp in ['pos', 'neg']:
#        min_diff_dat[ff + '_' + fp] = np.zeros(tot);
#
##%%
#for kk in range(jumpF, len(segworm_id), jumpF):
#    next_dat = {};
#    next_dat['pos'] = { 'H': contrastmap_fid['/worm_H_pos'][segworm_id[kk]],
#     'T':contrastmap_fid['/worm_T_pos'][segworm_id[kk]]};
#    next_dat['neg'] = { 'H': contrastmap_fid['/worm_H_neg'][segworm_id[kk]],
#     'T':contrastmap_fid['/worm_T_neg'][segworm_id[kk]]};
#
#    for f1 in ['H', 'T']:
#        for f2 in ['H', 'T']:
#            for fp in ['pos', 'neg']:
#                key = f1 + f2 + '_' + fp
#                diff_dat[key][(kk//jumpF)-1] = np.sum(np.abs(prev_dat[fp][f1]-next_dat[fp][f2]));
    
#%%
#plt.figure()
#plt.plot(diff_dat['HH_pos'])
#plt.plot(diff_dat['HT_pos'])

#plt.figure()
#plt.plot(diff_dat['TT_pos'])
#plt.plot(diff_dat['TH_pos'])



#for kk in range(860,870):
#    f, ax = plt.subplots(4,1)
#    
#    ax[0].imshow(contrastmap_fid['/worm_H_pos'][segworm_id[kk]])
#    ax[1].imshow(contrastmap_fid['/worm_T_pos'][segworm_id[kk]])
#    ax[2].imshow(contrastmap_fid['/worm_H_neg'][segworm_id[kk]])
#    ax[3].imshow(contrastmap_fid['/worm_T_neg'][segworm_id[kk]])
