# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:21:02 2015

@author: ajaver
"""

import pandas as pd
import tables
import h5py
import numpy as np
from min_avg_difference import min_avg_difference, avg_difference_mat, min_avg_difference2
#from getWormAngles import calWormAngles
#from image_difference import image_difference
import os
import time

import matplotlib.pylab as plt

def calWormAngles(x,y):
    assert(len(x.shape)==1)
    assert(len(y.shape)==1)
    
    dx = np.diff(x);
    dy = np.diff(y);
    angles = np.arctan2(dx,dy)
    dAngles = np.diff(angles)
    
    #    % need to deal with cases where angle changes discontinuously from -pi
    #    % to pi and pi to -pi.  In these cases, subtract 2pi and add 2pi
    #    % respectively to all remaining points.  This effectively extends the
    #    % range outside the -pi to pi range.  Everything is re-centred later
    #    % when we subtract off the mean.
    #    
    #    % find discontinuities larger than pi (skeleton cannot change direction
    #    % more than pi from one segment to the next)
    positiveJumps = np.where(dAngles > np.pi)[0] + 1; #%+1 to cancel shift of diff
    negativeJumps = np.where(dAngles <-np.pi)[0] + 1;
    
    #% subtract 2pi from remainging data after positive jumps
    for jump in positiveJumps:
        angles[jump:] = angles[jump:] - 2*np.pi;
    
    #% add 2pi to remaining data after negative jumps
    for jump in negativeJumps:
        angles[jump:] = angles[jump:] + 2*np.pi;
    
    #% rotate skeleton angles so that mean orientation is zero
    meanAngle = np.mean(angles);
    angles = angles - meanAngle;
    
    return (angles, meanAngle)

def calWormAnglesAll(skeleton):
    angles_all = np.zeros((skeleton.shape[0], skeleton.shape[2]-1))
    meanAngles_all = np.zeros(skeleton.shape[0])
    for ss in range(skeleton.shape[0]):
        angles_all[ss,:],meanAngles_all[ss] = calWormAngles(skeleton[ss,0,:],skeleton[ss,1,:])
    return angles_all, meanAngles_all
        

#%%
#contrastmap_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_cmap-short.hdf5';
contrastmap_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_cmap.hdf5';
segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_segworm.hdf5';

#contrastmap_fid = h5py.File(contrastmap_file, 'r');

segworm_file_fix = segworm_file[:-5] + '_fix2' + segworm_file[-5:];
os.system('cp "%s" "%s"' % (segworm_file, segworm_file_fix))

cmap_fid = pd.HDFStore(contrastmap_file, 'r');
block_index = cmap_fid['/block_index'];
cmap_fid.close()

contrastmap_fid = tables.File(contrastmap_file, 'r');
segworm_fid = h5py.File(segworm_file_fix, 'r+');


#%%
worm_id= 3;

worm_block = block_index[block_index['worm_index_joined']==worm_id]

block_count = worm_block['block_id'].value_counts();
block_order = block_count.index;

skeleton_std = np.zeros((block_order.max(), segworm_fid['/segworm_results/skeleton'].shape[2]-1))

for ii, bb in enumerate(block_order):
    good = worm_block['block_id'] == bb;
    segworm_id = worm_block.loc[good, 'segworm_id'].values
    if len(segworm_id) > 1:
        skeleton = segworm_fid['/segworm_results/skeleton'][segworm_id,:]
        skeleton_angles,_ = calWormAnglesAll(skeleton)
        skeleton_std[ii,:] = np.std(skeleton_angles, axis=0)

        
#%%
block_range = [worm_block['cmap_id'].min(), worm_block['cmap_id'].max()]
for map_type in ['pos' , 'neg']:
    for N in [('D', 'V')]:
        key_map1 = '/block_cmap/worm_%s_%s' % (N[0], map_type);
        key_map2 = '/block_cmap/worm_%s_%s' % (N[1], map_type);
        
        buff1 = contrastmap_fid.get_node(key_map1)[block_range[0]:block_range[1]+1,:,:].astype(np.int)
        buff2 = contrastmap_fid.get_node(key_map2)[block_range[0]:block_range[1]+1,:,:].astype(np.int)
    
    
        plt.plot(np.sum(np.abs(buff1-buff2), axis=(1,2)))
#%%
            
        
#%%
valid = np.where(block_count>2)[0];
ind_prev = block_order[valid[0]];
for bb_check in range(1,len(valid)):
    ind_curr = block_order[valid[bb_check]];
    
    good_prev = worm_block['block_id'] == ind_prev;
    cmaps_ids = worm_block[good_prev]['cmap_id'].values
    prev_range = (cmaps_ids[0], cmaps_ids[-1])
    
    segworm_id = worm_block.loc[good_prev, 'segworm_id'].values
    skeleton_prev = segworm_fid['/segworm_results/skeleton'][segworm_id,:]
    angle_prev,_ = calWormAnglesAll(skeleton_prev)
    
    good_curr = worm_block['block_id'] == ind_curr;
    cmaps_ids = worm_block[good_curr]['cmap_id'].values
    curr_range = (cmaps_ids[0], cmaps_ids[-1])
    
    segworm_id = worm_block.loc[good_curr, 'segworm_id'].values
    skeleton_curr = segworm_fid['/segworm_results/skeleton'][segworm_id,:]
    angle_curr,_ = calWormAnglesAll(skeleton_curr)
    
    
    
    all_min_diff = {}
    #all_min_diff_index = {}
    for map_type in ['pos' , 'neg']:
    #    key_map = '/block_cmap/worm_all_%s' % map_type;
    #    buff_prev = contrastmap_fid.get_node(key_map)[prev_range[0]:prev_range[1],:,:]
    #    buff_curr = contrastmap_fid.get_node(key_map)[curr_range[0]:curr_range[1],:,:]      
    #    min_avg, index2check_prev = min_avg_difference(buff_prev, buff_curr)
    #    all_min_diff[map_type] = min_avg
    #    
        for N in [('H', 'T')]:#, ('D', 'V')]:
            for ii_curr in range(2):
                key_map_curr = '/block_cmap/worm_%s_%s' % (N[ii_curr], map_type);
                buff_curr = contrastmap_fid.get_node(key_map_curr)[curr_range[0]:curr_range[1],:,:]#.astype(np.int)    
                for ii_prev in range(2):
                    key_map_prev = '/block_cmap/worm_%s_%s' % (N[ii_prev], map_type);
                    buff_prev = contrastmap_fid.get_node(key_map_prev)[prev_range[0]:prev_range[1],:,:]#.astype(np.int)
    #                buff_prev = buff_prev[index2check_prev,:,:]
    #                key_diff = N[ii_prev] + N[ii_curr] + '_' + map_type
    #                all_min_diff[key_diff] = np.sum(np.abs(buff_prev-buff_curr), axis=(1,2))
    #                #print tot, buff_prev
                    key_min = N[ii_prev] + N[ii_curr] + '_' + map_type
                    all_min_diff[key_min] =  min_avg_difference(buff_prev, buff_curr)[0];       
    
    
    plt.figure()
            
    plt.subplot(2,2,1)
    plt.plot(all_min_diff[N[0]+N[0]+'_pos'])
    plt.plot(all_min_diff[N[0]+N[1]+'_pos'])
    plt.title(N[0] + '_pos')
    
    plt.subplot(2,2,2)
    plt.plot(all_min_diff[N[1]+N[1]+'_pos'])
    plt.plot(all_min_diff[N[1]+N[0]+'_pos'])
    plt.title(N[1] + '_pos')
    
    plt.subplot(2,2,3)
    plt.plot(all_min_diff[N[0]+N[0]+'_neg'])
    plt.plot(all_min_diff[N[0]+N[1]+'_neg'])
    plt.title(N[0] + '_neg')

    plt.subplot(2,2,4)
    plt.plot(all_min_diff[N[1]+N[1]+'_neg'])
    plt.plot(all_min_diff[N[1]+N[0]+'_neg'])
    plt.title(N[1] + '_neg')
#%%
#plt.figure()
#for ii,key_map in enumerate(['pos', 'neg']):
#    for ni in range(0,2):
#        factor = all_min_diff[key_map].astype(np.float)
#        max_factor = np.max(factor);
#        min_factor = np.min(factor);
#        factor = 1-(factor-min_factor)/(max_factor-min_factor+1)
#        hh_norm = all_min_diff[N[ni]+N[ni]+'_' + key_map]
#        ht_norm = all_min_diff[N[ni]+N[not ni]+'_' + key_map]
#        
#        plt.subplot(2,2, 2*ii + ni + 1)
#        plt.plot((ht_norm-hh_norm)*factor)
#        plt.title(N[ni]+N[ni]+'_' + key_map)
#
#        print np.mean((ht_norm-hh_norm)*factor)
#%%
for ii_n in range(0,0,100):
    h_curr = contrastmap_fid.get_node('/block_cmap/worm_H_pos')[curr_range[0]+ii_n,:,:]
    t_curr = contrastmap_fid.get_node('/block_cmap/worm_T_pos')[curr_range[0]+ii_n,:,:]
    
    h_prev = contrastmap_fid.get_node('/block_cmap/worm_H_pos')[index2check_prev[ii_n],:,:]
    t_prev = contrastmap_fid.get_node('/block_cmap/worm_T_pos')[index2check_prev[ii_n],:,:]
    
    
    
    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(h_curr, interpolation = 'none')
    plt.title('head current')
    
    plt.subplot(2,3,2)
    plt.imshow(t_curr, interpolation = 'none')
    plt.title('tail current')
    
    plt.subplot(2,3,4)
    plt.imshow(h_prev, interpolation = 'none')
    plt.title('head previous')
    
    plt.subplot(2,3,5)
    plt.imshow(t_prev, interpolation = 'none')
    plt.title('tail previous')
    
    w_curr = contrastmap_fid.get_node('/block_cmap/worm_all_pos')[curr_range[0] + ii_n,:,:]
    w_prev = contrastmap_fid.get_node('/block_cmap/worm_all_pos')[prev_range[0] + index2check_prev[ii_n],:,:]
    plt.subplot(2,3,3)
    plt.imshow(w_curr, interpolation = 'none')
    plt.subplot(2,3,6)
    plt.imshow(w_prev, interpolation = 'none')
    print np.sum(np.abs(w_curr.astype(np.int)-w_prev.astype(np.int)))

#%%
#for N in [('H', 'T')]:#, ('D', 'V')]:
#    all_min_diff = {}
#    #all_min_diff_index = {}
#    for map_type in ['neg', 'pos']:
#        for ii_curr in range(2):
#            for ii_prev in range(2):
#                key_map_curr = '/block_cmap/worm_%s_%s' % (N[ii_curr], map_type)
#                key_map_prev = '/block_cmap/worm_%s_%s' % (N[ii_prev], map_type);
#                key_map_prev_switch = '/block_cmap/worm_%s_%s' % (N[not ii_prev], map_type);
#                
#                buff_curr = contrastmap_fid.get_node(key_map_curr)[curr_range[0]:curr_range[1],:,:]      
#                buff_prev = contrastmap_fid.get_node(key_map_prev)[prev_range[0]:prev_range[1],:,:]
#                #print tot, buff_prev
#                key_min = N[ii_prev] + N[ii_curr] + '_' + map_type
#                all_min_diff[key_min] =  avg_difference_mat(buff_prev, buff_curr);       
#%%                        
       
#prob_switch = {}
#for ind in [(N[0]+N[0], N[0]+N[1]), (N[1]+N[1], N[1]+N[0])]:
#    for map_key in ['_neg', '_pos']:
#        min_eq = all_min_diff[ind[0] + map_key]
#        min_dif = all_min_diff[ind[1] + map_key]
#        prob_switch[ind[0] + map_key] = np.mean((min_dif-min_eq)/min_eq) #this will be negative only if the change is prefered
            
#            print np.sum([prob_switch[key] for key in prob_switch])
#            direction = -1 if np.sum([prob_switch[key] for key in prob_switch])<0 else 1;
#            switch_direction[N[0]][block_id-1] = switch_direction[N[0]][block_id-2]*(direction); #cummulative switch
#%%
#for ax_n in [0,1]:
#    plt.figure()
#    
#    plt.subplot(2,2,1)
#    plt.plot(np.min(all_min_diff[N[0]+N[0]+'_pos'], axis=ax_n))
#    plt.plot(np.min(all_min_diff[N[0]+N[1]+'_pos'], axis=ax_n))
#    plt.title(N[0] + '_pos')
#    
#    plt.subplot(2,2,2)
#    plt.plot(np.min(all_min_diff[N[1]+N[1]+'_pos'], axis=ax_n))
#    plt.plot(np.min(all_min_diff[N[1]+N[0]+'_pos'], axis=ax_n))
#    plt.title(N[1] + '_pos')
#    
#    plt.subplot(2,2,3)
#    plt.plot(np.min(all_min_diff[N[0]+N[0]+'_neg'], axis=ax_n))
#    plt.plot(np.min(all_min_diff[N[0]+N[1]+'_neg'], axis=ax_n))
#    plt.title(N[0] + '_neg')
#    
#    plt.subplot(2,2,4)
#    plt.plot(np.min(all_min_diff[N[1]+N[1]+'_neg'], axis=ax_n))
#    plt.plot(np.min(all_min_diff[N[1]+N[0]+'_neg'], axis=ax_n))
#    plt.title(N[1] + '_neg')
#%%            
#    block2switch = {}
#    for key in switch_direction:
#        block2switch[key] = np.where(switch_direction[key]==-1)[0]+2;
#        #dum = np.cumprod(np.where(is_switch[key]<0, -1, 1));
#        #dum = np.where(is_switch[key]<0, -1, 1)
#        #block2switch[key] = np.where(dum==-1)[0]+2;
###%%    
##    
##    
#    for bb in block2switch['H']:
#        kernel = '(worm_index_joined==%d) & (block_id==%d)' % (worm_id, bb)
#        block_segworm_id = block_index.query(kernel)['segworm_id']
#        for cc in ['skeleton', 'contour_dorsal', 'contour_ventral']:
#            print len(block_segworm_id)
#            for nn in block_segworm_id:
#                aa = segworm_fid['/segworm_results/' + cc][nn,:,:]
#                segworm_fid['/segworm_results/' + cc][nn,:,:] = aa[:,::-1]
#    
#    for bb in block2switch['D']:
#        kernel = '(worm_index_joined==%d) & (block_id==%d)' % (worm_id, bb)
#        block_segworm_id = block_index.query(kernel)['segworm_id']
#        for nn in block_segworm_id:
#            vv = segworm_fid['/segworm_results/contour_ventral'][nn,:,:]
#            dd = segworm_fid['/segworm_results/contour_dorsal'][nn,:,:]
#            
#            segworm_fid['/segworm_results/contour_ventral'][nn,:,:] = dd
#            segworm_fid['/segworm_results/contour_dorsal'][nn,:,:] = vv
#    
#    print ii_worm, len(worm_ids), time.time()-tic, time.time() - tic_ini
#%%
#segworm_fid.close()
#contrastmap_fid.close()
