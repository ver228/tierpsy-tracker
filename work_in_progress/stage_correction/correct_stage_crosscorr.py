# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:53:42 2016

@author: ajaver
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pylab as plt
from itertools import combinations

import h5py
import numpy as np
from itertools import combinations, chain
from scipy.misc import comb

def comb_index(n, k):
    #http://stackoverflow.com/questions/16003217/n-d-version-of-itertools-combinations-in-numpy
    count = comb(n, k, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), k)), 
                        int, count=count*k)
    return index.reshape(-1, k)


def calculate_best_seq(moves_x_mag, moves_y_mag, worm_ini, stage_dx, stage_dy, stage_ini, 
                       samples_take = 10, delta_search = 10, tolerance = 0.1, th = 31):
    #%%
    #worm_ini, stage_ini = w_cur, s_cur
    #worm_ini, stage_ini = 0, 0
                       
    search_w = samples_take + delta_search

    assert moves_x_mag.size - worm_ini > search_w 
    assert stage_dx.size - stage_ini > samples_take

    #probably i can avoid having to calculate this several times.
    comb_ind = comb_index(search_w, samples_take) 
    
    
    
    stage_fin = stage_ini + samples_take
    comb_ind_n = comb_ind + worm_ini
    
    dx = moves_x_mag[comb_ind_n] - stage_dx[stage_ini:stage_fin]
    dy = moves_y_mag[comb_ind_n] - stage_dy[stage_ini:stage_fin]
    r = np.sqrt(dx*dx + dy*dy)
    
    good_rs = np.sum(r <= th,axis=1)
    
    ind2check = np.where(good_rs == np.max(good_rs))[0]
    ii = ind2check[0]
    #i don't want the minimum, but the first index within a given tolerance of the minimum.
    r_mean = np.mean(r[ind2check],axis=1);
    r_min = np.min(r_mean)
    ii = np.where(np.abs(r_mean-r_min) < tolerance)[0][0]
    
    ii = ind2check[ii]
    valid_ind = comb_ind_n[ii]
    best_r = r[ii]
    #%%
    return valid_ind, best_r
#%%

from MWTracker.intensityAnalysis.correctHeadTailIntensity import createBlocks
#%%
import glob
import os

main_dir = '/Users/ajaver/Desktop/Videos/single_worm/agar_2/Results/'

file_list = glob.glob(os.path.join(main_dir, '*_stagemov.mat' ))


for stage_file in [file_list[0]]:#file_list:
    #%%
    skeletons_file = stage_file.replace('_stagemov.mat', '_skeletons.hdf5')
    intensity_file =  stage_file.replace('_stagemov.mat', '_intensities.hdf5')
    feat_file = stage_file.replace('_stagemov.mat', '_features.mat').replace('/Results/', '/Features/')
#    
#    #%%
#    with h5py.File(intensity_file, 'r') as fid:
#        int_map = fid['/straighten_worm_intensity'][:].astype(np.float)
#        int_map_med = fid['/straighten_worm_intensity_median'][:].astype(np.float)
#        
#    #%%
#    from scipy.ndimage.filters import median_filter
#    #med = np.median(int_map, axis=(1,2))    
#    #int_dv = np.median(np.abs(med-int_map), axis=(1,2))
#    #int_dv = np.std(int_map_med, axis=1)
#    int_dv = np.std(int_map, axis=(1,2))
#    
#    int_dv_s = median_filter(int_dv, 20);
#    plt.figure()    
#    #plt.plot(int_dv)
#    plt.plot(int_dv-int_dv_s)
    
    #%%
    mvars = loadmat(stage_file)
    
    xShift = np.squeeze(mvars['xShift']);
    yShift = np.squeeze(mvars['yShift']);
    stage_x = np.squeeze(mvars['stage_x']);
    stage_y = np.squeeze(mvars['stage_y']);
    stage_time = np.squeeze(mvars['stage_time']);
    video_timestamp_time = np.squeeze(mvars['video_timestamp_time']);
    video_timestamp_ind = np.squeeze(mvars['video_timestamp_ind']);
    
    if stage_x.size <= 1:
        continue
    #%%
    stage_dx = np.diff(stage_x)
    stage_dy = np.diff(stage_y)
    
    
    moves_lims = createBlocks(np.abs(yShift) + np.abs(xShift) >0.1 , min_block_size=0)
    moves_y_mag = np.array([np.sum(yShift[ini:fin+1]) for ini,fin in moves_lims])
    
    #moves_x_lims = createBlocks(np.abs(xShift)>0.1, min_block_size=0)
    moves_x_mag = np.array([np.sum(xShift[ini:fin+1]) for ini,fin in moves_lims])
    
    #%%
    samples_take = 5
    delta_search = 5
    
    #best_start, start_r = calculate_best_seq(moves_x_mag, moves_y_mag, 0, \
    #            stage_dx, stage_dy, 0, samples_take, delta_search)
    
    #w_cur = best_start[-1] + 1
    #s_cur = len(best_start)
    #r = list(start_r)
    #valid_ind_l = list(best_start)
    
    w_cur, s_cur = 0, 0
    r, valid_ind_l = [], []
    
    #th = 25
    #w = 10
    
    dum = 0
    while w_cur < len(moves_y_mag):
        #if len(r) >= 10:        
        med = np.median(r)
        mad = np.median(np.abs(med-r))
        th = 31#med + mad*6
            
        dx_n = moves_x_mag[w_cur] - stage_dx[s_cur]
        dy_n = moves_y_mag[w_cur] - stage_dy[s_cur]
        r_n = np.sqrt(dx_n*dx_n + dy_n*dy_n)
        
        if r_n < th:
            r.append(r_n)
            valid_ind_l.append(w_cur)
            s_cur += 1
            
#        else:
#            #if dum>3: break
#            #dum += 1   
#            
#            
#            perm_best, r_best = calculate_best_seq(moves_x_mag, moves_y_mag, w_cur, \
#                stage_dx, stage_dy, s_cur, samples_take, delta_search)
#                
#            next_valid = w_cur + samples_take
#            good = perm_best<=next_valid
#            perm_best = perm_best[good]
#            r_best = r_best[good]
#            
#            valid_ind_l += list(perm_best)
#            r += list(r_best)
#            s_cur += len(perm_best)
#            #w_cur = valid_ind_l[-1]
#            w_cur = next_valid
            
            
        w_cur += 1
        if s_cur >= len(stage_dx):
            break
        
    #%%  
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(stage_dx)
    plt.plot(moves_x_mag[valid_ind_l])
    #if len(valid_ind_l) <  stage_dx.size:
    #    plt.plot(moves_x_mag)
#%%    
    plt.subplot(3,1,2)
    plt.plot(stage_dy)
    plt.plot(moves_y_mag[valid_ind_l])
    
    plt.subplot(3,1,3)   
    #dy = stage_dy-moves_y_mag[valid_ind_l]
    #dx = stage_dx-moves_x_mag[valid_ind_l]
    #r = dx*dx + dy*dy
    plt.plot(r, 'g')
    
    if len(valid_ind_l) != len(stage_dx):
        continue

    
    #%%
    worm_stage_x = np.zeros_like(video_timestamp_time)
    worm_stage_y = np.zeros_like(video_timestamp_time)
    
    tot_moves = len(valid_ind_l)
    for ii in range(len(stage_x)):
        
        prev_lim = (-1,-1) if ii == 0 else moves_lims[valid_ind_l[ii-1]]
        next_lim = (worm_stage_x.size,worm_stage_x.size) if ii == tot_moves else moves_lims[valid_ind_l[ii]]
        
        worm_stage_x[prev_lim[1]+1:next_lim[0]] = stage_x[ii]
        worm_stage_x[next_lim[0]:next_lim[1]+1] = np.nan
        
        worm_stage_y[prev_lim[1]+1:next_lim[0]] = stage_y[ii]
        worm_stage_y[next_lim[0]:next_lim[1]+1] = np.nan
    
    tot_ind = np.max(video_timestamp_ind)+1
    skeletons = np.full((tot_ind, 49, 2), np.nan)
    with h5py.File(skeletons_file, 'r') as fid:
        skeletons[video_timestamp_ind,:,:] = fid['/skeleton'][:, :, :]
    #%%
    worm_stage_y_n = np.full(tot_ind, np.nan)
    worm_stage_y_n[video_timestamp_ind] = worm_stage_y
    worm_stage_x_n = np.full(tot_ind, np.nan)
    worm_stage_x_n[video_timestamp_ind] = worm_stage_x
    

    skeletons_x = skeletons[:,:,0] - worm_stage_x_n[:, np.newaxis]
    skeletons_y = skeletons[:,:,1] - worm_stage_y_n[:, np.newaxis]
    #%%
    if os.path.exists(feat_file):
        fvars = loadmat(feat_file)
        segworm_x = fvars['worm']['posture'][0][0]['skeleton'][0][0]['x'][0][0]      
        segworm_y = fvars['worm']['posture'][0][0]['skeleton'][0][0]['y'][0][0]     
        
        micronsPerPixels_x = fvars['info']['video'][0][0]['resolution'][0][0]['micronsPerPixels'][0][0]['x'][0][0][0][0]
        micronsPerPixels_y = fvars['info']['video'][0][0]['resolution'][0][0]['micronsPerPixels'][0][0]['y'][0][0][0][0]
        #%%
        max_n_skel = min(segworm_x.shape[1], skeletons_x.shape[0])   
        
        #%%
        
        delT = 15
        ske_x = skeletons_x[:max_n_skel:delT,:]
        ske_y = skeletons_y[:max_n_skel:delT,:]
        
        seg_x = -segworm_x[:, :max_n_skel:delT]/micronsPerPixels_x
        seg_y = -segworm_y[:, :max_n_skel:delT]/micronsPerPixels_y
    
        dX = np.nanmedian(ske_x-seg_x.T)
        dY = np.nanmedian(ske_y-seg_y.T)
        
        #%%
        
        dx = -segworm_x[:,:max_n_skel]/micronsPerPixels_x-skeletons_x[:max_n_skel].T + dX
        dy = -segworm_y[:,:max_n_skel]/micronsPerPixels_y-skeletons_y[:max_n_skel].T + dY
        skel_error = np.mean(np.sqrt(dx*dx + dy*dy), axis=0)
                
        #%%
        plt.figure()
        plt.subplot(2,1,1)        
        plt.plot(ske_x.T-dX,ske_y.T-dY, 'r')    
        plt.plot(seg_x, seg_y, 'b')
        plt.axis('equal')
        
        plt.subplot(2,1,2)
        plt.plot(skel_error, '.')
        plt.ylim((0, np.nanmax(skel_error)))

#%%
    
#    plt.figure()
#    plt.subplot(2,1,1)
#    plt.plot(stage_dx)
#    plt.plot(moves_x_mag)
#    
#    plt.subplot(2,1,2)
#    plt.plot(stage_dy)
#    plt.plot(moves_y_mag)
#    
    #%%

    
    #%%
    
    #valid_ind = np.array(valid_ind_l)
    #dy = moves_y_mag[valid_ind] - stage_dy[s_ini:s_cur]
    #dx = moves_x_mag[valid_ind] - stage_dx[s_ini:s_cur]
    #r = list(dx*dx + dy*dy)
    
    
#%%
#plt.figure()
#plt.plot(r)

##%%
#bot = 0
#top = bot+10
#
#
#group = np.abs(moves_y_mag[bot:top]-stage_dy[bot:top]);
#
#med = np.median(group)
#mad = np.median(np.abs(med-group))
#bad = group > med+6*mad
#
#
#vv = np.arange(top-bot)
#plt.figure()
#plt.plot(stage_dx[bot:top])
#plt.plot(moves_x_mag[bot:top])
#plt.plot(vv[bad], moves_x_mag[bot:top][bad], 'or')
#
#plt.figure()
#plt.plot(stage_dy[bot:top])
#plt.plot(moves_y_mag[bot:top])
#plt.plot(vv[bad], moves_y_mag[bot:top][bad], 'or')
##%%
#np.arange(len(stage_dy))





#%%





#%%
#err_x = [] 
#err_y = []
#for sind in combinations(range(bot,top + search_w), samples_take):
#    #print(sind)
#    sint_a = np.array(sind)
#    err_x.append(np.max(np.abs(moves_x_mag[sint_a]-stage_wx)))
#    err_y.append(np.max(np.abs(moves_y_mag[sint_a]-stage_wy)))
    







#%%
#for xx 



#%%

#probability of a given stage motion given an image motion.
#dx = np.abs(stage_dx[np.newaxis,:] - moves_x_mag[:,np.newaxis])
#dx = np.exp(-dx)
#tot_diff = np.sum(dx, axis=0)
#ps_m = dx/tot_diff



#%%    


#def permutationG(input, s):
#    if len(s) == len(input): yield s
#    for i in input:
#        if i in s: continue
#        s=s+i
#        for x in permutationG(input, s): yield x
#        s=s[:-1]

#%%

#def groupShifts():
#    labels_x = np.zeros(xShift.shape);
#    group_n = 0;
#    is_group = False;
#    for ii, xx in enumerate(xShift);
#        if xx != 0:
#            if not is_group
#                group_n += 1;
#                is_group = True;
#            
#            labels_x[ii] = group_n;
#        else:
#            is_group = False;
#            
#    
#    
#    last_group = np.max(labels_x));
#    
#    
#    x_img_dist = zeros(1, tot_groups);
#    
#    for ii = 1:numel(xShift);
#        if labels_x(ii) > 0
#            group_n = labels_x(ii);
#            x_img_dist(group_n) = x_img_dist(group_n) + xShift(ii);
#        end
#    end