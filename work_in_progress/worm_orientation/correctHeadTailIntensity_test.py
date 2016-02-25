# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 19:00:43 2016

@author: ajaver
"""
import pandas as pd
import tables
import matplotlib.pylab as plt
import numpy as np
import collections

from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import minimum_filter
from scipy.ndimage.filters import maximum_filter
from scipy.signal import savgol_filter

def medabsdev(x): return np.median(np.abs(np.median(x)-x))    

def get_range(mm, W, maxN):
    ini = max(0, mm-W)
    fin = min(maxN-1, mm+W)
    return np.arange(ini,fin+1)

def createBlocks(flags_vector, min_block_size = 0):
    #divide data into groups of continous indexes 
    blocks = np.zeros(flags_vector.shape, np.int)
    
    lab_ind = 0
    prev_ind = False
    group_ini = []
    group_fin = []
    for ii, flag_ind in enumerate(flags_vector):
        
        if not prev_ind and flag_ind: 
            group_ini.append(ii)
        if prev_ind and not flag_ind: 
            group_fin.append(ii-1) #substract one size this is the condition one after the end of the block
        prev_ind = flag_ind
    #append the last index if the group ended in the last index
    if len(group_ini) - len(group_fin) == 1: group_fin.append(ii)
    assert len(group_ini) == len(group_fin)
    
    #change this into a single list of tuples
    groups = list(zip(group_ini, group_fin))

    #remove any group smaller than the min_block_size
    groups = [gg for gg in groups if gg[1]-gg[0] >= min_block_size] 
    return groups


def _fuseOverlapingGroups(corr_groups, gap_size=0):
    '''Helper function of correctBlock.
        -- gap_size, gap between blocks    
    '''
    #ensure the groups are sorted
    corr_groups = sorted(corr_groups)
    
    if len(corr_groups)==1:
        return corr_groups
    else:
        #fuse groups that overlap
        ini, fin = corr_groups[0]        
        corr_groups_f = []#[(ini,fin)]
        for gg in corr_groups[1:]:
            #print(ini, fin, 'a', gg)
            if fin + gap_size>= gg[0]:
                fin = gg[1]
            else:
                corr_groups_f.append((ini,fin))
                ini,fin = gg
        
        corr_groups_f.append((ini,fin))

        return corr_groups_f

def correctBlock(groups, new_flag_vec, gap_size):
    assert len(groups)>0
    corr_groups = []        
    maxInd = len(new_flag_vec)-1
    for gg in groups:         
        #loop until it reaches the window borders or find an false index
        ini = gg[0]
        while ini > 0:# and ini > gg[0]-smooth_W:
            if not new_flag_vec[ini-1]: break
            ini -= 1
        
        fin = gg[1]
        #print('a',fin)
        
        while fin < maxInd:# and fin < gg[1]+smooth_W:
            if not new_flag_vec[fin+1]: break
            fin += 1
        
        #print('b',fin)
                
        corr_groups.append((ini,fin))
    assert len(groups) == len(corr_groups)
    
    return _fuseOverlapingGroups(corr_groups, gap_size = gap_size)
    

def checkLocalVariation(worm_int_profile, groups, local_avg_win = 10):
    corr_groups = []
    
    groups = sorted(groups)
    
    tot_groups = len(groups)
    max_index = len(groups)-1
    
    min_loc_avg_win = max(1, local_avg_win//2)

    for ii in range(tot_groups):
        
        gg = groups[ii]
        
        #get the limits from the previous and enxt index
        prev_group = (-1,-1) if ii == 0 else groups[ii-1]
        next_group = (tot_groups,tot_groups) if ii == max_index else groups[ii+1]
        
        med_block = np.median(worm_int_profile[gg[0]:gg[1]+1], axis=0)
                            
        m_dif_ori_left = 0
        m_dif_inv_left = 0
        m_dif_ori_right = 0
        m_dif_inv_right = 0
        
        #get previous contigous map limits
        bot = max(gg[0] - local_avg_win, prev_group[1]+1)
        top = gg[0]-1
        
        if top-bot+1 >= min_loc_avg_win:
            med_block_left =  np.median(worm_int_profile[bot:top+1], axis=0)
                    
            m_dif_ori_left = np.sum(np.abs(med_block-med_block_left))
            m_dif_inv_left = np.sum(np.abs(med_block-med_block_left[::-1]))
        
        #get next contigous map limits
        bot = gg[1]+1
        top = min(gg[1] + local_avg_win, next_group[0]-1);
            
        if top-bot+1 >= min_loc_avg_win:
            #med_block = np.median(worm_avg[min(gg[1]-local_avg_win, gg[0]):gg[1]+1], axis=0)
            med_block_right =  np.median(worm_int_profile[bot:top+1], axis=0)
        
            m_dif_ori_right = np.sum(np.abs(med_block-med_block_right))
            m_dif_inv_right = np.sum(np.abs(med_block-med_block_right[::-1]))
                
        #combine both, we only need to have a size that show a very big change when the intensity map is switch
        #if m_dif_inv_left+m_dif_inv_right < m_dif_ori_left+m_dif_ori_right:
        if m_dif_inv_left <= m_dif_ori_left and m_dif_inv_right <= m_dif_ori_right:
            corr_groups.append(gg)
    
    return corr_groups

def removeBadSkelBlocks(groups, int_skeleton_id, trajectories_worm, min_frac_in):
    if len(groups) == 0:
        return groups #nothing to do here
    
    assert trajectories_worm['worm_index_joined'].unique().size == 1
#%%    
    #get the index of the skeletons that delimited the candiate block to be inverted
    skel_group = [(int_skeleton_id[ini], int_skeleton_id[fin]) for ini, fin in groups]
    
    #change index in the original worm skeletons matrix
    first_skel = trajectories_worm.index[0]        
    int_skel_group = [(x-first_skel,y-first_skel) for x,y in skel_group]
    
    #create globs according if consecutive frames have an skeleton map (if the have valid filtered  skeletons)
    good = (trajectories_worm['int_map_id']>0).values          
    has_skel_group = createBlocks(good, min_block_size = 0)
    
    #to test for overlaps let's created a vector with the labeled groups            
    has_blocks_flags = np.full(len(trajectories_worm), -1, np.int)
    for kk, gg in enumerate(has_skel_group):
        has_blocks_flags[gg[0]:gg[1]+1] = kk
    
    #%%
    #total number of skeletons for each group
    blocks_sizes = collections.Counter(has_blocks_flags)
    
    #total number of skeletons of a given group inside a block to be switched
    blocks_in = []
    for gg in int_skel_group:
        blocks_in += list(has_blocks_flags[gg[0]:gg[1]+1])
    blocks_in_size = collections.Counter(blocks_in)
    
    #calculate the fraction of skeletons of each group insde a block
    blocks_in_frac = {x:(blocks_in_size[x]/blocks_sizes[x]) 
    for x in blocks_in_size if x != -1}
    
    #only keep groups that has at least blocks_in_frac skeletons inside the block
    corr_skel_group = [has_skel_group[x] for x in blocks_in_frac if blocks_in_frac[x] >= min_frac_in]
    
    #shift the index to match the general trajectories_table
    corr_skel_group = [(x+first_skel,y+first_skel) for x,y in corr_skel_group]

    #convert from skeleton row id in the worm profile_intensities
    int_map_ord = {dd:kk for kk, dd in enumerate(int_skeleton_id)} 
    corr_groups = [(int_map_ord[x],int_map_ord[y]) for x,y in corr_skel_group]
    #correct for contingous groups        
    if len(corr_groups) > 1: 
        corr_groups = _fuseOverlapingGroups(corr_groups, gap_size = 1)        
    
    return corr_groups

#def removeBadSkelBlocks(skel_group, trajectories_worm):
#    #%%
#    assert trajectories_worm['worm_index_joined'].unique().size == 1
#    
#    #change index in the original worm skeletons matrix
#    first_skel = trajectories_worm.index[0]        
#    int_skel_group = [(x-first_skel,y-first_skel) for x,y in skel_group]
#    
#    #create globs according if consecutive frames have an skeleton map (if the have valid filtered  skeletons)
#    good = (trajectories_worm['int_map_id']>0).values          
#    has_skel_group = createBlocks(good, min_block_size = 0)
#    
#    #to test for overlaps let's created a vector with the labeled groups            
#    has_blocks_flags = np.full(len(trajectories_worm), -1, np.int)
#    for kk, gg in enumerate(has_skel_group):
#        has_blocks_flags[gg[0]:gg[1]+1] = kk
#    
#    #%%
#    #total number of skeletons for each group
#    blocks_sizes = collections.Counter(has_blocks_flags)
#    
#    #total number of skeletons of a given group inside a block to be switched
#    blocks_in = []
#    for gg in int_skel_group:
#        blocks_in += list(has_blocks_flags[gg[0]:gg[1]+1])
#    blocks_in_size = collections.Counter(blocks_in)
#    
#    #calculate the fraction of skeletons of each group insde a block
#    blocks_in_frac = {x:(blocks_in_size[x]/blocks_sizes[x]) 
#    for x in blocks_in_size if x != -1}
#    
#    #only keep groups that has at least blocks_in_frac skeletons inside the block
#    corr_skel_group = [has_skel_group[x] for x in blocks_in_frac if blocks_in_frac[x] >= min_frac_in]
#    
#    #shift the index to match the general trajectories_table
#    corr_skel_group = [(x+first_skel,y+first_skel) for x,y in corr_skel_group]
#    return corr_skel_group
    #%%
#    #%%
#    #Let's keep only blocks of skeletons that are mostly inside the cut
#    corr_skel_group = []
#    for gg in int_skel_group:
#        #get the groups and number of skeletons in each group, inside the candidate block to invert
#        block_N_in = collections.Counter(has_blocks_flags[gg[0]:gg[1]+1])
#        
#        new_bounds = []
#        for hh in block_N_in:
#            if hh == -1: continue
#            
#            #test if there are enough skeletons inside the block, otherwise move the limitis accordingly
#            curr_block = has_skel_group[hh]
#            block_N = curr_block[1] - curr_block[0] + 1
#            frac_in = block_N_in[hh]/float(block_N)
#            
#            
#            if frac_in > min_frac_in:
#                if len(new_bounds) == 0: new_bounds = list(curr_block)
#            
#                new_bounds[0] = min(new_bounds[0], curr_block[0])
#                new_bounds[1] = max(new_bounds[1], curr_block[1])
#        
#        if len(new_bounds) > 0:
#            assert len(new_bounds) == 2
#            corr_skel_group.append(tuple(new_bounds))
#    
#    #shift the index to match the general trajectories_table
#    corr_skel_group = [(x+first_skel,y+first_skel) for x,y in corr_skel_group]
#    #%%    
#    return corr_skel_group

def getBlockSkelInd(groups, int_skeleton_id, has_skeleton):
    skel_group = []
    
    first_skel_worm = has_skeleton.index[0]
    last_skel_worm = has_skeleton.index[-1]
    last_ind_skel = len(int_skeleton_id)-1
    
    for ini, fin in groups:
        s_ini = int_skeleton_id[ini]
        
        #if the index is the first in skeleton_id look outside that range for the first skeleton in the block
        s_ini_lim = 0 if ini == 0 else int_skeleton_id[ini-1]
        while s_ini > s_ini_lim and s_ini > first_skel_worm:
            if has_skeleton[s_ini-1] == 0:
                break
            else:
                s_ini -= 1
        
        s_fin = int_skeleton_id[fin]
        
        #if the index is the last in skeleton_id look outside that range for the last skeleton in the block
        s_fin_lim = np.inf if fin == last_ind_skel else int_skeleton_id[fin+1]
        
        #print('b', fin, s_fin, s_fin_lim)        
        while s_fin < s_fin_lim and s_fin < last_skel_worm:
            if has_skeleton[s_fin+1] == 0:
                break
            else:
                s_fin += 1
        #print(s_fin)
        skel_group.append((s_ini, s_fin))
        
        
    return skel_group

def getMinShiftDist(vec, vec_med, shift_W=10):
    diff_mat = np.zeros((vec.shape[0], 2*shift_W+1))
    
    
    diff_mat[:,shift_W] = np.sum(np.abs(vec[:,shift_W:-shift_W]-vec_med[shift_W:-shift_W]), axis = 1)
    N = vec.shape[0]
    for kk in range(1, shift_W+1):
        diff_mat[:,shift_W-kk] = np.sum(np.abs(vec[:,shift_W-kk:-shift_W-kk]-vec_med[shift_W-kk:-shift_W-kk]), axis = 1)
        diff_mat[:,shift_W+kk] = np.sum(np.abs(vec[:,shift_W+kk:N+-shift_W+kk]-vec_med[shift_W+kk:N+-shift_W+kk]), axis = 1)
    
    
    return np.min(diff_mat, axis=1)

def contrast_maps(worm_avg):
    #i need to generalize this scaling, but for the moment it is fine
    min_val = np.min(worm_avg)
    max_val = np.max(worm_avg)  
    
    assert min_val != max_val
    worm_avg = (worm_avg - min_val)/(max_val-min_val)
    
    
    bin_size = 64;
    all_mapCP = np.zeros((worm_avg.shape[0], bin_size,bin_size+1))
    all_mapCS = np.zeros((worm_avg.shape[0], bin_size,bin_size+1))
    
    for mm in range(worm_avg.shape[0]):
        XX = worm_avg[mm]
        N = len(XX)        
        
        CS = np.zeros(0)
        CP = np.zeros(0)
        II = []
        
        for delta in range(1, N):
            
            CS = np.hstack((CS, np.abs(XX[delta:] - XX[:-delta])))
            CP = np.hstack((CP, XX[delta:] + XX[:-delta]))
            II += [delta]*(N-delta)            
        
        
        
        CS = np.round(CS*(bin_size-1)).astype(np.int)
        CP = np.round(CP/2*(bin_size-1)).astype(np.int)
        
        II = ((np.array(II)-1)/(N-2)*(bin_size)).astype(np.int)
        
        
        IIrS = np.ravel_multi_index(np.vstack((CS,II)),(bin_size,bin_size+1))        
        binCS = np.bincount(IIrS, minlength=bin_size*(bin_size+1))
        binCS = binCS.reshape((bin_size,bin_size+1))
        
        IIrP = np.ravel_multi_index(np.vstack((CP,II)),(bin_size,bin_size+1))
        binCP = np.bincount(IIrP, minlength=bin_size*(bin_size+1))
        binCP = binCP.reshape((bin_size,bin_size+1))
        
        all_mapCP[mm] = binCP
        all_mapCS[mm] = binCS    
    return all_mapCP, all_mapCS
  
def dat_switch(X, r_range):
    fin = r_range[1]+1
    dat = X[r_range[0]:fin]
    X[r_range[0]:fin] = dat[:, ::-1]

def dat_swap(X, Y, r_range):
    fin = r_range[1]+1
    dat_x = X[r_range[0]:fin]
    dat_y = Y[r_range[0]:fin]
    X[r_range[0]:fin] = dat_y
    Y[r_range[0]:fin] = dat_x

def dat_switch_swap(X,Y, r_range):
    fin = r_range[1]+1
    dat_x = X[r_range[0]:fin]
    dat_y = Y[r_range[0]:fin]
    X[r_range[0]:fin] = dat_y[:, ::-1]
    Y[r_range[0]:fin] = dat_x[:, ::-1]

if __name__ == '__main__':

    #%%
    masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch1_18112015_075624.hdf5'
    #masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch1_17112015_205616.hdf5'
    #masked_image_file = '/Users/ajaver/Desktop/Videos/04-03-11/MaskedVideos/575 JU440 swimming_2011_03_04__13_16_37__8.hdf5'    
    #masked_image_file = '/Users/ajaver/Desktop/Videos/04-03-11/MaskedVideos/575 JU440 on food Rz_2011_03_04__12_55_53__7.hdf5'    
    
    skeletons_file = masked_image_file.replace('MaskedVideos', 'Results')[:-5] + '_skeletons.hdf5'
    intensities_file = skeletons_file.replace('_skeletons', '_intensities')

    smooth_W = 5
    min_block_size = 2*smooth_W 
    local_avg_win = 25#2*smooth_W 
    min_frac_in = 0.85
    
    gap_size = 0#local_avg_win #otherwise i cannot really say a local average is valid
    
    with pd.HDFStore(intensities_file, 'r') as fid:
        trajectories_data_int = fid['/trajectories_data_valid']
        
    with pd.HDFStore(skeletons_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
        
        ind = trajectories_data_int['skeleton_id'].values
        trajectories_data['int_map_id'] = np.array(-1, np.int)
        trajectories_data.loc[ind, 'int_map_id'] = trajectories_data_int['int_map_id'].values
        
        assert not np.any(np.isnan(trajectories_data.loc[ind, 'int_map_id'].values))
    #%%
        #,
    #ind2check =  [2157]#[190, 2523]#[16, 190, 812]#190, 901, 2945, 2919, 2665, 2494, 2470, 2432, 2217, 2102, 1293, 832, 268, 152]
    #ind2check = [1970]#[2918, 2494, 2037, 1970, 1235, 832, 788, 731, 599]
    #ind2check = [2999, 2157, 1699, 1214, 717, 303, 9, 6]    
    ind2check = [6]    
    for ind, trajectories_worm in trajectories_data.groupby('worm_index_joined'):
        if not ind in ind2check: continue 
        #if ind < 433: continue    
        #if ind > 3: break
        
        print(ind, len(trajectories_worm))        
        good = trajectories_worm['int_map_id']>0;
        
        int_map_id = trajectories_worm.loc[good, 'int_map_id']
        int_skeleton_id = trajectories_worm.loc[good, 'skeleton_id'].values
        int_frame_number = trajectories_worm.loc[good, 'frame_number'].values
        
        if int_map_id.size < min_block_size:
            continue
        assert int_map_id.size > 0
        
        #skel_id = trajectories_worm.loc[trajectories_worm['has_skeleton'] == 1, 'skeleton_id'].values
        
        with tables.File(intensities_file, 'r') as fid:
            #worm_maps = fid.get_node('/straighten_worm_intensity')[int_map_id,:,:]
            worm_avg = fid.get_node('/straighten_worm_intensity_median')[int_map_id.values,:]
        
        #%%
        #normalize intensities for each level    
        #worm_avg -= np.median(worm_avg, axis = 1)
        
        for ii in range(worm_avg.shape[0]):
            #worm_avg[ii,:] = savgol_filter(worm_avg[ii,:], 11, 3)
            worm_avg[ii,:] -= np.median(worm_avg[ii,:])
            
        
        #%%    
        med_int = np.median(worm_avg, axis=0).astype(np.float)
        
        #%%
        diff_ori = np.sum(np.abs(med_int-worm_avg), axis = 1)
        diff_inv = np.sum(np.abs(med_int[::-1]-worm_avg), axis = 1)
        
        #diff_ori = getMinShiftDist(worm_avg, med_int, shift_W=15)
        #diff_inv = getMinShiftDist(worm_avg, med_int[::-1], shift_W=15)
        
        #%%
        #check if signal noise will allow us to distinguish between the two signals
        #I am assuming that most of the images will have a correct head tail orientation and the robust estimates will give us a good representation
        #we should label as bad    
        #%%
        half_N = len(diff_inv)//2+1
        top = np.sort(diff_inv)[half_N:]
        bot = np.sort(diff_ori)[:half_N]
        
        #if np.mean(top) - np.std(top) < np.mean(bot) + np.std(bot):
        #    print('A', worm_index)
        #    continue
        
        #%%
        if np.median(diff_inv) - medabsdev(diff_inv)/2 < np.median(diff_ori) + medabsdev(diff_ori)/2:
            print('AAA')            
            continue
        
#%%
        #smooth data, it is easier for identification
        diff_ori_med = median_filter(diff_ori,smooth_W)
        diff_inv_med = median_filter(diff_inv,smooth_W)
            
        #this will increase the distance between the original and the inversion. Therefore it will become more stringent on detection
        diff_orim = minimum_filter(diff_ori_med, smooth_W)    
        diff_invM = maximum_filter(diff_inv_med, smooth_W)   
                
        
        #%%
        bad_orientationM = diff_orim>diff_invM
        if np.all(bad_orientationM): continue
        
        groups = createBlocks(bad_orientationM, min_block_size)
        if not groups: continue
        #refine groups using the original diferences
        bad_orientation = diff_ori>diff_inv
        groups = correctBlock(groups, bad_orientation, gap_size)
        #%%
        
        #correct for contingous groups        
        corr_groups = removeBadSkelBlocks(groups, int_skeleton_id, trajectories_worm, min_frac_in)       
        #%%     
        #filter groups to check if there is really a better local match is the block is inverted 
        corr_groups = checkLocalVariation(worm_avg, groups, local_avg_win = local_avg_win)
        
        groups = corr_groups  
        #redefine the skel limits using the final groups
        skel_group = [(int_skeleton_id[ini], int_skeleton_id[fin]) for ini, fin in groups]
        int_group = [(int_map_id.values[ini], int_map_id.values[fin]) for ini, fin in groups]
        frame_group = [(int_frame_number[ini], int_frame_number[fin]) for ini, fin in groups]
        
        if not groups: continue
        
        #%%
        if True:
        
            fig, ax1 = plt.subplots()
            
            if True:            
                ax1.imshow(worm_avg.T, interpolation='none', cmap='gray')        
                for ini, fin in groups:
                    ax1.plot((ini, ini), ax1.get_ylim(), 'r:')
                    ax1.plot((fin, fin), ax1.get_ylim(), 'm--')
                for ini, fin in corr_groups:
                    ax1.plot((ini, ini), ax1.get_ylim(), 'c-')
                    ax1.plot((fin, fin), ax1.get_ylim(), 'c-')
                
            if False:
                ax2=ax1.twinx()
                ax2.plot(diff_ori, '.-')
                ax2.plot(diff_inv, '.-')
                #ax2.plot(diff_orim)
                #ax2.plot(diff_invM)
                
                for ini, fin in groups:
                    ax2.plot((ini, ini), ax2.get_ylim(), 'r:')
                    ax2.plot((fin, fin), ax2.get_ylim(), 'm--')
                    
                for ini, fin in corr_groups:
                    ax2.plot((ini, ini), ax2.get_ylim(), 'c-')
                    ax2.plot((fin, fin), ax2.get_ylim(), 'c-')
                
            dd = '' #if badInd.size == 0 else ' BAD'
            plt.title(str(ind) +  dd)
                
            plt.xlim((-1, len(diff_ori)))    
            #plt.xlim((42303, 43913))
        #%%
        if False:
            for ini, fin in groups:
                plt.figure()
                plt.imshow(worm_avg.T, interpolation='none', cmap='gray')        
                plt.plot((ini, ini), plt.gca().get_ylim(), 'c:')
                plt.plot((fin, fin), plt.gca().get_ylim(), 'c--')
                plt.title(str(ind))
                plt.xlim((ini-100, fin+100)) 
        
        #%% switch block
        break
        continue
        with tables.File(skeletons_file, 'r+') as fid:
            contour_side1 = fid.get_node('/contour_side1')
            contour_side2 = fid.get_node('/contour_side2')
            skeleton = fid.get_node('/skeleton')
            contour_width = fid.get_node('/contour_width')
            
            
            cnt1_length = fid.get_node('/contour_side1_length')
            cnt2_length = fid.get_node('/contour_side2_length')
            
            w_head_t = fid.get_node('/width_head_tip')
            w_head_b = fid.get_node('/width_head_base')
            w_neck = fid.get_node('/width_neck')
            w_hips = fid.get_node('/width_hips')
            w_tail_b = fid.get_node('/width_tail_base')
            w_tail_t = fid.get_node('/width_tail_tip')
            
            for gg in skel_group:
                dat_switch_swap(contour_side1, contour_side2, gg)
                
                dat_switch(skeleton, gg)
                dat_switch(contour_width, gg)
                
                dat_swap(cnt1_length, cnt2_length, gg)
                dat_swap(w_head_t, w_tail_t, gg)
                dat_swap(w_head_b, w_tail_b, gg)
                dat_swap(w_hips, w_neck, gg)
            fid.flush()
        
        with tables.File(intensities_file, 'r+') as fid:
            worm_int_med = fid.get_node('/straighten_worm_intensity_median')
            for gg in int_group:
                dat_switch(worm_int_med, gg)
            
            if '/straighten_worm_intensity' in fid:
                worm_int = fid.get_node('/straighten_worm_intensity')
            
                for ini, fin in int_group:
                    dat = worm_int[ini:fin+1, :, :]
                    worm_int[ini:fin+1, :, :] = dat[:, ::-1, ::-1]
            fid.flush()
                
        


        
        
#%%
def testSkeletonsLoc():
    #broken
    win_skel_check = 5
    
    corr_groups = []     
    
    N = len(int_skeleton_id)
    with tables.File(skeletons_file, 'r') as fid:
        for block_ini, block_fin in groups:
            skel_HT = (0,1,2,-3,-2,-1)
            nomr1_ori = 0
            nomr1_inv = 0
            
            if block_ini - win_skel_check > 0:
                bot = block_ini-win_skel_check
                top = min(block_ini + win_skel_check, block_fin)                    
                skeletons_out_f = fid.get_node('/skeleton')[int_skeleton_id[bot:block_ini],:,:]
                
                coord_out_first = np.median(skeletons_out_f[:,skel_HT,:], axis=0)
                
                skeletons_in_f = fid.get_node('/skeleton')[int_skeleton_id[block_ini:top+1],:,:]
                coord_in_first = np.median(skeletons_in_f[:,skel_HT,:], axis=0)
                
                nomr1_ori += np.sum(np.abs(coord_in_first - coord_out_first))
                nomr1_inv += np.sum(np.abs(coord_in_first[::-1,:] - coord_out_first))

                
            if block_fin + win_skel_check < N:
                top = block_fin+win_skel_check
                bot = max(block_fin - win_skel_check, block_ini)
                
                skeletons_in_l = fid.get_node('/skeleton')[int_skeleton_id[bot:block_fin],:,:]
                coord_in_last = np.median(skeletons_in_l[:,skel_HT,:], axis=0)
                
                skeletons_out_l = fid.get_node('/skeleton')[int_skeleton_id[block_fin:top+1],:,:]
                coord_out_last = np.median(skeletons_out_l[:,skel_HT,:], axis=0)
                
                nomr1_ori += np.sum(np.abs(coord_in_last - coord_out_last))
                nomr1_inv += np.sum(np.abs(coord_in_last[::-1,:] - coord_out_last))

            if nomr1_inv < nomr1_ori:
                corr_groups.append((block_ini, block_fin))
            
            #%%
            plt.figure()
            plt.subplot(2,1,1)
            if block_ini - win_skel_check > 0:
                for ii in range(skeletons_out_f.shape[0]):
                    plt.plot(skeletons_out_f[ii,:,0], skeletons_out_f[ii,:,1])
                    plt.plot(skeletons_out_f[ii,0,0], skeletons_out_f[ii,0,1], 'sr')
                for ii in range(skeletons_in_f.shape[0]):
                    plt.plot(skeletons_in_f[ii,:,0], skeletons_in_f[ii,:,1])
                    plt.plot(skeletons_in_f[ii,0,0], skeletons_in_f[ii,0,1], 'og')
                
                
            plt.subplot(2,1,2)
            if block_fin + win_skel_check < N:
                for ii in range(skeletons_out_l.shape[0]):
                    plt.plot(skeletons_out_l[ii,:,0], skeletons_out_l[ii,:,1])
                    plt.plot(skeletons_out_l[ii,0,0], skeletons_out_l[ii,0,1], 'sr')
                for ii in range(skeletons_in_l.shape[0]):
                    plt.plot(skeletons_in_l[ii,:,0], skeletons_in_l[ii,:,1])
                    plt.plot(skeletons_in_l[ii,0,0], skeletons_in_l[ii,0,1], 'og')
                
            plt.title(ind)
        