# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:46:36 2016

@author: ajaver
"""
import pandas as pd
import os
import tables
import numpy as np
import matplotlib.pylab as plt
from collections import OrderedDict

import networkx as nx

import cv2
from scipy.signal import savgol_filter
from scipy.signal import medfilt

import sys
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking')
from MWTracker.helperFunctions.timeCounterStr import timeCounterStr

from MWTracker.trackWorms.getSkeletonsTables import getWormROI, getWormMask, binaryMask2Contour
from MWTracker.featuresAnalysis.obtainFeaturesHelper import WLAB
from MWTracker.featuresAnalysis.getFilteredFeats import saveModifiedTrajData

#%%
def getStartEndTraj(trajectories_data):
    traj_limits = OrderedDict()
    
    assert 'worm_index_auto' in trajectories_data
    grouped_trajectories = trajectories_data.groupby('worm_index_auto')
    
    tot_worms = len(grouped_trajectories)
    base_name = skeletons_file.rpartition('.')[0].rpartition(os.sep)[-1].rpartition('_')[0]
    progress_timer = timeCounterStr('');
    
    win_area = 10    
    
    traj_limits = np.recarray(tot_worms, [('worm_index',np.int), ('t0',np.int), ('tf',np.int), \
    ('x0',np.float), ('xf',np.float), ('y0',np.float), ('yf',np.float), \
    ('a0',np.float), ('af',np.float),  ('th0',np.float), ('thf',np.float), \
    ('roi_size',np.int)])
    
    fields_needed = ['coord_x', 'coord_y', 'roi_size', 'frame_number', 'threshold']
    for index_n, (worm_index, trajectories_worm) in enumerate(grouped_trajectories):
        good = trajectories_worm['area'] != 0
        dd = trajectories_worm.loc[good, 'frame_number']
        row0 = dd.argmin();
        rowf = dd.argmax();
        
        worm_areas = trajectories_worm['area'].values;
        
        a0 = np.median(worm_areas[:win_area])
        
        af = np.median(worm_areas[-win_area:])
        
        dd = trajectories_worm.loc[[row0, rowf], fields_needed]
        
        t0, tf = dd['frame_number'].values
                
        roi_size = dd['roi_size'].values[0]

        x0, xf = dd['coord_x'].values
        y0, yf = dd['coord_y'].values
        
        th0, th1 = dd['threshold'].values     
        
        traj_limits[index_n] = (worm_index, t0, tf, x0,xf, y0,yf, a0, af, th0, th1, roi_size)
        #if worm_index == 50: break;
    traj_limits = pd.DataFrame(traj_limits, index=traj_limits['worm_index'])
    return traj_limits
#%%
class imageRigBuff:
    #ring buffer class to avoid having to read the same image several times
    def __init__(self, mask_group, buf_size):
        
        self.buf_size = buf_size
        
        self.ind = 0
        self.current_time = 0
        self.win_size = buf_size//2
        self.last_frame = mask_group.shape[0]
        self.mask_group = mask_group
        
        #start the buffer
        self.I_buff = mask_group[:buf_size]
        
    def get_buffer(self, next_frame):
        bot = max(next_frame-self.win_size, self.current_time+self.win_size+1)        
        top = min(self.last_frame, next_frame + self.win_size)
        self.current_time = next_frame
        
        
        Idum = self.mask_group[bot:top+1]
        #print('A', bot, top, Idum.shape)
                
        for ii in range(Idum.shape[0]):
            self.I_buff[self.ind] = Idum[ii]
            self.ind += 1
            if self.ind >= self.buf_size:
                self.ind = 0
        
        return self.I_buff

def getIndCnt(img, x, y, roi_size, thresh, max_area):
    
    
    worm_img, roi_corner = getWormROI(img, x, y, roi_size)
    worm_mask = getWormMask(worm_img, thresh)
    worm_cnt, _ = binaryMask2Contour(worm_mask, min_mask_area = max_area)
    if worm_cnt.size > 0:
        worm_cnt += roi_corner   
    return worm_cnt                 
                    

def extractWormContours(masked_image_file, traj_limits, buf_size = 5):
    grouped_t0 = traj_limits.groupby('t0')
    grouped_tf = traj_limits.groupby('tf')
    
    uT0 = np.unique(traj_limits['t0'])
    uTf = np.unique(traj_limits['tf'])
    
    initial_cnt = OrderedDict()
    final_cnt = OrderedDict()
    
    with tables.File(masked_image_file, 'r') as fid:
        mask_group = fid.get_node('/mask')
        ir = imageRigBuff(mask_group, buf_size)
        
        for frame_number in np.unique(np.concatenate((uT0,uTf))):
            #img = mask_group[frame_number]
            img = ir.get_buffer(frame_number)
            img = np.min(img, axis=0)
            if frame_number in uT0:
                dd = grouped_t0.get_group(frame_number)
                for ff, row in dd.iterrows():
                    worm_cnt = getIndCnt(img, row['x0'], row['y0'], 
                                         row['roi_size'], row['th0'], row['a0']/2)
                    initial_cnt[int(row['worm_index'])] = worm_cnt
            if frame_number in uTf:
                dd = grouped_tf.get_group(frame_number) 
                for ff, row in dd.iterrows():
                    worm_cnt = getIndCnt(img, row['xf'], row['yf'], 
                                         row['roi_size'], row['thf'], row['af']/2)
                    final_cnt[int(row['worm_index'])] = worm_cnt
                    
        return initial_cnt, final_cnt 
#%% 
def getPossibleConnections(traj_limits, max_gap = 25):
    connect_before = OrderedDict()
    connect_after = OrderedDict()
    
    for worm_index in traj_limits.index:
        curr_data = traj_limits.loc[worm_index]
        other_data = traj_limits[traj_limits.index != worm_index].copy()
        
        other_data['gap'] = curr_data['t0'] - other_data['tf']
        good = (other_data['gap']> 0) & (other_data['gap']<= max_gap)
        before_data = other_data[good].copy()
    
        other_data['gap'] = other_data['t0'] - curr_data['tf']
        good = (other_data['gap']> 0) & (other_data['gap']<= max_gap)
        after_data =  other_data[good].copy()
        
        Rlim = curr_data['roi_size']**2
        
        delXb = curr_data['x0'] - before_data['xf']
        delYb = curr_data['y0'] - before_data['yf']
        before_data['R2'] = delXb*delXb + delYb*delYb
        before_data = before_data[before_data['R2']<=Rlim]
        
        #before_data['AR'] =  curr_data['a0']/before_data['af']
        before_data = before_data[(curr_data['a0']!=0) & (before_data['af']!=0)]
        
        delXa = curr_data['xf'] - after_data['x0']
        delYa = curr_data['yf'] - after_data['y0']
        after_data['R2'] = delXa*delXa + delYa*delYa
        after_data = after_data[after_data['R2']<=Rlim]
        
        #after_data['AR'] =  curr_data['af']/after_data['a0']
        after_data = after_data[(curr_data['af']!=0) & (after_data['a0']!=0)]
        
        assert worm_index == curr_data['worm_index']
        if len(before_data) > 0:        
            connect_before[worm_index] = list(before_data.index.values)
        
        if len(after_data) > 0:        
            connect_after[worm_index] = list(after_data.index.values)
            
    return connect_before, connect_after



def getAreaIntersecRatio(connect_dict, node1_cnts, node2_cnts):    

    intersect_ratio = {}
    for current_ind in connect_dict:
        current_cnt = node1_cnts[current_ind]
        if current_cnt.size == 0:
            continue
        bot = np.min(current_cnt, axis=0);
        top = np.max(current_cnt, axis=0);
        
        for pii in connect_dict[current_ind]:
            if node2_cnts[pii].size == 0:
                continue
            bot_p = np.min(node2_cnts[pii],axis=0);
            top_p = np.max(node2_cnts[pii],axis=0);
            
            bot = np.min((bot, bot_p), axis=0)
            top = np.max((top, top_p), axis=0)
        
        roi_size = top-bot + (1,1)
        roi_size = roi_size[::-1]
    
        mask_curr = np.zeros(roi_size, np.int32)
        worm_cnt = [(current_cnt-bot).astype(np.int32)];
        cv2.drawContours(mask_curr, worm_cnt, 0, 1, -1)
        area_curr = np.sum(mask_curr)    
        
        for pii in connect_dict[current_ind]:
            if node2_cnts[pii].size == 0:
                continue
            mask_possible = np.zeros(roi_size, np.int32)
            worm_cnt = [(node2_cnts[pii]-bot).astype(np.int32)];
            cv2.drawContours(mask_possible, worm_cnt, 0, 1, -1)
            
            area_intersect = np.sum(mask_curr & mask_possible)
        
            intersect_ratio[(current_ind, pii)] = area_intersect/area_curr
        
    return intersect_ratio

def selectNearNodes(connect_dict, ratio_dict, time_table, min_intersect = 0.5):
    posible_nodes = []
    for node1 in connect_dict:
        node2_dat = []
        for node2 in connect_dict[node1]:
            if (node1, node2) in ratio_dict:
                node2_t0 = time_table[node2]
                ratio12 = ratio_dict[(node1, node2)]
                node2_dat.append((node2, node2_t0,ratio12))
        
        node2_dat = [x for x in node2_dat if x[2] > min_intersect]
        if len(node2_dat)==0: 
            continue

        node2_dat = min(node2_dat, key=lambda a:a[1])
        
        posible_nodes.append((node1, node2_dat[0]))
    return posible_nodes

#%%
def cleanRedundantNodes(DG, trajectories_data_f):
    
    #join trajectories that only has as parent and child each other. (A->B->C)
    same_nodes = []
    for ind1 in DG.nodes():
        next_nodes = DG.successors(ind1)
        if len(next_nodes) == 1:
            pp = DG.predecessors(next_nodes[0]);
            if len(pp) == 1 and pp[0] == ind1:
                same_nodes.append((ind1, next_nodes[0]))
    
    #create path form this trajectories
    G_redundant = nx.Graph()
    G_redundant.add_nodes_from(set([i for sub in same_nodes for i in sub]))
    G_redundant.add_edges_from(same_nodes)
    
    index2rename = {}
    for subgraph in nx.connected_component_subgraphs(G_redundant):
        
        nodes2remove = sorted(subgraph.nodes())
        edges2remove = subgraph.edges()
        
        #find the first node
        first_node = []
        for x in nodes2remove:
            pred = DG.predecessors(x)
            if len(pred) != 1 or not pred in nodes2remove:
                first_node.append(x)
        assert len(first_node) == 1
        first_node = first_node[0]
        
        #find the last node
        last_node = []
        for x in nodes2remove:
            suc = DG.successors(x)
            if len(suc) != 1 or not suc in nodes2remove:
                last_node.append(x)
        assert len(last_node) == 1
        last_node = last_node[0]
        
        #we are only keeping the first node
        nodes2remove.remove(first_node)
        index2rename[first_node] = nodes2remove
        
        #connect the edges of the last node with the first no
        edges2add = []
        for ind2 in DG.successors(last_node):
                edges2add.append((first_node,ind2))
        DG.add_edges_from(edges2add)
        DG.remove_edges_from(edges2remove)
        DG.remove_nodes_from(nodes2remove)
    
    for new_index in index2rename:
        for ind in index2rename[new_index]:
            row2chg = trajectories_data_f.worm_index_auto.isin(index2rename[new_index])
            trajectories_data_f.loc[row2chg, 'worm_index_auto'] = new_index
    
    return DG, trajectories_data_f
#%%
def getTrajGraph(trajectories_data, masked_image_file, max_gap = 25, min_area_intersect = 0.5):
    print('Getting the trajectories starting and ending points.')
    traj_limits = getStartEndTraj(trajectories_data) 
    
    print('Getting possible connecting point.')
    connect_before, connect_after = getPossibleConnections(traj_limits, max_gap = max_gap)
    
    print('Extracting worm contours from trajectory limits.')
    initial_cnt, final_cnt = extractWormContours(masked_image_file, traj_limits)

    print('Looking for overlaping fraction between contours.')
    after_ratio = getAreaIntersecRatio(connect_after, final_cnt, initial_cnt)
    before_ratio = getAreaIntersecRatio(connect_before, initial_cnt, final_cnt)
    #maybe a graph reduction algorithm would work better...
    
    print('Getting connections between trajectories.')    
    edges_after = selectNearNodes(connect_after, after_ratio, traj_limits['t0'], min_intersect = min_area_intersect)
    edges_before = selectNearNodes(connect_before, before_ratio, -traj_limits['tf'], min_intersect = min_area_intersect)
    #switch so the lower index is first    
    edges_before = [(y,x) for x,y in edges_before]
    
    #get unique nodes
    trajectories_edges = set(edges_after+edges_before)
        
    print('Removing redundant connections.')
    DG=nx.DiGraph()
    DG.add_nodes_from(traj_limits.index)
    DG.add_edges_from(trajectories_edges)
    DG, index2rename = cleanRedundantNodes(DG)
    
    for new_index in index2rename:
        for ind in index2rename[new_index]:
            row2chg = trajectories_data.worm_index_auto.isin(index2rename[new_index])
            trajectories_data.loc[row2chg, 'worm_index_auto'] = new_index
    return DG, trajectories_data, traj_limits


def getRealWormsIndexes(trajectories_data, n_min_skel = 5, min_frac_skel = 0.25):
    
    N = trajectories_data.groupby('worm_index_auto').agg({'is_good_skel': ['sum', 'count']})
    frac_skel = N['is_good_skel']['sum']/N['is_good_skel']['count']
    good = (N['is_good_skel']['count'] > 5) & (frac_skel> min_frac_skel)
    worm_indexes = set(N[good].index.tolist())    
    return worm_indexes

def getPossibleClusters(DG, worm_indexes):
    #%%
    possible_cluster = []
    for node in DG.nodes():
        node_inputs = DG.predecessors(node)
        node_outpus = DG.successors(node)
        
        if (any(x in worm_indexes for x in node_inputs) or  \
            any(x in worm_indexes for x in node_outpus)):
            possible_cluster.append(node)
    possible_cluster = set(possible_cluster)
#%%    
    return possible_cluster

def filterTableByArea(trajectories_data, min_area_limit = 50, n_sigma = 6):
    area_med = trajectories_data['area'].median()
    area_mad = (trajectories_data['area']-area_med).abs().median();
    
    min_area = max(min_area_limit, area_med-n_sigma*area_mad)
    
    median_area = trajectories_data.groupby('worm_index_auto').agg({'area':'median'})
    is_small =  (median_area<min_area).values
    small_index = median_area[is_small].index.tolist()
    
    
    trajectories_data_f = trajectories_data[~trajectories_data.worm_index_auto.isin(small_index)]    
    return trajectories_data_f

#%%
if __name__ == '__main__':
    #base directory
    #masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch6_17112015_205616.hdf5'
    #masked_image_file = '/Users/ajaver/Desktop/Videos/Camille_151030/MaskedVideos/CSTCTest_Ch2_30102015_212430.hdf5'    
    masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch1_18112015_075624.hdf5'
    #masked_image_file = '/Users/ajaver/Desktop/Videos/04-03-11/MaskedVideos/575 JU440 swimming_2011_03_04__13_16_37__8.hdf5'    
    #masked_image_file = '/Users/ajaver/Desktop/Videos/04-03-11/MaskedVideos/575 JU440 on food Rz_2011_03_04__12_55_53__7.hdf5'    
    
    skeletons_file = masked_image_file.replace('MaskedVideos', 'Results')[:-5] + '_skeletons.hdf5'
    intensities_file = skeletons_file.replace('_skeletons', '_intensities')
    
    min_area_limit = 50
    
    #get the trajectories table
    with pd.HDFStore(skeletons_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
        trajectories_data['worm_index_auto'] = trajectories_data['worm_index_joined'] 
    
    trajectories_data_f = filterTableByArea(trajectories_data, min_area_limit=min_area_limit, n_sigma = 6)
    
    del trajectories_data
    
    #%%
    trajgrouped = trajectories_data_f.groupby('frame_number')    

    traj_limits = getStartEndTraj(trajectories_data_f) 
    
    first_index = traj_limits['t0'].min()
    last_index = traj_limits['tf'].max()

    break_points = []
    #check if the trajectories end or the begining intersect the middle of other trajectories    
    #for ind_check, row2check in traj_limits.loc[436:437].iterrows():
    for ind_check, row2check in traj_limits.iterrows():
        
        Rsearch = row2check['roi_size']**2
        
        t_prev = row2check['t0']-1
        try:      
            
            dat_prev = trajgrouped.get_group(t_prev);
            
            #only cosider points that are in the middle of another trajectory (do not use start or ending points)
            possible_ind = dat_prev['worm_index_auto'].values                
            traj_tt = traj_limits.loc[possible_ind,  'tf']
                            
            dat_prev = dat_prev[(traj_tt != t_prev).values]# & (traj_tt['tf'] != t_next)).values]
            
            delX = row2check['x0'] - dat_prev['coord_x']
            delY = row2check['y0'] - dat_prev['coord_y']
            R = delX*delX + delY*delY
            dat_prev = dat_prev[R <= Rsearch]
            for _, rr in dat_prev.iterrows():
                assert t_prev == rr['frame_number'] == row2check['t0']-1
                            
                ind_split = int(rr['worm_index_auto'])
                dat_split = (rr['coord_x'], rr['coord_y'], rr['roi_size'], rr['threshold'], min_area_limit)#rr['area']/2)
                dat_check = (row2check['x0'], row2check['y0'], row2check['roi_size'], row2check['th0'], min_area_limit)#row2check['a0']/2)
                break_points.append(((int(ind_split), int(t_prev), dat_split), \
                                    (int(ind_check), int(row2check['t0']), dat_check)))      
        except KeyError:
             pass
        
        t_next = row2check['tf']+1
        try:    
            
            dat_next = trajgrouped.get_group(t_next);
            
            #only cosider points that are in the middle of another trajectory (do not use start or ending points)
            possible_ind = dat_next['worm_index_auto'].values                
            traj_tt = traj_limits.loc[possible_ind,  't0']
            dat_next = dat_next[(traj_tt != t_next).values]# & (traj_tt['tf'] != t_next)).values]
            
            delX = row2check['xf'] - dat_next['coord_x']
            delY = row2check['yf'] - dat_next['coord_y']
            R = delX*delX + delY*delY
            dat_next = dat_next[R <= Rsearch]
            
            for _, rr in dat_next.iterrows():
                assert t_next == rr['frame_number'] == row2check['tf']+1
                            
                ind_split = int(rr['worm_index_auto'])
                dat_split = (rr['coord_x'], rr['coord_y'], rr['roi_size'], rr['threshold'], min_area_limit)#rr['area']/2)
                dat_check = (row2check['xf'], row2check['yf'], row2check['roi_size'], row2check['thf'], min_area_limit)#row2check['af']/2)
                break_points.append(((int(ind_split), int(t_next), dat_split), \
                                    (int(ind_check), int(row2check['tf']), dat_check)))            
        except KeyError:
             pass

    print('B', len(break_points))
    #AA    

#%%
    indexInFrame = {} 
    roi_data = {}
    for dd in break_points:
        for ind, t, dat in dd:
            if not t in indexInFrame:
                indexInFrame[t] = []
            indexInFrame[t].append(ind)
            roi_data[ind,t] = dat

#%%
    
    buf_size = 5
    worm_cnt = OrderedDict() 
    with tables.File(masked_image_file, 'r') as fid:
        mask_group = fid.get_node('/mask')
        ir = imageRigBuff(mask_group, buf_size)
        
        for t in sorted(indexInFrame):
            img = ir.get_buffer(t)
            img = np.min(img, axis=0)
            for ind in indexInFrame[t]:
                worm_cnt[ind, t] = getIndCnt(img, *roi_data[ind, t])
    
    
#%%
    area_overlap = {}
    for (ind_split, t_split, dat_split), (ind_check, t_check, dat_check) in break_points:
        key_tuple = (ind_split, t_split, ind_check, t_check)
        cnt_split = worm_cnt[ind_split, t_split]
        cnt_check = worm_cnt[ind_check, t_check]
        
        if len(cnt_check) == 0: 
            area_overlap[key_tuple] = -2#'bad_check'
        elif len(cnt_split) == 0: 
            area_overlap[key_tuple] = -1#'bad_split'
            
        elif not key_tuple in area_overlap:
        
            bot = np.minimum(np.amin(cnt_split,0), np.amin(cnt_check, 0))
            top = np.maximum(np.amax(cnt_split,0), np.amax(cnt_check, 0))
            
            roi_size = roi_size = top-bot + (1,1)
            roi_size = roi_size[::-1]
            
            mask_split = np.zeros(roi_size, np.int32)
            cnt_split = [(cnt_split-bot).astype(np.int32)];
            cv2.drawContours(mask_split, cnt_split, 0, 1, -1)
            
            mask_check = np.zeros(roi_size, np.int32)
            cnt_check = [(cnt_check-bot).astype(np.int32)];
            cv2.drawContours(mask_check, cnt_check, 0, 1, -1)
            
            area_intersect = np.sum(mask_check & mask_split)
            area_check = np.sum(mask_check)
            
            area_overlap[key_tuple] = area_intersect/area_check
            
    points2split = {}
    for x in area_overlap:
        if area_overlap[x]>0.5:
            if not x[0] in points2split: 
                points2split[x[0]] = []
            t_split = max(x[1],x[3]);
            if not t_split in points2split[x[0]]:
                points2split[x[0]].append(t_split)
    
    
    
    print('S', len(points2split))    
#%%
    #def splitTrajectories(trajectories_data_f, points2split):
    
    last_index = trajectories_data_f['worm_index_auto'].max()
    traj_grouped_ind = trajectories_data_f.groupby('worm_index_auto')
    for worm_ind in points2split:
        worm_dat = traj_grouped_ind.get_group(worm_ind)
        
        frames = worm_dat['frame_number']           
        
        last_index += 1
        new_index = pd.Series(last_index, index=frames.index)            
        for t_split in sorted(points2split[worm_ind]):
            last_index += 1
            new_index[frames>=t_split] += 1
            
        assert len(np.unique(new_index)) == len(points2split[worm_ind])+1
        assert np.all(trajectories_data_f.loc[new_index.index, 'worm_index_auto'] == worm_ind)
        
        trajectories_data_f.loc[new_index.index, 'worm_index_auto'] = new_index
#%%
#            new_ind1 = last_index + 1
#            new_ind2 = last_index + 2
#            last_index =  new_ind2
#            
#            worm_dat = traj_grouped_ind.get_group(worm_ind)
#            
#            frames = worm_dat['frame_number'].sort_values(inplace=False)
#            good = frames < split_t
#            traj1_rows = frames[good].index
#            traj2_rows = frames[~good].index
#            
#            assert np.all(trajectories_data_f.ix[traj1_rows, 'worm_index_auto'] == worm_ind)
#            assert np.all(trajectories_data_f.ix[traj2_rows, 'worm_index_auto'] == worm_ind)
#            
#            trajectories_data_f.loc[traj1_rows, 'worm_index_auto'] = new_ind1
#            trajectories_data_f.loc[traj2_rows, 'worm_index_auto'] = new_ind2
    
    #trajectories_data_f = splitTrajectories(trajectories_data_f, points2split)
#        #print('L', len(trajectories_data_f))
#%%
    max_gap = 25
    print('Getting the trajectories starting and ending points.')
    traj_limits = getStartEndTraj(trajectories_data_f) 
    
    print('Getting possible connecting point.')
    connect_before, connect_after = getPossibleConnections(traj_limits, max_gap = 25)
    
    print('Extracting worm contours from trajectory limits.')
    initial_cnt, final_cnt = extractWormContours(masked_image_file, traj_limits)

    print('Looking for overlaping fraction between contours.')
    after_ratio = getAreaIntersecRatio(connect_after, final_cnt, initial_cnt)
    before_ratio = getAreaIntersecRatio(connect_before, initial_cnt, final_cnt)
    
    
    
    #maybe a graph reduction algorithm would work better...
    #%%
    print('Getting connections between trajectories.')    
    edges_after = selectNearNodes(connect_after, after_ratio, traj_limits['t0'], min_intersect = 0.5)
    edges_before = selectNearNodes(connect_before, before_ratio, -traj_limits['tf'], min_intersect = 0.5)
    #switch so the lower index is first    
    edges_before = [(y,x) for x,y in edges_before]
    
    #get unique nodes
    trajectories_edges = set(edges_after+edges_before)
    #%%
    #print('Removing redundant connections.')
    DG=nx.DiGraph()
    DG.add_nodes_from(traj_limits.index)
    DG.add_edges_from(trajectories_edges)
    
    #remove nodes that bypass earlier childs in the trajectory
    for n1 in DG.nodes():
        for n2 in DG.successors(n1):
             for path in nx.all_simple_paths(DG, source=n1, target=n2, cutoff=5):
                 if len(path) > 2:
                     #there is anothr way to arrive to this node we miss an earlier child
                     DG.remove_edge(n1,n2)
                     break
    #%%
    DG, trajectories_data_f = cleanRedundantNodes(DG, trajectories_data_f)
    
    
    #%%
    with pd.HDFStore(skeletons_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    
    trajectories_data['worm_index_auto'] = trajectories_data_f['worm_index_auto']
    trajectories_data.loc[np.isnan(trajectories_data['worm_index_auto']), 'worm_index_auto'] = -1 


    #%%
    worm_indexes = getRealWormsIndexes(trajectories_data, n_min_skel = 5, min_frac_skel = 0.25)
    possible_cluster = getPossibleClusters(DG, worm_indexes)

    good_worms = worm_indexes - possible_cluster
    good_clusters = possible_cluster - worm_indexes

    bad_index = []
    for subgraph in nx.connected_component_subgraphs(DG.to_undirected()):
        subgraph_nodes = subgraph.nodes()
        if not any(x in worm_indexes for x in subgraph_nodes):
            bad_index += subgraph_nodes

    #%%
    trajectories_data['auto_label'] = WLAB['U']
    
    good = trajectories_data.worm_index_auto.isin(good_worms)
    trajectories_data.loc[good, 'auto_label'] = WLAB['WORM']
    
    good = trajectories_data.worm_index_auto.isin(good_clusters)
    trajectories_data.loc[good, 'auto_label'] = WLAB['WORMS']
    
    good = trajectories_data.worm_index_auto.isin(bad_index)
    trajectories_data.loc[good, 'auto_label'] = WLAB['BAD']

    #let's save this data into the skeletons file
    saveModifiedTrajData(skeletons_file, trajectories_data)


#%%

#    #create a dictionary to map from old to new indexes  
#    dd = trajectories_data.groupby('worm_index_auto').agg({'frame_number':'min'})
#    dd = dd.to_records()
#    dd.sort(order=['frame_number', 'worm_index_auto'])
#    new_ind_dict = {x:ii+1 for ii,x in enumerate(dd['worm_index_auto'])}
#    
#    #replace the data from the new indexes (pandas replace do not work because the dict and keys values must be different)
#    worm_index_auto = trajectories_data['worm_index_auto'].values
#    worm_index_auto = [new_ind_dict[x] for x in worm_index_auto]
#    trajectories_data['worm_index_auto'] = worm_index_auto
#%%
         
