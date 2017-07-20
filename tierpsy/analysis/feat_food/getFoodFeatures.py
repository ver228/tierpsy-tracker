#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:55:14 2017

@author: ajaver
"""

import tables
import numpy as np
import matplotlib.path as mplPath
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from tierpsy.analysis.feat_food.getFoodContourNN import get_food_contour_nn
from tierpsy.analysis.feat_food.getFoodContourMorph import get_food_contour_morph

from tierpsy.helper.params import read_microns_per_pixel
from tierpsy.helper.misc import TABLE_FILTERS, remove_ext, \
TimeCounter, print_flush, get_base_name

def get_worm_partitions(n_segments=49):
    worm_partitions = { 'head': (0, 8),
                        'neck': (8, 16),
                        'midbody': (16, 33),
                        'hips': (33, 41),
                        'tail': (41, 49),
                        'head_tip': (0, 3),
                        'head_base': (5, 8),
                        'tail_base': (41, 44),
                        'tail_tip': (46, 49),
                        'all': (0, 49),
                        'body': (8, 41)
                        }
    
    if n_segments != 49:
        r_fun = lambda x : int(round(x/49*n_segments))
        for key in worm_partitions:
            worm_partitions[key] = tuple(map(r_fun, worm_partitions[key]))
        
    return worm_partitions

def get_partition_transform(data, func=np.mean, partitions=None):
    n_segments = data.shape[1]
    worm_partitions = get_worm_partitions(n_segments)
    if partitions is None:
        partitions = worm_partitions.keys()
    
    data_transformed = {}
    for pp in partitions:
        ini, fin = worm_partitions[pp]
        data_transformed[pp] = func(data[:, ini:fin, :], axis=1)
    return data_transformed
    
def get_cnt_feats(skeletons, 
                  food_cnt,
                  _is_debug = False):
    #%%
    partitions = ['head_base', 'tail_base', 'midbody']
    skel_avg = get_partition_transform(skeletons,
                                       func=np.mean,
                                       partitions=partitions
                                       )
    
    midbody_cc = skel_avg['midbody']
    
    def _get_food_ind(cc):
        rr = np.linalg.norm(cc - food_cnt, axis=1)
        ip = np.argmin(rr)
        rp = rr[ip]
        return ip, rp
    
    cnt_ind, dist_from_cnt = map(np.array, zip(*map(_get_food_ind, midbody_cc)))
    
    #find if the trajectory points are inside the closed polygon (outside will be negative)
    bbPath = mplPath.Path(food_cnt)
    outside = ~bbPath.contains_points(midbody_cc)
    dist_from_cnt[outside] = -dist_from_cnt[outside]
    
    get_unit_vec = lambda x : x/np.linalg.norm(x, axis=1)[:, np.newaxis]
    
    top = cnt_ind+1
    top[top>=food_cnt.shape[0]] -= food_cnt.shape[0] #fix any overflow index
    bot = cnt_ind-1 #it is not necessary to correct because we can use negative indexing
    
    food_u =  get_unit_vec(food_cnt[top]-food_cnt[bot])
    worm_u = get_unit_vec(skel_avg['head_base'] - skel_avg['tail_base'])
    
    dot_prod = np.sum(food_u*worm_u, axis=1)
    orientation_food_cnt = np.arccos(dot_prod)*180/np.pi
    #%%
    if _is_debug:
        import matplotlib.pylab as plt
        plt.figure(figsize=(12,12))
        
        plt.subplot(2,2,2)
        plt.plot(orientation_food_cnt)
        plt.title('Orientation respect to the food contour')
        
        plt.subplot(2,2,4)
        plt.plot(dist_from_cnt)
        plt.title('Distance from the food contour')
        
        plt.subplot(1,2,1)
        plt.plot(food_cnt[:,0], food_cnt[:,1])
        plt.plot(midbody_cc[:,0], midbody_cc[:,1], '.')
        plt.plot(food_cnt[cnt_ind,0], food_cnt[cnt_ind,1], 'r.')
        plt.axis('equal')
    #%%  
    
    return orientation_food_cnt, dist_from_cnt, cnt_ind
#%%
def _is_valid_cnt(x):
    return x is not None and \
           x.size >= 2 and \
           x.ndim ==2 and \
           x.shape[1] == 2
    

#%%
def calculate_food_cnt(mask_file, method='NN', _is_debug=False, solidity_th=0.98):
    
    assert method in ['NN', 'MORPH']
    if method == 'NN':
        food_cnt, food_prob,cnt_solidity = get_food_contour_nn(mask_file, _is_debug=_is_debug)
        if cnt_solidity < solidity_th:
            food_cnt = np.zeros(0)
        
        
    elif method == 'MORPH':
        food_cnt = get_food_contour_morph(mask_file, _is_debug=_is_debug)
    else:
        raise ValueError('Invalid method argument.')
    
    #transform contour from pixels to microns
    microns_per_pixel = read_microns_per_pixel(mask_file)
    food_cnt *= microns_per_pixel
    
    #smooth contours
    food_cnt = smooth_cnt(food_cnt, _is_debug=_is_debug)
    
    return food_cnt

def smooth_cnt(food_cnt, resampling_N = 1000, smooth_window=None, _is_debug=False):
    if smooth_window is None:
        smooth_window = resampling_N//20
    
    if not _is_valid_cnt(food_cnt):
        #invalid contour arrays
        return food_cnt
        
    smooth_window = smooth_window if smooth_window%2 == 1 else smooth_window+1
    # calculate the cumulative length for each segment in the curve
    dx = np.diff(food_cnt[:, 0])
    dy = np.diff(food_cnt[:, 1])
    dr = np.sqrt(dx * dx + dy * dy)
    lengths = np.cumsum(dr)
    lengths = np.hstack((0, lengths))  # add the first point
    tot_length = lengths[-1]
    fx = interp1d(lengths, food_cnt[:, 0])
    fy = interp1d(lengths, food_cnt[:, 1])
    subLengths = np.linspace(0 + np.finfo(float).eps, tot_length, resampling_N)
    
    rx = fx(subLengths)
    ry = fy(subLengths)
    
    pol_degree = 3
    rx = savgol_filter(rx, smooth_window, pol_degree)
    ry = savgol_filter(ry, smooth_window, pol_degree)
    
    food_cnt_s = np.stack((rx, ry), axis=1)
    
    if _is_debug:
        import matplotlib.pylab as plt
        plt.figure()
        plt.plot(food_cnt[:, 0], food_cnt[:, 1], '.-')
        plt.plot(food_cnt_s[:, 0], food_cnt_s[:, 1], '.-')
        plt.axis('equal')
        plt.title('smoothed contour')
    
    return food_cnt_s

def getFoodFeatures(mask_file, 
                    skeletons_file,
                    features_file = None,
                    cnt_method = 'NN',
                    solidity_th=0.98,
                    batch_size = 100000,
                    _is_debug = False
                    ):
    if features_file is None:
        features_file = remove_ext(skeletons_file) + '_featuresN.hdf5'
    
    base_name = get_base_name(mask_file)
    
    progress_timer = TimeCounter('')
    print_flush("{} Calculating food features {}".format(base_name, progress_timer.get_time_str()))
    
    food_cnt = calculate_food_cnt(mask_file,  
                                  method=cnt_method, 
                                  solidity_th=solidity_th,
                                  _is_debug=_is_debug)
    microns_per_pixel = read_microns_per_pixel(skeletons_file)
    
    #store contour coordinates in pixels into the skeletons file for visualization purposes
    food_cnt_pix = food_cnt/microns_per_pixel
    with tables.File(skeletons_file, 'r+') as fid:
        if '/food_cnt_coord' in fid:
            fid.remove_node('/food_cnt_coord')
        if _is_valid_cnt(food_cnt):
            tab = fid.create_array('/', 
                                   'food_cnt_coord', 
                                   obj=food_cnt_pix)
            tab._v_attrs['method'] = cnt_method
    
    print_flush("{} Calculating food features {}".format(base_name, progress_timer.get_time_str()))
    
    feats_names = ['orient_to_food_cnt', 'dist_from_food_cnt', 'closest_cnt_ind']
    feats_dtypes = [(x, np.float32) for x in feats_names]
    
    with tables.File(skeletons_file, 'r') as fid:
        tot_rows = fid.get_node('/skeleton').shape[0]
        features_df = np.full(tot_rows, np.nan, dtype = feats_dtypes)
        
        if food_cnt.size > 0:
            for ii in range(0, tot_rows, batch_size):
                skeletons = fid.get_node('/skeleton')[ii:ii+batch_size]
                skeletons *= microns_per_pixel
                
                outputs = get_cnt_feats(skeletons, food_cnt, _is_debug = _is_debug)
                for irow, row in enumerate(zip(*outputs)):
                    features_df[irow]  = row
    
    
    with tables.File(features_file, 'a') as fid: 
        if '/food' in fid:
            fid.remove_node('/food', recursive=True)
        fid.create_group('/', 'food')
        if _is_valid_cnt(food_cnt):
            fid.create_carray(
                    '/food',
                    'cnt_coordinates',
                    obj=food_cnt,
                    filters=TABLE_FILTERS) 
        
        fid.create_table(
                '/food',
                'features',
                obj=features_df,
                filters=TABLE_FILTERS) 
    #%%
    print_flush("{} Calculating food features {}".format(base_name, progress_timer.get_time_str()))
    
#%%
if __name__ == '__main__':
    
    #mask_file = '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/short_movies_new/MaskedVideos/Double_picking_020317/trp-4_worms6_food1-3_Set4_Pos5_Ch3_02032017_153225.hdf5'
    #mask_file = '/Users/ajaver/OneDrive - Imperial College London/optogenetics/Arantza/MaskedVideos/control_pulse/pkd2_5min_Ch1_11052017_121414.hdf5'
    mask_file = '/Users/ajaver/OneDrive - Imperial College London/aggregation/N2_1_Ch1_29062017_182108_comp3.hdf5'
    mask_file = '/Users/ajaver/Tmp/MaskedVideos/Development_C1_170617/development_c1_Set0_Pos0_Ch2_17062017_123028.hdf5'
    skeletons_file = mask_file.replace('MaskedVideos','Results').replace('.hdf5', '_skeletons.hdf5')
    getFoodFeatures(mask_file, skeletons_file, _is_debug = False)



