#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:55:14 2017

@author: ajaver
"""

import numpy as np
import matplotlib.path as mplPath

from getFoodContourNN import get_food_contour_nn
from getFoodContourMorph import get_food_contour_morph

from tierpsy.analysis.feat_create.obtainFeatures import getGoodTrajIndexes
from tierpsy.analysis.feat_create.obtainFeaturesHelper import WormFromTableSimple
from tierpsy.helper.params import read_microns_per_pixel


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
                  food_r, 
                  food_centroid,
                  is_debug = False):
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
    #%%
    get_unit_vec = lambda x : x/np.linalg.norm(x, axis=1)[:, np.newaxis]
    
    top = cnt_ind+1
    top[top>=food_cnt.shape[0]] -= food_cnt.shape[0] #fix any overflow index
    bot = cnt_ind-1 #it is not necessary to correct because we can use negative indexing
    
    food_u =  get_unit_vec(food_cnt[top]-food_cnt[bot])
    
    #%%
    worm_u = get_unit_vec(skel_avg['head_base'] - skel_avg['tail_base'])
    
    
    
    
    dot_prod = np.sum(food_u*worm_u, axis=1)
    orientation_food_cnt = np.arccos(dot_prod)*180/np.pi
    
    if is_debug:
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
        
    
    return orientation_food_cnt, dist_from_cnt, cnt_ind

#%%
def calculate_food_cnt(mask_video, method='NN', _is_debug=False):
    
    assert method in ['NN', 'MORPH']
    if method == 'NN':
        food_cnt, food_prob, cnt_solidity = get_food_contour_nn(mask_file, _is_debug=True)
        if cnt_solidity < 0.98:
            food_cnt = np.zeros([])
        
    elif method == 'MORPH':
        food_cnt = get_food_contour_morph(mask_file, _is_debug=_is_debug)
    else:
        raise ValueError('Invalid method argument.')
    
    microns_per_pixel = read_microns_per_pixel(mask_video)
    food_cnt *= microns_per_pixel
    #polar coordinates from the centroid
    food_centroid = np.mean(food_cnt, axis=0)
    food_r = np.linalg.norm(food_cnt-food_centroid, axis=1)
    
    return food_cnt, food_r, food_centroid
#%%
if __name__ == '__main__':
    from tierpsy.helper.misc import TimeCounter, print_flush
    #mask_file = '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/short_movies_new/MaskedVideos/Double_picking_020317/trp-4_worms6_food1-3_Set4_Pos5_Ch3_02032017_153225.hdf5'
    #mask_file = '/Users/ajaver/OneDrive - Imperial College London/optogenetics/Arantza/MaskedVideos/oig8/oig-8_ChR2_control_males_3_Ch1_11052017_161018.hdf5'
    mask_file = '/Users/ajaver/OneDrive - Imperial College London/optogenetics/Arantza/MaskedVideos/control_pulse/pkd2_5min_Ch1_11052017_121414.hdf5'
    #%%
    skeletons_file = mask_file.replace('MaskedVideos','Results').replace('.hdf5', '_skeletons.hdf5')
    
    food_cnt, food_r, food_centroid = calculate_food_cnt(mask_file)
    
    good_traj_index, worm_index_type = getGoodTrajIndexes(skeletons_file)
    for iw, worm_index in enumerate(good_traj_index):
        worm = WormFromTableSimple(skeletons_file,
                            worm_index,
                            worm_index_type=worm_index_type)
        
        orientation_food_cnt, dist_from_cnt, cnt_ind = \
        get_cnt_feats(worm.skeleton, 
              food_cnt, 
              food_r, 
              food_centroid,
              is_debug = True)
            
             
            #skeletons = worm.skeleton
            #is_debug = True
    #%%
#    food_cnt_nn, _, _ = calculate_food_cnt(mask_file, method='NN', _is_debug=False)
#    food_cnt_morph, _, _ = calculate_food_cnt(mask_file, method='MORPH', _is_debug=False)
#    
#    import matplotlib.pylab as plt
#    import tables
#    with tables.File(mask_file, 'r') as fid:
#        bgnd_o = fid.get_node('/full_data')[0]
#    plt.figure()
#    plt.imshow(bgnd_o, cmap='gray')
#    plt.plot(food_cnt_nn[:,0], food_cnt_nn[:,1], '.')
#    plt.plot(food_cnt_morph[:,0], food_cnt_morph[:,1], '.')
#    plt.grid('off')
    #%%

    #calculate the solidity, it is a good way to filter bad contours
    hull = cv2.convexHull(cnts[ind])
    hull_area = cv2.contourArea(hull)
    cnt_solidity = cnt_areas[ind]/hull_area
    
    
    
    