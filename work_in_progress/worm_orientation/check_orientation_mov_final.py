# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:37:12 2016

@author: ajaver
"""

import tables
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import glob, os

from MWTracker.trackWorms.checkHeadOrientation import isWormHTSwitched
from MWTracker.intensityAnalysis.getIntensityProfile import getWidthWinLimits

peak_search_limits = [0.054, 0.192, 0.269, 0.346]

def searchIntPeaks(median_int, d_search = [7, 25, 35, 45]):
    length_resampling = median_int.shape[0]
    
    peaks_ind = []
    hh = 0
    tt = length_resampling
    for ii, ds in enumerate(d_search):
        func_search = np.argmin if ii % 2 == 0 else np.argmax
        
        hh = func_search(median_int[hh:ds]) + hh
        
        dd = length_resampling-ds
        tt = func_search(median_int[dd:tt]) + dd
        
        peaks_ind.append((hh,tt))
    return peaks_ind
    
    
    
def wholeWormHeadTail(skeletons_file, intensities_file):
    with pd.HDFStore(trajectories_file, 'r') as fid:
        plate_worms = fid['/plate_worms']
    
    with pd.HDFStore(skeletons_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    
    with tables.File(skeletons_file, 'r') as fid:
        skeletons = fid.get_node('/skeleton')[:]
    
    with tables.File(intensities_file, 'r') as fid:
        worm_int = fid.get_node('/straighten_worm_intensity_median')[:].astype(np.float)
        #worm_int_map = fid.get_node('/straighten_worm_intensity')[:].astype(np.float)

    worm_int -= np.median(worm_int, axis=1)[:, np.newaxis]

    #%%
    is_switch_skel, roll_std = isWormHTSwitched(skeletons, segment4angle = 5, max_gap_allowed = 10, \
                         window_std = 25, min_block_size=250)
    head_angle = np.nanmedian(roll_std['head_angle'])
    tail_angle = np.nanmedian(roll_std['tail_angle'])
    
    p_mov = head_angle/(head_angle + tail_angle)
    
    #%%
    
    median_int = np.median(worm_int, axis=0)
    all_median.append((base_name, median_int))
    
    peaks_ind = searchIntPeaks(median_int, d_search = [7, 25, 35, 45])
    
    headbot2neck = median_int[peaks_ind[3][0]] - median_int[peaks_ind[2][0]]
    headbot2neck = 0 if headbot2neck < 0 else headbot2neck
    
    tailbot2waist = median_int[peaks_ind[3][1]] - median_int[peaks_ind[2][1]]
    tailbot2waist = 0 if tailbot2waist < 0 else tailbot2waist
    
    p_int_bot = headbot2neck/(headbot2neck+tailbot2waist)
    
    headtop2bot = median_int[peaks_ind[1][0]] - median_int[peaks_ind[2][0]]
    headtop2bot = 0 if headtop2bot < 0 else headtop2bot
    
    tailtop2bot = median_int[peaks_ind[1][1]] - median_int[peaks_ind[2][1]]
    tailtop2bot = 0 if tailtop2bot < 0 else tailtop2bot
    p_int_top = headtop2bot/(headtop2bot+tailtop2bot)
    
    p_tot = 0.75*p_mov + 0.15*p_int_bot + 0.1*p_int_top
    
    #if it is nan, both int changes where negatives, equal the probability to the p_mov
    if p_tot != p_tot:
        p_tot = p_mov
    
    print('M %2.2f | T1 %2.2f | T2 %2.2f | tot %2.2f' % (p_mov, p_int_bot, p_int_top, p_tot))
    
    #%%
    plt.figure()
    plt.title(base_name)
    plt.plot(median_int, label ='0.3')
    
    strC = 'rgck'
    for ii, dd in enumerate(peaks_ind):
        for xx in dd:
            plt.plot(xx, median_int[xx], 'o' + strC[ii])
        
    
    
    
    
    #%%    


#%%
#                         
#    offset_a, offset_b = 5, 20
#    offset_c = offset_b + 2
#    offset_d =   offset_c + (offset_b-offset_a)
#    int_range = np.max(worm_int, axis=1) - np.min(worm_int, axis=1)    
#    
#    #the head is likely to have a peak so we take the maximum
#    head_int = np.max(worm_int[:, offset_a:offset_b], axis = 1)
#    tail_int = np.median(worm_int[:, -offset_b:-offset_a], axis= 1)
#    
#    #while the neck is more a plateau so we take the median
#    neck_int = np.max(worm_int[:, offset_c:offset_d], axis = 1)
#    waist_int = np.median(worm_int[:, -offset_d:-offset_c], axis= 1)
#    
#    
#    
#    A = np.nanmedian(head_angle/tail_angle)
#    I = np.nanmedian((head_int-tail_int)/int_range)
#    
#    In = np.nanmedian((head_int-neck_int)/int_range)
#    Iw = np.nanmedian((tail_int-waist_int)/int_range)
#    
#    print(base_name)
#    print('A: %f | I:% f |In:% f  |Iw:% f ' % (A, I, In, Iw))
#    #print('head_angle: %f : tail_angle %f' % (np.nanmedian(roll_std['head_angle']), np.nanmedian(roll_std['tail_angle'])))
#    #print('head_int: %f : tail_int %f' % (np.median(head_int), np.median(tail_int)))
#    print('')
    
    #%%
    
    #%%
    #signal = median_int[3:-3]
    #peakind_min, peakind_max = getExtrema(signal,0.1, min_dist)
    
    #max_head = signal[peakind_max[0]]
    #max_tail = signal[peakind_max[-1]]
    
    #range_int = (max(signal) - min(signal))
    
    #head_tail_ratio = (max_head-max_tail)/range_int


#%%
#for base_name,median_int in all_median:
    
#    min_dist = 10
#
#    plt.figure()
#    signal = median_int[3:-3]
#    peakind_min, peakind_max = getExtrema(signal,0.1, min_dist)
#    
#    max_head = signal[peakind_max[0]]
#    max_tail = signal[peakind_max[-1]]
#    
#    range_int = (max(signal) - min(signal))
#    
#    head_tail_ratio = (max_head-max_tail)/range_int
#    
#    good = (peakind_min >= peakind_max[0]+min_dist) & (peakind_min <= peakind_max[-1]-min_dist)
#    peakind_min = peakind_min[good]
#    
#    min_neck = signal[peakind_min[0]]
#    min_waist = signal[peakind_min[-1]]
#    
#    head_neck_ratio = (max_head-min_neck)/range_int
#    tail_waist_ratio = (max_tail-min_waist)/range_int
#    
#    print(head_tail_ratio, head_neck_ratio, tail_waist_ratio)    
#    print('')

#    plt.plot(signal, label ='0.3')
#    plt.plot(peakind_max, signal[peakind_max], 'or')
#    plt.plot(peakind_min, signal[peakind_min], 'og')
#    plt.title(base_name)
#    
    
    #%%
#    for mm in [5, 20, 22, 35]:
#        plt.plot((mm, mm), plt.gca().get_ylim(), 'r:')
#        nn = length_resampling-mm
#        plt.plot((nn, nn), plt.gca().get_ylim(), 'r:')
##    
        #%%
#    plt.figure()
#    plt.imshow(worm_int.T, interpolation='none', cmap='gray')
#    plt.grid('off')
#    plt.xlim((0, 1000))
#    #%%    
#    
#    wlim = getWidthWinLimits(15, 0.5)
#
#    worm_int2 = np.zeros_like(worm_int)
#        
#    for kk in range(worm_int_map.shape[0]):
#        worm_int2[kk,:] = np.median(worm_int_map[kk,:,wlim[0]:wlim[1]], axis=1)
#
#    worm_int2 -= np.median(worm_int2, axis=1)[:, np.newaxis]
    #%%
#    plt.figure()
#    #plt.imshow(worm_int_map[8000,:,wlim[0]:wlim[1]].T, interpolation='none', cmap='gray')
#    plt.imshow(worm_int2.T, interpolation='none', cmap='gray')
#    plt.grid('off')
#    plt.xlim((0, 1000))
#%%
    #plt.figure()
    #plt.plot(np.std(worm_int_map[:, 40:-40,:], axis=(1,2)))
#%%
#roll_std[['head_angle', 'tail_angle']].plot()
    

    #%%
    #if False:
    #%%
    #plt.figure()
    #plt.plot(head_int)
    #plt.plot(tail_int)
    #good_frames = trajectories_data.loc[trajectories_data['int_map_id']!=-1, 'frame_number'].values
    #convert_f = {ff:ii for ii,ff in enumerate(good_frames)}
    #%%
    #plt.figure()
    #plt.imshow(worm_int.T, interpolation='none', cmap='gray')
    #plt.grid('off')
#%%    
#    ini = convert_f[20873]
#    fin = convert_f[21033]
#    
#    plt.plot((ini, ini), plt.gca().get_ylim(), 'c:')
#    plt.plot((fin, fin), plt.gca().get_ylim(), 'c--')
#    plt.xlim((ini-100, fin+100)) 
#%%

#
#plt.figure()
##plt.imshow(worm_int_map[8000,:,wlim[0]:wlim[1]].T, interpolation='none', cmap='gray')
#plt.imshow(worm_int2.T, interpolation='none', cmap='gray')
#plt.grid('off')
#plt.xlim((5000, 6000))
#
if __main__ == '__name__':
    check_dir = '/Users/ajaver/Desktop/Videos/single_worm/swimming/MaskedVideos/'
    
    all_median = []
    for ff in glob.glob(os.path.join(check_dir, '*')):
        ff = ff.replace('MaskedVideos', 'Results')
        base_name = os.path.split(ff)[1].rpartition('.')[0]
        print(base_name)

        trajectories_file = ff[:-5] + '_trajectories.hdf5'
        skeletons_file = ff[:-5] + '_skeletons.hdf5'
        intensities_file = ff[:-5] + '_intensities.hdf5'
        
        try:
            with tables.File(skeletons_file, 'r') as fid:
                if fid.get_node('/skeleton')._v_attrs['has_finished'] != 4:
                    raise
        except:
            continue
    