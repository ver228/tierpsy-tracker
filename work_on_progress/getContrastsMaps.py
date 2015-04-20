# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:39:41 2015

@author: ajaver
"""
import pandas as pd
#import matplotlib.pylab as plt
import numpy as np
import h5py
import cv2
import time
import tables
import matplotlib.pylab as plt

from calContrastMaps import calContrastMaps

masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/CaptureTest_90pc_Ch1_02022015_141431.hdf5';
trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_trajectories.hdf5';
segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_segworm.hdf5';
contrastmap_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_cmap.hdf5';

ROI_SIZE = 128;

#open the hdf5 with the masked images
mask_fid = h5py.File(masked_image_file, 'r');
mask_dataset = mask_fid["/mask"]

    #read that frame an select trajectories that were considered valid by join_trajectories
table_fid = pd.HDFStore(trajectories_file, 'r');
df = table_fid['/plate_worms'];
df =  df[df['worm_index_joined'] > 0]
good_index = df[df['segworm_id']>=0]['worm_index_joined'].unique();
df = df[df.worm_index_joined.isin(good_index)];
table_fid.close()

results_fid = tables.open_file(trajectories_file, 'r')
tracking_table = results_fid.get_node('/plate_worms')

segworm_fid = h5py.File(segworm_file, 'r')

map_R_range = 120;
map_pos_range = 511;
map_neg_range = 256;
N_pix_HT = 20; #segment size taken for the head and the tail

tot_segworm = segworm_fid['/segworm_results/skeleton'].shape[0]
contrastmap_fid = h5py.File(contrastmap_file, 'w');

maps_ID = {};

for key in ['worm', 'worm_H', 'worm_T', 'worm_V', 'worm_D']:
    key_map = key + "_pos";
    maps_ID[key_map] = contrastmap_fid.create_dataset("/"+key_map , (tot_segworm, map_R_range, map_pos_range), 
                                        dtype = np.int, maxshape = (tot_segworm, map_R_range, map_pos_range), 
                                        chunks = (1, map_R_range, map_pos_range),
                                        compression="lzf", shuffle=True);
    key_map = key + "_neg";
    maps_ID[key_map] = contrastmap_fid.create_dataset("/"+key_map , (tot_segworm, map_R_range, map_neg_range), 
                                        dtype = np.int, maxshape = (tot_segworm, map_R_range, map_neg_range), 
                                        chunks = (1, map_R_range, map_neg_range),
                                        compression="lzf", shuffle=True);
    

tic = time.time()
tic_first = tic
for frame in range(0, df['frame_number'].max()):
    
    img = mask_dataset[frame,:,:]
    
    for (ii, worm) in df[df.frame_number==frame+1].iterrows():
        
        worm_index = int(worm['worm_index_joined']); 
        segworm_id = worm['segworm_id']
        if segworm_id == -1:
            continue
        #initialize dictionary
        
        range_x = np.round(worm['coord_x']) + [-ROI_SIZE/2, ROI_SIZE/2]
        range_y = np.round(worm['coord_y']) + [-ROI_SIZE/2, ROI_SIZE/2]
        
        if (range_y[0] <0) or (range_y[0]>= img.shape[0]) or \
        (range_x[0] <0) or (range_x[0]>= img.shape[1]):
            continue
        
        worm_img =  img[range_y[0]:range_y[1], range_x[0]:range_x[1]]
        
        dat = {}
        for key in ['contour_ventral', 'skeleton', 'contour_dorsal']:
            dat[key] = np.squeeze(segworm_fid['/segworm_results/' + key][segworm_id,:,:]);
            dat[key][1,:] = dat[key][1,:]-range_x[0]
            dat[key][0,:] = dat[key][0,:]-range_y[0]
        
        contours = {};
        contours['worm'] = np.hstack((dat['contour_ventral'][::-1,:],dat['contour_dorsal'][::-1,::-1]));
        contours['worm_H'] = np.hstack((dat['contour_ventral'][::-1,0:N_pix_HT],dat['contour_dorsal'][::-1,N_pix_HT-1::-1]));
        contours['worm_T'] = np.hstack((dat['contour_ventral'][::-1,-N_pix_HT:],dat['contour_dorsal'][::-1,:-N_pix_HT-1:-1]));
        contours['worm_V'] = np.hstack((dat['contour_ventral'][::-1,:],dat['skeleton'][::-1,::-1]));
        contours['worm_D'] = np.hstack((dat['contour_dorsal'][::-1,:],dat['skeleton'][::-1,::-1]));
        
        #print frame, ii
        for key in contours:
            worm_mask = np.zeros(worm_img.shape)
            cc = [contours[key].astype(np.int32).T];
            cv2.drawContours(worm_mask, cc, 0, 1, -1)
            pix_list = np.where(worm_mask==1);
            pix_val = worm_img[pix_list].astype(np.int);
            pix_dat = np.array((pix_list[0], pix_list[1], pix_val))
            
            #print key, pix_dat.shape
            Ipos, Ineg = calContrastMaps(pix_dat, map_R_range, map_pos_range, map_neg_range);
            maps_ID[key + "_pos"][segworm_id,:,:] = Ipos.copy()
            maps_ID[key + "_neg"][segworm_id,:,:] = Ineg.copy()
            
#    
    results_fid.flush()
    if frame%25 == 0:
        print frame, time.time() - tic
        tic = time.time()
    
mask_fid.close()
results_fid.close()
contrastmap_fid.close()
