# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:03:15 2015

@author: ajaver
"""

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
#from image_difference import image_difference

masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/CaptureTest_90pc_Ch1_02022015_141431.hdf5';
trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_trajectories.hdf5';
segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_segworm.hdf5';
contrastmap_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_cmap-short.hdf5';

MAX_DELT = 5;
jump_frames = 3; #3
del_frames = 100; #100
ROI_SIZE = 128;


table_fid = pd.HDFStore(trajectories_file, 'r');
df = table_fid['/plate_worms'];
df =  df[df['worm_index_joined'] > 0]
df = df[df['segworm_id']>=0];
table_fid.close()

#track_counts = df['worm_index_joined'].value_counts()

all_ind_ini = {};
all_ind_last = {};

all_ind_block = [];
for worm_ind in df.worm_index_joined.unique():
    worm = df[(df['worm_index_joined']==worm_ind)]
    delTs = np.diff(worm['frame_number']);
    block_ind = np.zeros(len(worm), dtype = np.int);
    
    block_ind[0] = 1;
    for ii, delT in enumerate(delTs):
        if delT <= MAX_DELT:
            block_ind[ii+1] = block_ind[ii];
        else:
            block_ind[ii+1] = block_ind[ii]+1;
    
    
    worm_segworm_id = worm['segworm_id'].values  
    plate_worms_id = worm.index.values;
    
    all_ind_block += zip(*[block_ind.size*[worm_ind], list(block_ind), list(plate_worms_id), list(worm_segworm_id)])
    
    block_ind_ini = {};
    block_ind_last = {};
    for ii in range(1, block_ind[-1]+1):
        curr_block = plate_worms_id[block_ind==ii]
        
        if curr_block.size/jump_frames > del_frames:
            ini_block = curr_block[:del_frames*jump_frames:jump_frames]
            last_block = curr_block[-del_frames*jump_frames::jump_frames]
        else:
            ini_block = curr_block[:del_frames]
            last_block = curr_block[-del_frames:]
        
        block_ind_ini[ii] = ini_block
        block_ind_last[ii] = last_block
    
    all_ind_ini[worm_ind] = block_ind_ini;
    all_ind_last[worm_ind] = block_ind_last;

ind_list_ini = [];
ind_list_last = [];
for key1 in all_ind_ini:
    for key2 in all_ind_ini[key1]:
        for ind in all_ind_ini[key1][key2]:
            ind_list_ini.append((key1,key2, ind))
        
        for ind in all_ind_last[key1][key2]:
            ind_list_last.append((key1,key2, ind))

#this sorting should speed up reading operations for h5py 
ind_list_ini = sorted(ind_list_ini, key = lambda indexes : indexes[0]);
ind_list_last = sorted(ind_list_last, key = lambda indexes : indexes[0]);
#%%
ind_list = zip(*ind_list_ini)[-1]+zip(*ind_list_last)[-1]
ind_list, cmap_ind  = np.unique(ind_list, return_inverse=True)

ind_list_ini = zip(*ind_list_ini)+[tuple(cmap_ind[:len(ind_list_ini)])]
ind_list_last = zip(*ind_list_last)+[tuple(cmap_ind[len(ind_list_last):])]
##
contrastmap_fid = tables.File(contrastmap_file, 'w');

ind_block_table = contrastmap_fid.create_table('/', 'block_index', {
    'worm_index_joined':    tables.Int32Col(pos=0),
    'block_id'         : tables.Int32Col(pos=1),
    'plate_worms_id'   : tables.Int32Col(pos=2),
    'segworm_id'         : tables.Int32Col(pos=3)
    }, filters = tables.Filters(complevel=5, complib='zlib', shuffle=True))

ind_ini_table = contrastmap_fid.create_table('/', 'block_initial', {
    'worm_index_joined':    tables.Int32Col(pos=0),
    'block_id'         : tables.Int32Col(pos=1),
    'plate_worms_id'   : tables.Int32Col(pos=2),
    'cmap_id'         : tables.Int32Col(pos=3)
    }, filters = tables.Filters(complevel=5, complib='zlib', shuffle=True))
ind_last_table = contrastmap_fid.create_table('/', 'block_last', {
    'worm_index_joined':    tables.Int32Col(pos=0),
    'block_id'         : tables.Int32Col(pos=1),
    'plate_worms_id'   : tables.Int32Col(pos=2),
    'cmap_id'         : tables.Int32Col(pos=3)
    }, filters = tables.Filters(complevel=5, complib='zlib', shuffle=True))
ind_ini_table.append(zip(*ind_list_ini));
ind_last_table.append(zip(*ind_list_last));
ind_block_table.append(all_ind_block);


contrastmap_fid.flush()
contrastmap_fid.close()

table_fid = pd.HDFStore(trajectories_file, 'r');
df = table_fid['/plate_worms'];
df = df.irow(ind_list);
df.loc[:,'cmaps_id'] = np.arange(len(df))
table_fid.close();


#%%
#open the hdf5 with the masked images
mask_fid = h5py.File(masked_image_file, 'r');
mask_dataset = mask_fid["/mask"]

results_fid = tables.open_file(trajectories_file, 'r')
tracking_table = results_fid.get_node('/plate_worms')

segworm_fid = h5py.File(segworm_file, 'r')

map_R_range = 120;
map_pos_range = 511;
map_neg_range = 256;
N_pix_HT = 20; #segment size taken for the head and the tail
N_pix_DV = 10; #segment size taken for the head and the tail

tot_maps = len(df)#segworm_fid['/segworm_results/skeleton'].shape[0]

contrastmap_fid = h5py.File(contrastmap_file, 'r+');
#%%


maps_ID = {};

for key in ['worm_H', 'worm_T', 'worm_V', 'worm_D']:
    key_map = key + "_pos";
    
    maps_ID[key_map] = contrastmap_fid.create_dataset("/"+key_map , (tot_maps, map_R_range, map_pos_range), 
                                        dtype = np.int, maxshape = (tot_maps, map_R_range, map_pos_range), 
                                        chunks = (1, map_R_range, map_pos_range),
                                        compression="lzf", shuffle=True);
    key_map = key + "_neg";
    maps_ID[key_map] = contrastmap_fid.create_dataset("/"+key_map , (tot_maps, map_R_range, map_neg_range), 
                                        dtype = np.int, maxshape = (tot_maps, map_R_range, map_neg_range), 
                                        chunks = (1, map_R_range, map_neg_range),
                                        compression="lzf", shuffle=True);

tic = time.time()
tic_first = tic
for frame, wormsInFrame in df.groupby('frame_number'):
    frame
    img = mask_dataset[frame,:,:]
    
    for ii, worm in wormsInFrame.iterrows():
        worm_index = int(worm['worm_index_joined']); 
        cmap_id = int(worm['cmaps_id'])
        segworm_id = worm['segworm_id']

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

        #initialize dictionary
        contours = {};
        contours['worm_H'] = np.hstack((dat['contour_ventral'][::-1,0:N_pix_HT],dat['contour_dorsal'][::-1,N_pix_HT-1::-1]));
        contours['worm_T'] = np.hstack((dat['contour_ventral'][::-1,-N_pix_HT:],dat['contour_dorsal'][::-1,:-N_pix_HT-1:-1]));
        contours['worm_V'] = np.hstack((dat['contour_ventral'][::-1,N_pix_DV:-N_pix_DV],dat['skeleton'][::-1,-N_pix_DV:N_pix_DV:-1]));
        contours['worm_D'] = np.hstack((dat['contour_dorsal'][::-1,N_pix_DV:-N_pix_DV],dat['skeleton'][::-1,-N_pix_DV:N_pix_DV:-1]));
        
        for key in contours:
            worm_mask = np.zeros(worm_img.shape)
            cc = [contours[key].astype(np.int32).T];
            cv2.drawContours(worm_mask, cc, 0, 1, -1)
            pix_list = np.where(worm_mask==1);
            pix_val = worm_img[pix_list].astype(np.int);
            pix_dat = np.array((pix_list[0], pix_list[1], pix_val))
            
            #print key, pix_dat.shape
            Ipos, Ineg = calContrastMaps(pix_dat, map_R_range, map_pos_range, map_neg_range);
            maps_ID[key + "_pos"][cmap_id,:,:] = Ipos.copy()
            maps_ID[key + "_neg"][cmap_id,:,:] = Ineg.copy()
##    
    results_fid.flush()
    if frame%25 == 0:
        print frame, time.time() - tic
        tic = time.time()
    
mask_fid.close()
results_fid.close()
contrastmap_fid.close()


