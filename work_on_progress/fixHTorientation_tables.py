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
import operator
import matplotlib.pylab as plt

from calContrastMaps import calContrastMaps
#from image_difference import image_difference

masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/CaptureTest_90pc_Ch1_02022015_141431.hdf5';
trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_trajectories.hdf5';
segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_segworm.hdf5';
contrastmap_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_cmap-short.hdf5';

MAX_DELT = 5;
jump_frames = 5; #3
del_frames = 100; #100
ROI_SIZE = 128;


table_fid = pd.HDFStore(trajectories_file, 'r');
df = table_fid['/plate_worms'];
df =  df[df['worm_index_joined'] > 0]
df = df[df['segworm_id']>=0]; #select only rows with a valid segworm skeleton
table_fid.close()

#track_counts = df['worm_index_joined'].value_counts()

#tot_ind_ini = 0;
#tot_ind_last = 0;

all_ind_block = [];
for worm_ind in df['worm_index_joined'].unique():
    #select data from the same worm (trajectory)
    worm = df[(df['worm_index_joined']==worm_ind)].sort(columns='frame_number')
    
    #identify blocks as segments where the skeletons are separated by more than MAX_DELT frames    
    delTs = np.diff(worm['frame_number']); 
    block_ind = np.zeros(len(worm), dtype = np.int);    
    block_ind[0] = 1;
    for ii, delT in enumerate(delTs):
        if delT <= MAX_DELT:
            block_ind[ii+1] = block_ind[ii];
        else:
            block_ind[ii+1] = block_ind[ii]+1;
    
    #construct index for the blocks
    worm_segworm_id = worm['segworm_id'].values  
    plate_worms_id = worm.index.values;
    
    
        
    #identify segments that correspond to the begining or ending of a block
    
    block_ind_ini = np.ones_like(block_ind)*-1; 
    block_ind_last = np.ones_like(block_ind)*-1;
    for ii in range(1, block_ind[-1]+1):
        curr_block = np.where(block_ind==ii)[0]
        
        if curr_block.size/jump_frames > del_frames:
            ini_block = curr_block[:del_frames*jump_frames:jump_frames]
            last_block = curr_block[-del_frames*jump_frames::jump_frames]
        else:
            ini_block = curr_block[:del_frames]
            last_block = curr_block[-del_frames:]
        
        #vv_ini = range(tot_ind_ini, tot_ind_ini+ini_block.size)
        block_ind_ini[ini_block] = 1
        #tot_ind_ini = vv_ini[-1]
        
        #vv_last = range(tot_ind_last, tot_ind_last+ini_block.size)
        block_ind_last[last_block] = 1
        #tot_ind_last = vv_last[-1]
    
    all_ind_block += zip(*[block_ind.size*[worm_ind], list(block_ind), list(plate_worms_id), list(worm_segworm_id), list(block_ind_ini), list(block_ind_last)])

all_ind_block = sorted(all_ind_block, key = operator.itemgetter(0, 1)); #sort by worm index first, and the by block index
all_ind_block = zip(*all_ind_block)

for kk in range(4,6):
    all_ind_block[kk] = np.array(all_ind_block[kk])
    good = (np.array(all_ind_block[kk])>=0)
    all_ind_block[kk][good] = np.arange(np.sum(good))
    all_ind_block[kk] = list(all_ind_block[kk])

all_ind_block = zip(*all_ind_block)
#%%

#good = (np.array(all_ind_block[4])>=0)
#np.array(all_ind_block[4])>0 = np.arange(np.sum(good))
#cmap_id = np.ones_like(all_ind_block[4])*-1;
#cmap_id[good] = np.arange(np.sum(good));
#all_ind_block = all_ind_block + [cmap_id]; 
#all_ind_block = zip(*all_ind_block)

contrastmap_fid = tables.File(contrastmap_file, 'w');

ind_block_table = contrastmap_fid.create_table('/', 'block_index', {
    'worm_index_joined':    tables.Int32Col(pos=0),
    'block_id'         : tables.Int32Col(pos=1),
    'plate_worms_id'   : tables.Int32Col(pos=2),
    'segworm_id'         : tables.Int32Col(pos=3),
    'block_ini_id'     : tables.Int32Col(pos=4),
    'block_last_id'     : tables.Int32Col(pos=5)
    }, filters = tables.Filters(complevel=5, complib='zlib', shuffle=True))

ind_block_table.append(all_ind_block);

contrastmap_fid.flush()
contrastmap_fid.close()
#%%

#read the valid contrastmap data (again)
contrastmap_fid = pd.HDFStore(contrastmap_file, 'r');
df_map_ind = contrastmap_fid['/block_index'].query('block_ini_id>=0 | block_last_id>=0')
contrastmap_fid.close()

#read the trajectories data and add a column for the cmap index
trajectories_fid = pd.HDFStore(trajectories_file, 'r');
df = trajectories_fid['/plate_worms'];
df = df.irow(df_map_ind['plate_worms_id']);
df.loc[:, 'block_ini_id'] = pd.Series(df_map_ind['block_ini_id'].values, df_map_ind['plate_worms_id'].values)
df.loc[:, 'block_last_id'] = pd.Series(df_map_ind['block_last_id'].values, df_map_ind['plate_worms_id'].values)
#df.loc[:,'cmap_id'] = df_map_ind['cmap_id'].values #copy values otherwise it would try to mach the index


#%%
#open the hdf5 with the masked images
mask_fid = h5py.File(masked_image_file, 'r');
mask_dataset = mask_fid["/mask"]

segworm_fid = h5py.File(segworm_file, 'r')

map_R_range = 120;
map_range = {'pos':511, 'neg': 256};
N_pix_HT = 20; #segment size taken for the head and the tail
N_pix_DV = 10; #segment size taken for the head and the tail

tot_maps = {'block_ini':(df['block_last_id']>=0).sum(), 
            'block_last':(df['block_last_id']>=0).sum()};
assert df['block_ini_id'].max() + 1 == tot_maps['block_ini']
assert df['block_last_id'].max() + 1 == tot_maps['block_last']

contrastmap_fid = tables.File(contrastmap_file, 'r+');
#%%


maps_ID = {};

for key_block in ['block_ini', 'block_last']:
    maps_ID[key_block] = {}
    contrastmap_fid.create_group("/", key_block)
    for key_part in ['worm_H', 'worm_T', 'worm_V', 'worm_D']:
        for key_mapT in ["pos", "neg"]:
            key_map = key_part + '_'+ key_mapT;
            maps_ID[key_block][key_map] = contrastmap_fid.create_carray("/" + key_block, key_map , atom = tables.UInt32Atom(), 
                                                shape = (tot_maps[key_block], map_R_range, map_range[key_mapT]),
                                                filters = tables.Filters(complevel=5, complib='blosc', shuffle = True), 
                                                chunkshape = (1, map_R_range, map_range[key_mapT]));
#    maps_ID[key_map] = contrastmap_fid.create_dataset("/"+key_map , (tot_maps, map_R_range, map_pos_range), 
#                                        dtype = np.int, maxshape = (tot_maps, map_R_range, map_pos_range), 
#                                        chunks = (1, map_R_range, map_pos_range),
#                                        compression="lzf", shuffle=True);

tic = time.time()
tic_first = tic


for frame, wormsInFrame in df.groupby('frame_number'):
    #if frame > 300:
    #    break
    img = mask_dataset[frame,:,:]
    
    for ii, worm in wormsInFrame.iterrows():
        worm_index = int(worm['worm_index_joined']); 
        block_id = {'block_ini': int(worm['block_ini_id']), 
                    'block_last': int(worm['block_last_id'])}
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
            Ipos, Ineg = calContrastMaps(pix_dat, map_R_range, map_range['pos'], map_range['neg']);
            
            for key_block in block_id:
                ind = block_id[key_block]
                if ind>=0:
#                    maps_ID[key_block][key + "_pos"][ind,:,:] = \
#                        cv2.GaussianBlur(Ipos.astype(np.double), (3,3), 0).astype(np.int)
#                    maps_ID[key_block][key + "_neg"][ind,:,:] = \
#                        cv2.GaussianBlur(Ineg.astype(np.double), (3,3), 0).astype(np.int)
                    maps_ID[key_block][key + "_pos"][ind,:,:] = Ipos.copy()
                    maps_ID[key_block][key + "_neg"][ind,:,:] = Ineg.copy()
##    
    contrastmap_fid.flush()
    if frame%25 == 0:
        print frame, time.time() - tic, time.time() - tic_first
        tic = time.time()
    
mask_fid.close()
contrastmap_fid.close()


