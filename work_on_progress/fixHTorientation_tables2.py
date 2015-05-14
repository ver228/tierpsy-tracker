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
from scipy.spatial.distance import cdist

from calContrastMaps import calContrastMapsBinned
#from image_difference import image_difference
if __name__  == "__main__":
    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/CaptureTest_90pc_Ch1_02022015_141431.hdf5';
    trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_trajectories.hdf5';
    segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_segworm_short.hdf5';
    contrastmap_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_cmap.hdf5';
    
    MAX_DELT = 1;
    #jump_frames = 3; #3
    #del_frames = 100; #100
    ROI_SIZE = 128;
    
    
    table_fid = pd.HDFStore(trajectories_file, 'r');
    df = table_fid['/plate_worms'];
    #df =  df[df['worm_index_joined'] > 0]
    df =  df[df['worm_index_joined'] == 3]
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
        all_ind_block += zip(*[block_ind.size*[worm_ind], list(block_ind), list(plate_worms_id), list(worm_segworm_id)])
    
    all_ind_block = sorted(all_ind_block, key = operator.itemgetter(0, 1)); #sort by worm index first, and the by block index
    all_ind_block = zip(*all_ind_block)
    all_ind_block += [range(len(all_ind_block[0]))]
    all_ind_block = zip(*all_ind_block)
    #%%
    contrastmap_fid = tables.File(contrastmap_file, 'w');
    
    ind_block_table = contrastmap_fid.create_table('/', 'block_index', {
        'worm_index_joined':    tables.Int32Col(pos=0),
        'block_id'         : tables.Int32Col(pos=1),
        'plate_worms_id'   : tables.Int32Col(pos=2),
        'segworm_id'         : tables.Int32Col(pos=3),
        'cmap_id'     : tables.Int32Col(pos=4),
        }, filters = tables.Filters(complevel=5, complib='zlib', shuffle=True))
    
    ind_block_table.append(all_ind_block);
    
    contrastmap_fid.flush()
    contrastmap_fid.close()
    #%%
    #read the valid contrastmap data (again)
    contrastmap_fid = pd.HDFStore(contrastmap_file, 'r');
    df_map_ind = contrastmap_fid['/block_index']
    contrastmap_fid.close()
    
    #read the trajectories data and add a column for the cmap index
    trajectories_fid = pd.HDFStore(trajectories_file, 'r');
    df = trajectories_fid['/plate_worms'];
    df = df.irow(df_map_ind['plate_worms_id']);
    df.loc[:, 'cmap_id'] = pd.Series(df_map_ind['cmap_id'].values, df_map_ind['plate_worms_id'].values)
    assert(len(df) == len(df_map_ind))
    #%%
    #open the hdf5 with the masked images
    mask_fid = h5py.File(masked_image_file, 'r');
    mask_dataset = mask_fid["/mask"]
    
    segworm_fid = h5py.File(segworm_file, 'r')
    
    bins_size = {'R':50., 'pos':50., 'neg':50.};  
    max_values = {'R':100., 'pos':511., 'neg':256.};  
    
    N_pix_HT = 10; #segment size taken for the head and the tail
    N_pix_DV = 10; #segment taken for the dorsal and ventral segments from the head/tail
    
    tot_maps = len(df_map_ind)
    
    contrastmap_fid = tables.File(contrastmap_file, 'r+');
    #%%
    maps_ID = {};
    
    contrastmap_fid.create_group("/", "block_cmap")
    for key_part in ['worm_H', 'worm_T', 'worm_V', 'worm_D', 'worm_all']:
        for key_mapT in ["pos", "neg"]:
            key_map = key_part + '_'+ key_mapT;
            maps_ID[key_map] = contrastmap_fid.create_carray("/block_cmap/", key_map , atom = tables.UInt32Atom(), 
                                                    shape = (tot_maps, bins_size['R'], bins_size[key_mapT]),
                                                    filters = tables.Filters(complevel=5, complib='blosc', shuffle = True), 
                                                    chunkshape = (1, bins_size['R'], bins_size[key_mapT]));
    
    tic = time.time()
    tic_first = tic
    
    label_range = (15, 479)
    for frame, wormsInFrame in df.groupby('frame_number'):
        #if frame > 300:
        #    break
        img = mask_dataset[frame,:,:]
        #img[:label_range[0],:label_range[1]] = 0; #make zero the label
        
        for ii, worm in wormsInFrame.iterrows():
            worm_index = int(worm['worm_index_joined']); 
            cmap_id = worm['cmap_id']
            segworm_id = worm['segworm_id']
    
            range_x = np.round(worm['coord_x']) + [-ROI_SIZE/2, ROI_SIZE/2]
            range_y = np.round(worm['coord_y']) + [-ROI_SIZE/2, ROI_SIZE/2]
            
            if (range_y[0] <0) or (range_y[0]>= img.shape[0]) or \
            (range_x[0] <0) or (range_x[0]>= img.shape[1]):
                continue
            
            worm_img =  img[range_y[0]:range_y[1], range_x[0]:range_x[1]]
            
            dat = {}
            for key in ['contour_ventral', 'skeleton', 'contour_dorsal']:
                dat[key] = segworm_fid['/segworm_results/' + key][segworm_id,:,:];
                dat[key][1,:] = dat[key][1,:]-range_x[0]
                dat[key][0,:] = dat[key][0,:]-range_y[0]
            #%%
            #initialize dictionary
            contours = {};
            p_ske = dat['skeleton'][:,[N_pix_HT, -N_pix_HT, N_pix_DV, -N_pix_DV]];
            v_closest = np.argmin(cdist(dat['contour_ventral'].T, p_ske.T), axis=0)
            d_closest = np.argmin(cdist(dat['contour_dorsal'].T, p_ske.T), axis=0)
            p_ske = p_ske[::-1,:]
            
            contours['worm_all'] = np.hstack((dat['contour_ventral'][::-1,:],dat['contour_dorsal'][::-1,::-1]));
            contours['worm_H'] = np.hstack((dat['contour_ventral'][::-1,0:v_closest[0]+1], \
                np.reshape(p_ske[:,0],(2,-1)), \
                dat['contour_dorsal'][::-1,d_closest[0]::-1]));
            
            contours['worm_T'] = np.hstack((dat['contour_ventral'][::-1, :v_closest[1]-1:-1], \
                np.reshape(p_ske[:,1],(2,-1)), \
                dat['contour_dorsal'][::-1, d_closest[1]:]));
            
            contours['worm_V'] = np.hstack((dat['contour_ventral'][::-1,v_closest[2]:v_closest[3]+1], \
                dat['skeleton'][::-1,-N_pix_DV:N_pix_DV-1:-1]));
            
            contours['worm_D'] = np.hstack((dat['contour_dorsal'][::-1,d_closest[2]:d_closest[3]+1], \
                dat['skeleton'][::-1,-N_pix_DV:N_pix_DV-1:-1]));
            
            
#            #plt.imshow(worm_img, interpolation = 'none', cmap = 'gray')
#            plt.plot(contours['worm_all'][0,:],contours['worm_all'][1,:], '.-')
#            plt.plot(contours['worm_H'][0,:],contours['worm_H'][1,:])
#            plt.plot(contours['worm_T'][0,:],contours['worm_T'][1,:])
#            plt.plot(contours['worm_V'][0,:],contours['worm_V'][1,:])
#            plt.plot(contours['worm_D'][0,:],contours['worm_D'][1,:])
#            
            #%%
            
            for key in contours:
                worm_mask = np.zeros(worm_img.shape)
                cc = [contours[key].astype(np.int32).T];
                cv2.drawContours(worm_mask, cc, 0, 1, -1)
                pix_list = np.where(worm_mask==1);
                pix_val = worm_img[pix_list].astype(np.int);
                pix_dat = np.array((pix_list[0], pix_list[1], pix_val))
                
                #print key, pix_dat.shape
                #Ipos, Ineg = calContrastMaps(pix_dat, map_R_range, map_range['pos'], map_range['neg']);
                Ipos, Ineg = calContrastMapsBinned(pix_dat, bins_size, max_values)
                maps_ID[key + "_pos"][cmap_id,:,:] = Ipos.copy()
                maps_ID[key + "_neg"][cmap_id,:,:] = Ineg.copy()
#    ##    
        contrastmap_fid.flush()
        if frame%25 == 0:
            print frame, time.time() - tic, time.time() - tic_first
            tic = time.time()
        
    mask_fid.close()
    contrastmap_fid.close()
    
    
