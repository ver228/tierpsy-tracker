#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:00:40 2018

@author: ajaver
"""
import tables
import tqdm
import glob
import os
import pickle

if __name__ == '__main__':
    mask_dir = '/Volumes/behavgenom_archive$/Serena/AggregationScreening/MaskedVideos'
    fnames = glob.glob(os.path.join(mask_dir, '**', '*.hdf5'), recursive=True)
    fnames = [x for x in fnames if os.path.basename(x).startswith('0')]
    to_skeletons_f = lambda x : x.replace('MaskedVideos', 'Results').replace('.hdf5', '_featuresN.hdf5')
    fnames = [x for x in fnames if not os.path.exists(to_skeletons_f(x))]
    
    
    all_bad_frames = []
    
    for fname in tqdm.tqdm(fnames):
        bad_frames = []
        with tables.File(fname, 'r') as fid:
            masks = fid.get_node('/mask')
            for nn in tqdm.tqdm(range(masks.shape[0])):
                try:
                    img = masks[nn]
                except:
                    bad_frames.append(nn)
            if bad_frames:
                all_bad_frames.append((fname, bad_frames))
    
        with open('bad_frames.p', 'wb') as fid:
            pickle.dump(all_bad_frames, fid)
#%%       
    if False:
        #Try to fix the wrong frames
        with open('bad_frames.p', 'rb') as fid:
            all_bad_frames = pickle.load(fid)
            
        
        import numpy as np
        for fname, bad_frames in all_bad_frames:
            if not os.path.exists(fname):
                continue
            
            if fname != '/Volumes/behavgenom_archive$/Serena/AggregationScreening/MaskedVideos/Agg_0.1_180108/0.1_n2_6b_Set0_Pos0_Ch6_08012018_152556.hdf5':
                continue
            print(fname)
            dd = [-1, *np.where(np.diff(bad_frames)>1)[0].tolist(), -1]
            
            jumps = [(bad_frames[dd[ii]+1], bad_frames[dd[ii+1]]) for ii in range(len(dd)-1)]
            max_jump = max(x[1]-x[0] for x in jumps)
            
            if max_jump > 100:
                print('BAD', fname, max_jump, len(bad_frames))
                
            with tables.File(fname, 'r+') as fid:
                masks = fid.get_node('/mask')
                for ini, fin in jumps:
                    
                    try:
                        img = masks[ini:fin+1]
                    except:
                        try:
                            
                            print('modifying...')
                            jump_size = fin-ini+1
                            mid = ini + jump_size//2
                            
                            assert ini != 0
                            assert fin != masks.shape[0]-1
                            
                            m1 = masks[ini-1]
                            masks[ini:mid] = m1
                            m2 = masks[fin+1]
                            masks[mid:fin+1] = m2
                        except:
                            print('BAD', fname)
                    
                
                
                
                