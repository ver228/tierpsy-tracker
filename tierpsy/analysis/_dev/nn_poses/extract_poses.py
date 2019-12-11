#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 13:11:35 2019

@author: avelinojaver
"""

import torch
import tables
from pathlib import Path
from models import CPM_PAF

_model_path = Path.home() / 'workspace/WormData/results/worm-poses/logs/manually-annotated-PAF_20190116_181119_CPMout-PAF_adam_lr0.0001_wd0.0_batch16/checkpoint.pth.tar'
n_segments = 25
n_affinity_maps = 20


def _get_device(cuda_id = 0):
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    return device
#%%
def read_images(mask_file, batch_size, frames2check = None):
    with tables.File(mask_file, 'r') as fid:
        masks = fid.get_node('/mask')
        
        if frames2check is None:
            frames2check = range(masks.shape[0])
        
        batch = []
        for frame in frames2check:
            img = masks[frame]
            
            batch.append((frame, img))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
            
            
def img2skelmaps(batch, model, device):
    frames, X = zip(*batch)
    with torch.no_grad():
        X = torch.tensor([x[None] for x in X]).float()
        X = X.to(device)
        X /= 255.
        outs = model(X)
    cpm_maps, paf_maps = outs[-1]
    cpm_maps = cpm_maps.detach().numpy()
    
    out_repacked = zip(frames, cpm_maps)
    
    return out_repacked
 
#%%
if __name__ == '__main__': 
    import tqdm
    
    #mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/skeletonize_training/manual_annotations/raw/Phase3/MaskedVideos/wildMating1.2_MY23_cross_CB4856_cross_PC2_Ch2_15082018_121839.hdf5'
    mask_file = Path.home() / 'WormData/results/worm-poses/raw/Phase4/MaskedVideos/JU2464_Ch1_07072017_102132.hdf5'
    
    cuda_id = 0
    device = _get_device(cuda_id)
    
    
    model = CPM_PAF(n_segments = n_segments, 
                             n_affinity_maps = n_affinity_maps, 
                             same_output_size = True)
    
    state = torch.load(_model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    
    model = model.to(device)
    model.eval()
    

    gen = read_images(mask_file, batch_size = 4)
    for batch in tqdm.tqdm(gen):
        out_repacked = img2skelmaps(batch, model, device)
    
    