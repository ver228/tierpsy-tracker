#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 14:31:16 2018

@author: avelinojaver
"""

from unet import UNet

import torch
import tables
import numpy as np
import tqdm

from tierpsy.analysis.compress.compressVideo import createImgGroup
from tierpsy.helper.misc import RESERVED_EXT

model_path = '/Users/avelinojaver/OneDrive - Nexus365/worms/bertie_worms_l1_20181210_162006_unet_adam_lr0.0001_wd0.0_batch16.pth.tar'

def _add_bgnd(fname, _model_path = model_path, _int_scale = (0, 255), cuda_id = 0):
    
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    model = UNet(n_channels = 1, n_classes = 1)
    state = torch.load(_model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    
    model = model.to(device)
    model.eval()
    
    with tables.File(fname, 'r+') as fid:
        full_data = fid.get_node('/full_data')
        
        if '/bgnd' in fid:
            fid.remove_node('/bgnd')
            
        bgnd = createImgGroup(fid, "/bgnd", *full_data.shape, is_expandable = False)
        bgnd._v_attrs['save_interval'] = full_data._v_attrs['save_interval']
        
        for ii in tqdm.trange(full_data.shape[0]):
            img = full_data[ii]
            
            x = img.astype(np.float32)
            x = (x - _int_scale[0])/(_int_scale[1] - _int_scale[0])
            
            
            with torch.no_grad():
                X = torch.from_numpy(x[None, None])
                X = X.to(device)
                Xhat = model(X)
            
            xhat = Xhat.squeeze().detach().cpu().numpy()
            
            bg = xhat*(_int_scale[1] - _int_scale[0]) + _int_scale[0]
            bg = bg.round().astype(img.dtype)
            bgnd[ii] = bg

if __name__ == '__main__':
    
    from pathlib import Path
    dname = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/')
    
    fnames = list(dname.rglob('*.hdf5'))
    fnames = [x for x in fnames if not any([x.name.endswith(e) for e in RESERVED_EXT])]
    
    for fname in tqdm.tqdm(fnames):
        #with tables.File(fname, 'r') as fid:
        #    if '/bgnd' in fid:
        #        continue
    
        _add_bgnd(fname)
        

        
        