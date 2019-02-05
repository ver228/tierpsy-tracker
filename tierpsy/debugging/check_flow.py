#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 17:59:36 2018

@author: avelinojaver
"""
import numpy as np
import cv2
from functools import partial
from  tierpsy.analysis.traj_create.getBlobTrajectories import generateImages, generateROIBuff, _thresh_bw, getBlobsSimple, _cnt_to_props
from tierpsy.analysis.ske_create.getSkeletonsTables import getWormMask
from pathlib import Path



mask_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/CX11314_Ch1_04072017_103259.hdf5'


#root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/'
#mask_file = list(Path(root_dir).glob('*.hdf5'))[0]


worm_bw_thresh_factor = 0.9
strel_size=(5, 5)
min_area = 50


blob_params = (min_area,
              worm_bw_thresh_factor,
              strel_size)
img_generator = generateImages(str(mask_file), 
                               bgnd_param={'is_light_background' : 1}
                               )
for frame_number, img in img_generator:
    plt.figure()
    plt.imshow(img)
    break
#%%

f_blob_data = partial(getBlobsSimple, blob_params = blob_params)

blobs_generator = map(f_blob_data,  img_generator)

all_props = []

for frame_props in blobs_generator:
    all_props += frame_props
    if len(all_props) > 100:
        break
        
    
    