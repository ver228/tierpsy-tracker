#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 17:59:36 2018

@author: avelinojaver
"""
import numpy as np
import cv2
from functools import partial
import json
from pathlib import Path


import pandas as pd
from tierpsy.analysis.ske_create.helperIterROI import generateMoviesROI



mask_file = Path('/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/CX11314_Ch1_04072017_103259.hdf5')


root_dir = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/'

for mask_file in list(Path(root_dir).glob('*.hdf5')):
    skeletons_file = mask_file.parent / 'Results' / (mask_file.stem + '_skeletons.hdf5')
    with pd.HDFStore(str(skeletons_file), "r") as ske_file_id:
    
        #attribute useful to understand if we are dealing with dark or light worms
        bgnd_param = ske_file_id.get_node('/plate_worms')._v_attrs['bgnd_param']
        bgnd_param = json.loads(bgnd_param.decode("utf-8"))
        print(bgnd_param)

#%%
ROIs_generator = generateMoviesROI(masked_image_file, 
                                         trajectories_data, 
                                         bgnd_param = bgnd_param,
                                         progress_prefix = '')

for frame_props in ROIs_generator:
    break
        
    
    