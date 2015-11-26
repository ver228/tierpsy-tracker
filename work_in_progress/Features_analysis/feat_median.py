# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:35:11 2015

@author: ajaver
"""

import pandas as pd
import numpy as np
import os
import time

#%%

import sys
if __name__ == '__main__':
    
    feat_file = sys.argv[1]
    
    tic = time.time()
    with pd.HDFStore(feat_file) as fid:
        base_name = feat_file.rpartition(os.sep)[-1].partition('_features.hdf5')[0]
        print(base_name + ': Calculating...')
        sys.stdout.flush()
        
        features_motion = fid['/features_motion']
        plate_med = features_motion.groupby('frame_number').agg(np.nanmedian)
        fid['/features_motion_t_med'] = plate_med
        
        print(base_name + ': Total time %2.2fs' % (time.time() - tic))
        sys.stdout.flush()
    


