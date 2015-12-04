# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:35:11 2015

@author: ajaver
"""

import pandas as pd
import numpy as np
import glob
import os
import time
from collections import OrderedDict

main_dir = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/'
#main_dir = '/Volumes/behavgenom$/GeckoVideo/Results/Avelino_17112015_2100/'
feat_files = glob.glob(main_dir + '*_features.hdf5')

#%%
plate_med = OrderedDict()


for feat_file in feat_files:
    tic = time.time()
    #[['frame_number', 'length', 'primary_wavelength', 'midbody_speed']]
    with pd.HDFStore(feat_file, 'r') as fid:
        base_name = feat_file.rpartition(os.sep)[-1].partition('_features.hdf5')[0]
        print(base_name)
        plate_med[base_name] = fid['/features_motion_t_med']
#%%
import matplotlib.pylab as plt
delta_t = 60*25*15

    
for exp in plate_med:
    #if not '205616' in exp:
    if not 'Ch1' in exp:
         continue
    
    plt.figure()

    dd = plate_med[exp]
    tot_t = len(dd)
    
    str_field = 'length'
    for t in range(0, tot_t, delta_t):
        val = dd.ix[t:(t+delta_t),str_field].dropna()
        cc, edges = np.histogram(val,25)
        cc = cc/len(val)/(edges[1]-edges[0])
        
        plt.plot(edges[:-1], cc, label=t)
    #plt.legend()
    plt.title(exp)
#%%

