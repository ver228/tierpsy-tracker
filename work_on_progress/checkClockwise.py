# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:45:06 2015

@author: ajaver
"""
import pandas as pd
import h5py
import numpy as np
import matplotlib.pylab as plt

trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Trajectories/Capture_Ch1_11052015_195105_trajectories.hdf5';
segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Trajectories/Capture_Ch1_11052015_195105_segworm_fix2.hdf5';


table_fid = pd.HDFStore(trajectories_file, 'r');
df = table_fid['/plate_worms'];
df =  df[df['worm_index_joined'] == 8]
df = df[df['segworm_id']>=0]; #select only rows with a valid segworm skeleton
table_fid.close()

segworm_fid = h5py.File(segworm_file, 'r');

vv = segworm_fid['/segworm_results/contour_ventral'][df['segworm_id'].values,:,:]
dd = segworm_fid['/segworm_results/contour_dorsal'][df['segworm_id'].values,:,:]
#%%
x =  np.concatenate((dd[:,0,:], vv[:, 0, ::-1]), axis=1);
y =  np.concatenate((dd[:,1,:], vv[:, 1, ::-1]), axis=1);

signedArea = np.sum(x[:, 0:-1]*y[:, 1:] - x[:, 1:]*y[:, 0:-1], axis = 1)

plt.plot(signedArea)
