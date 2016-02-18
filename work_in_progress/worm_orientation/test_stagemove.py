# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 22:01:59 2016

@author: ajaver
"""

import h5py
import numpy as np
import matplotlib.pylab as plt


#masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch5_17112015_205616.hdf5'
#masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch1_18112015_075624.hdf5'
masked_image_file = '/Users/ajaver/Desktop/Videos/04-03-11/MaskedVideos/575 JU440 on food Rz_2011_03_04__12_55_53__7.hdf5'    
#masked_image_file = '/Users/ajaver/Desktop/Videos/04-03-11/MaskedVideos/575 JU440 swimming_2011_03_04__13_16_37__8.hdf5'    
       
skeletons_file = masked_image_file.replace('MaskedVideos', 'Results')[:-5] + '_skeletons.hdf5'
intensities_file = skeletons_file.replace('_skeletons', '_intensities')

#intensities_file = '/Users/ajaver/Desktop/Videos/04-03-11/Results/575 JU440 swimming_2011_03_04__13_16_37__8_intensities.hdf5'    
intensities_file = '/Users/ajaver/Desktop/Videos/04-03-11/Results/575 JU440 on food Rz_2011_03_04__12_55_53__7_intensities.hdf5'    
    

skeletons_file = intensities_file.replace('intensities', 'skeletons')
masked_image_file = intensities_file.replace('Results', 'MaskedVideos').replace('_intensities', '')

#%%
with h5py.File(skeletons_file, 'r') as fid:
    skeletons = fid['/skeleton'][:]
    

with h5py.File(masked_image_file, 'r') as fid:
    pixels2microns_x = fid['/mask'].attrs['pixels2microns_x']
    pixels2microns_y = fid['/mask'].attrs['pixels2microns_y']
#%%
xx = np.diff(skeletons[:,:,0], axis=0)
xx = np.median(xx,axis=1)

yy = np.diff(skeletons[:,:,1], axis=0)
yy = np.median(yy,axis=1)

tt = np.arange(xx.size)*1/30

good = ~np.isnan(xx)
xx = np.cumsum(xx[good])*pixels2microns_x
yy = np.cumsum(yy[good])*pixels2microns_y
tt = tt[good]
#%%

fid = h5py.File(masked_image_file, 'r')
x_stage = fid['/stage_data']['stage_x']/10
y_stage = fid['/stage_data']['stage_y']/10
stage_time = fid['/stage_data']['stage_time']
#%%
delta_T = 10
plt.figure()
plt.plot(tt[delta_T:], xx[:-delta_T]-xx[delta_T:])
plt.plot(stage_time[:-1], np.diff(x_stage),'.')

plt.figure()
plt.plot(tt[delta_T:], yy[:-delta_T]-yy[delta_T:])
plt.plot(stage_time[:-1], np.diff(y_stage),'.')
#%%

#plt.figure()
#plt.plot(stage_time, x_stage-x_stage[0],'.')
#plt.plot(tt,xx)
#
#plt.figure()
#plt.plot(stage_time, y_stage-y_stage[0],'.')
#plt.plot(tt,-yy)
