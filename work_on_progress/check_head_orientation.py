# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:35:04 2015

@author: ajaver
"""

import pandas as pd
import h5py
import numpy as np
import matplotlib.pylab as plt

trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Trajectories/Capture_Ch1_11052015_195105_trajectories.hdf5';
segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150511/Trajectories/Capture_Ch1_11052015_195105_segworm.hdf5';


table_fid = pd.HDFStore(trajectories_file, 'r');
df = table_fid['/plate_worms'];
#df =  df[df['worm_index_joined'] == 2608] , 8, 538, 3433
df =  df[df['worm_index_joined'] == 8]
df = df[df['segworm_id']>=0]; #select only rows with a valid segworm skeleton
table_fid.close()

segworm_fid = h5py.File(segworm_file, 'r');



segworm_id = df['segworm_id'].values
skeletons = segworm_fid['/segworm_results/skeleton'][segworm_id,:,:]
#%%
def getAnglesDelta(dx,dy):
    angles = np.arctan2(dx,dy)
    dAngles = np.diff(angles)
        
    
    positiveJumps = np.where(dAngles > np.pi)[0] + 1; #%+1 to cancel shift of diff
    negativeJumps = np.where(dAngles <-np.pi)[0] + 1;
        
    #% subtract 2pi from remainging data after positive jumps
    for jump in positiveJumps:
        angles[jump:] = angles[jump:] - 2*np.pi;
        
    #% add 2pi to remaining data after negative jumps
    for jump in negativeJumps:
        angles[jump:] = angles[jump:] + 2*np.pi;
    
    #% rotate skeleton angles so that mean orientation is zero
    meanAngle = np.mean(angles);
    angles = angles - meanAngle;
    return (angles, meanAngle)
    
#%%


delta_ind = [5]#range(5,7)
for ii in delta_ind:
    dx = skeletons[:,0,ii] - skeletons[:,0,0]
    dy = skeletons[:,1,ii] - skeletons[:,1,0]

    angles_head,_ = getAnglesDelta(dx,dy)
    


    dx = skeletons[:,0,-ii-1] - skeletons[:,0,-1]
    dy = skeletons[:,1,-ii-1] - skeletons[:,1,-1]

    angles_tail,_ = getAnglesDelta(dx,dy)
    plt.figure()
    plt.plot(angles_head)
    plt.plot(angles_tail)
    
    #ts = pd.DataFrame({'head':angles_head, 'tail':angles_tail}, index = df['frame_number'].values)
    #pd.rolling_std(ts, 100).plot()
    
    
    
#%%
plt.figure()
dx = skeletons[:,0,0] - skeletons[:,0,-1]
dy = skeletons[:,1,0] - skeletons[:,1,-1]
R = np.sqrt(dx*dx + dy*dy)
plt.plot(R)
#%%



#%%



#vv = segworm_fid['/segworm_results/contour_ventral'][df['segworm_id'].values,:,:]
#dd = segworm_fid['/segworm_results/contour_dorsal'][df['segworm_id'].values,:,:]
##%%
#x =  np.concatenate((dd[:,0,:], vv[:, 0, ::-1]), axis=1);
#y =  np.concatenate((dd[:,1,:], vv[:, 1, ::-1]), axis=1);
#
#signedArea = np.sum(x[:, 0:-1]*y[:, 1:] - x[:, 1:]*y[:, 0:-1], axis = 1)
#
#plt.plot(signedArea)
