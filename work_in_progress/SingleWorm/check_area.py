# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 19:44:37 2015

@author: ajaver
"""

import pandas as pd
import glob
import matplotlib.pylab as plt
import h5py
import numpy as np

#main_dir = '/Users/ajaver/Desktop/Videos/Check_Align_samples/Results/'
main_dir = '/Users/ajaver/Desktop/Videos/copied_from_pc207-13/Results/'

files = glob.glob(main_dir + '*_skeletons.hdf5')

bad_files = [files[x-1] for x in [28, 24, 23, 19, 16, 14, 5]]

for ii, file_skel in enumerate(bad_files):
    cmd_next = file_skel.replace('_skeletons', '').replace('/Results', '/MaskedVideos/')
    cmd_next = 'python3 MWTrackerViewer.py ' + '"' + cmd_next + '"'
    print(ii, cmd_next)
    file_traj = file_skel.replace('_skeletons', '_trajectories')
    
    with pd.HDFStore(file_skel, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    
    with pd.HDFStore(file_traj, 'r') as fid:
        plate_worms = fid['/plate_worms']
    
    with h5py.File(file_skel, 'r') as fid:
        skel_area = fid['/contour_area'][:]
    
    plt.figure()
    #plt.plot(skel_area)
    plt.plot(plate_worms['area'])
    
    