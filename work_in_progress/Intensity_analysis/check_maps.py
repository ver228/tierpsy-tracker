# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 09:46:57 2016

@author: ajaver
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

if __name__ == '__main__':
    #base directory
    masked_image_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/MaskedVideos/CSTCTest_Ch1_18112015_075624.hdf5'
    skeletons_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch1_18112015_075624_skeletons.hdf5'
    intensities_file = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch1_18112015_075624_intensities.hdf5'

    with pd.HDFStore(intensities_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
        dd = trajectories_data.groupby('worm_index_joined').agg({'int_map_id':(np.min,np.max)})

    print(dd)
#%%
    worm_N = trajectories_data.groupby('frame_number').agg({'int_map_id':'count'})
    worm_N.plot()