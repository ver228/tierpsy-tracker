# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:35:46 2016

@author: ajaver
"""

#skeletons_file_auto = '/Users/ajaver/Desktop/Videos/Avelino_17112015/Results/CSTCTest_Ch1_17112015_205616_skeletons.hdf5'
#skeletons_file_manual = '/Volumes/behavgenom$/GeckoVideo/JoinedTrajManual/Results/20150521_1115/Capture_Ch1_21052015_111806_skeletons.hdf5'
#
#import pandas as pd
#with pd.HDFStore(skeletons_file_auto, 'r') as fid:
#    traj_auto = fid['/trajectories_data']
#    
#with pd.HDFStore(skeletons_file_manual, 'r') as fid:
#    traj_manual = fid['/trajectories_data']
#%%
import sys    
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking/')
from MWTracker.featuresAnalysis.obtainFeaturesHelper import calWormArea

import tables 
import numpy as np
skel_file = '/Users/ajaver/Tmp/Results/Capture_Ch1_12062015_142858_skeletons.hdf5'
with tables.File(skel_file, 'r+') as fid:

    #get the idea of valid skeletons    
    skeleton_length = fid.get_node('/skeleton_length')[:]
    has_skeleton = fid.get_node('/trajectories_data').col('has_skeleton')
    skeleton_id = fid.get_node('/trajectories_data').col('skeleton_id')
    
    skeleton_id = skeleton_id[has_skeleton==1]
    tot_rows = len(has_skeleton) #total number of rows in the arrays
    
    #remove node area if it existed before
    if '/contour_area' in fid: fid.remove_node('/', 'contour_area')
    
    #intiailize area arreay
    table_filters = tables.Filters(complevel=5, complib='zlib', shuffle=True, fletcher32=True)
    contour_area = fid.create_carray('/', "contour_area", \
    tables.Float32Atom(dflt = np.nan), (tot_rows,), filters = table_filters);
    
    cnt_side1 = fid.get_node('/contour_side1')
    cnt_side2 = fid.get_node('/contour_side2')
        
    print('Calculating skeletons area...')
    for skel_id in skeleton_id:
        print(skel_id)
        contour = np.hstack((cnt_side1[skel_id], cnt_side2[skel_id][::-1,:]))
        signed_area = np.sum(contour[:-1,0]*contour[1:,1]-contour[1:,0]*contour[:-1,1])
        contour_area[skel_id] =  np.abs(signed_area)/2
