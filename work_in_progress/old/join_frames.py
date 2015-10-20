# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 16:58:24 2015

@author: ajaver
"""

import tables
import itertools
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment #hungarian algorithm
from calculate_ratio import calculate_ratio
from scipy.spatial.distance import cdist
import matplotlib.pylab as plt
#from collections import defaultdict
import time

#featuresFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/A002 - 20150116_140923D_features-2m.hdf5';
featuresFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116-3/A002 - 20150116_140923H_features.hdf5';

feature_fid = tables.open_file(featuresFile, mode = 'r')
feature_table = feature_fid.get_node('/plate_worms')

tic = time.time()

data = []
data_prev = []
area = np.empty([])

max_frame = feature_table.cols.frame_number[-1];
frame_particles = np.zeros(max_frame+1)

tic = time.time();
def frame_selector(row):
    return row['frame_number']
for frame_number, rows in itertools.groupby(feature_table, frame_selector):
    if frame_number % 1000 == 0:   
        toc = time.time();
        print frame_number, toc-tic;
        tic = toc;
        #print frame_number
        
    #data =[[r.nrow,[r['coord_x'], r['coord_y']],r['area']] for r in rows];
    #data = zip(*data);
    frame_particles[frame_number] = sum(1 for x in rows);
    if len(data_prev) !=0:
        pass
        #costMatrix = cdist(data[1], data_prev[1])        
        #dm2 = calculate_ratio(np.array(data[2]), np.array(data_prev[2]))
        
        #costMatrix.shape (new_index, prev_index)
        #np.sum(costMatrix < 10, axis=1)
        
    dum = data_prev
    data_prev = data;
    #area_prev = data[1];
    
print feature_table
#print time.time() - tic
#feature_fid.close()