# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 23:39:22 2015

@author: ajaver
"""

import pandas as pd
import tables
import numpy as np
import matplotlib.pylab as plt
import time
import glob
import warnings
import os

from ..helperFunctions.timeCounterStr import timeCounterStr
from ..helperFunctions.miscFun import print_flush

from sklearn.covariance import EllipticEnvelope, MinCovDet
np.seterr(invalid='ignore') 

import warnings
warnings.filterwarnings('ignore', '.*det > previous_det*',)


from MWTracker.featuresAnalysis.obtainFeatures import getWormFeatures
from MWTracker.featuresAnalysis.obtainFeaturesHelper import getValidIndexes, calWormArea

getBaseName = lambda skeletons_file : skeletons_file.rpartition(os.sep)[-1].replace('_skeletons.hdf5', '')

worm_partitions = {'neck': (8, 16),
                'midbody':  (16, 33),
                'hips':  (33, 41),
                # refinements of ['head']
                'head_tip': (0, 4),
                'head_base': (4, 8),    # ""
                # refinements of ['tail']
                'tail_base': (40, 45),
                'tail_tip': (45, 49)}


name_width_fun = lambda part: 'width_' + part

def saveModifiedTrajData(skeletons_file, trajectories_data):
    trajectories_recarray = trajectories_data.to_records(index=False)
    with tables.File(skeletons_file, "r+") as ske_file_id:
        table_filters = tables.Filters(complevel=5, complib='zlib', shuffle=True, fletcher32=True)
        newT = ske_file_id.create_table('/', 'trajectories_data_d', obj = trajectories_recarray, filters=table_filters)
        ske_file_id.remove_node('/', 'trajectories_data')
        newT.rename('trajectories_data')
    

def readField(fid, field, valid_index):
    data = fid.get_node(field)[:]
    data = data[valid_index]
    return data

def nodes2Array(skeletons_file, valid_index = -1):

    nodes4fit = ['/skeleton_length', '/contour_area'] + \
    ['/' + name_width_fun(part) for part in worm_partitions]

    with tables.File(skeletons_file, 'r') as fid:
        assert all(node in fid for node in nodes4fit)
        
        if isinstance(valid_index, (float,int)) and valid_index <0:
            valid_index = np.arange(fid.get_node(nodes4fit[0]).shape[0])

        n_samples = len(valid_index)
        n_features = len(nodes4fit)
        
        X = np.zeros((n_samples, n_features))

        if valid_index.size > 0:
            for ii, node in enumerate(nodes4fit):
                X[:,ii] = readField(fid, node, valid_index)
            
        return X

def getSkelRows(skeletons_file):
    with tables.File(skeletons_file, 'r') as fid:
        #get the idea of valid skeletons    
        skeleton_length = fid.get_node('/skeleton_length')[:]
        has_skeleton = fid.get_node('/trajectories_data').col('has_skeleton')
        skeleton_id = fid.get_node('/trajectories_data').col('skeleton_id')
        
        skeleton_id = skeleton_id[has_skeleton==1]
        tot_rows = len(has_skeleton) #total number of rows in the arrays

    return skeleton_id, tot_rows

def calculateWidths(skeletons_file):
    with tables.File(skeletons_file, 'r+') as fid:
        #i am using an auxiliar field in case the function is interrupted before finisihing
        for part in worm_partitions:
            aux_name = name_width_fun(part) + '_AUX'
            if '/' + aux_name in fid: fid.remove_node('/', aux_name)
        
        if all('/' + name_width_fun(part) in fid for part in worm_partitions): return

    
    base_name = getBaseName(skeletons_file)
    progress_timer = timeCounterStr('');
    
    #get the list of valid indexes with skeletons in the table
    skeleton_id, tot_rows = getSkelRows(skeletons_file)

    #calculate the widhts
    with tables.File(skeletons_file, 'r+') as fid:
        widths = fid.get_node('/contour_width')
        tot_rows = widths.shape[0]
        
        table_filters = tables.Filters(complevel=5, complib='zlib', shuffle=True, fletcher32=True)
        widths_means = {}
        
        for part in worm_partitions:
            widths_means[part] = fid.create_carray('/', name_width_fun(part) + '_AUX', \
                                    tables.Float32Atom(dflt = np.nan), (tot_rows,), filters = table_filters)

        for ii, skel_id in enumerate(skeleton_id):
            skel_width = widths[skel_id]
            for part in worm_partitions:
                pp = worm_partitions[part]
                widths_means[part][skel_id] = np.mean(skel_width[pp[0]:pp[1]])

            if ii % 25000 == 0:
                dd = " Calculating mean widths: skeleton %i of %i done." % (ii, len(skeleton_id))
                print_flush(base_name + dd + ' Total time:' + progress_timer.getTimeStr())
                

        for part in worm_partitions:        
            widths_means[part].rename(name_width_fun(part)) #now we can use the real name
        print_flush(base_name + ' Calculating mean widths. Finished. Total time :' + progress_timer.getTimeStr())

def calculateAreas(skeletons_file):

    with tables.File(skeletons_file, 'r+') as fid:
        #i am using an auxiliar field in case the function is interrupted before finisihing
        if '/contour_area_AUX' in fid: fid.remove_node('/', 'contour_area_AUX')
        if '/contour_area' in fid: return
        
    base_name = getBaseName(skeletons_file)
    progress_timer = timeCounterStr('');
    print_flush(base_name + ' Calculating countour areas...')

    #get the list of valid indexes with skeletons in the table
    skeleton_id, tot_rows = getSkelRows(skeletons_file)
    
    #calculate areas
    with tables.File(skeletons_file, 'r+') as fid:
        
        #intiailize area arreay
        table_filters = tables.Filters(complevel=5, complib='zlib', shuffle=True, fletcher32=True)
        contour_area = fid.create_carray('/', "contour_area_AUX", \
        tables.Float32Atom(dflt = np.nan), (tot_rows,), filters = table_filters);
        
        cnt_side1 = fid.get_node('/contour_side1')
        cnt_side2 = fid.get_node('/contour_side2')
            
        for ii, skel_id in enumerate(skeleton_id):
            contour = np.hstack((cnt_side1[skel_id], cnt_side2[skel_id][::-1,:]))
            signed_area = np.sum(contour[:-1,0]*contour[1:,1]-contour[1:,0]*contour[:-1,1])
            contour_area[skel_id] =  np.abs(signed_area)/2

            if ii % 25000 == 0:
                dd = " Calculating countour areas: skeleton %i of %i done." % (ii, len(skeleton_id))
                print_flush(base_name + dd + ' Total time:' + progress_timer.getTimeStr())
            

        contour_area.rename('contour_area') #now we can use the real name
        fid.flush()
        print_flush(base_name + ' Calculating countour area. Finished. Total time :' + progress_timer.getTimeStr())


def labelValidSkeletons(skeletons_file, good_skel_row, fit_contamination = 0.05):
    base_name = getBaseName(skeletons_file)
    progress_timer = timeCounterStr('');
    
    print_flush(base_name + ' Filter Skeletons: Starting...')
    with pd.HDFStore(skeletons_file, 'r') as table_fid:
        trajectories_data = table_fid['/trajectories_data']

    trajectories_data['is_good_skel'] = trajectories_data['has_skeleton']
    
    if good_skel_row.size > 0:
        #nothing to do if there are not valid skeletons left. 
        
        print_flush(base_name + ' Filter Skeletons: Reading features for outlier identification.')
        #calculate classifier for the outliers    
        X4fit = nodes2Array(skeletons_file, good_skel_row)
        assert not np.any(np.isnan(X4fit))
        
        #%%
        print_flush(base_name + ' Filter Skeletons: Fitting elliptic envelope. Total time:' + progress_timer.getTimeStr())
        #TODO here the is a problem with singular covariance matrices that i need to figure out how to solve
        clf = EllipticEnvelope(contamination = fit_contamination)
        clf.fit(X4fit)
        
        print_flush(base_name + ' Filter Skeletons: Calculating outliers. Total time:' + progress_timer.getTimeStr())
        #calculate outliers using the fitted classifier
        X = nodes2Array(skeletons_file) #use all the indexes
        y_pred = clf.decision_function(X).ravel() #less than zero would be an outlier

        print_flush(base_name + ' Filter Skeletons: Labeling valid skeletons. Total time:' + progress_timer.getTimeStr())
        #labeled rows of valid individual skeletons as GOOD_SKE
        trajectories_data['is_good_skel'] = (y_pred>0).astype(np.int)
    
    #Save the new is_good_skel column
    saveModifiedTrajData(skeletons_file, trajectories_data)

    print_flush(base_name + ' Filter Skeletons: Finished. Total time:' + progress_timer.getTimeStr())

def getFrameStats(feat_file):
    
    stats_funs = {'median':np.nanmedian, 'mean':np.nanmean, \
    'std':np.nanstd, 'count':lambda a : np.count_nonzero(~np.isnan(a))}
    
    table_filters = tables.Filters(complevel=5, complib='zlib', shuffle=True, fletcher32=True)
    warnings.simplefilter("ignore")
            
    with pd.HDFStore(feat_file) as feat_fid:
        features_timeseries = feat_fid['/features_timeseries']            
    
    if len(features_timeseries['worm_index'].unique()) == 1:
        return #nothing to do here
    
    groupbyframe = features_timeseries.groupby('frame_number')
        
    with warnings.catch_warnings(), tables.File(feat_file, 'r+') as feat_fid:
        if '/frame_stats' in feat_fid:
            feat_fid.remove_node('/', 'frame_stats', recursive = True)
        
        frame_stats_group = feat_fid.create_group('/', 'frame_stats')
        
        for stat in stats_funs:
            #ignore wargings from frames/features with only nan's
            plate_stats = groupbyframe.agg(stats_funs[stat])
            
            feat_fid.create_table(frame_stats_group, stat, \
            obj = plate_stats.to_records(), filters = table_filters)

def getFilteredFeats(skeletons_file, use_skel_filter, min_num_skel = 100, bad_seg_thresh = 0.8, 
    min_dist = 5, fit_contamination = 0.05):
    
    #check if the skeletonization finished succesfully
    with tables.File(skeletons_file, "r") as ske_file_id:
        skeleton_table = ske_file_id.get_node('/skeleton')
        assert skeleton_table._v_attrs['has_finished'] >= 2

    #calculate valid widths and areas if they were not calculated before
    calculateWidths(skeletons_file)
    calculateAreas(skeletons_file)

    if use_skel_filter:
        #get valid rows using the trajectory displacement and the skeletonization success
        good_traj_index , good_skel_row = getValidIndexes(skeletons_file, \
        min_num_skel = min_num_skel, bad_seg_thresh = bad_seg_thresh, min_dist = min_dist)
        
        labelValidSkeletons(skeletons_file, good_skel_row, fit_contamination = fit_contamination)

    with tables.File(skeletons_file, "r+") as ske_file_id:
        skeleton_table = ske_file_id.get_node('/skeleton')
        #label as finished
        skeleton_table._v_attrs['has_finished'] = 3
        




