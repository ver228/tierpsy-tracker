# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 20:55:19 2016

@author: ajaver
"""

import tables
import pandas as pd
from collections import OrderedDict
import numpy as np
import json
import gzip
import os

from MWTracker.analysis.feat_create.obtainFeaturesHelper import WormStatsClass
from MWTracker.helper.misc import print_flush

def _getMetaData(features_file, READ_FEATURES=False):
    with tables.File(features_file, 'r') as fid:
        if not '/experiment_info' in fid:
            experiment_info = {}
        else:
            experiment_info = fid.get_node('/experiment_info').read()
            experiment_info = json.loads(experiment_info.decode('utf-8'))
        
        
        provenance_tracking = fid.get_node('/provenance_tracking/FEAT_CREATE').read()
        provenance_tracking = json.loads(provenance_tracking.decode('utf-8'))
        commit_hash = provenance_tracking['commit_hash']
        
        MWTracker_ver = {"name":"MWTracker (https://github.com/ver228/Multiworm_Tracking)",
            "version":commit_hash['MWTracker'],
            "featureID":"@OMG"}
        
        if not READ_FEATURES:
            experiment_info["software"] = MWTracker_ver
        else:
            open_worm_ver = {"name":"open_worm_analysis_toolbox (https://github.com/openworm/open-worm-analysis-toolbox)",
            "version":commit_hash['open_worm_analysis_toolbox'],
            "featureID":""}
            experiment_info["software"] = [MWTracker_ver, open_worm_ver]
        return experiment_info

def __reformatForJson(A):
    try:
        #round 
        good = ~np.isnan(A) & (A != 0)
        dd = A[good]
        if dd.size > 0:
            dd = np.abs(np.floor(np.log10(np.abs(dd)))-2)
            precision = max(2, int(np.min(dd)))
            A = np.round(A.astype(np.float64), precision)
        A = np.where(np.isnan(A), None, A)
    except:
        import pdb
        pdb.set_trace()
    
    #wcon specification require to return a single number if it is only one element list
    if A.size == 1:
        return A[0]
    else:
        return A.tolist()

def __addOMGFeat(fid, worm_feat_time, worm_id):
    worm_features = OrderedDict()
    #add time series features
    for col_name, col_dat in worm_feat_time.iteritems():
        if not col_name in ['worm_index', 'timestamp']:
            worm_features[col_name] = __reformatForJson(col_dat)
    
    worm_path = '/features_events/worm_%i' % worm_id
    worm_node = fid.get_node(worm_path)
    #add event features
    for feature_name in worm_node._v_children:
        feature_path = worm_path + '/' + feature_name
        worm_features[feature_name] = __reformatForJson(fid.get_node(feature_path)[:])
    
    

def _getData(features_file, READ_FEATURES=False):
    with pd.HDFStore(features_file, 'r') as fid:
        features_timeseries = fid['/features_timeseries']
        feat_time_group_by_worm = features_timeseries.groupby('worm_index');
        
        
    with tables.File(features_file, 'r') as fid:
        #fps used to adjust timestamp to real time
        fps = fid.get_node('/features_timeseries').attrs['fps']
        
        
        #get pointers to some useful data
        skeletons = fid.get_node('/coordinates/skeletons')
        dorsal_contours = fid.get_node('/coordinates/dorsal_contours')
        ventral_contours = fid.get_node('/coordinates/ventral_contours')
        
        #let's append the data of each individual worm as a element in a list
        all_worms_feats = []
        
        #group by iterator will return sorted worm indexes
        for worm_id, worm_feat_time in feat_time_group_by_worm:
            worm_id = int(worm_id)
            
            #read worm skeletons data
            worm_skel = skeletons[worm_feat_time.index]
            worm_dor_cnt = dorsal_contours[worm_feat_time.index]
            worm_ven_cnt = ventral_contours[worm_feat_time.index]
            
            
            #start ordered dictionary with the basic features
            worm_basic = OrderedDict()
            worm_basic['id'] = worm_id
            worm_basic['t'] = __reformatForJson(worm_feat_time['timestamp'].values/fps)
            worm_basic['x'] = __reformatForJson(worm_skel[:, :, 0])
            worm_basic['y'] = __reformatForJson(worm_skel[:, :, 1])
            worm_basic['x_ventral_contour'] = __reformatForJson(worm_ven_cnt[:, :, 0])
            worm_basic['y_ventral_contour'] = __reformatForJson(worm_ven_cnt[:, :, 1])
            worm_basic['x_dorsal_contour'] = __reformatForJson(worm_dor_cnt[:, :, 0])
            worm_basic['y_dorsal_contour'] = __reformatForJson(worm_dor_cnt[:, :, 1])
            
            worm_basic['@OMG'] = OrderedDict()
            worm_basic['@OMG']['x_ventral_contour'] = __reformatForJson(worm_ven_cnt[:, :, 0])
            worm_basic['@OMG']['y_ventral_contour'] = __reformatForJson(worm_ven_cnt[:, :, 1])
            worm_basic['@OMG']['x_dorsal_contour'] = __reformatForJson(worm_dor_cnt[:, :, 0])
            worm_basic['@OMG']['y_dorsal_contour'] = __reformatForJson(worm_dor_cnt[:, :, 1])
            
            
            if READ_FEATURES:
                worm_features = __addOMGFeat(fid, worm_feat_time, worm_id)
                for feat in worm_features:
                     worm_basic['@OMG'][feat] = worm_features[feat]

            #append features
            all_worms_feats.append(worm_basic)
    
    return all_worms_feats

def _getUnits(features_file, READ_FEATURES=False):
    
    with tables.File(features_file, 'r') as fid:
        micronsPerPixel = fid.get_node('/features_timeseries').attrs['micronsPerPixel']
    
    units = OrderedDict()
    units['t'] = 'seconds'
    field_names = ['x', 'y', 'x_ventral_contour', 'y_ventral_contour', 'x_dorsal_contour', 'y_dorsal_contour']
    if isinstance(micronsPerPixel, (float, np.float64)) and micronsPerPixel == 1:
        for field in field_names:
            units[field] = 'pixels'
    else:
        for field in field_names:
            units[field] = 'microns'
    units["size"] = "mm"
    
    if READ_FEATURES:
        ws = WormStatsClass()
        units.update(ws.features_info['units'].to_dict())
    
    return units
    
    
def exportWCONdict(features_file):
    metadata = _getMetaData(features_file)
    data = _getData(features_file)
    units = _getUnits(features_file)
    
    #units = {x:units[x].replace('degrees', '1') for x in units}
    #units = {x:units[x].replace('radians', '1') for x in units}
    
    wcon_dict = OrderedDict()
    
    wcon_dict['metadata'] = metadata
    wcon_dict['units'] = units
    wcon_dict['data'] = data
    
    

    return wcon_dict


def getWCOName(features_file):
    return features_file.replace('_features.hdf5', '.wcon.zip')

def exportWCON(features_file):
    base_name = os.path.basename(features_file).replace('_features.hdf5', '')
    
    print_flush("{} Exporting data to WCON...".format(base_name))
    wcon_file = getWCOName(features_file)
    wcon_dict = exportWCONdict(features_file)
    with gzip.open(wcon_file, 'wt') as fid:
        json.dump(wcon_dict, fid, allow_nan=False)
        
    print_flush("{} Finised to export to WCON.".format(base_name))

if __name__ == '__main__':
    features_file = 'N2 on food R_2011_02_24__16_35_59___2___9_features.hdf5'    
    #exportWCON(features_file)

    
    #%%
#    import wcon
#    wc = wcon.WCONWorms()
#    wc = wc.load_from_file(JSON_path, validate_against_schema = False)
