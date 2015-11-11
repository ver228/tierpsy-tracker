# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:30:53 2015

@author: ajaver
"""
import os, sys
#import sys
import tables
import pandas as pd
import numpy as np
from math import floor, ceil

import warnings
warnings.filterwarnings('ignore', '.*empty slice*',)
tables.parameters.MAX_COLUMNS = 1024 #(http://www.pytables.org/usersguide/parameter_files.html)

from collections import OrderedDict

from ..helperFunctions.timeCounterStr import timeCounterStr

#add movement_validation path
#from .. import config_param as param
#sys.path.append(movement_validation_dir)

from movement_validation import NormalizedWorm
from movement_validation import WormFeatures, VideoInfo
from movement_validation.statistics import specifications

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:16:41 2015

@author: ajaver
"""

np.seterr(invalid='ignore')

def calWormAngles(x,y, segment_size):
    '''
    Get the skeleton angles from its x, y coordinates
    '''
    assert(x.ndim==1)
    assert(y.ndim==1)
    
    dx = x[segment_size:]-x[:-segment_size]
    dy = y[segment_size:]-y[:-segment_size]
    #dx = np.diff(x);
    #dy = np.diff(y);
    angles = np.arctan2(dx,dy)
    dAngles = np.diff(angles)
    
    #    % need to deal with cases where angle changes discontinuously from -pi
    #    % to pi and pi to -pi.  In these cases, subtract 2pi and add 2pi
    #    % respectively to all remaining points.  This effectively extends the
    #    % range outside the -pi to pi range.  Everything is re-centred later
    #    % when we subtract off the mean.
    #    
    #    % find discontinuities larger than pi (skeleton cannot change direction
    #    % more than pi from one segment to the next)
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

def calWormAnglesAll(skeleton, segment_size=5):
    
    segment_half = segment_size/2
    ang_pad =  (floor(segment_half),ceil(segment_half))
    
    meanAngles_all = np.zeros(skeleton.shape[0])    
    angles_all = np.full((skeleton.shape[0], skeleton.shape[1]), np.nan)
    for ss in range(skeleton.shape[0]):
        angles_all[ss,ang_pad[0]:-ang_pad[1]],meanAngles_all[ss] = \
        calWormAngles(skeleton[ss,:,0],skeleton[ss,:,1], segment_size=segment_size)
    
    return angles_all, meanAngles_all


def calWormArea(cnt_side1, cnt_side2):
    #x1y2 - x2y1(http://mathworld.wolfram.com/PolygonArea.html)
    #contour = np.vstack((cnt_side1, cnt_side2[::-1,:])) 
    
    contour = np.hstack((cnt_side1, cnt_side2[:,::-1,:])) 
    signed_area = np.sum(contour[:, :-1,0]*contour[:, 1:,1]-contour[:, 1:,0]*contour[:, :-1,1], axis=1)
    
    assert signed_area.size == contour.shape[0]
    return np.abs(signed_area)/2



class WormFromTable(NormalizedWorm):
    """
    Encapsulates the notion of a worm's elementary measurements, scaled
    (i.e. "normalized") to 49 points along the length of the worm.
    In the future it might become a method of NormalizedWorm, 
    but for the moment let's test it as a separate identity
    """
    def __init__(self, other=None):
        """
        Populates an empty normalized worm.
        If other is specified, this becomes a copy constructor.
        
        """
        NormalizedWorm.__init__(self)
    
    def fromFile(self, file_name, worm_index, fps = 25, rows_range = (0,0), isOpenWorm = False):
        self.file_name = file_name;
        with tables.File(file_name, 'r') as file_id:
            #data range
            if rows_range[0] == 0 and rows_range[1] == 0:
                #try to read it from the file
                tab = file_id.get_node('/trajectories_data')
                skeleton_id = tab.read_where('worm_index_joined==%i' % 1, field='skeleton_id')
                
                #the indexes must be continous
                assert np.all(np.diff(skeleton_id) == 1) 
                
                rows_range = (np.min(skeleton_id), np.max(skeleton_id))
                del skeleton_id
            
            ini, end = rows_range

            #video info, for the moment we intialize it with the fps
            self.video_info = VideoInfo('', fps)  
            #flag as segmented flags should be marked by the has_skeletons column
            self.video_info.frame_code = file_id.get_node('/trajectories_data').cols.has_skeleton[ini:end+1]
                        
            self.worm_index = worm_index
            self.rows_range = rows_range
            
            self.frame_number = file_id.get_node('/trajectories_data').read(ini,end,1,'frame_number')
            
            self.skeleton = file_id.get_node('/skeleton')[ini:end+1,:,:]
            self.ventral_contour = file_id.get_node('/contour_side1')[ini:end+1,:,:]
            self.dorsal_contour = file_id.get_node('/contour_side2')[ini:end+1,:,:]
            
            self.angles, meanAngles_all = calWormAnglesAll(self.skeleton)            
            
            self.length = file_id.get_node('/skeleton_length')[ini:end+1] 
            self.widths = file_id.get_node('/contour_width')[ini:end+1,:]
            
            self.n_segments = self.skeleton.shape[1]
            self.n_frames = self.skeleton.shape[0]
            
            try: 
                self.length = file_id.get_node('/skeleton_area')[ini:end+1] 
            except:
                self.area = calWormArea(self.ventral_contour, self.dorsal_contour)
            
            assert self.angles.shape[1] == self.n_segments
            assert self.skeleton.shape[2] == 2
            assert self.ventral_contour.shape[2] == 2
            assert self.dorsal_contour.shape[2] == 2
            
            assert self.ventral_contour.shape[1] == self.n_segments            
            assert self.dorsal_contour.shape[1] == self.n_segments
            assert self.widths.shape[1] == self.n_segments
            
            assert self.length.ndim == 1
            
            if isOpenWorm:
                self.changeAxis()
            
    def changeAxis(self):
        for field in ['skeleton', 'ventral_contour', 'dorsal_contour', 'widths', 'angles']:
            A = getattr(self, field)
            #roll axis to have it as 49x2xn
            A = np.rollaxis(A,0,A.ndim)
            #change the memory order so the last dimension changes slowly
            A = np.asfortranarray(A)
            #finally copy it back to the field
            setattr(self, field, A)
            
            assert getattr(self, field).shape[0] == self.n_segments

def walk_obj(obj, path = '', main_dict = {}, chr_sep='.'):
    for leaf_name in dir(obj):
        leaf = getattr(obj, leaf_name)
        module_name = type(leaf).__module__
        new_path =  path + chr_sep + leaf_name
        if module_name == np.__name__:
            main_dict[new_path] = leaf
        elif 'movement_validation' in module_name:
            walk_obj(leaf, new_path, main_dict, chr_sep)
    
    return main_dict

class wormStatsClass():
    def __init__(self):
        #get the info for each feature chategory
        specs_simple = specifications.SimpleSpecs.specs_factory()
        specs_event = specifications.EventSpecs.specs_factory()
        self.specs_motion = specifications.MovementSpecs.specs_factory()
    
        #create a new category for events where data corresponds to variable size numpy arrays
        self.specs_events = specs_simple + [x for x in specs_event \
        if not x.sub_field in ['time_ratio', 'data_ratio', 'frequency']]
        
        #create a category for data whose output is a float number that goes in the final feature tables
        self.specs4table =  [x for x in specs_event \
        if x.sub_field in ['time_ratio', 'data_ratio', 'frequency']]
        
        #condition the names given in spec to a more pytables friendly format.
        #remove spaces, and special punctuation marks, and remplace '/' by Ratio
        
        self.spec2tableName = {} #used to translate the spec name into the table name
        for spec in self.specs_events + self.specs_motion + self.specs4table:
            feature = spec.name.split(' (')[0].replace(' ', '_').replace('.', '').replace('-', '_')
            if '/' in feature:
                feature = feature.replace('/', '_') + '_Ratio'
            self.spec2tableName[spec.name] = feature.lower()

    def featureStat(self, stat_func, data, name, is_signed, is_motion, motion_mode = np.zeros(0), stats={}):
        # I prefer to keep this function quite independend and pass the stats and moition_mode argument 
        #rather than save those values in the class
        if data is None:
            data = np.zeros(0)
        
        motion_types = {'all':np.nan};
        if is_motion:
            #if the the feature is motion type we can subdivide in Foward, Paused or Backward motion
            assert motion_mode.size == data.size
            motion_types['Foward'] = 1;
            motion_types['Paused'] = 0;
            motion_types['Backward'] = -1;
        
        
        for key in motion_types:
            if key == 'all':
                valid = ~np.isnan(data)
                sub_name = name
            else:
                valid = motion_mode == motion_types[key]
                sub_name = name + '_' + key
                
            stats[sub_name] = stat_func(data[valid]);
            if is_signed:
                # if the feature is signed we can subdivide in positive, negative and absolute 
                stats[sub_name + '_Abs'] = stat_func(np.abs(data[valid]))
                stats[sub_name + '_Neg'] = stat_func(data[data<0 & valid])
                stats[sub_name + '_Pos'] = stat_func(data[data>0 & valid])

    def getWormStats(self, worm_features, stat_func = np.mean):
        ''' Calculate the statistics of an object worm features, subdividing data
            into Backward/Foward/Paused and/or Positive/Negative/Absolute, when appropiated.
            The default is to calculate the mean value, but this can be changed 
            using stat_func.
            
            Return the feature list as an ordered dictionary. 
        '''
        #initialize the stats dictionary
        self.stats = OrderedDict()
        
        #motion type stats
        motion_mode = worm_features.locomotion.motion_mode;
        for spec in self.specs_motion:
            feature = self.spec2tableName[spec.name]
            tmp_data = spec.get_data(worm_features)
            self.featureStat(stat_func, tmp_data, feature, spec.is_signed, True, motion_mode, stats=self.stats)
        
        for spec in self.specs_events:
            feature = self.spec2tableName[spec.name]
            tmp_data = spec.get_data(worm_features)
            self.featureStat(stat_func, tmp_data, feature, spec.is_signed, False, stats=self.stats)
        
        for spec in self.specs4table:
            feature = self.spec2tableName[spec.name]
            tmp_data = spec.get_data(worm_features)
            self.stats[feature] = tmp_data
        
        return self.stats

def getWormFeatures(skeletons_file, features_file, bad_seg_thresh = 0.5, fps = 25):
    #%%
    
    #useful to display progress 
    base_name = skeletons_file.rpartition('.')[0].rpartition(os.sep)[-1]
    
    #read skeletons index data
    with pd.HDFStore(skeletons_file, 'r') as ske_file_id:
        indexes_data = ske_file_id['/trajectories_data']
    
    if 'has_skeleton' in indexes_data.columns:
        indexes_data = indexes_data[['worm_index_joined', 'skeleton_id', 'has_skeleton']]
    else:
        indexes_data = indexes_data[['worm_index_joined', 'skeleton_id']]
        with tables.File(skeletons_file, 'r') as ske_file_id:
            #this is slow but faster than having to recalculate all the skeletons
            indexes_data['has_skeleton'] = ~np.isnan(ske_file_id.get_node('/skeleton_length'))
            
    #%%
    
    #get the fraction of worms that were skeletonized per trajectory
    dum = indexes_data.groupby('worm_index_joined').agg({'has_skeleton':['mean', 'sum']})
    skeleton_fracc = dum['has_skeleton']['mean']
    skeleton_tot = dum['has_skeleton']['sum']
    valid_worm_index = skeleton_fracc[(skeleton_fracc>=bad_seg_thresh) & (skeleton_tot>=fps)].index
    
    #remove the bad worms, we do not care about them
    indexes_data = indexes_data[indexes_data['worm_index_joined'].isin(valid_worm_index)]
    
    #get the first and last frame of each worm_index
    rows_indexes = indexes_data.groupby('worm_index_joined').agg({'skeleton_id':[min, max]})
    rows_indexes = rows_indexes['skeleton_id']
    
    #remove extra variable to free memory
    del indexes_data
    
    #initialize by getting the specs data subdivision
    wStats = wormStatsClass()
    
    #list to save trajectories mean features
    all_stats = []
    
    progress_timer = timeCounterStr('');
    #filter used for each fo the tables
    filters_tables = tables.Filters(complevel = 5, complib='zlib', shuffle=True)
    with tables.File(features_file, 'w') as features_fid:

        group_events = features_fid.create_group('/', 'Features_events')
        
        #initialize motion table. All the features here are a numpy array having the same length as the worm trajectory
        motion_header = {'worm_index':tables.Int32Col(pos=0),\
        'frame_number':tables.Int32Col(pos=1),\
        'Motion_Modes':tables.Float32Col(pos=2)}
        
        for ii, spec in enumerate(wStats.specs_motion):
            feature = wStats.spec2tableName[spec.name]
            motion_header[feature] = tables.Float32Col(pos=ii+2)
        table_motion = features_fid.create_table('/', 'Features_motion', motion_header, filters=filters_tables)
        
        #get the is_signed flag for motion specs and store it as an attribute
        #is_signed flag is used by featureStat in order to subdivide the data if required
        is_signed_motion = np.zeros(len(motion_header), np.uint8);
        for ii, spec in enumerate(wStats.specs_motion):
            feature = wStats.spec2tableName[spec.name]
            is_signed_motion[motion_header[feature]._v_pos] = spec.is_signed

        table_motion._v_attrs['is_signed'] = is_signed_motion
        
        #start to calculate features for each worm trajectory      
        tot_worms = len(rows_indexes)        
        for ind, dat  in enumerate(rows_indexes.iterrows()):
            worm_index, row_range = dat
            
            #initialize worm object, and extract data from skeletons file
            worm = WormFromTable()
            worm.fromFile(skeletons_file, worm_index, fps = 25, rows_range= tuple(row_range.values), isOpenWorm=False)
            

            worm.changeAxis()
            assert not np.all(np.isnan(worm.skeleton))
            
            # Generate the OpenWorm movement validation repo version of the features
            worm_features = WormFeatures(worm)
            
            #get the average for each worm feature
            worm_stats = wStats.getWormStats(worm_features, np.mean)
            worm_stats['n_frames'] = worm.n_frames
            worm_stats['worm_index'] = worm_index
            worm_stats.move_to_end('n_frames', last=False)
            worm_stats.move_to_end('worm_index', last=False)
            
            all_stats.append(worm_stats)
            
            #save the motion data as a general table
            motion_data = [[]]*len(motion_header)
            motion_data[motion_header['frame_number']._v_pos] = worm.frame_number
            motion_data[motion_header['worm_index']._v_pos] = np.full(worm.n_frames, worm.worm_index)
            motion_data[motion_header['Motion_Modes']._v_pos] = worm_features.locomotion.motion_mode
            for spec in wStats.specs_motion:
                feature = wStats.spec2tableName[spec.name]
                tmp_data = spec.get_data(worm_features)
                motion_data[motion_header[feature]._v_pos] = tmp_data
            
            motion_data = list(zip(*motion_data))
            table_motion.append(motion_data)
            table_motion.flush()
            del motion_data
            
            #save events data as a subgroup for the worm
            worm_node = features_fid.create_group(group_events, 'worm_%i' % worm_index )
            worm_node._v_attrs['worm_index'] = worm_index
            worm_node._v_attrs['frame_range'] = (worm.frame_number[0], worm.frame_number[-1])
            worm_node._v_attrs['skeletons_rows_range'] = tuple(row_range.values)
            for spec in wStats.specs_events:
                feature = wStats.spec2tableName[spec.name]
                tmp_data = spec.get_data(worm_features)
                
                if tmp_data is not None and tmp_data.size > 0:
                    table_tmp = features_fid.create_carray(worm_node, feature, \
                                    obj = tmp_data, filters=filters_tables)
                    table_tmp._v_attrs['is_signed'] = int(spec.is_signed)
            
            dd = " Extracting features. Worm %i of %i done." % (ind+1, tot_worms)
            dd = base_name + dd + ' Total time:' + progress_timer.getTimeStr()
            print(dd)
            sys.stdout.flush()
            
            
        #create and save a table containing the averaged worm feature for each worm
        tot_rows = len(all_stats)
        dtype = [(x, np.float32) for x in (all_stats[0])]
        mean_features_df = np.recarray(tot_rows, dtype = dtype);
        for kk, row_dict in enumerate(all_stats):
            for key in row_dict:
                mean_features_df[key][kk] = row_dict[key]
        feat_mean = features_fid.create_table('/', 'Features_means', obj = mean_features_df, filters=filters_tables)
        
        feat_mean._v_attrs['has_finished'] = 1
        
        print('Feature extraction finished:' + progress_timer.getTimeStr())
        sys.stdout.flush()
        
if __name__ == "__main__":
    
#    base_name = 'Capture_Ch3_12052015_194303'
#    mask_dir = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/20150512/'
#    results_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150512/'    

    base_name = 'Capture_Ch5_11052015_195105'
    mask_dir = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/20150511/'
    results_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150511/'
    
    masked_image_file = mask_dir + base_name + '.hdf5'
    trajectories_file = results_dir + base_name + '_trajectories.hdf5'
    skeletons_file = results_dir + base_name + '_skeletons.hdf5'
    features_file = results_dir + base_name + '_features.hdf5'
    
    assert os.path.exists(masked_image_file)
    assert os.path.exists(trajectories_file)
    assert os.path.exists(skeletons_file)
        
    getWormFeatures(skeletons_file, features_file)
