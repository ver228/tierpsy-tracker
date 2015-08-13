# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:30:53 2015

@author: ajaver
"""
import os
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
from movement_validation import WormFeatures, VideoInfo, FeatureProcessingOptions
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
    '''calculate the angles of each of the skeletons'''

    segment_half = segment_size/2
    ang_pad =  (floor(segment_half),ceil(segment_half))
    
    meanAngles_all = np.zeros(skeleton.shape[0])    
    angles_all = np.full((skeleton.shape[0], skeleton.shape[1]), np.nan)
    for ss in range(skeleton.shape[0]):
        if skeleton[ss,0,0] == np.nan:
            continue #skip if skeleton is invalid

        angles_all[ss,ang_pad[0]:-ang_pad[1]],meanAngles_all[ss] = \
        calWormAngles(skeleton[ss,:,0],skeleton[ss,:,1], segment_size=segment_size)
    
    return angles_all, meanAngles_all


def calWormArea(cnt_side1, cnt_side2):
    '''calculate the contour area using the shoelace method'''
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
    
    def fromFile(self, file_name, worm_index, fps = 25, isOpenWorm = False, pix2mum = 1):
        
        #get the skeletons_id and frame_number in case they were not given by the user
        with pd.HDFStore(file_name, 'r') as ske_file_id:
            trajectories_data = ske_file_id['/trajectories_data']
            good = trajectories_data['worm_index_N']==worm_index
            #good = good & (trajectories_data['frame_number']<10000)
            trajectories_data = trajectories_data.loc[good, ['skeleton_id', 'frame_number', 'has_skeleton']]
            
            skeleton_id = trajectories_data['skeleton_id'].values
            frame_number = trajectories_data['frame_number'].values
            frame_code = trajectories_data['has_skeleton'].values
            
            del trajectories_data         
        
        
        self.file_name = file_name;
        self.worm_index = worm_index

        self.first_frame = np.min(frame_number)
        self.last_frame = np.max(frame_number)
        self.n_frames = self.last_frame - self.first_frame +1;
        
        with tables.File(file_name, 'r') as ske_file_id:
            n_ske_points =  ske_file_id.get_node('/skeleton').shape[1]
            self.n_segments = n_ske_points
            
            tot_frames = self.n_frames

            self.skeleton = np.full((tot_frames, n_ske_points,2), np.nan)
            self.ventral_contour = np.full((tot_frames, n_ske_points,2), np.nan)
            self.dorsal_contour = np.full((tot_frames, n_ske_points,2), np.nan)
            self.length = np.full(tot_frames, np.nan)
            self.widths = np.full((tot_frames,n_ske_points), np.nan)
    
            self.frame_number = np.full(tot_frames, -1, np.int32)

            ind_ff = frame_number - self.first_frame
            self.frame_number[ind_ff] = frame_number
            
            self.skeleton_id = np.full(tot_frames, -1, np.int32)
            self.skeleton_id[ind_ff] = skeleton_id

            #video info, for the moment we intialize it with the fps
            self.video_info = VideoInfo('', fps)  
            #flag as segmented flags should be marked by the has_skeletons column
            self.video_info.frame_code = np.zeros(tot_frames, np.int32)
            self.video_info.frame_code[ind_ff] = frame_code

            self.skeleton[ind_ff] = ske_file_id.get_node('/skeleton')[skeleton_id,:,:]*pix2mum
            self.ventral_contour[ind_ff] = ske_file_id.get_node('/contour_side1')[skeleton_id,:,:]*pix2mum
            self.dorsal_contour[ind_ff] = ske_file_id.get_node('/contour_side2')[skeleton_id,:,:]*pix2mum
            self.length[ind_ff] = ske_file_id.get_node('/skeleton_length')[skeleton_id]*pix2mum
            self.widths[ind_ff] = ske_file_id.get_node('/contour_width')[skeleton_id,:]*pix2mum

#             #READ DATA ON BLOCKS. This is faster than fancy indexing
#            block_ind, block_frame = self.getBlockInd(skeleton_id, frame_number, self.first_frame)
#            for bi, bf in zip(*(block_ind, block_frame)):
#                self.skeleton[bf[0]:bf[1]+1] = \
#                ske_file_id.get_node('/skeleton')[bi[0]:bi[1]+1]
#                
#                self.ventral_contour[bf[0]:bf[1]+1] = \
#                ske_file_id.get_node('/contour_side1')[bi[0]:bi[1]+1]
#    
#                self.dorsal_contour[bf[0]:bf[1]+1] = \
#                ske_file_id.get_node('/contour_side2')[bi[0]:bi[1]+1]
#    
#                self.length[bf[0]:bf[1]+1] = \
#                ske_file_id.get_node('/skeleton_length')[bi[0]:bi[1]+1]
#    
#                self.widths[bf[0]:bf[1]+1] = \
#                ske_file_id.get_node('/contour_width')[bi[0]:bi[1]+1]
                
            
            self.angles, meanAngles_all = calWormAnglesAll(self.skeleton)            
            self.area = calWormArea(self.ventral_contour, self.dorsal_contour)
            
            #assertions to check the data has the proper dimensions
            assert self.frame_number.shape[0] == self.n_frames
            assert self.skeleton_id.shape[0] == self.n_frames
            assert self.video_info.frame_code.shape[0] == self.n_frames
            
            assert self.skeleton.shape[0] == self.n_frames
            assert self.ventral_contour.shape[0] == self.n_frames
            assert self.dorsal_contour.shape[0] == self.n_frames
            assert self.length.shape[0] == self.n_frames
            assert self.widths.shape[0] == self.n_frames
            
            assert self.angles.shape[0] == self.n_frames
            assert self.area.shape[0] == self.n_frames
            
            
            assert self.skeleton.shape[2] == 2
            assert self.ventral_contour.shape[2] == 2
            assert self.dorsal_contour.shape[2] == 2
            
            assert self.ventral_contour.shape[1] == self.n_segments            
            assert self.dorsal_contour.shape[1] == self.n_segments
            assert self.widths.shape[1] == self.n_segments
            
            assert self.length.ndim == 1
            
            assert self.angles.shape[1] == self.n_segments
            
            #check the axis dimensions to make it compatible with openworm
            if isOpenWorm:
                self.changeAxis()
    
    def getBlockInd(self, skeleton_id, frame_number, first_frame):
        
        #use frame_number since it could contain more gaps than skeleton_id
        jumps = np.where(np.diff(frame_number) != 1)[0]
        block_ind = []
        block_frame = []
        for n in range(jumps.size+1):
            if n == 0:
                ini = skeleton_id[0]
                ini_f = frame_number[0] - first_frame
            else:
                ii = jumps[n-1]+1
                ini = skeleton_id[ii]
                ini_f = frame_number[ii] - first_frame
    
            if n >= jumps.size:
                fin = skeleton_id[-1]
                fin_f = frame_number[-1] - first_frame
            else:
                ii = jumps[n]
                fin = skeleton_id[ii]
                fin_f = frame_number[ii] - first_frame
    
            block_ind.append((ini, fin))
            block_frame.append((ini_f, fin_f))
            
        return block_ind, block_frame    
    
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

class wormStatsClass():
    def __init__(self):
        '''get the info for each feature chategory'''
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
                feature = feature.replace('/', '_') + '_ratio'
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


def getWormFeaturesLab(skeletons_file, features_file, worm_indexes, fps = 25):

    #overight processing options
    processing_options = FeatureProcessingOptions()
    #increase the time window (s) for the velocity calculation 
    processing_options.locomotion.velocity_tip_diff = 0.5
    processing_options.locomotion.velocity_body_diff = 1

    #useful to display progress 
    base_name = skeletons_file.rpartition('.')[0].rpartition(os.sep)[-1]
    
    #initialize by getting the specs data subdivision
    wStats = wormStatsClass()

    #list to save trajectories mean features
    all_stats = []
    
    progress_timer = timeCounterStr('');

    #filter used for each fo the tables
    filters_tables = tables.Filters(complevel = 5, complib='zlib', shuffle=True)
    
    #create the motion table header
    motion_header = {'frame_number':tables.Int32Col(pos=0),\
    'skeleton_id':tables.Int32Col(pos=1),\
    'motion_modes':tables.Float32Col(pos=2)}

    for ii, spec in enumerate(wStats.specs_motion):
        feature = wStats.spec2tableName[spec.name]
        motion_header[feature] = tables.Float32Col(pos=ii+2)

    #get the is_signed flag for motion specs and store it as an attribute
    #is_signed flag is used by featureStat in order to subdivide the data if required
    is_signed_motion = np.zeros(len(motion_header), np.uint8);
    for ii, spec in enumerate(wStats.specs_motion):
        feature = wStats.spec2tableName[spec.name]
        is_signed_motion[motion_header[feature]._v_pos] = spec.is_signed


    with tables.File(features_file, 'w') as features_fid:

        #features group
        group_features = features_fid.create_group('/', 'features')

        #Calculate features for each worm trajectory      
        tot_worms = len(worm_indexes)        
        for ind, worm_index  in enumerate(worm_indexes):
            #initialize worm object, and extract data from skeletons file
            worm = WormFromTable()
            worm.fromFile(skeletons_file, worm_index, fps = fps, isOpenWorm=False)
            assert not np.all(np.isnan(worm.skeleton))

            #save data as a subgroup for each worm
            worm_node = features_fid.create_group(group_features, 'worm_%i' % worm_index )
            worm_node._v_attrs['worm_index'] = worm_index
            worm_node._v_attrs['frame_range'] = (worm.frame_number[0], worm.frame_number[-1])

            #save skeleton
            features_fid.create_carray(worm_node, 'skeletons', \
                                    obj = worm.skeleton, filters=filters_tables)
            
            #change axis to an openworm format
            worm.changeAxis()

            # Generate the OpenWorm movement validation repo version of the features
            worm_features = WormFeatures(worm, processing_options=processing_options)
            

            #get the average for each worm feature
            worm_stats = wStats.getWormStats(worm_features, np.mean)
            worm_stats['n_frames'] = worm.n_frames
            worm_stats['worm_index'] = worm_index
            worm_stats.move_to_end('n_frames', last=False)
            worm_stats.move_to_end('worm_index', last=False)
            all_stats.append(worm_stats)
            
            
            #save event features
            events_node = features_fid.create_group(worm_node, 'events')
            for spec in wStats.specs_events:
                feature = wStats.spec2tableName[spec.name]
                tmp_data = spec.get_data(worm_features)
                
                if tmp_data is not None and tmp_data.size > 0:
                    table_tmp = features_fid.create_carray(events_node, feature, \
                                    obj = tmp_data, filters=filters_tables)
                    table_tmp._v_attrs['is_signed'] = int(spec.is_signed)
            
            dd = " Extracting features. Worm %i of %i done." % (ind+1, tot_worms)
            dd = base_name + dd + ' Total time:' + progress_timer.getTimeStr()
            print(dd)
            
            #initialize motion table. All the features here are a numpy array having the same length as the worm trajectory
            table_motion = features_fid.create_table(worm_node, 'locomotion', motion_header, filters=filters_tables)
            table_motion._v_attrs['is_signed'] = is_signed_motion
            
            #save the motion data as a general table
            motion_data = [[]]*len(motion_header)
            motion_data[motion_header['frame_number']._v_pos] = worm.frame_number
            #motion_data[motion_header['worm_index']._v_pos] = np.full(worm.n_frames, worm.worm_index)
            motion_data[motion_header['skeleton_id']._v_pos] = worm.skeleton_id
            motion_data[motion_header['motion_modes']._v_pos] = worm_features.locomotion.motion_mode
            for spec in wStats.specs_motion:
                feature = wStats.spec2tableName[spec.name]
                tmp_data = spec.get_data(worm_features)
                motion_data[motion_header[feature]._v_pos] = tmp_data
            
            motion_data = list(zip(*motion_data))
            table_motion.append(motion_data)
            
            table_motion.flush()
            del motion_data
            
        #create and save a table containing the averaged worm feature for each worm
        tot_rows = len(all_stats)
        dtype = [(x, np.float32) for x in (all_stats[0])]
        mean_features_df = np.recarray(tot_rows, dtype = dtype);
        for kk, row_dict in enumerate(all_stats):
            for key in row_dict:
                mean_features_df[key][kk] = row_dict[key]
        feat_mean = features_fid.create_table('/', 'features_means', obj = mean_features_df, filters=filters_tables)
        
        feat_mean._v_attrs['has_finished'] = 1
        
        print('Feature extraction finished:' + progress_timer.getTimeStr())


