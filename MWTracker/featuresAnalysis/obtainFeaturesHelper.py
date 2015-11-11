# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 2015

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

#add movement_validation path
#from .. import config_param as param
#sys.path.append(movement_validation_dir)
from MWTracker import config_param #import the movement_validation directory
from movement_validation import NormalizedWorm, VideoInfo
from movement_validation.statistics import specifications

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
    
    def fromFile(self, file_name, worm_index, fps = 25, isOpenWorm = False, pix2mum = 1, time_range = []):
        
        assert len(time_range) == 0 or len(time_range) == 2
        
        #get the skeletons_id and frame_number in case they were not given by the user
        with pd.HDFStore(file_name, 'r') as ske_file_id:
            trajectories_data = ske_file_id['/trajectories_data']
            good = trajectories_data['worm_index_N']==worm_index
            
            #if a time_range is given only consider frames within the time range 
            if len(time_range) == 2:
                good = good & (trajectories_data['frame_number']>=time_range[0]) \
                & (trajectories_data['frame_number']<=time_range[1])
                
            
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
            
            self.angles, meanAngles_all = calWormAnglesAll(self.skeleton)

            #if the area field exists read it otherwise calculated
            try: 
                self.length = file_id.get_node('/skeleton_area')[ini:end+1] 
            except:
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
            motion_types['foward'] = 1;
            motion_types['paused'] = 0;
            motion_types['backward'] = -1;
        
        
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
                stats[sub_name + '_abs'] = stat_func(np.abs(data[valid]))
                stats[sub_name + '_neg'] = stat_func(data[data<0 & valid])
                stats[sub_name + '_pos'] = stat_func(data[data>0 & valid])

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