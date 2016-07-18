# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 01:56:06 2015

@author: ajaver
"""
import os, sys
import tables
import argparse
import time, datetime
import json

from MWTracker.trackWorms.getWormTrajectories import getWormTrajectories, correctTrajectories
from MWTracker.trackWorms.getDrawTrajectories import drawTrajectoriesVideo
from MWTracker.trackWorms.getSkeletonsTables import trajectories2Skeletons, writeIndividualMovies
from MWTracker.trackWorms.checkHeadOrientation import correctHeadTail
from MWTracker.trackWorms.getFilteredSkels import getFilteredSkels

from MWTracker.intensityAnalysis.getIntensityProfile import getIntensityProfile
from MWTracker.intensityAnalysis.correctHeadTailIntensity import correctHeadTailIntensity

from MWTracker.featuresAnalysis.obtainFeatures import getWormFeaturesFilt
from MWTracker.featuresAnalysis.obtainFeaturesHelper import switchCntSingleWorm

from MWTracker.helperFunctions.tracker_param import tracker_param
from MWTracker.helperFunctions.trackProvenance import getGitCommitHash, execThisPoint
from MWTracker.helperFunctions.miscFun import print_flush

from MWTracker.stageAligment.alignStageMotion import alignStageMotion

from collections import OrderedDict



#the order of the list is very IMPORTANT, and reflects the order where is step is done
checkpoint_label = ['TRAJ_CREATE', 'TRAJ_JOIN', 'SKE_CREATE', 'SKE_FILT', 'SKE_ORIENT', 
'INT_PROFILE', 'INT_SKE_ORIENT', 
'FEAT_CREATE','FEAT_MANUAL_CREATE', 'END']
checkpoint = {ii:x for x,ii in enumerate(checkpoint_label)}


def getStartingPoint(masked_image_file, results_dir):    
    '''determine for where to start. This is useful to check if the previous analysis was 
    completely succesfully, or if it was interrupted restarted from the last succesful step'''
    
    base_name, trajectories_file, skeletons_file, features_file, \
    feat_manual_file, intensities_file = constructNames(masked_image_file, results_dir)

    accepted_errors = (tables.exceptions.HDF5ExtError, tables.exceptions.NoSuchNodeError, KeyError, IOError)
    try:
        with tables.open_file(trajectories_file, mode = 'r') as traj_fid:
             trajectories = traj_fid.get_node('/plate_worms')
             if trajectories._v_attrs['has_finished'] == 0:
                 return checkpoint['TRAJ_CREATE'];
             elif trajectories._v_attrs['has_finished'] == 1:
                 return checkpoint['TRAJ_JOIN'];
    except accepted_errors:
        #if there is any problem while reading the file, create it again
        return checkpoint['TRAJ_CREATE'];

    try:
        with tables.File(skeletons_file, "r") as ske_file_id:
            skeleton_table = ske_file_id.get_node('/skeleton')
            if skeleton_table._v_attrs['has_finished'] <= 0:
                return checkpoint['SKE_CREATE'];
            elif skeleton_table._v_attrs['has_finished'] <= 1:
                return checkpoint['SKE_FILT'];
            elif skeleton_table._v_attrs['has_finished'] <= 2:
                return checkpoint['SKE_ORIENT'];

    except accepted_errors:
        #if there is any problem while reading the file, create it again
        return checkpoint['SKE_CREATE'];
    
    try:
        with tables.File(intensities_file, "r") as fid:
            int_med = fid.get_node('/straighten_worm_intensity_median')
            if int_med._v_attrs['has_finished'] < 1:
                return checkpoint['INT_PROFILE'];
    except accepted_errors:   
        return checkpoint['INT_PROFILE'];

    #at this point the skeleton file must exists so we only check for the correct level in the skeleton file
    with tables.File(skeletons_file, "r") as ske_file_id:
        skeleton_table = ske_file_id.get_node('/skeleton')
        if skeleton_table._v_attrs['has_finished'] <= 3:
            return checkpoint['INT_SKE_ORIENT'];

    try:
        with tables.File(features_file, "r") as feat_file_id:
            features_table = feat_file_id.get_node('/features_means')
            if features_table._v_attrs['has_finished'] == 0:
                return checkpoint['FEAT_CREATE'];

    except  accepted_errors:
        #if there is any problem while reading the file, create it again
        return checkpoint['FEAT_CREATE'];

    try:
        with tables.File(feat_manual_file, "r") as feat_file_id:
            features_table = feat_file_id.get_node('/features_means')
            if features_table._v_attrs['has_finished'] == 0:
                return checkpoint['FEAT_MANUAL_CREATE'];
    except  accepted_errors:
        #if there is any problem while reading the file, create it again
        with tables.File(skeletons_file, 'r') as ske_file_id:
            if 'worm_label' in ske_file_id.get_node('/trajectories_data').colnames:
                return checkpoint['FEAT_MANUAL_CREATE'];

        
    return checkpoint['END'];

def isBadStageAligment(skeletons_file):
    with tables.File(skeletons_file, 'r') as fid:
        try:
            good_aligment = fid.get_node('/stage_movement')._v_attrs['has_finished'][:]
        except (KeyError,IndexError, tables.exceptions.NoSuchNodeError):
            good_aligment = 0;
        
        return not good_aligment in [1,2]

def hasExpCntInfo(skeletons_file):
    #i'm reading this data twice (one more in switchCntSingleWorm), but I think this is cleaner
    #from a function organization point of view.
    with tables.File(skeletons_file, 'r') as fid:
        if not '/experiment_info' in fid:
            return True
        exp_info_b = fid.get_node('/experiment_info').read()
        exp_info = json.loads(exp_info_b.decode("utf-8"))
        
        #print('ventral_side:{}'.format(exp_info['ventral_side']))
        #only clockwise and anticlockwise are valid contour orientations
        return not exp_info['ventral_side'] in ['clockwise', 'anticlockwise']
        

def constructNames(masked_image_file, results_dir):
    base_name = masked_image_file.rpartition('.')[0].rpartition(os.sep)[-1]

    output = [base_name]
    
    ext2add = ['trajectories', 'skeletons', 'features', 'feat_manual', 'intensities']
    for ext in ext2add:
        output += [os.path.abspath(os.path.join(results_dir, base_name + '_' + ext + '.hdf5'))]
    
    return output

            
class getTrajectoriesWorker():
    def __init__(self, masked_image_file, results_dir, json_file ='',
    start_point = -1, end_point = checkpoint['END'], is_single_worm = False, 
    use_skel_filter = True, use_manual_join = False, cmd_original=''):
        
        #assert not is_single_worm
        #get repository commit hash numbers (useful to determine what version of the code was executed)
        self.commit_hash = getGitCommitHash()

        #derivate output_file names from the input_file and output_dir
        self.masked_image_file = masked_image_file
        self.results_dir = results_dir
        self.base_name, self.trajectories_file, self.skeletons_file, self.features_file, \
        self.feat_manual_file, self.intensities_file = \
        constructNames(self.masked_image_file, self.results_dir)
        
        if json_file: assert os.path.exists(json_file)

        #if starting point is not given, calculate it again
        self.start_point = start_point
        self.end_point = end_point

        if self.start_point < 0: self.start_point = getStartingPoint(self.masked_image_file, self.results_dir)

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        #check if the file with the masked images exists
        if self.start_point <= checkpoint['SKE_CREATE']: assert os.path.exists(masked_image_file)
        
        #get function parameters
        self.param = tracker_param(json_file)
        self.cmd_original = cmd_original

        self.use_manual_join = use_manual_join
        self.use_skel_filter = use_skel_filter
        self.is_single_worm = is_single_worm
        
        if self.is_single_worm:
            #we need to force parameters to obtain the correct features
            self.use_manual_join = False
            #self.use_skel_filter = False
            #self.param.head_tail_param['min_dist'] = 0

        #derive the inputs, functions, and output requires for each point in the analysis
        self.getPointsParameters()

        #execute the analysis depending on the start and ending points:
        self.execAllPoints()


    def getPointsParameters(self):
        self.points_parameters = {
        'TRAJ_CREATE': {
            'func':getWormTrajectories,
            'argkws':{ **{'masked_image_file':self.masked_image_file, 'trajectories_file':self.trajectories_file}, \
                        **self.param.trajectories_param},
            'output_file':self.trajectories_file
            },
        'TRAJ_JOIN': {
            'func':correctTrajectories,
            'argkws':{'trajectories_file':self.trajectories_file, 'is_single_worm':self.is_single_worm, \
                    'join_traj_param':self.param.join_traj_param},
            'output_file':self.trajectories_file
            },
        'SKE_ORIENT': {
            'func':correctHeadTail, 
            'argkws':{**{'skeletons_file':self.skeletons_file}, **self.param.head_tail_param},
            'output_file':self.skeletons_file
            },
        'SKE_CREATE': {
            'func':trajectories2Skeletons,
            'argkws':{**{'masked_image_file':self.masked_image_file, 'skeletons_file':self.skeletons_file, \
                    'trajectories_file':self.trajectories_file}, **self.param.skeletons_param},
            'output_file':self.skeletons_file
            },
        'SKE_FILT': {
            'func':getFilteredSkels,
            'argkws':{**{'skeletons_file':self.skeletons_file, 'use_skel_filter':self.use_skel_filter}, 
                         **self.param.feat_filt_param},
            'output_file':self.skeletons_file
            },
        'INT_PROFILE': {
            'func':getIntensityProfile, 
            'argkws':{**{'masked_image_file':self.masked_image_file , 'skeletons_file':self.skeletons_file, 
            'intensities_file':self.intensities_file}, **self.param.int_profile_param},
            'output_file':self.intensities_file
            },
        'INT_SKE_ORIENT': {
            'func':correctHeadTailIntensity, 
            'argkws':{**{'skeletons_file':self.skeletons_file, 'intensities_file':self.intensities_file}, 
            **self.param.head_tail_int_param},
            'output_file':self.skeletons_file
            },
        'FEAT_CREATE': {
            'func':getWormFeaturesFilt,
            'argkws':{'skeletons_file':self.skeletons_file, 'features_file':self.features_file, 
                'expected_fps': self.param.expected_fps, 'is_single_worm':self.is_single_worm, 
                'use_skel_filter':self.use_skel_filter, 'use_manual_join':False,
                'feat_filt_param':self.param.feat_filt_param},
            'output_file':self.features_file
            },
        'FEAT_MANUAL_CREATE': {
            'func':getWormFeaturesFilt,
            'argkws':{'skeletons_file':self.skeletons_file, 'features_file':self.feat_manual_file,  
                'expected_fps': self.param.expected_fps, 'is_single_worm':self.is_single_worm, 
                'use_skel_filter':self.use_skel_filter, 'use_manual_join':True,
                'feat_filt_param':self.param.feat_filt_param},
            'output_file':self.feat_manual_file
            },
        #only for single worm
        'STAGE_ALIGMENT': {
            'func':alignStageMotion,
            'argkws':{'masked_image_file':self.masked_image_file, 'skeletons_file':self.skeletons_file},
            'output_file':self.skeletons_file
            },
        'CONTOUR_ORIENT': {
            'func':switchCntSingleWorm,
            'argkws':{'skeletons_file':self.skeletons_file},
            'output_file':self.skeletons_file
            }
        }

    
    def execAllPoints(self):
        initial_time = time.time()

        #if start_point is larger than end_point there is nothing else to do 
        if self.start_point > self.end_point: 
            print_flush(self.base_name + ' Finished in ' + checkpoint_label[self.end_point])
            return
        
        print_flush(self.base_name + ' Starting checkpoint: ' + checkpoint_label[self.start_point])
            

        for ii in range(self.start_point, self.end_point+1):
            current_point = checkpoint_label[ii]
            if not self.use_manual_join and current_point == 'FEAT_MANUAL_CREATE': continue
            if current_point == 'END': break

            #extra steps for the single worm case
            if self.is_single_worm:
                if current_point == 'INT_PROFILE': 
                    execThisPoint('STAGE_ALIGMENT', **self.points_parameters['STAGE_ALIGMENT'], 
                    commit_hash=self.commit_hash, cmd_original=self.cmd_original)
                    
                    if isBadStageAligment(self.skeletons_file):
                        #break, bad video we do not need to calculate anything else.
                        current_point = 'STAGE_ALIGMENT'
                        break

                if current_point == 'FEAT_CREATE':
                    if hasExpCntInfo(self.skeletons_file):
                        current_point = 'CONTOUR_ORIENT'
                        break
                    else:
                        execThisPoint('CONTOUR_ORIENT', **self.points_parameters['CONTOUR_ORIENT'], 
                        commit_hash=self.commit_hash, cmd_original=self.cmd_original)

            #print(current_point, self.points_parameters[current_point]['func'])
            #print(self.points_parameters[current_point]['argkws'])
            execThisPoint(current_point, **self.points_parameters[current_point], 
                commit_hash=self.commit_hash, cmd_original=self.cmd_original)
        
        time_str = str(datetime.timedelta(seconds=round(time.time()-initial_time)))
        print_flush('%s  Finished in %s. Total time %s' % (self.base_name, current_point, time_str))
        

track_worker_parser = argparse.ArgumentParser(description="Track woms in an individual video hdf5 file.")
track_worker_parser.add_argument('masked_image_file', help = 'hdf5 video file.')
track_worker_parser.add_argument('results_dir', help = 'Directory where results are going to be saved.')
track_worker_parser.add_argument('--json_file', default = '', help='File (.json) containing the tracking parameters.')
track_worker_parser.add_argument('--start_point', type=int, default = -1, help = 'Force the program to start at a specific point in the analysis.')
track_worker_parser.add_argument('--end_point', type=int, default = checkpoint['END'], help='End point of the analysis.')
track_worker_parser.add_argument('--is_single_worm', action='store_true', help = 'This flag indicates if the video corresponds to the single worm case.')
track_worker_parser.add_argument('--use_manual_join', action='store_true', help = 'Use this flag to calculate features on manually joined data.')
track_worker_parser.add_argument('--use_skel_filter', action='store_true', help = 'Flag to filter valid skeletons using the movie robust averages.')
track_worker_parser.add_argument('--cmd_original', default = '', help = 'Internal. Used in provenance tracking.')

if __name__ == '__main__':
    
    if len(sys.argv)>1: 
        args = track_worker_parser.parse_args()
        getTrajectoriesWorker(**vars(args)) 
    else:
        print('Bad', sys.argv)
        