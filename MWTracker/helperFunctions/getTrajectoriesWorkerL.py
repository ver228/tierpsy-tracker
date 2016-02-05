# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 01:56:06 2015

@author: ajaver
"""
import os, sys
import tables

from ..trackWorms.getWormTrajectories import getWormTrajectories, correctTrajectories
from ..trackWorms.getDrawTrajectories import drawTrajectoriesVideo
from ..trackWorms.getSkeletonsTables import trajectories2Skeletons, writeIndividualMovies
from ..trackWorms.checkHeadOrientation import correctHeadTail

from ..featuresAnalysis.getFilteredFeats import getFilteredFeats
from ..featuresAnalysis.obtainFeatures import getWormFeaturesFilt

from ..helperFunctions.tracker_param import tracker_param
from ..helperFunctions.trackProvenance import getGitCommitHash, execThisPoint
from ..helperFunctions.miscFun import print_flush

from collections import OrderedDict



#the order of the list is very IMPORTANT, and reflects the order where is step is done
checkpoint_label = ['TRAJ_CREATE', 'TRAJ_JOIN', 'SKE_CREATE', 'SKE_ORIENT', 'SKE_FILT', 
'FEAT_CREATE','FEAT_MANUAL_CREATE', 'END']
checkpoint = {ii:x for x,ii in enumerate(checkpoint_label)}


def getStartingPoint(masked_image_file, results_dir):    
    '''determine for where to start. This is useful to check if the previous analysis was 
    completely succesfully, or if it was interrupted restarted from the last succesful step'''
    
    base_name, trajectories_file, skeletons_file, features_file, feat_ind_file = constructNames(masked_image_file, results_dir)

    try:
        with tables.open_file(trajectories_file, mode = 'r') as traj_fid:
             trajectories = traj_fid.get_node('/plate_worms')
             if trajectories._v_attrs['has_finished'] == 0:
                 return checkpoint['TRAJ_CREATE'];
             elif trajectories._v_attrs['has_finished'] == 1:
                 return checkpoint['TRAJ_JOIN'];
    except:
        #if there is any problem while reading the file, create it again
        return checkpoint['TRAJ_CREATE'];

    try:
        with tables.File(skeletons_file, "r") as ske_file_id:
            skeleton_table = ske_file_id.get_node('/skeleton')
            if skeleton_table._v_attrs['has_finished'] == 0:
                return checkpoint['SKE_CREATE'];
            elif skeleton_table._v_attrs['has_finished'] == 1:
                return checkpoint['SKE_ORIENT'];
            elif skeleton_table._v_attrs['has_finished'] == 2:
                return checkpoint['SKE_FILT'];
    except:
        #if there is any problem while reading the file, create it again
        return checkpoint['SKE_CREATE'];
    

    try:
        with tables.File(features_file, "r") as feat_file_id:
            features_table = feat_file_id.get_node('/features_means')
            if features_table._v_attrs['has_finished'] == 0:
                return checkpoint['FEAT_CREATE'];

    except:
        #if there is any problem while reading the file, create it again
        return checkpoint['FEAT_CREATE'];
    
    try:
        with tables.File(feat_ind_file, "r") as feat_file_id:
            features_table = feat_file_id.get_node('/features_means')
            if features_table._v_attrs['has_finished'] == 0:
                return checkpoint['FEAT_MANUAL_CREATE'];
    except:
        #if there is any problem while reading the file, create it again
        with tables.File(skeletons_file, 'r') as ske_file_id:
            if 'worm_label' in ske_file_id.get_node('/trajectories_data').colnames:
                return checkpoint['FEAT_MANUAL_CREATE'];

        
    return checkpoint['END'];


def constructNames(masked_image_file, results_dir):
    base_name = masked_image_file.rpartition('.')[0].rpartition(os.sep)[-1]

    output = [base_name]
    
    ext2add = ['trajectories', 'skeletons', 'features', 'feat_manual']
    for ext in ext2add:
        output += [os.path.abspath(os.path.join(results_dir, base_name + '_' + ext + '.hdf5'))]
    
    return output

            
class getTrajectoriesWorkerL():
    def __init__(self, masked_image_file, results_dir, json_file ='',
    start_point = -1, end_point = checkpoint['END'], is_single_worm = False, 
    use_auto_label = True, use_manual_join = False, cmd_original=''):
    
        #get repository commit hash numbers (useful to determine what version of the code was executed)
        self.commit_hash = getGitCommitHash()

        #derivate output_file names from the input_file and output_dir
        self.masked_image_file = masked_image_file
        self.results_dir = results_dir
        self.base_name, self.trajectories_file, self.skeletons_file, self.features_file, self.feat_manual_file = \
        constructNames(self.masked_image_file, self.results_dir)
        

        #if starting point is not given, calculate it again
        self.start_point = start_point
        self.end_point = end_point

        if self.start_point < 0: self.start_point = getStartingPoint(self.masked_image_file, self.results_dir)

        assert os.path.exists(self.results_dir)
        #check if the file with the masked images exists
        if self.start_point <= checkpoint['SKE_CREATE']: assert os.path.exists(masked_image_file)
        
        
        #get function parameters
        if json_file: assert os.path.exists(json_file)
        
        self.param = tracker_param(json_file)
        self.cmd_original = cmd_original

        self.use_manual_join = use_manual_join
        self.use_auto_label = use_auto_label
        self.is_single_worm = is_single_worm
        
        if self.is_single_worm:
            #we need to force parameters to obtain the correct features
            self.use_manual_join = False
            self.use_auto_label = False
            self.param.head_tail_param['min_dist'] = 0

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
        'SKE_CREATE': {
            'func':trajectories2Skeletons,
            'argkws':{**{'masked_image_file':self.masked_image_file, 'skeletons_file':self.skeletons_file, \
                    'trajectories_file':self.trajectories_file}, **self.param.skeletons_param},
            'output_file':self.skeletons_file
            },
        'SKE_ORIENT': {
            'func':correctHeadTail, 
            'argkws':{**{'skeletons_file':self.skeletons_file}, **self.param.head_tail_param},
            'output_file':self.skeletons_file
            },
        'SKE_FILT': {
            'func':getFilteredFeats,
            'argkws':{**{'skeletons_file':self.skeletons_file, 'use_auto_label':self.use_auto_label}, 
                         **self.param.feat_filt_param},
            'output_file':self.skeletons_file
            },
        'FEAT_CREATE': {
            'func':getWormFeaturesFilt,
            'argkws':{'skeletons_file':self.skeletons_file, 'features_file':self.features_file, 
                'use_auto_label':self.use_auto_label, 'use_manual_join':False, 
                'feat_filt_param':self.param.feat_filt_param},
            'output_file':self.features_file
            },
        'FEAT_MANUAL_CREATE': {
            'func':getWormFeaturesFilt,
            'argkws':{'skeletons_file':self.skeletons_file, 'features_file':self.features_file, 
                'use_auto_label':self.use_auto_label, 'use_manual_join':True, 
                'feat_filt_param':self.param.feat_filt_param},
            'output_file':self.features_file
            }
        }

    
    def execAllPoints(self):
        #if start_point is larger than end_point there is nothing else to do 
        if self.start_point > self.end_point: 
            print_flush(self.base_name + ' Finished in ' + checkpoint_label[self.end_point])
            return
        
        print_flush(self.base_name + ' Starting checkpoint: ' + checkpoint_label[self.start_point])

        for ii in range(self.start_point, self.end_point+1):
            current_point = checkpoint_label[ii]
            
            if not self.use_manual_join and current_point == 'FEAT_MANUAL_CREATE': continue
            if current_point == 'END': break

            execThisPoint(current_point, **self.points_parameters[current_point], 
                commit_hash=self.commit_hash, cmd_original=self.cmd_original)
            
        print_flush(self.base_name + ' Finished in ' + checkpoint_label[self.end_point])
        
    