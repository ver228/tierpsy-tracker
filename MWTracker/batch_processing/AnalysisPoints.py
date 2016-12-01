# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 17:24:27 2016

@author: ajaver
"""

import os
from functools import partial

from MWTracker.analysis.vid_subsample.createSampleVideo import createSampleVideo, getSubSampleVidName
from MWTracker.analysis.compress.processVideo import processVideo, isGoodVideo
from MWTracker.analysis.contour_orient.correctVentralDorsal import switchCntSingleWorm, hasExpCntInfo
from MWTracker.analysis.int_ske_orient.correctHeadTailIntensity import correctHeadTailIntensity
from MWTracker.analysis.compress_add_data.getAdditionalData import storeAdditionalDataSW, hasAdditionalFiles
from MWTracker.analysis.feat_create.obtainFeatures import getWormFeaturesFilt, hasManualJoin
from MWTracker.analysis.int_profile.getIntensityProfile import getIntensityProfile
from MWTracker.analysis.ske_create.getSkeletonsTables import trajectories2Skeletons
from MWTracker.analysis.ske_filt.getFilteredSkels import getFilteredSkels
from MWTracker.analysis.ske_orient.checkHeadOrientation import correctHeadTail
from MWTracker.analysis.stage_aligment.alignStageMotion import alignStageMotion, isGoodStageAligment
from MWTracker.analysis.traj_create.getWormTrajectories import getWormTrajectories
from MWTracker.analysis.traj_join.correctTrajectories import correctTrajectories
from MWTracker.batch_processing.CheckFinished import CheckFinished
from MWTracker.helper.tracker_param import tracker_param


class AnalysisPoints(object):
    def __init__(self, video_file, masks_dir, results_dir, json_file = '',
                 is_single_worm = False, use_skel_filter=True):
        
        self.getFileNames(video_file, masks_dir, results_dir)
        
        self.video_file = video_file
        self.masks_dir = masks_dir
        self.results_dir = results_dir
        
        self.param = tracker_param(json_file)
        self.is_single_worm = is_single_worm
        self.use_skel_filter = use_skel_filter
        
        self.buildPoints()
        self.checker = CheckFinished(output_files = self.getField('output_files'))
        
    def getFileNames(self, video_file, masks_dir, results_dir):
        base_name = video_file.rpartition('.')[0].rpartition(os.sep)[-1]
        results_dir = os.path.abspath(results_dir)
        
        output = {'base_name' : base_name, 'original_video' : video_file}
        output['masked_image'] = os.path.join(masks_dir, base_name + '.hdf5')
    
        ext2add = [
            'trajectories',
            'skeletons',
            'features',
            'feat_manual',
            'intensities']

        for ext in ext2add:
            output[ext] = os.path.join(results_dir, base_name + '_' + ext + '.hdf5')
        
        output['subsample'] = getSubSampleVidName(output['masked_image'])

        self.file_names =  output
        self.file2dir_dict = {fname:dname for dname, fname in map(os.path.split, self.file_names.values())}
    
    def getField(self, key, points2get = 0):
        if isinstance(points2get, int):
            points2get = self.checkpoints
        return {x:self.checkpoints[x][key] for x in points2get}
        
    def getArgs(self, point):
        return {x:self.checkpoints[point][x] for x in ['func', 'argkws', 'provenance_file']}
    
    def buildPoints(self):
         #simple alias to reduce the writing
        fn = self.file_names
        param = self.param
        is_single_worm = self.is_single_worm
        use_skel_filter = self.use_skel_filter
        
        #THE FIRST ELEMENT IN OUTPUT_FILES MUST BE AND HDF5 AND WILL BE USED AS FLAG TO 
        #STORE THE PROVENANCE TRACKING
        
        self.checkpoints = {
            'compress' : {
                'func':processVideo,
                'argkws' : {'video_file': fn['original_video'], 
                            'masked_image_file' : fn['masked_image'],
                            'compress_vid_param' : param.compress_vid_param},
                'input_files' : [fn['original_video']],
                'output_files': [fn['masked_image']],
                'requirements' : [('can_read_video', partial(isGoodVideo, fn['original_video']))]
            },
            'vid_subsample': {
                'func':createSampleVideo,
                'argkws' : {**{'masked_image_file': fn['masked_image'], 'sample_video_name':fn['subsample']}, 
                            **param.subsample_vid_param},
                'input_files' : [fn['masked_image']],
                'output_files': [fn['masked_image'], fn['subsample']],
                'requirements' : ['compress']
            },
            'traj_create': {
                'func': getWormTrajectories,
                'argkws': {**{'masked_image_file': fn['masked_image'], 'trajectories_file': fn['trajectories']},
                           **param.trajectories_param},
                'input_files' : [fn['masked_image']],
                'output_files': [fn['trajectories']],
                'requirements' : ['compress']
            },
            'traj_join': {
                'func': correctTrajectories,
                'argkws': {'trajectories_file': fn['trajectories'], 'is_single_worm': is_single_worm,
                           'join_traj_param': param.join_traj_param},
                'input_files' : [fn['trajectories']],
                'output_files': [fn['trajectories']],
                'requirements' : ['traj_create']
            },
            'ske_create': {
                'func': trajectories2Skeletons,
                'argkws': {**{'masked_image_file': fn['masked_image'], 'skeletons_file': fn['skeletons'],
                              'trajectories_file': fn['trajectories']}, **param.skeletons_param},
                'input_files' : [fn['trajectories'],fn['masked_image']],
                'output_files': [fn['skeletons']],
                'requirements' : ['traj_join']
            },
            'ske_orient': {
                'func': correctHeadTail,
                'argkws': {**{'skeletons_file': fn['skeletons']}, **param.head_tail_param},
                'input_files' : [fn['skeletons']],
                'output_files': [fn['skeletons']],
                'requirements' : ['ske_create']
            },
            'ske_filt': {
                'func': getFilteredSkels,
                'argkws': {**{'skeletons_file': fn['skeletons']}, **param.feat_filt_param},
                'input_files' : [fn['skeletons']],
                'output_files': [fn['skeletons']],
                'requirements' : ['ske_create']
            },
            'int_profile': {
                'func': getIntensityProfile,
                'argkws': {**{'masked_image_file': fn['masked_image'], 'skeletons_file': fn['skeletons'],
                              'intensities_file': fn['intensities']}, **param.int_profile_param},
                'input_files' : [fn['skeletons'],fn['masked_image']],
                'output_files': [fn['intensities']],
                'requirements' : ['ske_create']
            },
            'int_ske_orient': {
                'func': correctHeadTailIntensity,
                'argkws': {**{'skeletons_file': fn['skeletons'], 'intensities_file': fn['intensities']},
                           **self.param.head_tail_int_param},
                'input_files' : [fn['skeletons'], fn['intensities']],
                'output_files': [fn['skeletons']],
                'requirements' : ['int_profile']
            },
            'feat_create': {
                'func': getWormFeaturesFilt,
                'argkws': {'skeletons_file': fn['skeletons'], 'features_file': fn['features'],
                           **param.feats_param,
                           'is_single_worm': is_single_worm, 'use_skel_filter': use_skel_filter, 'use_manual_join': False
                           },
                'input_files' : [fn['skeletons']],
                'output_files': [fn['features']],
                'requirements' : ['ske_create']
            },
            'feat_manual_create': {
                'func': getWormFeaturesFilt,
                'argkws': {'skeletons_file': fn['skeletons'], 'features_file': fn['feat_manual'],
                           **param.feats_param,
                           'is_single_worm': False, 'use_skel_filter': use_skel_filter, 'use_manual_join': True,
                           },
                'input_files' : [fn['skeletons']],
                'output_files': [fn['feat_manual']],
                'requirements' : ['ske_create',
                                  ('has_manual_joined_traj', partial(hasManualJoin, fn['skeletons']))]
            },
        }
        
        # points only for single worm
        if is_single_worm:
            self.checkpoints['compress_add_data'] = {
                'func':storeAdditionalDataSW,
                'argkws' : {'video_file': fn['original_video'], 'masked_image_file': fn['masked_image']},
                'input_files' : [fn['original_video'], fn['masked_image']],
                'output_files': [fn['masked_image']],
                'requirements' : ['compress']
            }
            self.checkpoints['stage_aligment'] = {
                'func': alignStageMotion,
                'argkws': {'masked_image_file': fn['masked_image'], 'skeletons_file': fn['skeletons']},
                'input_files' : [fn['skeletons'], fn['masked_image']],
                'output_files': [fn['skeletons']],
                'requirements' : ['compress_add_data', 'ske_create']
            }
            self.checkpoints['contour_orient'] = {
                'func': switchCntSingleWorm,
                'argkws': {'skeletons_file': fn['skeletons']},
                'input_files' : [fn['skeletons']],
                'output_files': [fn['skeletons']],
                'requirements' : ['ske_create',
                                  ('has_contour_info', partial(hasExpCntInfo, fn['skeletons']))]
            }
            #make sure the file has the additional files, even before start compression
            for key in ['compress', 'compress_add_data']:
                self.checkpoints[key]['requirements'] += \
            [('has_additional_files', partial(hasAdditionalFiles, fn['original_video']))]
            
            #make sure the stage was aligned correctly
            for key in ['ske_filt', 'ske_orient', 'int_profile', 'int_ske_orient','feat_create']:
                self.checkpoints['feat_create']['requirements'] += \
                        [('is_stage_aligned', partial(isGoodStageAligment, fn['skeletons']))]
            
            
        
        #add provenance file field if it is not explicity added
        for point in self.checkpoints:
            self.checkpoints[point]['provenance_file'] = self.checkpoints[point]['output_files'][0]
            assert self.checkpoints[point]['provenance_file'].endswith('.hdf5')
    
    def hasRequirements(self, point):
        requirements_results = {}

        #check the requirements of a given point
        for requirement in self.checkpoints[point]['requirements']:
            if isinstance(requirement, str):
                #if the requirement is a string, check the requirement with the checker 
                requirements_results[requirement] = self.checker.get(requirement)
            else:
                try:
                    requirements_results[requirement[0]] = requirement[1]()
                except (OSError): 
                    #if there is a problem with the file return back requirement
                    requirements_results[requirement[0]] = False
        
        self.unmet_requirements = [x for x in requirements_results if not requirements_results[x]]
        
        return self.unmet_requirements
    
    def getUnfinishedPoints(self, checkpoints2process):
        return self.checker.getUnfinishedPoints(checkpoints2process)
    
