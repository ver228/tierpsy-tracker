# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 01:56:06 2015

@author: ajaver
"""
import os, sys
import tables

from .. import config_param

from ..trackWorms.getWormTrajectories import getWormTrajectories, joinTrajectories, correctSingleWormCase
from ..trackWorms.getDrawTrajectories import drawTrajectoriesVideo
from ..trackWorms.getSkeletonsTables import trajectories2Skeletons, writeIndividualMovies
from ..trackWorms.checkHeadOrientation import correctHeadTail

from ..featuresAnalysis.obtainFeatures import getWormFeatures
from ..featuresAnalysis.obtainFeatures_N import featFromLabSkel

from ..helperFunctions.tracker_param import tracker_param


checkpoint = {'TRAJ_CREATE':0, 'TRAJ_JOIN':1, 'TRAJ_VID':2, 
'SKE_CREATE':3, 'SKE_ORIENT':4, 'FEAT_CREATE':5, 'FEAT_IND': 6,
 'END':1e6} 

checkpoint_label = {}
for key in checkpoint:
    checkpoint_label[checkpoint[key]] = key

def print_flush(pstr):
        print(pstr)
        sys.stdout.flush()

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
                return checkpoint['FEAT_IND'];
    except:
        #if there is any problem while reading the file, create it again
        with tables.File(skeletons_file, 'r') as ske_file_id:
            if 'worm_label' in ske_file_id.get_node('/trajectories_data').colnames:
                return checkpoint['FEAT_IND'];

        
    return checkpoint['END'];


def constructNames(masked_image_file, results_dir):
    base_name = masked_image_file.rpartition('.')[0].rpartition(os.sep)[-1]

    output = [base_name]
    
    ext2add = ['trajectories', 'skeletons', 'features', 'feat_ind']
    for ext in ext2add:
        output += [os.path.abspath(os.path.join(results_dir, base_name + '_' + ext + '.hdf5'))]
    
    return output




def getTrajectoriesWorkerL(masked_image_file, results_dir, param_file ='', overwrite = False, 
    start_point = -1, end_point = checkpoint['END'], is_single_worm = False):
    
    base_name, trajectories_file, skeletons_file, features_file, feat_ind_file = constructNames(masked_image_file, results_dir)
    print(trajectories_file, skeletons_file, features_file, feat_ind_file)

    #if starting point is not given, calculate it again
    if overwrite:
        start_point = checkpoint['TRAJ_CREATE']
    elif start_point < 0:
        start_point = getStartingPoint(masked_image_file, results_dir)

    #if start_point is larger than end_point there is nothing else to do 
    if start_point > end_point:
        print_flush(base_name + ' Finished in ' + checkpoint_label[end_point])
        return

    if start_point < checkpoint['FEAT_CREATE']:
        #check if the file with the masked images exists
        assert os.path.exists(masked_image_file)
    
    if results_dir[-1] != os.sep:
        results_dir += os.sep
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
        except:
            pass
    
    #%%
    #get function parameters
    param = tracker_param(param_file)
    
    execThisPoint = lambda current_point : (checkpoint[current_point] >= start_point ) &  (checkpoint[current_point] <= end_point)
    

    print_flush(base_name + ' Starting checkpoint: ' + checkpoint_label[start_point])
    #get trajectory data
    if execThisPoint('TRAJ_CREATE'):
        getWormTrajectories(masked_image_file, trajectories_file, **param.trajectories_param)
        if is_single_worm: correctSingleWormCase(trajectories_file)

    if execThisPoint('TRAJ_JOIN'):        
        joinTrajectories(trajectories_file, **param.join_traj_param)

    #get skeletons data    
    if execThisPoint('SKE_CREATE'):
        trajectories2Skeletons(masked_image_file, skeletons_file, trajectories_file, **param.skeletons_param)

    if execThisPoint('SKE_ORIENT'):
        correctHeadTail(skeletons_file, **param.head_tail_param)
    
    if execThisPoint('FEAT_CREATE'):
        #extract features
        getWormFeatures(skeletons_file, features_file, **param.features_param)

    if execThisPoint('FEAT_IND'):
        #extract individual features if the worms are labeled
        featFromLabSkel(skeletons_file, feat_ind_file, param.fps)

    
    print_flush(base_name + ' Finished')
    
    