# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:29:30 2015

@author: ajaver
"""

import tables
#import h5py
import pandas as pd
import numpy as np
import time
from math import floor, ceil

movement_validation_dir = '/Users/ajaver/Documents/GitHub/movement_validation'
import sys
sys.path.append(movement_validation_dir)

from movement_validation import NormalizedWorm
from movement_validation import WormFeatures, VideoInfo

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
    
    meanAngles_all = np.full(skeleton.shape[0], np.nan)    
    angles_all = np.full((skeleton.shape[0], skeleton.shape[1]), np.nan)
    for ss in range(skeleton.shape[0]):
        if skeleton[ss,0,0] == np.nan:
            continue
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
    
    def fromFile(self, file_name, worm_index, fps = 25, isOpenWorm = False):
        
        #get the skeletons_id and frame_number in case they were not given by the user
        with pd.HDFStore(file_name, 'r') as ske_file_id:
            trajectories_data = ske_file_id['/trajectories_data']
            good = trajectories_data['worm_index_N']==worm_index
            good = good & (trajectories_data['frame_number']<10000)
            trajectories_data = trajectories_data.loc[good, ['skeleton_id', 'frame_number', 'has_skeleton']]
            
            skeleton_id = trajectories_data['skeleton_id'].values
            frame_number = trajectories_data['frame_number'].values
            frame_code = trajectories_data['has_skeleton'].values
            
            del trajectories_data         
        
        
        self.file_name = file_name;
        
        self.first_frame = np.min(frame_number)
        self.last_frame = np.max(frame_number)
        self.n_frames = self.last_frame - self.first_frame +1;
        
        with tables.File(file_name, 'r') as ske_file_id:
            n_ske_points =  ske_file_id.get_node('/skeleton').shape[1]
            self.n_segments = n_ske_points
            
            tot_frames = self.n_frames            
            #try to read data by blocks. This is faster than fancy indexing
            block_ind, block_frame = self.getBlockInd(skeleton_id, frame_number, self.first_frame)
            
            self.skeleton = np.full((tot_frames, n_ske_points,2), np.nan)
            self.ventral_contour = np.full((tot_frames, n_ske_points,2), np.nan)
            self.dorsal_contour = np.full((tot_frames, n_ske_points,2), np.nan)
            self.length = np.full(tot_frames, np.nan)
            self.widths = np.full((tot_frames,n_ske_points), np.nan)
    
            self.frame_number = np.full(tot_frames, -1, np.int32)
            self.frame_number[frame_number] = frame_number
            
            self.skeleton_id = np.full(tot_frames, -1, np.int32)
            self.skeleton_id[frame_number] = skeleton_id

            #video info, for the moment we intialize it with the fps
            self.video_info = VideoInfo('', fps)  
            #flag as segmented flags should be marked by the has_skeletons column
            self.video_info.frame_code = np.zeros(tot_frames, np.int32)
            self.video_info.frame_code[frame_number] = frame_code
            
            for bi, bf in zip(*(block_ind, block_frame)):
                self.skeleton[bf[0]:bf[1]+1] = \
                ske_file_id.get_node('/skeleton')[bi[0]:bi[1]+1]
                
                self.ventral_contour[bf[0]:bf[1]+1] = \
                ske_file_id.get_node('/contour_side1')[bi[0]:bi[1]+1]
    
                self.dorsal_contour[bf[0]:bf[1]+1] = \
                ske_file_id.get_node('/contour_side2')[bi[0]:bi[1]+1]
    
                self.length[bf[0]:bf[1]+1] = \
                ske_file_id.get_node('/skeleton_length')[bi[0]:bi[1]+1]
    
                self.widths[bf[0]:bf[1]+1] = \
                ske_file_id.get_node('/contour_width')[bi[0]:bi[1]+1]
                
            
            self.angles, meanAngles_all = calWormAnglesAll(self.skeleton)            
            self.area = calWormArea(self.ventral_contour, self.dorsal_contour)
            
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
            
            if isOpenWorm:
                self.changeAxis()
    
    def getBlockInd(self, skeleton_id, frame_number, first_frame):
        jumps = np.where(np.diff(skeleton_id) != 1)[0]
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


if __name__ == '__main__':
    skeletons_file = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150521_1115/Capture_Ch2_21052015_111806_skeletons.hdf5'
    
    #n_ske_points = 49
    
    with pd.HDFStore(skeletons_file, 'r') as ske_file_id:
        trajectories_data = ske_file_id['/trajectories_data']
    
    #select only data labelled as worms
    trajectories_data = trajectories_data[trajectories_data['worm_label']==1]
    worm_inds = trajectories_data['worm_index_N'].unique()
    
    worm_index = worm_inds[0]
    
    tic = time.time()
    
    worm = WormFromTable()
    worm.fromFile(skeletons_file, worm_index, fps = 25, isOpenWorm=True)
            
    print(time.time()-tic)
#%%
    tic = time.time()
    worm_features = WormFeatures(worm)
    print(time.time()-tic)
    


        