# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:34:11 2015

@author: ajaver
"""

import h5py
import numpy as np
import matplotlib.pylab as plt

def calWormAngles(x,y):
        
    assert(len(x.shape)==1)
    assert(len(y.shape)==1)
    
    #if edge_length == 0:
    #    edge_length = int(round(x.size/12));
    
    dx = np.diff(x);
    dy = np.diff(y);
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

def calWormAnglesAll(skeleton):
    #edge_length = int(round(x.size/12));
    
    angles_all = np.full((skeleton.shape[0], skeleton.shape[2]), np.nan)
    meanAngles_all = np.zeros(skeleton.shape[0])
    for ss in range(skeleton.shape[0]):
        angles_all[ss,:],meanAngles_all[ss] = calWormAngles(skeleton[ss,0,:],skeleton[ss,1,:])
    return angles_all, meanAngles_all


#if __name__ == "__main__":
#    segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_segworm.hdf5';
#    
#    segworm_fid = h5py.File(segworm_file, 'r+')
#    skeleton = segworm_fid['/segworm_results/skeleton']
#    
#    if "/segworm_results/skeleton_angles" in segworm_fid:
#        del segworm_fid["/segworm_results/skeleton_angles"]
#    if "/segworm_results/skeleton_mean_angles2" in segworm_fid:
#        del segworm_fid["/segworm_results/skeleton_mean_angles"]
#    
#    angles_shape = (skeleton.shape[0], skeleton.shape[2]-1); #substract one size substract one point (difference)
#    skeleton_angles = segworm_fid.create_dataset("/segworm_results/skeleton_angles" , angles_shape, 
#                               dtype = np.float64, maxshape =angles_shape, 
#                               chunks = (5, angles_shape[1]),
#                            compression="gzip", shuffle=True);
#    
#    mean_angles_shape = (angles_shape[0],1);
#    skeleton_mean_angles = segworm_fid.create_dataset("/segworm_results/skeleton_mean_angles" , (angles_shape[0],),
#                               dtype = np.float64, maxshape = (angles_shape[0],), 
#                               chunks = True, 
#                            compression="gzip", shuffle=True);
#    
#    
#    for kk in range(skeleton.shape[0]):
#        print kk, skeleton.shape[0]
#        angles, mean_angle = calWormAngles(skeleton[kk,0,:], skeleton[kk,1,:]);
#        skeleton_mean_angles[kk] = mean_angle;
#        skeleton_angles[kk,:] = angles;
#    
#    
#    segworm_fid.close()
#function [angleArray, meanAngles] = makeAngleArray(x, y)
#
#%MAKEANGLEARRAY Get tangent angles for each frame of normBlocks and rotate
#%               to have zero mean angle
#%
#%   [ANGLEARRAY, MEANANGLES] = MAKEANGLEARRAY(X, Y)
#%
#%   Input:
#%       x - the x coordinates of the worm skeleton (equivalent to 
#%           dataBlock{4}(:,1,:)
#%       y - the y coordinates of the worm skeleton (equivalent to 
#%           dataBlock{4}(:,2,:)
#%
#%   Output:
#%       angleArray - a numFrames by numSkelPoints - 1 array of tangent
#%                    angles rotated to have mean angle of zero.
#%       meanAngles - the average angle that was subtracted for each frame
#%                    of the video.
#
#[numFrames, lengthX] = size(x);
#
#% initialize arrays
#angleArray = zeros(numFrames, lengthX-1);
#meanAngles = zeros(numFrames, 1);
#
#% for each video frame
#for i = 1:numFrames
#    
#    % calculate the x and y differences
#    dX = diff(x(i,:));
#    dY = diff(y(i,:));
#    
#    % calculate tangent angles.  atan2 uses angles from -pi to pi instead...
#    % of atan which uses the range -pi/2 to pi/2.
#    angles = atan2(dY, dX);
#    
#    % need to deal with cases where angle changes discontinuously from -pi
#    % to pi and pi to -pi.  In these cases, subtract 2pi and add 2pi
#    % respectively to all remaining points.  This effectively extends the
#    % range outside the -pi to pi range.  Everything is re-centred later
#    % when we subtract off the mean.
#    
#    % find discontinuities larger than pi (skeleton cannot change direction
#    % more than pi from one segment to the next)
#    positiveJumps = find(diff(angles) > pi) + 1; %+1 to cancel shift of diff
#    negativeJumps = find(diff(angles) < -pi) + 1;
#    
#    % subtract 2pi from remainging data after positive jumps
#    for j = 1:length(positiveJumps)
#        angles(positiveJumps(j):end) = angles(positiveJumps(j):end) - 2*pi;
#    end
#    
#    % add 2pi to remaining data after negative jumps
#    for j = 1:length(negativeJumps)
#        angles(negativeJumps(j):end) = angles(negativeJumps(j):end) + 2*pi;
#    end
#    
#    % rotate skeleton angles so that mean orientation is zero
#    meanAngle = mean(angles(:));
#    meanAngles(i) = meanAngle;
#    angles = angles - meanAngle;
#    
#    % append to angle array
#    angleArray(i,:) = angles;
#end