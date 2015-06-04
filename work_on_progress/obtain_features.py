# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:30:53 2015

@author: ajaver
"""

import sys
sys.path.append('../../movement_validation')

#from movement_validation.pre_features import WormParsing
from movement_validation import NormalizedWorm
from movement_validation import WormFeatures, VideoInfo
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:16:41 2015

@author: ajaver
"""
import tables
import numpy as np
np.seterr(invalid='ignore')

def calWormAngles(x,y):
        
    assert(x.ndim==1)
    assert(y.ndim==1)
    
    
    
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
    #after the inversion...
    meanAngles_all = np.zeros(skeleton.shape[0])
    
    angles_all = np.full((skeleton.shape[0], skeleton.shape[1]), np.nan)
    for ss in range(skeleton.shape[0]):
        angles_all[ss,:-1],meanAngles_all[ss] = calWormAngles(skeleton[ss,:,0],skeleton[ss,:,1])
        
    
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
    
    def fromFile(self, file_name, worm_index, rows_range = (0,0)):
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
            
            self.worm_index = worm_index
            self.rows_range = rows_range
            
            self.frame_code = file_id.get_node('/trajectories_data').read(ini,end,1,'frame_number')
            
            self.skeleton = file_id.get_node('/skeleton')[ini:end+1,:,:]
            self.vulva_contour = file_id.get_node('/contour_side1')[ini:end+1,:,:]
            self.non_vulva_contour = file_id.get_node('/contour_side2')[ini:end+1,:,:]
            
            self.length = file_id.get_node('/skeleton_length')[ini:end+1] 
            self.widths = file_id.get_node('/contour_width')[ini:end+1,:]
            self.angles, meanAngles_all = calWormAnglesAll(self.skeleton)
            
            self.n_segments = self.skeleton.shape[1]
            self.area = calWormArea(self.vulva_contour, self.non_vulva_contour)
            
            assert self.skeleton.shape[2] == 2
            assert self.vulva_contour.shape[2] == 2
            assert self.non_vulva_contour.shape[2] == 2
            
            assert self.vulva_contour.shape[1] == self.n_segments            
            assert self.non_vulva_contour.shape[1] == self.n_segments
            assert self.widths.shape[1] == self.n_segments
            
            assert self.angles.shape[1] == self.n_segments
            assert self.length.ndim == 1
            
            
    def changeAxis(self):
        for field in ['skeleton', 'vulva_contour', 'non_vulva_contour', 'widths', 'angles']:
            A = getattr(self, field)
            #roll axis to have it as 49x2xn
            A = np.rollaxis(A,0,A.ndim)
            #change the memory order so the last dimension changes slowly
            A = np.asfortranarray(A)
            #finally copy it back to the field
            setattr(self, field, A)
            
            assert getattr(self, field).shape[0] == self.n_segments

if __name__ == "__main__":
    root_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150511/'
    base_name = 'Capture_Ch1_11052015_195105'
    
    masked_image_file = root_dir + '/Compressed/' + base_name + '.hdf5'
    trajectories_file = root_dir + '/Trajectories/' + base_name + '_trajectories.hdf5'
    skeleton_file = root_dir + '/Trajectories/' + base_name + '_skeletons.hdf5'
    
    worm_index = 773
    rows_range = (0,0)
    #assert rows_range[0] <= rows_range[1]
    
    file_name = skeleton_file;
    worm = WormFromTable()
    worm.fromFile(file_name, worm_index)
    worm.changeAxis()

    vi = VideoInfo(masked_image_file, 25)    
#    # Generate the OpenWorm movement validation repo version of the features
    openworm_features = WormFeatures(worm, vi)


    


