# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:30:53 2015

@author: ajaver
"""
import os
import sys
import tables
import pandas as pd
import numpy as np
from math import floor, ceil
import time

sys.path.append('../../../movement_validation')
#from movement_validation.pre_features import WormParsing

from movement_validation import NormalizedWorm
from movement_validation import WormFeatures, VideoInfo
from movement_validation.statistics.histogram_manager import HistogramManager

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
    
    def fromFile(self, file_name, worm_index, rows_range = (0,0), isOpenWorm = False):
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
            
            self.angles, meanAngles_all = calWormAnglesAll(self.skeleton)            
            
            self.length = file_id.get_node('/skeleton_length')[ini:end+1] 
            self.widths = file_id.get_node('/contour_width')[ini:end+1,:]
            
            self.n_segments = self.skeleton.shape[1]
            self.n_frames = self.skeleton.shape[0]
            
            self.segmentation_status = np.zeros(self.n_frames, dtype='unicode_')
            #flag as segmented (s) evertyhing that is not nan
            self.segmentation_status[~np.isnan(self.skeleton[:,0,0])] = 's' 
            
            self.area = calWormArea(self.vulva_contour, self.non_vulva_contour)
            
            
            assert self.angles.shape[1] == self.n_segments            
            
            assert self.skeleton.shape[2] == 2
            assert self.vulva_contour.shape[2] == 2
            assert self.non_vulva_contour.shape[2] == 2
            
            assert self.vulva_contour.shape[1] == self.n_segments            
            assert self.non_vulva_contour.shape[1] == self.n_segments
            assert self.widths.shape[1] == self.n_segments
            
            assert self.length.ndim == 1
            
            
            if isOpenWorm:
                self.changeAxis()
            
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

if __name__ == "__main__":
#    base_name = 'Capture_Ch3_12052015_194303'
#    mask_dir = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/20150512/'
#    results_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150512/'    
    
    base_name = 'Capture_Ch1_11052015_195105'
    mask_dir = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/20150511/'
    results_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150511/'
    
    masked_image_file = mask_dir + base_name + '.hdf5'
    trajectories_file = results_dir + base_name + '_trajectories.hdf5'
    skeletons_file = results_dir + base_name + '_skeletons.hdf5'
    features_file = results_dir + base_name + '_features.hdf5'
    
    assert os.path.exists(masked_image_file)
    assert os.path.exists(trajectories_file)
    assert os.path.exists(skeletons_file)
    
    
    
    
    bad_seg_thresh = 0.5
    
    with pd.HDFStore(skeletons_file, 'r') as ske_file_id:
        indexes_data = ske_file_id['/trajectories_data'][['worm_index_joined', 'skeleton_id', 'has_skeleton']]
    
    skeleton_fracc = indexes_data.groupby('worm_index_joined').agg({'has_skeleton':'mean'})
    skeleton_fracc = skeleton_fracc['has_skeleton']
    valid_worm_index = skeleton_fracc[skeleton_fracc>=bad_seg_thresh].index
        
    #remove the bad worms, we do not care about them
    indexes_data = indexes_data[indexes_data['worm_index_joined'].isin(valid_worm_index)]
    
    #get the first and last frame of each worm_index
    rows_indexes = indexes_data.groupby('worm_index_joined').agg({'skeleton_id':[min, max]})
    rows_indexes = rows_indexes['skeleton_id']
    
    tot_rows = len(indexes_data)
    del indexes_data



        
    
    histMng = HistogramManager()
    filters_tables = tables.Filters(complevel=5, complib='zlib', shuffle=True)
    
    with tables.File(features_file, 'w') as features_fid:
        features_group = features_fid.create_group('/', 'worms_features')
        
        tot_worms = len(rows_indexes)
        for ind, dat  in enumerate(rows_indexes.iterrows()):
            worm_index, row_range = dat
            print(ind, tot_worms, worm_index)
            
            if worm_index>2: break
            
            worm = WormFromTable()
            worm.fromFile(skeletons_file, worm_index, rows_range= tuple(row_range.values), isOpenWorm=True)
#            
#            vi = VideoInfo(masked_image_file, 25)    
#            
#            if np.all(np.isnan(worm.skeleton)):
#                continue
#            # Generate the OpenWorm movement validation repo version of the features
#            tic = time.time()
#            openworm_features = WormFeatures(worm, vi)
#            tic_f = time.time() - tic
#            
#            tic = time.time()
#            openworm_histograms = histMng.init_histograms(openworm_features)
#            tic_h = time.time() - tic
#            
#            print(tic_f, tic_h)
         
         
            #initialize features table
            worm_node = features_fid.create_group(features_group, 'worm_%i' % worm_index )
            worm_node._v_attrs['worm_id'] = worm_index
            worm_node._v_attrs['frame_range'] = (worm.frame_code[0], worm.frame_code[-1])
            worm_node._v_attrs['skeletons_rows_range'] = tuple(row_range.values)
            
#        for feature_name in feature_names:
#            path, _, feature = feature_name.rpartition('/')
#            features_fid.create_carray(path, feature, shape = (tot_rows,), \
#                            atom = tables.Float32Atom(dflt=np.nan), \
#                            createparents=True, filters=filters_tables)
            

#    worm_index = 773
#    rows_range = (0,0)
    
#    file_name = skeletons_file;
#    worm = WormFromTable()
#    worm.fromFile(file_name, worm_index)
#    #worm.changeAxis()

    #import matplotlib.pylab as plt
    #plt.plot(worm.angles[0,:])
    #vi = VideoInfo(masked_image_file, 25)    
    # Generate the OpenWorm movement validation repo version of the features
    #openworm_features = WormFeatures(worm, vi)
##%%    
#    all_features = {}
#    for main_feature in ['path', 'locomotion', 'morphology', 'posture']:
#        all_features = walk_obj(getattr(openworm_features, main_feature), main_feature, all_features)
#    
#    with open('./feature_list.txt', 'w') as f:
#        for feature in sorted(all_features.keys()):
#            f.write(feature + '\n')




    


