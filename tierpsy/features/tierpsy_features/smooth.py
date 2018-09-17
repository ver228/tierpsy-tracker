# -*- coding: utf-8 -*-
"""
This module defines the NormalizedWorm class

"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def _h_resample_curve(curve, resampling_N=49, widths=None):
    '''Resample curve to have resampling_N equidistant segments
    I give width as an optional parameter since I want to use the 
    same interpolation as with the skeletons
    
    I calculate the length here indirectly
    '''

    # calculate the cumulative length for each segment in the curve
    dx = np.diff(curve[:, 0])
    dy = np.diff(curve[:, 1])
    dr = np.sqrt(dx * dx + dy * dy)

    lengths = np.cumsum(dr)
    lengths = np.hstack((0, lengths))  # add the first point
    tot_length = lengths[-1]

    # Verify array lengths
    if len(lengths) < 2 or len(curve) < 2:
        return None, None, None

    fx = interp1d(lengths, curve[:, 0])
    fy = interp1d(lengths, curve[:, 1])

    subLengths = np.linspace(0 + np.finfo(float).eps, tot_length, resampling_N)

    # I add the epsilon because otherwise the interpolation will produce nan
    # for zero
    try:
        resampled_curve = np.zeros((resampling_N, 2))
        resampled_curve[:, 0] = fx(subLengths)
        resampled_curve[:, 1] = fy(subLengths)
        if widths is not None:
            fw = interp1d(lengths, widths)
            widths = fw(subLengths)
    except ValueError:
        resampled_curve = np.full((resampling_N, 2), np.nan)
        widths = np.full(resampling_N, np.nan)

    return resampled_curve, tot_length, widths


def _h_smooth_curve(curve, window=5, pol_degree=3):
    '''smooth curves using the savgol_filter'''

    if curve.shape[0] < window:
        # nothing to do here return an empty array
        return np.full_like(curve, np.nan)

    # consider the case of one (widths) or two dimensions (skeletons, contours)
    if curve.ndim == 1:
        smoothed_curve = savgol_filter(curve, window, pol_degree)
    else:
        smoothed_curve = np.zeros_like(curve)
        for nn in range(curve.ndim):
            smoothed_curve[:, nn] = savgol_filter(
                curve[:, nn], window, pol_degree)

    return smoothed_curve

def get_group_borders(index_o, pad_val = False):
    
    #add zeros at the edge to consider any block in the edges
    index = np.hstack([pad_val, index_o , pad_val])
    switches = np.diff(index.astype(np.int))
    turn_on, = np.where(switches==1)
    turn_off, = np.where(switches==-1)
    assert turn_off.size == turn_on.size
    
    #fin if fin<index.size else fin-1)
    ind_ranges = list(zip(turn_on, turn_off))
    return ind_ranges
                                
def _h_fill_small_gaps(index_o, max_gap_size):
    ind_ranges = get_group_borders(index_o)
    #ifter by the gap size
    ind_ranges = [(ini, fin) for ini, fin in ind_ranges if fin-ini > max_gap_size]
    
    index_filled = np.zeros_like(index_o)
    for ini, fin in ind_ranges:
        index_filled[ini:fin+1] = True

    return index_filled  
#%%

class SmoothedWorm():
    """
    Encapsulates the notion of a worm's elementary measurements, scaled
    (i.e. "normalized") to 49 points along the length of the worm.
    """

    def __init__(self, 
                 skeleton, 
                 widths = None, 
                 ventral_contour = None, 
                 dorsal_contour = None,
                 skel_smooth_window = None,
                 coords_smooth_window = None,
                 frames_to_interpolate = None,
                 gap_to_interp = 0
                 ):
        """
        I assume data is evenly distributed in time, and missing frames are nan.
        """
        
        


        self.ventral_contour = ventral_contour
        self.dorsal_contour = dorsal_contour
        self.skeleton = skeleton
        self.widths = widths
        self._h_validate_dims()
        
        self.pol_degree = 3
        self.gap_to_interp = gap_to_interp
        
        skel_smooth_window = self._h_fix_smooth(skel_smooth_window)
        coords_smooth_window = self._h_fix_smooth(coords_smooth_window)
        
        self._smooth_coords(frames_to_interpolate, s_win = coords_smooth_window)
        self._smooth_skeletons(s_win = skel_smooth_window)
        self._resample_coords()
        self._h_validate_dims()
        
        
    def _h_validate_dims(self):
        #validate dimenssions
        n_frames, n_segments, n_dims = self.skeleton.shape
        assert n_dims == 2
        if self.ventral_contour is not None:
            assert self.dorsal_contour is not None
            assert self.ventral_contour.shape == (n_frames, n_segments, n_dims)
            assert self.ventral_contour.shape == self.dorsal_contour.shape
        
        if self.widths is not None:
            #TODO I might be able to calculate the widths if the dorsal and ventral contour are given
            assert self.widths.shape == (n_frames, n_segments)
        
        
    def _h_fix_smooth(self, smooth_window):
        if smooth_window is None:
            return smooth_window
        
        if smooth_window <= self.pol_degree:
            #if the smoot window is too small do not smooth
            return None
        
        if smooth_window % 2 == 0:
            smooth_window += 1
        
        return smooth_window
        
    
    def _h_resample_coords(self, A, W = None):
        #I am adding the W as width, in the case of skeletons, 
        #I want to interpolate the widths using the same spacing
        L = np.full(A.shape[0], np.nan)
        for ii in range(A.shape[0]):
            w = None if W is None else W[ii]
                
            A[ii], L[ii], w = \
                _h_resample_curve(A[ii], A.shape[1], w)
            
            if not w is None:
                W[ii] = w
        
        return A, L, W
    
    def _resample_coords(self):
        
        self.skeleton, self.length, self.widths = \
            self._h_resample_coords(self.skeleton, W = self.widths)
        
        if self.dorsal_contour is not None:
            self.ventral_contour, _, _ = \
                self._h_resample_coords(self.ventral_contour)
            self.dorsal_contour, _, _ = \
                self._h_resample_coords(self.dorsal_contour)
    
    
    def _h_smooth_skeletons(self, curves, s_win, pol_degree=3):
        if curves is not None:
            for ii in range(curves.shape[0]):
                if not np.any(np.isnan(curves[ii])):
                    curves[ii] = _h_smooth_curve(
                        curves[ii], 
                        window = s_win, 
                        pol_degree = self.pol_degree
                        )
        return curves
    
    def _smooth_skeletons(self, s_win):
        if s_win is None:
            return
        self.skeleton = self._h_smooth_skeletons(self.skeleton, s_win)
        self.widths = self._h_smooth_skeletons(self.widths, s_win)
        self.ventral_contour = self._h_smooth_skeletons(self.ventral_contour, s_win)
        self.dorsal_contour = self._h_smooth_skeletons(self.dorsal_contour, s_win)

    def _h_interp_and_smooth(self, x, y, x_pred, s_win):
        f = interp1d(x, y)
        y_interp = f(x_pred)

        if (s_win is None) or (y_interp.size <= s_win):
            return y_interp
            
        y_smooth = savgol_filter(y_interp, s_win, self.pol_degree)
        return y_smooth


    def _h_smooth_coords(self, 
                         dat_o, 
                         s_win, 
                         good_frames_index, 
                         frames_to_interpolate, 
                         frames_to_nan):
        '''
        Interpolate coordinates for each segment
        '''
        
        if dat_o is None:
            return dat_o
        
        dat_all = dat_o[good_frames_index]
        if dat_all.shape[0] <= 2:
            #not enough data to smooth
            return dat_all
            
        new_shape = (frames_to_interpolate.size, dat_o.shape[1], dat_o.shape[2]) 
        dat_all_s = np.full(new_shape, np.nan)
        
        #add data in the borders to be able to interpolate within those regions
        tt = np.hstack([-1, good_frames_index, dat_o.shape[0]]) 
        for i_seg in range(dat_all.shape[1]):
            for i_coord in range(2):
                c = dat_all[:, i_seg, i_coord]
                c = np.hstack([c[0], c, c[-1]])

                c_s = self._h_interp_and_smooth(tt, c, frames_to_interpolate, s_win)
                dat_all_s[:, i_seg, i_coord] = c_s
                
        dat_all_s[frames_to_nan, :, :] = np.nan
        return dat_all_s

    def _smooth_coords(self, frames_to_interpolate, s_win):
        
        if frames_to_interpolate is None:
            frames_to_interpolate = np.arange(self.skeleton.shape[0])
        
        bad = np.isnan(self.skeleton[:, 0, 0])
        good_frames_index, = np.where(~bad)
        
        #get indexes of nan's after removing small gaps and interpolating
        bad_filled = _h_fill_small_gaps(bad, self.gap_to_interp)
        f = interp1d(np.arange(bad_filled.size), bad_filled)
        frames_to_nan = np.ceil(f(frames_to_interpolate)).astype(np.bool)
        assert frames_to_interpolate.size == frames_to_nan.size
        
        #interpolate all the fields
        args = (s_win, good_frames_index, frames_to_interpolate, frames_to_nan)
        self.skeleton = self._h_smooth_coords(self.skeleton,  *args)
        self.ventral_contour = self._h_smooth_coords(self.ventral_contour, *args)
        self.dorsal_contour = self._h_smooth_coords(self.dorsal_contour, *args)



if __name__ == '__main__':
    '''
    Code for testing...
    '''
    from tierpsy.analysis.feat_create.obtainFeaturesHelper import WormFromTable
    from tierpsy.analysis.feat_create.obtainFeatures import getGoodTrajIndexes
    from tierpsy.helper.misc import RESERVED_EXT
    from tierpsy.helper.params import read_fps
    
    import os
    
    if False:
        #use if if you want to get the file names
        import glob
        import fnmatch
        
        exts = ['']
        exts = ['*'+ext+'.hdf5' for ext in exts]
        
        mask_dir = '/Users/ajaver/OneDrive - Imperial College London/optogenetics/Arantza/MaskedVideos/**/'
        #mask_dir = '/Users/ajaver/OneDrive - Imperial College London/tests/join/'
        fnames = glob.glob(os.path.join(mask_dir, '*.hdf5'))
        fnames = [x for x in fnames if any(fnmatch.fnmatch(x, ext) for ext in exts)]
        fnames = [x for x in fnames if not any(x.endswith(ext) for ext in RESERVED_EXT)]
        
    
    #save_dir = '/Users/ajaver/OneDrive - Imperial College London/smooth_examples'
    save_dir = './'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    
#    mask_video = '/Users/ajaver/OneDrive - Imperial College London/optogenetics/Arantza/MaskedVideos/control_pulse/pkd2_5min_Ch1_11052017_121414.hdf5'
#    save_prefix = 'worm_example.npz'
#    is_WT2 = False
    
#    mask_video = '/Volumes/behavgenom_archive$/single_worm/finished/WT/N2/food_OP50/XX/30m_wait/clockwise/N2 on food L_2011_03_29__17_02_06___8___14.hdf5'
#    save_prefix = 'worm_example_big_W{}.npz'
#    is_WT2 = True
    
#    mask_video = '/Volumes/behavgenom_archive$/single_worm/finished/WT/N2/food_OP50/XX/30m_wait/anticlockwise/N2 on food R_2009_09_04__10_59_59___8___5.hdf5'
#    save_prefix = 'worm_example_small_W{}.npz'
#    is_WT2 = True

    mask_video = '/Volumes/behavgenom_archive$/Lidia/MaskedVideos/Optogenetics-day1/AQ3071-ATR_Set1_Ch1_18072017_191322.hdf5'
    is_WT2 = False
    
    skeletons_file = mask_video.replace('MaskedVideos','Results').replace('.hdf5', '_skeletons.hdf5')
    #%%
    import pandas as pd
    with pd.HDFStore(skeletons_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    trajectories_data[trajectories_data['worm_index_joined'] == 2]
    
    #%%
    fps = read_fps(skeletons_file)
    coords_smooth_window = int(np.round(fps/3))
    if coords_smooth_window <= 3:
        coords_smooth_window = None
    
    good_traj_index, worm_index_type = getGoodTrajIndexes(skeletons_file)
    for iw, worm_index in enumerate(good_traj_index):
        print(iw, len(good_traj_index))
        worm = WormFromTable(skeletons_file,
                            worm_index,
                            worm_index_type=worm_index_type
                            )
        if is_WT2: worm.correct_schafer_worm()
        
        wormN = SmoothedWorm(
                 worm.skeleton, 
                 worm.widths, 
                 worm.ventral_contour, 
                 worm.dorsal_contour,
                 skel_smooth_window = 5,
                 coords_smooth_window = coords_smooth_window,
                 gap_to_interp = 5
                )
        
        
#        save_file = os.path.join(save_dir, save_prefix.format(worm_index))
#        np.savez_compressed(save_file, 
#                 skeleton=wormN.skeleton, 
#                 ventral_contour=wormN.ventral_contour, 
#                 dorsal_contour=wormN.dorsal_contour,
#                 widths = wormN.widths
#                 )
#            
#            
#        
#        break
#    #%%
#    import matplotlib.pyplot as plt
#    plt.figure()
#    plt.subplot(4,1,1)
#    plt.plot(wormN.skeleton[: ,0,0])
#    plt.subplot(4,1,2)
#    plt.plot(wormN.skeleton[: ,0,1])
#    plt.subplot(2,1,2)
#    plt.plot(wormN.skeleton[: ,0,0], wormN.skeleton[: ,0,1])
#    plt.axis('equal')
    