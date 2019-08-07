#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:55:41 2019

@author: lferiani
"""


#%% import statements

import re
import cv2
import pdb
import itertools
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import colors
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from sklearn.neighbors import NearestNeighbors

from tierpsy.helper.params import read_unit_conversions

#%% constants

# dictionary to go from camera name to channel
# to be updated as we get more copies of the LoopBio rig
CAM2CH_DICT_legacy = {"22594549":'Ch1',
                      "22594548":'Ch2',
                      "22594546":'Ch3',
                      "22436248":'Ch4',
                      "22594559":'Ch5',
                      "22594547":'Ch6'}

CAM2CH_DICT = {"22956818":'Ch1', # Hydra01
               "22956816":'Ch2',
               "22956813":'Ch3',
               "22956805":'Ch4',
               "22956807":'Ch5',
               "22956832":'Ch6',
               "22956839":'Ch1', # Hydra02
               "22956837":'Ch2',
               "22956836":'Ch3',
               "22956829":'Ch4',
               "22956822":'Ch5',
               "22956806":'Ch6',
               "22956814":'Ch1', # Hydra03
               "22956827":'Ch2',
               "22956819":'Ch3',
               "22956833":'Ch4',
               "22956823":'Ch5',
               "22956840":'Ch6',
               "22956812":'Ch1', # Hydra04
               "22956834":'Ch2',
               "22956817":'Ch3',
               "22956811":'Ch4',
               "22956831":'Ch5',
               "22956809":'Ch6',
               "22594559":'Ch1', # Hydra05
               "22594547":'Ch2',
               "22594546":'Ch3',
               "22436248":'Ch4',
               "22594549":'Ch5',
               "22594548":'Ch6'}

# dictionaries to go from channel/(col, row) to well name.
# there will be many as it depends on total number of wells, upright/upsidedown,
# and in case of the 48wp how many wells in the fov

UPRIGHT_48WP_669999 = pd.DataFrame.from_dict({ ('Ch1',0):['A1','B1','C1'],
                                               ('Ch1',1):['A2','B2','C2'],
                                               ('Ch2',0):['D1','E1','F1'],
                                               ('Ch2',1):['D2','E2','F2'],
                                               ('Ch3',0):['A3','B3','C3'],
                                               ('Ch3',1):['A4','B4','C4'],
                                               ('Ch3',2):['A5','B5','C5'],
                                               ('Ch4',0):['D3','E3','F3'],
                                               ('Ch4',1):['D4','E4','F4'],
                                               ('Ch4',2):['D5','E5','F5'],
                                               ('Ch5',0):['A6','B6','C6'],
                                               ('Ch5',1):['A7','B7','C7'],
                                               ('Ch5',2):['A8','B8','C8'],
                                               ('Ch6',0):['D6','E6','F6'],
                                               ('Ch6',1):['D7','E7','F7'],
                                               ('Ch6',2):['D8','E8','F8']})

UPRIGHT_96WP = pd.DataFrame.from_dict({('Ch1',0):[ 'A1', 'B1', 'C1', 'D1'],
                                       ('Ch1',1):[ 'A2', 'B2', 'C2', 'D2'],
                                       ('Ch1',2):[ 'A3', 'B3', 'C3', 'D3'],
                                       ('Ch1',3):[ 'A4', 'B4', 'C4', 'D4'],
                                       ('Ch2',0):[ 'E1', 'F1', 'G1', 'H1'],
                                       ('Ch2',1):[ 'E2', 'F2', 'G2', 'H2'],
                                       ('Ch2',2):[ 'E3', 'F3', 'G3', 'H3'],
                                       ('Ch2',3):[ 'E4', 'F4', 'G4', 'H4'],
                                       ('Ch3',0):[ 'A5', 'B5', 'C5', 'D5'],
                                       ('Ch3',1):[ 'A6', 'B6', 'C6', 'D6'],
                                       ('Ch3',2):[ 'A7', 'B7', 'C7', 'D7'],
                                       ('Ch3',3):[ 'A8', 'B8', 'C8', 'D8'],
                                       ('Ch4',0):[ 'E5', 'F5', 'G5', 'H5'],
                                       ('Ch4',1):[ 'E6', 'F6', 'G6', 'H6'],
                                       ('Ch4',2):[ 'E7', 'F7', 'G7', 'H7'],
                                       ('Ch4',3):[ 'E8', 'F8', 'G8', 'H8'],
                                       ('Ch5',0):[ 'A9', 'B9', 'C9', 'D9'],
                                       ('Ch5',1):['A10','B10','C10','D10'],
                                       ('Ch5',2):['A11','B11','C11','D11'],
                                       ('Ch5',3):['A12','B12','C12','D12'],
                                       ('Ch6',0):[ 'E9', 'F9', 'G9', 'H9'],
                                       ('Ch6',1):['E10','F10','G10','H10'],
                                       ('Ch6',2):['E11','F11','G11','H11'],
                                       ('Ch6',3):['E12','F12','G12','H12']})


#%% Class definition
class FOVMultiWellsSplitter(object):
    """Class tasked with finding how to split a full-FOV image into single-wells images, 
    and then splitting new images that are passed to it."""
    
    def __init__(self, masked_image_file=None, 
                 img=None, camera_serial=None, 
                 total_n_wells=96, whichsideup='upright', 
                 well_shape='square', px2um=None):
        """Class constructor. 
        Creates wells, and parses the image to fill up the wells property
        Either give the masked_image_file name, or pass an image AND camera serial number.
        If the masked_image_file name is given, any img, camera_serial, or px2um
        will be ignored and read from the masked_image_file
        img = a brightfield frame that will be used for well-finding
        n_wells = how many wells *in the entire multiwell plate*"""
        # parse input
        if masked_image_file is not None:
            self.img, self.camera_serial, self.px2um = \
            read_data_from_masked(masked_image_file)
        # if there was no masked_image_file and one other parameter is missing    
        elif (img is None) or (camera_serial is None) or (px2um is None): 
            raise ValueError('Either provide the masked video filename or an' +\
                             ' image, camera_serial, and px2um.')
        # no masked_image_file, and all other parameters exist    
        else:
            # save the input image just to make some things easier
            if len(img.shape) == 2:
                self.img = img.copy()
            elif len(img.shape) == 3:
                # convert to grey
                self.img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # and store serial number and px2um    
            self.camera_serial = camera_serial
            self.px2um = px2um         # px2um = 12.4 is default for Hydra rig
            
        # assert that other input was given correctly
        # whichsideup: was the cell upright or upside-down
        assert whichsideup in ['upside-down', 'upright'], \
            "Whichsideup can only be 'upside-down' or 'upright'"
        # well_shape: either 'square' or 'circle'
        assert well_shape in ['square', 'circle'], \
            "well_shape can only be 'square' or 'circle'"    

        # save height and width of image
        self.img_shape = self.img.shape
        # where was the camera on the rig? 
        self.channel = CAM2CH_DICT[self.camera_serial]
        # number of wells in the multiwell plate: 6 12 24 48 96?
        # TODO: input check. Dunno if it will be kept like this or parsed from a filename
        self.n_wells = total_n_wells
        # whichsideup: was the cell upright or upside-down
        self.whichsideup = whichsideup
        # well_shape: either 'square' or 'circle'
        self.well_shape = well_shape
        # according to n_wells and whichsideup choose a dictionary for 
        #(channel,position) <==> well 
        # TODO: the dictionaries will be imported from a helper module
        # TODO?: make user specify (if 48wp) which channels have 6 wells and not 9
        # TODO: make this its own function
        if (self.n_wells == 48) and (self.whichsideup == 'upright'):
            self.mwp_df = UPRIGHT_48WP_669999[self.channel]
        elif (self.n_wells == 96) and (self.whichsideup == 'upright'):
            self.mwp_df = UPRIGHT_96WP[self.channel]
        else:
            raise Exception("This case hasn't been coded for yet")
        # create a scaled down, blurry image which is faster to analyse
        self.blur_im = self.get_blur_im()
        # wells is the most important property. 
        # It's the dataframe that contains the coordinates of each recognised 
        # well in the original image
        # In particular
        #   x, y         = coordinates of the well's centre, in pixel (so x is a column index, y a row index)
        #   r            = radius of the circle, in pixel, if the wells are circular, 
        #                   or the circle inscribed into the wells template if squares
        #   row, col     = indices of a well in the grid of detected wells
        #   *_max, *_min = coordinates for cropping the FOV so 1 roi = 1 well
        self.wells = pd.DataFrame(columns = ['x','y','r','row','col',
                                          'x_min','x_max','y_min','y_max',
                                          'well_name'])
        
        # METHODS
        # call method to fill in the wells variable
        if self.well_shape == 'circle':
            self.find_circular_wells()
            self.remove_half_circles()
        elif self.well_shape == 'square':
            self.find_square_wells()
        self.find_row_col_wells()
        self.fill_lattice_defects()
        self.find_wells_boundaries()
        self.calculate_wells_dimensions()
#        print(self.wells)
        self.name_wells()
#        print(self.wells)

    
    def get_blur_im(self):
        """downscale and blur the image"""
        # preprocess image
        dwnscl_factor = 4; # Hydra images' shape is divisible by 4
        blr_sigma = 17; # blur the image a bit, seems to work better
        new_shape = (self.img.shape[1]//dwnscl_factor, # as x,y, not row,columns
                     self.img.shape[0]//dwnscl_factor)
        
        dwn_gray_im = cv2.resize(self.img, new_shape)
        # apply blurring
        blur_im = cv2.GaussianBlur(dwn_gray_im, (blr_sigma,blr_sigma),0)
        # normalise between 0 and 255
        blur_im = cv2.normalize(blur_im, None, alpha=0, beta=255, 
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return blur_im
    
    
    def find_square_wells(self, xcorr_threshold=0.85):
        """Cross-correlate the frame with a template approximating a square well.
        Then clean up the results by removing points too close together, or not on a nice grid
        - xcorr_threshold: how high the value of the cross-correlation between 
            image and template has to be
        """
        is_debug = False
        # Downscale factor for blur_im. was not saved as a property of the class
        dwnscl_factor = self.img_shape[0]/self.blur_im.shape[0]
        # How many pixels in a square well, in the downscaled image?
        if self.n_wells == 96:
            # TODO: make a dictionary global and call it
            well_size_mm = 8 # roughly, well size of 96wp square wells. Maybe have it as input?
            well_size_px = well_size_mm*1000/self.px2um/dwnscl_factor
        else: 
            raise Exception("This case hasn't been coded for yet")
            
        # make square template approximating a well    
        template = make_square_template(n_pxls=well_size_px, rel_width=0.7, blurring=0.1)
        
        # find template in image: cross correlation, then threshold and segment
        res = cv2.matchTemplate(self.blur_im, template, cv2.TM_CCORR_NORMED)
        # find local maxima that are at least well_size_px apart
        X=peak_local_max(res, min_distance=well_size_px//2, 
                         indices=True, 
                         exclude_border=False,
                         threshold_abs=xcorr_threshold)
        xcorr_peaks = np.array( [res[r, c] for r,c in X] )
        
        if is_debug:
            plt.figure()
            ax = plt.axes()
            hi = ax.imshow(res)
            plt.axis('off')
            plt.colorbar(hi, ax=ax, fraction=0.03378, pad=0.01)
            plt.tight_layout()
            plt.plot(X[:,1],X[:,0],'ro', mfc='none')
#            plt.plot(X2[:,1],X2[:,0],'bx')
            fig, axs = plt.subplots(1,2, figsize=(12.8, 4.8))
            axs[0].imshow(self.img, cmap='gray')
            axs[1].imshow(res)
            axs[1].plot(X[:,1], X[:,0], 'ro', mfc='none')
        
        #NOTE: X contains row and column (y and x) of corner of template matching, 
        # and are values within res.shape.  
        # To get the centre, add half the template size. 
        # This is to be done at the end, to avoid out-of-bound problems in res 

        # now get rid of points not nicely on a grid.
        
        # first get a good estimate of the lattice parameter
        # points on a grid will always have 2 neighbours at 
        # distance=lattice spacing
        nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X)
        distances, _ = nbrs.kneighbors(X)
        distances = distances[:,1:]         # throw away the column of zeros (dist from itself)
        lattice_dist = np.median(distances) # this is average lattice spacing
        delta = lattice_dist/10             # this is "grace" interval
        # now look for wells not aligned along the x or y:
        # samecoord is N-by-N matrix. Element [i,j] is true when 
        # the coordinate of well i and well j is (roughly) the same
        alonealongcoord = np.zeros(X.shape)
        for i in [0,1]: # dimension counter: y and x
            samecoord = abs(X[:,[i,]] - X[:,i]) < delta
            if is_debug: print(samecoord.astype(int))
            alonealongcoord[:,i] =  samecoord.sum(axis=1)==1
            if is_debug: print(alonealongcoord)
        # a point is not on a grid if it doesn't share its x or y coordinates 
        # with at least another point
        idx_not_on_grid = alonealongcoord.any(axis=1)
        if is_debug: print(idx_not_on_grid)
        # apply the filtering
        X = X[~idx_not_on_grid]
        xcorr_peaks = xcorr_peaks[~idx_not_on_grid]
        if is_debug: print(X)
        
        if is_debug:
            axs[1].plot(X[:,1], X[:,0], 'rx')
            ax.plot(X[:,1], X[:,0], 'rx')
        
        # write in self.wells
        X = X * dwnscl_factor
        well_size_px *= dwnscl_factor
        self.wells['y'] = X[:,0] + well_size_px/2
        self.wells['x'] = X[:,1] + well_size_px/2
        self.wells['r'] = well_size_px/2
        
        if is_debug:
            axs[0].plot(self.wells['x'], self.wells['y'], 'rx')
            
        return
        
        
    def find_circular_wells(self):
        """Simply use Hough transform to find circles in MultiWell Plate rgb image.
        The parameters used are optimised for 24 or 48WP"""
                    
        
        dwnscl_factor = self.img_shape[0]/self.blur_im.shape[0]
        
        # find circles
        # parameters in downscaled units
        circle_goodness = 70;
        highest_canny_thresh = 10;
        min_well_dist = self.blur_im.shape[1]/3;    # max 3 wells along short side. bank on FOV not taking in all the entirety of the well 
        min_well_radius = self.blur_im.shape[1]//7; # if 48WP 3 wells on short side ==> radius <= side/6
        max_well_radius = self.blur_im.shape[1]//4; # if 24WP 2 wells on short side. COnsidering intrawells space, radius <= side/4 
        # find circles
        _circles = cv2.HoughCircles(self.blur_im,
                                   cv2.HOUGH_GRADIENT, 
                                   dp=1,
                                   minDist=min_well_dist, 
                                   param1=highest_canny_thresh,
                                   param2=circle_goodness,
                                   minRadius=min_well_radius,
                                   maxRadius=max_well_radius)
        _circles = np.squeeze(_circles); # because why the hell is there an empty dimension at the beginning?
        
        # convert back to pixels
        _circles *= dwnscl_factor;
        
        # output back into class property
        self.wells['x'] = _circles[:,0].astype(int)
        self.wells['y'] = _circles[:,1].astype(int)
        self.wells['r'] = _circles[:,2].astype(int)
        return
        

    def find_row_col_wells(self):
        """
        The circular wells are aligned in a grid, but are not found in such 
        order by the Hough Transform. 
        Find (row, column) index for each well.
        Algorithm: (same for rows and columns)
            - Scan the found wells, pick up the topmost [leftmost] one
            - Find all other wells that are within the specified interval of the 
                first along the considered dimension.
            - Assign the same row [column] label to all of them
            - Repeat for the wells that do not yet have an assigned label, 
                increasing the label index
        """
        
        # number of pixels within which found wells are considered to be within the same row
        if self.well_shape == 'circle':
            interval = self.wells['r'].mean() # average radius across circles (3rd column)
        elif self.well_shape == 'square':
            interval = self.wells['r'].mean() # this is just half the template size anyway
            # maybe change that?
        
        # execute same loop for both rows and columns
        for d,lp in zip(['x','y'],['col','row']): # d = dimension, lp = lattice place
            # initialise array or row/column labels. This is a temporary variable, I could have just used self.wells[lp]
            d_ind = np.full(self.wells.shape[0],np.nan)
            cc = 0; # what label are we assigning right now
            # loop until all the labels have been assigned
            while any(np.isnan(d_ind)):
                # find coordinate of first (leftmost or topmost) non-labeled well
                idx_unlabelled_wells = np.isnan(d_ind)
                unlabelled_wells = self.wells.loc[idx_unlabelled_wells]
                coord_first_well = np.min(unlabelled_wells[d])
                # find distance between this and *all* wells along the considered dimension
                d_dists = self.wells[d] - coord_first_well;
                # find wells within interval. d_dists>=0 discards previous rows [columns]
                # could have taken the absolute value instead but meh I like this logic better
                idx_same = np.logical_and((d_dists >= 0),(d_dists < interval))
                # doublecheck we are not overwriting an existing label:
                # idx_same should point to positions that are still nan in d_ind 
                if any(np.isnan(d_ind[idx_same])==False):
                    pdb.set_trace()
                elif not any(idx_same): # if no wells found within the interval
                    pdb.set_trace()
                else:
                    # assign the row [col] label to the wells closer than
                    # interval to the topmost [leftmost] unlabelled well
                    d_ind[idx_same] = cc
                # increment label
                cc+=1
            # end while
            # assign label array to right dimension
            self.wells[lp] = d_ind.astype(int)
        
        # checks: if 24 wells => 4 entries only, if 48 either 3x3 or 3x2
        if self.n_wells == 24:
            _is_2x2 = self.wells.shape[0] == 4 and \
                        self.wells.row.max() == 1 and \
                        self.wells.col.max() == 1
            if not _is_2x2:
                self.plot_wells()
                raise Exception("Found wells not in a 2x2 arrangement, results are unreliable");
        elif self.n_wells == 48:
            _is_3x2 = self.wells.shape[0] == 6 and \
                        self.wells.row.max() == 2 and \
                        self.wells.col.max() == 1
            _is_3x3 = self.wells.shape[0] == 9 and \
                        self.wells.row.max() == 2 and \
                        self.wells.col.max() == 2
            if not (_is_3x2 or _is_3x3):
                self.plot_wells()
                raise Exception("Found wells not in a 3x2 or 3x3 arrangement, results are unreliable");
        
        return 
        

    def remove_half_circles(self, max_radius_portion_missing=0.5):
        """
        Only keep circles whose centre is at least 
        (1-max_radius_portion_missing)*radius away
        from the edge of the image
        """
        
        assert self.well_shape == 'circle', "This method is only to be used with circular wells"
        
        # average radius across circles (3rd column)
        avg_radius = self.wells['r'].mean()
        # keep only circles missing less than 0.5 radius
        extra_space = avg_radius*(1-max_radius_portion_missing); 
        # bad circles = centre of circles is not too close to image edge
        idx_bad_circles =   (self.wells['x'] - extra_space < 0) | \
                            (self.wells['x'] + extra_space >= self.img_shape[1]) | \
                            (self.wells['y'] - extra_space < 0) | \
                            (self.wells['y'] + extra_space >= self.img_shape[0])
        # remove entries that did not satisfy the initial requests
        self.wells.drop(self.wells[idx_bad_circles].index, inplace=True)
        return 


    def find_wells_boundaries(self):
        """
        Find lines along which to crop the FOV.
        Lines separating rows/columns are halfway between the grouped medians of 
        the relevant coordinate.
        Lines before the first and after the last row/column are the median 
        coordinate +- 0.5 the median lattice spacing.
        """
        # loop on dimension (and lattice place). di = dimension counter
        # di is needed to index on self.img_shape
        for di,(d,lp) in enumerate(zip(['x','y'],['col','row'])):
            # only look at correct column of dataframe. temporary variables for shortening purposes
            labels = self.wells[lp]
            coords = self.wells[d]
            # average distance between rows [cols]
            avg_lattice_spacing = np.diff(coords.groupby(labels).mean()).mean()
            max_ind = np.max(labels) # max label of rows [columns]
            # initialise array that will hold info re where to put lines
            # N lines = N rows + 1 = max row + 2 b.c. 0 indexing
            lines_coords = np.zeros(max_ind+2)
            # take care of lfirst and last edge
            lines_coords[0] = np.median(coords[labels==0]) - avg_lattice_spacing/2
            lines_coords[0] = max(lines_coords[0], 0); # line has to be within image bounds
            lines_coords[-1] = np.median(coords[labels==max_ind]) + avg_lattice_spacing/2
            lines_coords[-1] = min(lines_coords[-1], self.img_shape[1-di]); # line has to be within image bounds
            # for each row [col] find the middle point with the next one, 
            # write it into the lines_coord variable
            for ii in range(max_ind):
                jj = ii+1; # index on lines_coords 
                lines_coords[jj] = np.median(np.array([
                        np.mean(coords[labels==ii]),
                        np.mean(coords[labels==ii+1])]));
            # store into self.wells for return
            self.wells[d+'_min'] = lines_coords.copy().astype(np.int)[labels]
            self.wells[d+'_max'] = lines_coords.copy().astype(np.int)[labels+1]
    
        return


    def fill_lattice_defects(self):
        """
        If a grid of wells was detected but there are entries missing, try to
        return a guesstimate of where the missing well(s) may be.
        """
        # find, in theory, how many wells does the detected grid allow for
        n_rows = self.wells['row'].max()+1
        n_cols = self.wells['col'].max()+1
        n_expected_wells = n_rows * n_cols;
        n_detected_wells = len(self.wells)

        if n_detected_wells == n_expected_wells:
            # nothing to do here
            return
        elif n_detected_wells > n_expected_wells:
            # uncropped image? other errors?
            raise Exception("Found more wells than expected. Aborting now.")
        # I only get here if n_detected_wells < n_expected_wells
        assert n_detected_wells < n_expected_wells, \
            "Something wrong in the logic in fill_lattice_defects()"
        # some wells got missed. Using the lattice structure to find them
        expected_rowcols = set(itertools.product(range(n_rows), range(n_cols)))
        detected_rowcols = set((rr,cc) for rr,cc in self.wells[['row','col']].values)
        missing_rowcols = list(expected_rowcols - detected_rowcols)
        # now add the missing rowcols combinations
        for rr,cc in missing_rowcols:
            new_well = {}
            # calculate x,y,r
            y = self.wells[self.wells['row'] == rr]['y'].median()
            x = self.wells[self.wells['col'] == cc]['x'].median()
            r = self.wells['r'].mean()
            # append to temporary dict
            new_well['x'] = [x,]
            new_well['y'] = [y,]
            new_well['r'] = [r,]
            new_well['row'] = [rr,]
            new_well['col'] = [cc,]
            new_df = pd.DataFrame(new_well)
#            print(new_df)
            self.wells = pd.concat([self.wells, new_df], ignore_index=True, sort=False)
        return


    def calculate_wells_dimensions(self):
        """
        Finds width, height of each well
        """
        self.wells['width'] = self.wells['x_max']-self.wells['x_min']
        self.wells['height'] = self.wells['y_max']-self.wells['y_min']
        return


    def name_wells(self):
        """
        Assign name to the detected wells.
        Need to know what channel, how many wells in total, if mwp was upright,
        and in the future where was A1 or if the video with A1 has got 6 or 9 wells
        """
        
        max_row = self.wells['row'].max()
        max_col = self.wells['col'].max()
        
        # odd and even channels have opposite orientation 
        # ("up" in the camera is always towards outside of the rig)
        # so flip the row [col] labels before going to read from the MWP_dataframe
        # for odd channels
        if int(self.channel[-1])%2==1:
            self.wells['well_name'] = \
                [self.mwp_df.iloc[max_row-r, max_col-c] \
                 for r,c in self.wells[['row','col']].values]
        else:
            self.wells['well_name'] = [self.mwp_df.iloc[r,c] \
                         for r,c in self.wells[['row','col']].values]
        # the above code is equivalent (but faster than) the following two alternatives:
#        ################### alternative 1
#        def flip_oddchannels_rowcol(r, c, chname):
#            if int(chname[-1])%2==1:
#                row = max_row-r
#                col = max_col-c
#            else:
#                row = r
#                col = c
#            return (row, col)
#        # define function that acts on each row
#        def _apply_dict_to_row(_well):
#            row, col = flip_oddchannels_rowcol(_well['row"], _well['col"], self.channel)
#            return self.mwp_df.iloc[row,col]
#        # apply function to each row
#        self.wells['well_name'] = self.wells.apply(_apply_dict_to_row, axis=1)
#        #################### alternative 2
#        # the lines above are equivalent to:
#        for _i, _well in self.wells.iterrows():
#            row, col = flip_oddchannels_rowcol(_well['row"], _well['col"], self.channel)
#            self.wells.loc[_i,'well_name'] = self.mwp_df.iloc[row,col]
            
        return 


    def tile_FOV(self, img_or_stack):
        """
        Function that tiles the input image or stack and returns a dictionary of (well_name, ROI).
        When integrating in Tierpsy, check if you can use Avelino's function 
        for ROI making, could be a lot quicker
        """
        if len(img_or_stack.shape) == 2:
            return self.tile_FOV_2D(img_or_stack)
        elif len(img_or_stack.shape) == 3:
            return self.tile_FOV_3D(img_or_stack)
        else:
            raise Exception("Can only tile 2D or 3D objects")
            return
        
    def tile_FOV_2D(self, img):
        """
        Function that chops an image according to the x/y_min/max coordinates in
        wells, and returns a dictionary of (well_name, ROI).
        When integrating in Tierpsy, check if you can use Avelino's function 
        for ROI making, could be a lot quicker"""
        # initialise output
        out_list = []
        # loop on rois
        for rc, well in self.wells.iterrows():
            # extract roi name and roi data
            roi_name = well['well_name']
            roi_img = img[well['y_min']:well['y_max'],well['x_min']:well['x_max']]
            # grow output dictionary
            out_list.append((roi_name, roi_img))
        return out_list
    
    
    def tile_FOV_3D(self, img):
        """
        Function that chops an image stack (1st dimension is n_frames)  
        according to the x/y_min/max coordinates in
        wells, and returns a dictionary of (well_name, ROI).
        When integrating in Tierpsy, check if you can use Avelino's function 
        for ROI making, could be a lot quicker"""
        # initialise output
        out_list = []
        # loop on rois
        for rc, well in self.wells.iterrows():
            # extract roi name and roi data
            roi_name = well['well_name']
            roi_img = img[:,well['y_min']:well['y_max'],well['x_min']:well['x_max']]
            # grow output dictionary
            out_list.append((roi_name, roi_img))
        return out_list


    def plot_wells(self):
        """
        Plot the fitted wells, the wells separation, and the name of the well.
        (only if these things are present!)"""

        # make sure I'm not working on the original image
        _img = cv2.cvtColor(self.img.copy(),cv2.COLOR_GRAY2BGR)
#        pdb.set_trace()
        # flags: according to dataframe state, do or do not do
        _is_wells = self.wells.shape[0] > 0;
        _is_rois = np.logical_not(self.wells['x_min'].isnull()).all() and _is_wells;
        _is_wellnames = np.logical_not(self.wells['well_name'].isnull()).all() and _is_rois;
        # TODO: deal with grayscale image
        # burn the circles into the rgb image
        if _is_wells and self.well_shape == 'circle':
            for i, _circle in self.wells.iterrows():
                # draw the outer circle
                cv2.circle(_img,(_circle.x,_circle.y),_circle.r,(255,0,0),5)
                # draw the center of the circle
                cv2.circle(_img,(_circle.x,_circle.y),5,(0,255,255),5)
        # burn the boxes edges into the RGB image
        if _is_rois:
            #normalize item number values to colormap
            normcol = colors.Normalize(vmin=0, vmax=self.wells.shape[0])
            for i, _well in self.wells.iterrows():
                rgba_color = cm.Set1(normcol(i),bytes=True)
                rgba_color = tuple(map(lambda x : int(x), rgba_color))
#                pdb.set_trace()
                # same as:
#                rgba_color = tuple(np.array(rgba_color).astype(np.int))
                cv2.rectangle(_img,
                              (_well.x_min, _well.y_min),
                              (_well.x_max, _well.y_max),
                              rgba_color[:-1], 20)
        # add names of wells
        # plot, don't close
        hf = plt.figure(figsize=(10.06,7.59));
        plt.imshow(_img)
        if _is_wellnames:
            for i, _well in self.wells.iterrows():
                txt = "{} ({:d},{:d})".format(_well['well_name'],
                       int(_well['row']),
                       int(_well['col']))
                plt.text(_well['x'], _well['y'], txt,
                         fontsize=12,
                         color='r')
        elif _is_rois:
            for i, _well in self.wells.iterrows():
                plt.text(_well['x'], _well['y'],
                         "({:d},{:d})".format(int(_well['row']),int(_well['col'])),
                         fontsize=12,
                         color='r')
        plt.axis('off')
        return hf
    
    
    def find_well_of_xy(self, x, y):
        """
        Takes two numpy arrays (or pandas columns), returns an array of strings
        of the same with the name of the well each x,y, pair falls into
        """
        # I think a quick way is by using implicit expansion
        # treat the x array as column, and the *_min and *_max as rows
        # these are all matrices len(x)-by-len(self.wells)
        within_x = np.logical_and((x[:,None] - self.wells['x_min'][None,:]) >= 0,  # none creates new axis
                                  (x[:,None] - self.wells['x_max'][None,:]) <= 0)
        within_y = np.logical_and((y[:,None] - self.wells['y_min'][None,:]) >= 0,  # none creates new axis
                                  (y[:,None] - self.wells['y_max'][None,:]) <= 0)
        within_well = np.logical_and(within_x, within_y) 
        # in each row of within_well, the column index of the "true" value is the well index
        
        # sanity check:
        assert (within_well.sum(axis=1)>1).any() == False, \
        "a coordinate is being assigned to more than one well?"
        # now find index
        ind_worms_in_wells, ind_well = np.nonzero(within_well)

        # prepare the output panda series (as long as the input variable)
        well_names = pd.Series(data=['n/a']*len(x), dtype='S3', name='well_name')
        # and assign the well name (read using the ind_well variable from the self.well)
        well_names.loc[ind_worms_in_wells] = self.wells.iloc[ind_well]['well_name'].astype('S3').values

        return well_names
    
    
    def find_well_from_trajectories_data(self, trajectories_data):
        """Wrapper for find_well_of_xy, 
        reads the coordinates from the right columns of trajectories_data"""
        return self.find_well_of_xy(trajectories_data['coord_x'], 
                                    trajectories_data['coord_y'])
        
    def get_wells_data(self):
        """
        Returns info about the wells for storage purposes, in hdf5 friendly format
        """
        wells_out = self.wells[['x_min','x_max','y_min','y_max']].copy()
        wells_out['well_name'] = self.wells['well_name'].astype('S3')
        return wells_out
    
    
    def read_worm_counts(self, annotated_img, path_to_templates):
        """
        cross correlation of image with characters
        """
        
        is_debug = True
        
        templates = []
        for ii in range(10):
            c = str(ii)
            digit_name = path_to_templates / (c+".png")
            digit_img = cv2.imread(str(digit_name))
            digit_img = cv2.cvtColor(digit_img,cv2.COLOR_BGR2GRAY)
            digit_img = 255*(digit_img==0).astype(np.uint8)
            templates.append(digit_img)
        
        # loop on wells
        worm_count_dict = {}
        for i, well in self.wells.iterrows():
            # cut out region of interest
            roi = annotated_img[well['y_min']:well['y_max'],
                           well['x_min']:well['x_max']]
            roi_bw = 255*(roi==0).astype(np.uint8)

            # compare roi wit all digits
            is_digit_a_hit = np.zeros(10).astype(np.bool) # here store how well the digit matches the roi
            is_digit_repeated = np.zeros(10).astype(np.bool)
            x_coord_of_digit = np.nan * np.ones(10) 
            for dc, digit in enumerate(templates):
                res = cv2.matchTemplate(roi_bw, digit, cv2.TM_CCORR_NORMED)
            
                # find local maxima that are at least well_size_px apart
                X=peak_local_max(res, min_distance=np.min(digit.shape)//2, 
                                 indices=True, 
                                 exclude_border=False,
                                 threshold_abs=0.85)
                print(res.max())
                # save hits
                if X.shape[0] == 1:
                    is_digit_a_hit[dc] = True
                    x_coord_of_digit[dc] = X[0,1] # X is a list of row, column couples
                elif X.shape[0] == 2:
                    is_digit_a_hit[dc] = True
                    is_digit_repeated[dc] = True
                    x_coord_of_digit[dc] = X[0,1] # since repeated we don't really care about this
                elif X.shape[0] > 2:
                    plt.imshow(res)
                    import pdb
                    pdb.set_trace()
                    
                else:
#                    if is_debug:
#                    plt.figure()
#                    plt.imshow(roi)
                    plt.figure(str(dc))
                    plt.imshow(res)
                    pass # do nothing

            # how may different digits were found?
            # only one:
            if is_digit_a_hit.sum()==1:
                which_digit_was_it = np.argmax(is_digit_a_hit) # position in is_digit_a_hit is indeed digit
                # was the only found digit repeated?
                if is_digit_repeated[which_digit_was_it]: # it was repeated
                    number_out = which_digit_was_it * 11 # assuming <100 worms
                else: # it was not
                    number_out = which_digit_was_it
            # two different digits
            elif is_digit_a_hit.sum()==2:
                which_digits = np.argwhere(is_digit_a_hit).flatten()
                digits_coordinates = x_coord_of_digit[is_digit_a_hit]
                assert len(which_digits)==len(digits_coordinates), 'different lengths'
                digits_sorted = which_digits[np.argsort(digits_coordinates)]
                number_out = 10*digits_sorted[0] + digits_sorted[1]
            elif is_digit_a_hit.sum()==0:
                import pdb
                pdb.set_trace()
                raise ValueError('No digits were found!')
            elif is_digit_a_hit.sum()>2:
                raise ValueError('Too many digits were found!')    
            
            # now give output
            print('well name: {}, worm number: {}'.format(well['well_name'], number_out))
            worm_count_dict[well['well_name']] = number_out
            
            if is_debug:
                plt.figure('well name: {}, worm number: {}'.format(well.well_name, number_out))
                plt.imshow(roi)
                    
        return worm_count_dict
    
    
    
# end of class


def read_data_from_masked(masked_image_file):
    """
    - Opens the masked_image_file hdf5 file, reads the /full_data node and 
      creates a "background" by taking the maximum value of each pixel over time.
    - Parses the file name to find a camera serial number
    - reads the pixel/um ratio from the masked_image_file
    """
    # read attributes of masked_image_file
    _, (microns_per_pixel, xy_units) , is_light_background = read_unit_conversions(masked_image_file)
    # get "background" and px2um
    with pd.HDFStore(masked_image_file, 'r') as fid:
        assert is_light_background, \
        'MultiWell recognition is only available for brightfield at the moment'
        img = np.max(fid.get_node('/full_data'), axis=0)
    # find camera name
    regex = r"(?<=20\d{6}\_\d{6}\.)\d{8}"
    camera_serial = re.findall(regex, str(masked_image_file).lower())[0]
    
    return img, camera_serial, microns_per_pixel


def make_square_template(n_pxls=150, rel_width=0.8, blurring=0.1):
    """Function that creates a template that approximates a square well"""
    n_pxls = int(np.round(n_pxls))
    x = np.linspace(-0.5, 0.5, n_pxls)
    y = np.linspace(-0.5, 0.5, n_pxls)
    xx, yy = np.meshgrid(x, y, sparse=False, indexing='ij')

    # inspired by Mark Shattuck's function to make a colloid's template
    zz = (1 - np.tanh( (abs(xx)-rel_width/2)/blurring ))
    zz = zz * (1-np.tanh( (abs(yy)-rel_width/2)/blurring ))
    zz = (zz/4*255).astype(np.uint8)
    
    return zz
    
#%% main
        
if __name__ == '__main__':
    
    import re
    from pathlib import Path
#    from scipy.signal import find_peaks

    plt.close("all")

    # test from images:
    
#    wd = Path.home() / 'Desktop/Data_FOVsplitter'
#    img_dir = wd / 'RawVideos/96wpsquare_upright_150ulagar_l1dispensed_1_20190614_105312_firstframes'
#    wd = Path.home() / 'Desktop/Data_FOVsplitter'
#    img_dir = wd / 'RawVideos/drugexperiment_1hrexposure'
#    wd = Path('/Volumes/behavgenom$/Luigi/Data/LoopBio_calibrations/wells_mapping/20190710/')
#    img_dir = wd
#    img_dir = wd / 'Hydra04'
    
#    fnames = list(img_dir.rglob('*.png'))
##    fnames = fnames[2:3] # for code-review only
#    for fname in fnames:
#        # load image
#        img_ = cv2.imread(str(fname))
#        img = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
#        # find camera name
#        regex = r"(?<=20\d{6}\_\d{6}\.)\d{8}"
#        camera_serial = re.findall(regex, str(fname).lower())[0]
#        # run fov splitting
#        fovsplitter = FOVMultiWellsSplitter(img=img, camera_serial=camera_serial, total_n_wells=96,
#                                            whichsideup='upright', well_shape='square', px2um=12.4)
#        fig = fovsplitter.plot_wells()
#        plt.tight_layout()
#        fig.savefig(camera_serial + '.png', bbox_inches='tight', pad_inches=0, transparent=True)
     
#    %% test on filename
    
    masked_image_file = '/Users/lferiani/Desktop/Data_FOVsplitter/MaskedVideos/drugexperiment_1hrexposure_set1_20190712_131508.22436248/metadata.hdf5'   
    features_file = masked_image_file.replace('MaskedVideos','Results').replace('.hdf5','_featuresN.hdf5')
    import shutil
    shutil.copy(features_file.replace('.hdf5','.bk'), features_file)
    
    fovsplitter = FOVMultiWellsSplitter(masked_image_file=masked_image_file, total_n_wells=96,
                                            whichsideup='upright', well_shape='square')   
    
    foo_wells = fovsplitter.get_wells_data()
    
#    fig = fovsplitter.plot_wells()
    
#    with pd.HDFStore(features_file, 'r') as fid:
#        traj_data = fid['/trajectories_data']
#        well_names = fovsplitter.find_well_from_trajectories_data(traj_data)
        
        
    #%% test worm counting
#    img_dir = Path('/Volumes/behavgenom$/Ida/Data/LoopBio/20190719_MWPtest/firstframes/')
#
#    fnames = list(img_dir.rglob('*.png'))
#    fnames = [f for f in fnames if 'annotated' not in str(f)]
#     
#    path_to_templates = Path('/Volumes/behavgenom$/Ida/numbers')
#     
#    for fname in fnames[6:7]:
#        # load image
#        img_ = cv2.imread(str(fname))
#        annotated_img = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
#        # find camera name
#        regex = r"(?<=20\d{6}\_\d{6}\.)\d{8}"
#        camera_serial = re.findall(regex, str(fname).lower())[0]
#        # run fov splitting
#        fovsplitter = FOVMultiWellsSplitter(img=annotated_img, camera_serial=camera_serial, total_n_wells=96,
#                                            whichsideup='upright', well_shape='square', px2um=12.4)
#        # read worms count
#        fovsplitter.read_worm_counts(annotated_img, path_to_templates)
        
    