#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:55:41 2019

@author: lferiani
"""


#%% import statements

import cv2
import pdb
import numpy as np
import pandas as pd

from matplotlib import cm
from matplotlib import colors
from matplotlib import pyplot as plt


#%% constants

# dictionary to go from camera name to channel
# to be updated as we get more copies of the LoopBio rig
CAM2CH_DICT = {"22594549":'Ch1',
               "22594548":'Ch2',
               "22594546":'Ch3',
               "22436248":'Ch4',
               "22594559":'Ch5',
               "22594547":'Ch6'}

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



#%% Class definition
class FOVMultiWellsSplitter(object):
    """Class tasked with finding how to split a full-FOV image into single-wells images, 
    and then splitting new images that are passed to it."""
    
    def __init__(self, img, camera, total_n_wells=48, whichsideup='upright'):
        """Class constructor. 
        Creates circles, and parses the image to fill up the circles property
        img = a brightfield frame that will be used for circle-finding
        n_wells = how many wells *in the entire multiwell plate*"""
        # save the input image just to make some things easier
        if len(img.shape) == 2:
            self.img = img.copy()
        elif len(img.shape) == 3:
            # convert to grey
            self.img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # save height and width of image
        self.img_shape = img.shape
        self.camera_name = camera
        # where was the camera on the rig? 
        # TODO: input check. Dunno if it will be kept like this or parsed from a filename
        self.channel = CAM2CH_DICT[self.camera_name]
        # number of wells in the multiwell plate: 6 12 24 48 96?
        # TODO: input check. Dunno if it will be kept like this or parsed from a filename
        self.n_wells = total_n_wells
        # whichsideup: was the cell upright or upside-down
        # TODO: only allow upside-down or upright. Dunno if it will be kept like this or parsed from a filename
        self.whichsideup = whichsideup
        # according to n_wells and whichsideup choose a dictionary for 
        #(channel,position) <==> well 
        # TODO: the dictionaries will be imported from a helper module
        # TODO: make user specify (if 48wp) which channels have 6 wells and not 9
        if (self.n_wells == 48) and (self.whichsideup == 'upright'):
            self.mwp_df = UPRIGHT_48WP_669999[self.channel]
        else:
            raise Exception("This case hasn't been coded for yet")
        # circles is the most important property. 
        # It's the dataframe that contains the coordinates of each recognised 
        # well in the original image
        # In particular
        #   x, y         = coordinates of the circle's centre, in pixel (so x is a column index, y a row index)
        #   r            = radius of the circle, in pixel
        #   row, col     = indices of a circle in the grid of detected wells
        #   *_max, *_min = coordinates for cropping the FOV so 1 roi = 1 well
        self.circles = pd.DataFrame(columns = ['x','y','r','row','col',
                                          'x_min','x_max','y_min','y_max',
                                          'well'])
    
        # METHODS
        # call method to fill in the circles variable
        self.find_circular_wells()
        self.remove_half_circles()
        self.find_row_col_wells()
        self.find_wells_boundaries()
        self.calculate_wells_dimensions()
#        print(self.circles)
        self.name_wells()
        print(self.circles)



    def find_circular_wells(self):
        """Simply use Hough transform to find circles in MultiWell Plate rgb image.
        The parameters used are optimised for 24 or 48WP"""
                    
        # preprocess image
        dwnscl_factor = 4; # Hydra01 images' shape is divisible by 4
        blr_sigma = 17; # blur the image a bit, seems to work better
        new_shape = (self.img.shape[1]//dwnscl_factor, # as x,y, not row,columns
                     self.img.shape[0]//dwnscl_factor)
        
        dwn_gray_im = cv2.resize(self.img, new_shape)
        # apply blurring
        blur_im = cv2.GaussianBlur(dwn_gray_im, (blr_sigma,blr_sigma),0)
        # normalise between 0 and 255
        blur_im = cv2.normalize(blur_im, None, alpha=0, beta=255, 
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # find circles
        # parameters in downscaled units
        circle_goodness = 70;
        highest_canny_thresh = 10;
        min_well_dist = new_shape[1]/3;    # max 3 wells along short side. bank on FOV not taking in all the entirety of the well 
        min_well_radius = new_shape[1]//7; # if 48WP 3 wells on short side ==> radius <= side/6
        max_well_radius = new_shape[1]//4; # if 24WP 2 wells on short side. COnsidering intrawells space, radius <= side/4 
        # find circles
        _circles = cv2.HoughCircles(blur_im,
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
        self.circles['x'] = _circles[:,0].astype(int)
        self.circles['y'] = _circles[:,1].astype(int)
        self.circles['r'] = _circles[:,2].astype(int)
        return
        

    def find_row_col_wells(self):
        """
        The circular wells are aligned in a grid, but are not found in such 
        order by the Hough Transform. 
        Find (row, column) index for each circle.
        Algorithm: (same for rows and columns)
            - Scan the found wells, pick up the topmost [leftmost] one
            - Find all other circles that are within the average radius of the 
                first along the consddered dimension.
            - Assign the same row [column] label to all of them
            - Repeat for the wells that do not yet have an assigned label, 
                increasing the label index
        """
        # average radius across circles (3rd column)
        avg_radius = self.circles["r"].mean()
        
        # execute same loop for both rows and columns
        for d,lp in zip(['x','y'],['col','row']): # d = dimension, lp = lattice place
            # initialise array or row/column labels. This is a temporary variable, I could have just used self.circles[lp]
            d_ind = np.full(self.circles.shape[0],np.nan)
            cc = 0; # what label are we assigning right now
            # loop until all the labels have been assigned
            while any(np.isnan(d_ind)):
                # find coordinate of first (leftmost or topmost) non-labeled circle
                idx_unlabelled_circles = np.isnan(d_ind)
                unlabelled_circles = self.circles.loc[idx_unlabelled_circles]
                coord_first_circle = np.min(unlabelled_circles[d])
                # find distance between this and *all* circles along the considered dimension
                d_dists = self.circles[d] - coord_first_circle;
                # find circles within avg_radius. d_dists>=0 discards previous rows [columns]
                # could have taken the absolute value instead but meh I like this logic better
                idx_same = np.logical_and((d_dists >= 0),(d_dists < avg_radius))
                # doublecheck we are not overwriting an existing label:
                # idx_same should point to positions that are still nan in d_ind 
                if any(np.isnan(d_ind[idx_same])==False):
                    pdb.set_trace()
                elif not any(idx_same): # if no wells found within the avg_radius
                    pdb.set_trace()
                else:
                    # assign the row [col] label to the circles closer than
                    # avg_radius to the topmost [leftmost] unlabelled circle
                    d_ind[idx_same] = cc
                # increment label
                cc+=1
            # end while
            # assign label array to right dimension
            self.circles[lp] = d_ind.astype(int)
        
        # checks: if 24 wells => 4 entries only, if 48 either 3x3 or 3x2
        if self.n_wells == 24:
            _is_2x2 = self.circles.shape[0] == 4 and \
                        self.circles.row.max() == 1 and \
                        self.circles.col.max() == 1
            if not _is_2x2:
                self.plot_wells()
                raise Exception("Found wells not in a 2x2 arrangement, results are unreliable");
        elif self.n_wells == 48:
            _is_3x2 = self.circles.shape[0] == 6 and \
                        self.circles.row.max() == 2 and \
                        self.circles.col.max() == 1
            _is_3x3 = self.circles.shape[0] == 9 and \
                        self.circles.row.max() == 2 and \
                        self.circles.col.max() == 2
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
        # average radius across circles (3rd column)
        avg_radius = self.circles["r"].mean()
        # keep only circles missing less than 0.5 radius
        extra_space = avg_radius*(1-max_radius_portion_missing); 
        # bad circles = centre of circles is not too close to image edge
        idx_bad_circles =   (self.circles["x"] - extra_space < 0) | \
                            (self.circles["x"] + extra_space >= self.img_shape[1]) | \
                            (self.circles["y"] - extra_space < 0) | \
                            (self.circles["y"] + extra_space >= self.img_shape[0])
        # remove entries that did not satisfy the initial requests
        self.circles.drop(self.circles[idx_bad_circles].index, inplace=True)
        return 


    def find_wells_boundaries(self):
        """
        Find lines along which to crop the FOV.
        Lines separating rows/columns are halfway between the grouped average of 
        the relevant coordinate.
        Lines before the first and after the last row/column are the average 
        coordinate +- 0.5 the average lattice spacing.
        """
        # loop on dimension (and lattice place). di = dimension counter
        # di is needed to index on self.img_shape
        for di,(d,lp) in enumerate(zip(['x','y'],['col','row'])):
            # only look at correct column of dataframe. temporary variables for shortening purposes
            labels = self.circles[lp]
            coords = self.circles[d]
            # average distance between rows [cols]
            avg_lattice_spacing = np.diff(coords.groupby(labels).mean()).mean()
            max_ind = np.max(labels) # max label of rows [columns]
            # initialise array that will hold info re where to put lines
            # N lines = N rows + 1 = max row + 2 b.c. 0 indexing
            lines_coords = np.zeros(max_ind+2)
            # take care of lfirst and last edge
            lines_coords[0] = np.mean(coords[labels==0]) - avg_lattice_spacing/2
            lines_coords[0] = max(lines_coords[0], 0); # line has to be within image bounds
            lines_coords[-1] = np.mean(coords[labels==max_ind]) + avg_lattice_spacing/2
            lines_coords[-1] = min(lines_coords[-1], self.img_shape[1-di]); # line has to be within image bounds
            # for each row [col] find the middle point with the next one, 
            # write it into the lines_coord variable
            for ii in range(max_ind):
                jj = ii+1; # index on lines_coords 
                lines_coords[jj] = np.mean(np.array([
                        np.mean(coords[labels==ii]),
                        np.mean(coords[labels==ii+1])]));
            # store into self.circles for return
            self.circles[d+'_min'] = lines_coords.copy().astype(np.int)[labels]
            self.circles[d+'_max'] = lines_coords.copy().astype(np.int)[labels+1]
    
        return

    def calculate_wells_dimensions(self):
        """
        Finds width, height of each well
        """
        self.circles["width"] = self.circles["x_max"]-self.circles["x_min"]
        self.circles["height"] = self.circles["y_max"]-self.circles["y_min"]
        return

    def name_wells(self):
        """
        Assign name to the detected wells.
        Need to know what channel, how many wells in total, if mwp was upright,
        and in the future where was A1 or if the video with A1 has got 6 or 9 wells
        """
        
        max_row = self.circles.row.max()
        max_col = self.circles.col.max()
        
        # odd and even channels have opposite orientation 
        # ("up" in the camera is always towards outside of the rig)
        # so flip the row [col] labels before going to read from the MWP_dataframe
        # for odd channels
        if int(self.channel[-1])%2==1:
            self.circles["well"] = \
                [self.mwp_df.iloc[max_row-r, max_col-c] \
                 for r,c in self.circles[["row","col"]].values]
        else:
            self.circles["well"] = [self.mwp_df.iloc[r,c] \
                         for r,c in self.circles[["row","col"]].values]
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
#        def _apply_dict_to_row(_circle):
#            row, col = flip_oddchannels_rowcol(_circle["row"], _circle["col"], self.channel)
#            return self.mwp_df.iloc[row,col]
#        # apply function to each row
#        self.circles["well"] = self.circles.apply(_apply_dict_to_row, axis=1)
#        #################### alternative 2
#        # the lines above are equivalent to:
#        for _i, _circle in self.circles.iterrows():
#            row, col = flip_oddchannels_rowcol(_circle["row"], _circle["col"], self.channel)
#            self.circles.loc[_i,"well"] = self.mwp_df.iloc[row,col]
            
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
        circles, and returns a dictionary of (well_name, ROI).
        When integrating in Tierpsy, check if you can use Avelino's function 
        for ROI making, could be a lot quicker"""
        # initialise output
        out_list = []
        # loop on rois
        for rc, circle in self.circles.iterrows():
            # extract roi name and roi data
            roi_name = circle["well"]
            roi_img = img[circle["y_min"]:circle["y_max"],circle["x_min"]:circle["x_max"]]
            # grow output dictionary
            out_list.append((roi_name, roi_img))
        return out_list
    
    
    def tile_FOV_3D(self, img):
        """
        Function that chops an image stack (1st dimension is n_frames)  
        according to the x/y_min/max coordinates in
        circles, and returns a dictionary of (well_name, ROI).
        When integrating in Tierpsy, check if you can use Avelino's function 
        for ROI making, could be a lot quicker"""
        # initialise output
        out_list = []
        # loop on rois
        for rc, circle in self.circles.iterrows():
            # extract roi name and roi data
            roi_name = circle["well"]
            roi_img = img[:,circle["y_min"]:circle["y_max"],circle["x_min"]:circle["x_max"]]
            # grow output dictionary
            out_list.append((roi_name, roi_img))
        return out_list


    def plot_wells(self):
        """
        Plot the fitted circles, the wells separation, and the name of the well.
        (only if these things are present!)"""

        # make sure I'm not working on the original image
        _img = cv2.cvtColor(self.img.copy(),cv2.COLOR_GRAY2BGR)
#        pdb.set_trace()
        # flags: according to dataframe state, do or do not do
        _is_circles = self.circles.shape[0] > 0;
        _is_rois = np.logical_not(self.circles["x_min"].isnull()).all() and _is_circles;
        _is_wellnames = np.logical_not(self.circles["well"].isnull()).all() and _is_rois;
        # TODO: deal with grayscale image
        # burn the circles into the rgb image
        if _is_circles:
            for i, _circle in self.circles.iterrows():
                # draw the outer circle
                cv2.circle(_img,(_circle.x,_circle.y),_circle.r,(255,0,0),5)
                # draw the center of the circle
                cv2.circle(_img,(_circle.x,_circle.y),5,(0,255,255),5)
        # burn the boxes edges into the RGB image
        if _is_rois:
            #normalize item number values to colormap
            normcol = colors.Normalize(vmin=0, vmax=self.circles.shape[0])
            for i, _circle in self.circles.iterrows():
                rgba_color = cm.Set1(normcol(i),bytes=True)
                rgba_color = tuple(map(lambda x : int(x), rgba_color))
#                pdb.set_trace()
                # same as:
#                rgba_color = tuple(np.array(rgba_color).astype(np.int))
                cv2.rectangle(_img,
                              (_circle.x_min, _circle.y_min),
                              (_circle.x_max, _circle.y_max),
                              rgba_color[:-1], 20)
        # add names of wells
        # plot, don't close
        hf = plt.figure(figsize=(10.06,7.59));
        plt.imshow(_img)
        if _is_wellnames:
            for i, _circle in self.circles.iterrows():
                txt = "{} ({:d},{:d})".format(_circle.well,
                       int(_circle.row),
                       int(_circle.col))
                plt.text(_circle.x, _circle.y, txt,
                         fontsize=12,
                         color='r')
        elif _is_rois:
            for i, _circle in self.circles.iterrows():
                plt.text(_circle.x, _circle.y,
                         "({:d},{:d})".format(int(_circle.row),int(_circle.col)),
                         fontsize=12,
                         color='r')
        plt.axis('off')
        return hf
    
# end of class
    
#%% main
        
if __name__ == '__main__':
    
    import re
    import collections
    from pathlib import Path
    from scipy.signal import find_peaks
    from skimage.measure import label, regionprops

    wd = Path.home() / 'Desktop/Data_FOVsplitter'
    img_dir = wd / 'RawVideos/96wpsquare_upright_150ulagar_l1dispensed_1_20190614_105312_firstframes'
#    img_path = img_dir / '96wpsquare_upright_150ulagar_l1dispensed_1_20190614_105312.22436248.png'
#    img_path = img_dir / '96wpsquare_upright_150ulagar_l1dispensed_1_20190614_105312.22594546.png'
#    img_path = img_dir / '96wpsquare_upright_150ulagar_l1dispensed_1_20190614_105312.22594547.png'
    img_path = img_dir / '96wpsquare_upright_150ulagar_l1dispensed_1_20190614_105312.22594548.png'
#    img_path = img_dir / '96wpsquare_upright_150ulagar_l1dispensed_1_20190614_105312.22594549.png'   
#    img_path = img_dir / '96wpsquare_upright_150ulagar_l1dispensed_1_20190614_105312.22594559.png'

    img_ = cv2.imread(str(img_path))
    img = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    
    # find camera name
    regex = r"(?<=(20\d{6}\_\d{6}\.))\d{8}"
    camera_name = re.findall(regex, str(img_path).lower())[0]
    
    
    
    # parameters - this will go in other function
    # leave untouched from here
    dwnscl_factor = 4; # Hydra01 images' shape is divisible by 4
    blr_sigma = 17; # blur the image a bit, seems to work better
    new_shape = (img.shape[1]//dwnscl_factor, # as x,y, not row,columns
                 img.shape[0]//dwnscl_factor)
    dwn_gray_im = cv2.resize(img, new_shape)
    # apply blurring
    blur_im = cv2.GaussianBlur(dwn_gray_im, (blr_sigma,blr_sigma),0)
    # normalise between 0 and 255
    blur_im = cv2.normalize(blur_im, None, alpha=0, beta=255, 
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
#    plt.close("all")
#    # to here
#    
#    edge = cv2.Canny(blur_im, 5, 50, None, 3)
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#    edgedil = cv2.dilate(edge,kernel,iterations = 2)
#
#    bwim = cv2.adaptiveThreshold(blur_im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv2.THRESH_BINARY,51,2)
#    edgebwim = cv2.Canny(bwim, 5, 50, None, 3)
##    edge = cv2.
##    plt.imshow(edge)
##    lines = cv2.HoughLines(edge,5,np.pi/2,500)
##    for line in lines:
##        for rho,theta in line:
##            a = np.cos(theta)
##            b = np.sin(theta)
##            x0 = a*rho
##            y0 = b*rho
##            x1 = int(x0 + 1000*(-b))
##            y1 = int(y0 + 1000*(a))
##            x2 = int(x0 - 1000*(-b))
##            y2 = int(y0 - 1000*(a))
##        plt.plot(np.array([x1,x2]),np.array([y1,y2]),'r')
#      
#        
#    fig, axs = plt.subplots(1,3, figsize=(19.2, 4.8))
#    axs[0].imshow(blur_im, cmap="gray")
#    axs[1].imshow(edge, cmap="gray")
#    axs[2].imshow(np.dstack((bwim, bwim-edgebwim, bwim-edgebwim)), cmap="gray")
#    lines = cv2.HoughLinesP(edge,1,np.pi/2,5, 2, 20)
#    linesalt = cv2.HoughLinesP(edgebwim,1,np.pi/2,5, 2, 20)
#    for line in lines:
#        for x1,y1,x2,y2 in line:
#            axs[0].plot(np.array([x1,x2]),np.array([y1,y2]),'r')
#            axs[1].plot(np.array([x1,x2]),np.array([y1,y2]),'r')
#    for line in linesalt:
#        for x1,y1,x2,y2 in line:
#            axs[0].plot(np.array([x1,x2]),np.array([y1,y2]),'g')
#            axs[1].plot(np.array([x1,x2]),np.array([y1,y2]),'g')
#    fig.tight_layout()
#    
#    hor_lines = []
#    ver_lines = []
#    
#    for line in lines:
#        x1,y1,x2,y2 = line[0]
#        if x1==x2 and y1!=y2:
#            ver_lines.append(line[0])
#        elif x1!=x2 and y1==y2:
#            hor_lines.append(line[0])
#        else:
#            print('weird')
#            print(line)
#    
#    x_coords = np.array([line[0] for line in ver_lines])            
#    y_coords = np.array([line[1] for line in hor_lines])            
#    
#    #%% Figure
#    
#    def peaks_of_signal(_signal, _label, _ax, distance=100, color='k'):
#        pks,_ = find_peaks(_signal, distance=distance)
#        _ax.plot(_signal, label = _label, color=color)
#        _ax.plot(pks, _signal[pks],'x', label=None, color=color)
#        return pks, color
#        
#    colormap = plt.cm.gist_ncar
#    colours = [colormap(i) for i in np.linspace(0, 0.9, 5)]
#    
#    fig, axs = plt.subplots(2,4, figsize=(19.2, 9.6))
#    gs = axs[1,2].get_gridspec()
#    # remove the underlying axes
#    for ax in axs[:, 2:].flatten():
#        ax.remove()
#    axbig = fig.add_subplot(gs[:, 2:])
#
#    axs = axs.flatten()
#    
#    Pks = collections.namedtuple('pks', 'data color')
#    
#    # x coordinate
#    
#    x_pks = {}
#    
#    # look at lines density with hough transform
#    data_label = "density"
#    x_nonsmooth = np.zeros(blur_im.shape[1])
#    x_nonsmooth[x_coords] = 1
#    x_smooth = cv2.GaussianBlur(255*x_nonsmooth , (101,101),0 ).flatten()
#
#    axs[0].plot(255*x_nonsmooth)
#    pks, col = peaks_of_signal(x_smooth, data_label, axs[0], color=colours[0])
#    x_pks[data_label] = Pks(data=pks, color=col)
#    
#    axs[0].legend()
#    
#    
#    # now look at peaks detected onto the image profiles 
#    data_label="min"
#    signal = np.min(blur_im, axis=0)
#    pks, col = peaks_of_signal(signal, data_label, axs[1], color=colours[1])
#    x_pks[data_label] = Pks(data=pks, color=col)
#    
#    data_label="-std"
#    signal = 255-np.std(blur_im,axis=0)
#    pks, col = peaks_of_signal(signal, data_label, axs[1], color=colours[2])
#    x_pks[data_label] = Pks(data=pks, color=col)
#
#    data_label = "min-max"
#    signal = np.min(blur_im, axis=0) - np.max(blur_im, axis=0)
#    pks, col = peaks_of_signal(signal, "min - max", axs[1], color=colours[3])
#    x_pks[data_label] = Pks(data=pks, color=col)
#    
#    data_label="smooth(-std)"
#    signal = cv2.GaussianBlur(255-np.std(blur_im,axis=0), (51, 51), 0).flatten()
#    pks, col = peaks_of_signal(signal, data_label, axs[1], color=colours[4])
#    x_pks[data_label] = Pks(data=pks, color=col)
#
#    axs[1].legend()
#    
#
#    # y coordinate
#    y_pks = {}
#
#    data_label = "density"
#    y_nonsmooth = np.zeros(blur_im.shape[0])
#    y_nonsmooth[y_coords] = 1
#    y_smooth = cv2.GaussianBlur(255*y_nonsmooth , (101,101),0 ).flatten()
#    
#    axs[4].plot(255*y_nonsmooth)
#    pks, col = peaks_of_signal(y_smooth, "lines density", axs[4], color=colours[0])
#    y_pks[data_label] = Pks(data=pks, color=col)
#
#    axs[4].legend()
#    
#    # now look at peaks detected onto the image profiles
#    data_label="min"    
#    signal = np.min(blur_im, axis=1)
#    pks, col = peaks_of_signal(signal, data_label, axs[5], color=colours[1])
#    y_pks[data_label] = Pks(data=pks, color=col)
#
#    data_label="-std"
#    signal = 255-np.std(blur_im,axis=1)
#    pks, col = peaks_of_signal(signal, data_label, axs[5], color=colours[2])
#    y_pks[data_label] = Pks(data=pks, color=col)
#
#    data_label="min-max"
#    signal = np.min(blur_im, axis=1) - np.max(blur_im, axis=1)
#    pks, col = peaks_of_signal(signal, data_label, axs[5], color=colours[3])
#    y_pks[data_label] = Pks(data=pks, color=col)
#
#    data_label="smooth(-std)"
#    signal = cv2.GaussianBlur(255-np.std(blur_im,axis=1), (51, 51), 0).flatten()
#    pks, col = peaks_of_signal(signal, data_label, axs[5], color=colours[4])
#    y_pks[data_label] = Pks(data=pks, color=col)
#
#
#    axs[5].legend()
#    
#    fig.tight_layout()
#    
#    
#    
#    axbig.imshow(blur_im, cmap="gray")
#    
#    for k, x in x_pks.items():
#        if len(x.data) > 0:
#            axbig.plot([x.data, x.data], [0,blur_im.shape[0]], color=x.color, label=k)
#    
#    for k, y in y_pks.items():
#        if len(y.data) > 0:
#            axbig.plot([0,blur_im.shape[1]], [y.data, y.data], color=y.color)
##    axbig.legend()
#    
    
#    peaks
    
    #%% Xcorr method
    
    def make_template(n_pxls=150, rel_width=0.4, blurring=0.1):

        x = np.linspace(-0.5, 0.5, n_pxls)
        y = np.linspace(-0.5, 0.5, n_pxls)
        xx, yy = np.meshgrid(x, y, sparse=False, indexing='ij')

        zz = (1 - np.tanh( (abs(xx)-rel_width)/blurring ))
        zz = zz * (1-np.tanh( (abs(yy)-rel_width)/blurring ))
        zz = (zz/4*255).astype(np.uint8)
        
        return zz
    
    template = make_template(n_pxls=140, blurring=0.1)
    
    res = cv2.matchTemplate(blur_im,template,cv2.TM_CCORR_NORMED)
    
    match = np.uint8(res > 0.95)
#    match = cv2.copyMakeBorder(match, 75, 74, 75, 74, cv2.BORDER_CONSTANT)
    
    labelled_match = label(match) # needed for regionprops
    regions = regionprops(labelled_match)
    regions = [r for r in regions if (r.area > 1) and (r.perimeter > 0)]
    
    fig, axs = plt.subplots(1,2, figsize=(12.8, 4.8))
    axs[0].imshow(blur_im)
    axs[1].imshow(res)
    for region in regions:
        yc, xc = region.centroid
        axs[0].plot(xc + 75, yc+75, 'rx')
        axs[1].plot(xc, yc, 'rx')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    