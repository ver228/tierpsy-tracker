#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:00:17 2017

@author: ajaver
"""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import tables
import cv2
import multiprocessing as mp
import glob

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.measure import regionprops
from scipy import ndimage as ndi
from tierpsy.analysis.ske_create.getSkeletonsTables import binaryMask2Contour, generateMoviesROI
from tierpsy.helper.misc import TABLE_FILTERS, get_base_name


DEBUG = False

def get_pharynx_orient(worm_img, min_blob_area):#, min_dist_btw_peaks=5):
    #%%
    
    blur = cv2.GaussianBlur(worm_img,(5,5),0) 
    
    #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    th, worm_mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    worm_cnt, cnt_area = binaryMask2Contour(worm_mask, min_blob_area=min_blob_area)
    
    worm_mask = np.zeros_like(worm_mask)
    cv2.drawContours(worm_mask, [worm_cnt.astype(np.int32)], 0, 1, -1)
    
    local_maxi = peak_local_max(blur,
                                indices=True, 
                                labels=worm_mask)
    
    #%%
    markers = np.zeros_like(worm_mask, dtype=np.uint8)
    kernel = np.ones((3,3),np.uint8)
    for x in local_maxi:
        markers[x[0], x[1]] = 1
    markers = cv2.dilate(markers,kernel,iterations = 1)
    markers = ndi.label(markers)[0]
    #strel = ndi.generate_binary_structure(3, 3)
    #markers = binary_dilation(markers, iterations=3)
    
    labels = watershed(-blur, markers, mask=worm_mask)
    props = regionprops(labels)
    
    #sort coordinates by area (the larger area is the head)
    props = sorted(props, key=lambda x: x.area, reverse=True)
    peaks_dict = {labels[x[0], x[1]]:x[::-1] for x in local_maxi}
    peaks_coords = np.array([peaks_dict[x.label] for x in props])
    
    if DEBUG:
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(markers, cmap='gray', interpolation='none')
        
        plt.subplot(1,3,2)
        plt.imshow(labels)
        
        plt.subplot(1,3,3)
        plt.imshow(blur, cmap='gray', interpolation='none')
        
        for x,y in peaks_coords:
            plt.plot(x,y , 'or')
            
    if len(props) != 2:
        return np.full((2,2), np.nan) #invalid points return empty
        
    #%%
    return peaks_coords

def generateIndividualROIs(ROIs_generator):
    for worms_in_frame in ROIs_generator:
        for worm_index, (worm_img, roi_corner) in worms_in_frame.items():
            row_data = trajectories_data.loc[worm_index]
            min_blob_area=row_data['area'] / 2
            skeleton_id = int(row_data['skeleton_id'])
            
            yield (skeleton_id, worm_img, min_blob_area, roi_corner)
    

def _process_row(input_data):
    skeleton_id, worm_img, min_blob_area, roi_corner = input_data
    peaks_coords = get_pharynx_orient(worm_img, min_blob_area)
    peaks_coords += roi_corner
    output = skeleton_id, peaks_coords
    return output

def process_batch(batch_input):
    output = list(p.map(_process_row, batch_input))
    skeletons_id, peaks_coords = map(np.array, zip(*output))
    
    skeletons[skeletons_id, :, :] = peaks_coords
    has_skeleton[skeletons_id] = True
    
def init_data(ske_file_id, tot_rows):
    #create and reference all the arrays
    field = 'skeleton'
    dims = (tot_rows,2,2)
    
    if '/' + field in ske_file_id:
        ske_file_id.remove_node('/', field)
        
    skeletons = ske_file_id.create_carray('/', 
                              field, 
                              tables.Float32Atom(dflt=np.nan), 
                              dims, 
                              filters=TABLE_FILTERS)
    traj_dat = ske_file_id.get_node('/trajectories_data')
    has_skeleton = traj_dat.cols.has_skeleton
    has_skeleton[:] = np.zeros_like(has_skeleton) #delete previous
    
    return skeletons, has_skeleton

if __name__ == '__main__':
    filenames = glob.glob("/data2/shared/data/twoColour/MaskedVideos/*/*/*52.1g_X1.hdf5")

    for masked_file in filenames:
        skeletons_file = masked_file.replace('MaskedVideos', 'Results').replace('.hdf5', '_skeletons.hdf5')
        
        base_name = get_base_name(masked_file)
        progress_prefix =  base_name + ' Calculating skeletons.'
        
        with pd.HDFStore(skeletons_file, 'r') as ske_file_id:
            trajectories_data = ske_file_id['/trajectories_data']
        
        n_batch = 8 #mp.cpu_count()
        
        frame_generator = generateMoviesROI(masked_file, 
                                           trajectories_data, 
                                           progress_prefix=progress_prefix)
        ROIs_generator = generateIndividualROIs(frame_generator)
        
        tot = 0
        with tables.File(skeletons_file, "r+") as ske_file_id: 
            tot_rows = len(trajectories_data)
            skeletons, has_skeleton = init_data(ske_file_id, tot_rows)
            
            p = mp.Pool(processes=n_batch)
            batch_input = [] 
            for input_data in ROIs_generator:
                if DEBUG:
                    #run a no parallelize version of the code and do not save the data into the hdf5
                    _process_row(input_data)
                    tot += 1
                    
                else:
                    batch_input.append(input_data)
                    if len(batch_input) >= n_batch:
                        output = list(p.map(_process_row, batch_input))
                        skeletons_id, peaks_coords = zip(*output)
                        skeletons_id, peaks_coords = map(np.array, zip(*output))
                        
                        skeletons[skeletons_id, :, :] = peaks_coords
                        for ind in skeletons_id:
                            has_skeleton[ind] = True
                        
                        
                        batch_input = [] 
                    tot += 1
