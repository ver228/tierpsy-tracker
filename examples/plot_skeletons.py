#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:01:20 2018

@author: ajaver
"""

import pandas as pd
import tables
import matplotlib.pylab as plt
import matplotlib.patches as patches

mask_file = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/MaskedVideos/CeNDR_Set1_020617/N2_worms10_food1-10_Set3_Pos4_Ch3_02062017_123419.hdf5'
features_file = mask_file.replace('MaskedVideos','Results').replace('.hdf5', '_featuresN.hdf5')

with pd.HDFStore(features_file, 'r') as fid:
    #all the worm coordinates and how the skeletons matrix related with a given frame is here
    trajectories_data = fid['/trajectories_data']

    #data in coordinates is in micrometers so we need the conversion in order to plot
    microns_per_pixel = fid.get_node('/trajectories_data')._v_attrs['microns_per_pixel']

#select data from a given frame
frame_number = 100

#read image. If you want the image without the masked backgroudn use "/full_data"
img_field = '/mask'

traj_g = trajectories_data.groupby('frame_number')
frame_data = traj_g.get_group(frame_number)

#select the rows that have the skeletons in this frame
#worms that where not succesfully skeletonized will have a -1 here
skel_id = frame_data['skeleton_id'].values
skel_id = skel_id[skel_id>=0]

#get video frame
with tables.File(mask_file, 'r') as fid:
    img = fid.get_node(img_field)[frame_number]

#read worms skeletons
with tables.File(features_file, 'r') as fid:
    skel = fid.get_node('/coordinates/skeletons')[skel_id, :, :]/microns_per_pixel
    cnt1 = fid.get_node('/coordinates/ventral_contours')[skel_id, :, :]/microns_per_pixel
    cnt2 = fid.get_node('/coordinates/dorsal_contours')[skel_id, :, :]/microns_per_pixel


#plot frame
plt.figure(figsize=(20,20))
plt.imshow(img, interpolation='none', cmap='gray')

#add all the objects identified
for _, row in frame_data.iterrows():
    lw = 2 #linewidth for the plot
    x_i = row['coord_x'] - row['roi_size']/2
    y_i = row['coord_y'] - row['roi_size']/2
    
    r_p = patches.Rectangle((x_i, y_i), 
                            row['roi_size'] - lw, 
                            row['roi_size'] - lw,
                            fill = False,
                            color='g',
                            lw = lw
                            )
    plt.gca().add_patch(r_p)


#add all the worms  that well succesfully skeletonized
for (ss, cc1, cc2) in zip(skel, cnt1, cnt2):
    plt.plot(ss[:, 0], ss[:, 1], 'r')
    plt.plot(cc1[:, 0], cc1[:, 1], 'tomato')
    plt.plot(cc2[:, 0], cc2[:, 1], color='salmon')
plt.show()