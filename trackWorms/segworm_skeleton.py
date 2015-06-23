# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:46:20 2015

@author: ajaver
"""

import matplotlib.pylab as plt
import cv2
import numpy as np
import time
import os
#from scipy.interpolate import interp1d

import sys
sys.path.append('../../movement_validation') 
from movement_validation.pre_features import WormParsing
from movement_validation import user_config, BasicWorm, NormalizedWorm

sys.path.append('./segWormPython/') #add path for the segworm data 
from segWormPython.main_segworm import contour2Skeleton

#used to "normalized" worm
from segWormPython.cythonFiles.curvspace import curvspace

if __name__ == '__main__':
    base_path = os.path.abspath(user_config.EXAMPLE_DATA_PATH)
    schafer_bw_file_path = os.path.join(base_path,
                                     "example_contour_and_skeleton_info.mat")  
    bw = BasicWorm.from_schafer_file_factory(schafer_bw_file_path)
    #%%
    tic = time.time()
    tot_worms = len(bw._h_dorsal_contour)
    skeletons = []
    cnt_widths = []
    
    pixelSize = -1
    for ii in range(tot_worms):
        
        if bw._h_dorsal_contour[ii] is None:
            skeletons.append(None)
            cnt_widths.append(None)
            continue
        
        #the segmentation code needs a circular contour with a counter-clockwise orientation
        contour = np.hstack([bw._h_dorsal_contour[ii][:, -1:1:-1], bw._h_ventral_contour[ii]])
        
        #this code, as well as the numpy library uses numpy C contingous array (last dimension change fast)
        #the structures used in openworm are numpy fortran arrays (I assume it is a heritage from MATLAB)
        #we have to change the array to nuympy C contigous arrays before.
        contour = np.ascontiguousarray(contour.T)
        
        #This code assumes pixels. It tries to "clean the skeleton" by interpolating missing
        #pixels between points. If there are a lot of missing pixels (non-contingous integer numbers),
        #the resulting skeleton can be very large and even overflow the buffer. Try to reduce rescale the
        #contour to avoid this problem. I am assuming the contour is in micrometers and the pixel size is 3um
        #additionally, the contour must be rounded or some assertions might not be passed
        
        if pixelSize == -1:
            #let's assume the pixel size is the median distance between between contour segments
            pixelSize = np.sqrt(np.nanmedian(np.diff(contour[:,0])**2+np.diff(contour[:,1])))
        
        contour = np.round(contour/pixelSize)
        
        skeleton, cnt_side1, cnt_side2, cnt_width, err_msg = contour2Skeleton(contour)
        
        #comeback to the correct size, and to a fortran array
        skeleton = np.asfortranarray(skeleton.T*pixelSize)
        cnt_width = cnt_width*pixelSize
        
        
        #typically the orientation of the worm depends on the previous frame. Otherwise it 
        #try to find it from the contour angles, and it is not very precise. For the moment
        # I will just patch it to match the contour orientation. If this code becomes useful, I'll fix it
        
        if np.sqrt(np.sum((skeleton[:, 0]-bw._h_dorsal_contour[ii][:,0])**2)) > 3:
            skeleton = skeleton[:, ::-1]
            cnt_width = cnt_width[::-1]
        
        
        skeletons.append(skeleton)
        cnt_widths.append(cnt_width)
        
    print('python-segworm code: %2.2f' % (time.time() - tic))

#%%

tic = time.time()
parsing = WormParsing()
w, ske = parsing.compute_skeleton_and_widths(bw._h_dorsal_contour, bw._h_ventral_contour)
print('current openworm code: %2.2f' % (time.time() - tic))

#%%


tic = time.time()

all_norm_cnt = []
for kk, contour_tuple in enumerate(zip(bw._h_dorsal_contour, bw._h_ventral_contour)):
    norm_cnt = []
    for contour in contour_tuple:
        if contour is None:
            norm_cnt.append(None)
        else:
            dum = np.ascontiguousarray((contour.T))
            dum,_ = curvspace(dum, 49)
            dum = np.asfortranarray(dum.T)
            norm_cnt.append(dum)
    all_norm_cnt.append(norm_cnt)
norm_dorsal, norm_ventral = list(zip(*all_norm_cnt))

#compute skeleton using openworm code
w_norm, ske_norm = parsing.compute_skeleton_and_widths(norm_dorsal, norm_ventral)

print('normalized openworm code: %2.2f' % (time.time() - tic))

#%%
schafer_nw_file_path = os.path.join(base_path, "example_video_norm_worm.mat")
nw = NormalizedWorm.from_schafer_file_factory(schafer_nw_file_path)
ind = 4400

plt.figure()
plt.plot(skeletons[ind][1,:], skeletons[ind][0,:], 'xb', label = 'Avelino')
plt.plot(ske[ind][1,:], ske[ind][0,:], 'r',  label = 'OpenWorm')
plt.plot(ske_norm[ind][1,:], ske_norm[ind][0,:], 'og', label = 'OpenWorm Norm')
plt.plot(bw._h_dorsal_contour[ind][1,:], bw._h_dorsal_contour[ind][0,:], 'k')
plt.plot(bw._h_ventral_contour[ind][1,:], bw._h_ventral_contour[ind][0,:], 'k')

plt.plot(bw._h_ventral_contour[ind][1,0], bw._h_ventral_contour[ind][0,0], 'sk')

plt.plot(skeletons[ind][1,0], skeletons[ind][0,0], 'sk')

plt.legend(loc= 4)
plt.axis('equal')
plt.savefig('Skeletons.png')

plt.figure()
plt.plot(np.linspace(0, 1, cnt_widths[ind].size), cnt_widths[ind], 'xb', label = 'Avelino')
plt.plot(np.linspace(0, 1, w[ind].size), w[ind], 'r', label = 'OpenWorm')
plt.plot(np.linspace(0, 1, w_norm[ind].size), w_norm[ind], 'og', label = 'OpenWorm Norm')
plt.plot(np.linspace(0, 1, nw.widths.shape[0]), nw.widths[:,ind], 'k', label = 'Shafer Norm')
plt.legend(loc= 8)
plt.ylabel('worm width')
plt.xlabel('normalized worm length')
plt.savefig('Widths.png')
