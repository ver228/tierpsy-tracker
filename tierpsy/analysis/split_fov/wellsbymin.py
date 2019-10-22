#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:13:57 2019

@author: lferiani
"""

import cv2
import pdb
import time
import pandas as pd
import numpy as np
import scipy.optimize
from pathlib import Path
from numpy.fft import fft2, ifft2, fftshift

from matplotlib import pyplot as plt
from tierpsy.analysis.split_fov.FOVMultiWellsSplitter import FOVMultiWellsSplitter
from tierpsy.analysis.split_fov.helper import read_data_from_masked
from tierpsy.analysis.split_fov.helper import parse_camera_serial
from tierpsy.analysis.split_fov.helper import make_square_template



def get_blur_im(img):
    """downscale and blur the image"""
    # preprocess image
    dwnscl_factor = 4; # Hydra images' shape is divisible by 4
    blr_sigma = 17; # blur the image a bit, seems to work better
    new_shape = (img.shape[1]//dwnscl_factor, # as x,y, not row,columns
                 img.shape[0]//dwnscl_factor)
    
    try:        
        dwn_gray_im = cv2.resize(img, new_shape)
    except:
        pdb.set_trace()
    # apply blurring
    blur_im = cv2.GaussianBlur(dwn_gray_im, (blr_sigma,blr_sigma),0)
    # normalise between 0 and 255
    blur_im = cv2.normalize(blur_im, None, alpha=0, beta=255, 
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return blur_im



def fft_convolve2d(x,y):
    """ 2D convolution, using FFT"""
    fr = fft2(x)
    fr2 = fft2(y)
    m,n = fr.shape
    cc = np.real(ifft2(fr*fr2))
    cc = fftshift(cc)
    return cc

#%%
def cmfov(img_shape, x_off, y_off, sp, nwells): 
    """create mock fov"""
    # convert fractions into integers
    x_offset = int(x_off*img_shape[0])
    y_offset = int(y_off*img_shape[0])
    spacing = int(sp*img_shape[0])
    
    canvas = np.zeros(img_shape)    
    canvas[y_offset:y_offset+nwells*spacing:spacing, 
           x_offset:x_offset+nwells*spacing:spacing] = 1

    padding = canvas.shape[0]//4
    padded_canvas = np.pad(canvas, padding, 'edge')
    
    tmpl = make_square_template(n_pxls=spacing, 
                                rel_width=0.7, 
                                blurring=0.1)
#    tmpl[:,:5] = 255
#    tmpl[:,-5:] = 255
#    tmpl[:5,:] = 255
#    tmpl[-5:,:] = 255
    tmpl = tmpl.astype(float)/255
    tmpl = 1-tmpl
    lp = (padded_canvas.shape[1]-spacing)//2
    rp = lp+1 if lp*2+spacing<padded_canvas.shape[1] else lp
    tp = (padded_canvas.shape[0]-spacing)//2
    bp = tp+1 if tp*2+spacing<padded_canvas.shape[0] else tp
    
    padded_tmpl = np.pad(tmpl,((tp,bp),(lp,rp)),'constant')

    padded_canvas = fft_convolve2d(padded_canvas, padded_tmpl)
    padded_canvas = 1-padded_canvas
    cutout_canvas = padded_canvas[padding:padding+canvas.shape[0],
                                  padding:padding+canvas.shape[1]]
    
    return 1-cutout_canvas

#%%

which_image = 1
which_method = 'diffevo' #fmin
desktop = Path.home() / 'Desktop'

if which_image == 1:
    masked_image_file = str(desktop / 'foo/MaskedVideos/pilotdrugs_run1_20191003_161308.22956807/metadata.hdf5')
    img, camera_serial, px2um = read_data_from_masked(masked_image_file)
elif which_image == 2:
    fname = str(desktop / 'Data_FOVsplitter/short/RawVideos/firstframes/96wpsquare_upright_150ulagar_l1dispensed_1_20190614_105312_firstframes/96wpsquare_upright_150ulagar_l1dispensed_1_20190614_105312.22594548.png')
    img_ = cv2.imread(str(fname))
    img = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    camera_serial = parse_camera_serial(fname)
    px2um = 13.0

    
#%%
img1 = get_blur_im(img)
if which_image==1:
    img2 = np.pad(img1,((4,5),(4,5)),'edge')
elif which_image==2:
    img2 = np.pad(img1,((4,5),(9,9)),'edge')


#%% minimisation
    
# preprocess
_img = img2;
sz = _img.shape
img_flt = 1-(_img - _img.min())/(_img.max()-_img.min())

# fit
tic = time.time()
nw = 4

fun_to_minimise = lambda x: np.abs(img_flt - cmfov(sz, x[0],x[1],x[2], nw)).flatten().sum()

if which_method=='fmin':
    x0=[1/5, 1/6, 1/5]
    xopt = scipy.optimize.fmin(func=fun_to_minimise,
                               x0=x0,
                               ftol=0.000001)
    print(time.time()-tic)
    print(xopt)
    
elif which_method=='diffevo':
    bounds = [(1/8, 1/4), (1/8,1/4), (1/5, 1/3)]
    result = scipy.optimize.differential_evolution(fun_to_minimise, bounds)
    xopt = result.x

#%% plots

fitout = cmfov(sz, *xopt, nw)
mask = fitout<0.2
mask = fitout>0.85

fig,axs = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
axs[0].imshow(img_flt, cmap='gray')
if which_method=='fmin':
    axs[0].imshow(cmfov(sz, *x0, nw)<0.2, cmap='plasma', alpha=0.5)
axs[0].axis('off')

axs[1].imshow(img_flt, cmap='gray')
axs[1].imshow(mask, cmap='plasma', alpha=0.5)
axs[1].axis('off')

fig.tight_layout()
fig.savefig('/Users/lferiani/OneDrive - Imperial College London/Slides/20191021_group_meeting/grid_fitting_'+which_method+'.pdf')


#%%

def make_template(n_pxls=150):
    import numpy as np
    """Function that creates a template that approximates a square well"""
    n_pxls = int(np.round(n_pxls))
    theta=np.pi/4
    x = np.linspace(-0.5, 0.5, n_pxls)
    y = np.linspace(-0.5, 0.5, n_pxls)
    xx, yy = np.meshgrid(x, y, sparse=False, indexing='ij')

    zz = xx**4 + yy**4 - (xx*yy)**2
    zz *= -1
    zz -= zz.min()
    zz /= zz.max()
    return zz


plt.close("all")
plt.imshow(make_template())