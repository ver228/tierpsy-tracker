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
from scipy.fftpack import next_fast_len
from matplotlib import cm
from matplotlib import colors
from matplotlib import pyplot as plt
from tierpsy.analysis.split_fov.FOVMultiWellsSplitter import FOVMultiWellsSplitter
from tierpsy.analysis.split_fov.helper import calculate_bgnd_from_masked_fulldata
from tierpsy.analysis.split_fov.helper import parse_camera_serial
from tierpsy.analysis.split_fov.helper import make_square_template, fft_convolve2d
from tierpsy.analysis.split_fov.helper import naive_normalise, simulate_wells_lattice
from scipy.signal import fftconvolve


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




def draw_rects(input_img, x_off, y_off, spacing):
    _img = cv2.cvtColor(input_img.copy(),cv2.COLOR_GRAY2BGR)
    normcol = colors.Normalize(vmin=0, vmax=16)
    wc=0
    for ii in range(4):
        for jj in range(4):
            # colors
            rgba_color = cm.Set1(normcol(wc),bytes=True)
            rgba_color = tuple(map(lambda x : int(x), rgba_color))
            # coordinates
            xoff = int(x_off*_img.shape[0])
            yoff = int(y_off*_img.shape[0])
            well_width = int(spacing*_img.shape[0])
            ymin = yoff + ii*well_width - well_width//2
            ymax = ymin + well_width
            xmin = xoff + jj*well_width - well_width//2
            xmax = xmin + well_width
            # draw
            print(xmin,xmax,ymin,ymax)
            cv2.rectangle(_img,
                          (xmin,ymin),
                          (xmax,ymax),
                          rgba_color[:-1],
                          20
                          )
            wc+=1
    plt.figure()
    plt.imshow(_img)


def find_wells_on_grid(input_image):
    # first downscale for speed
    blur_im = get_blur_im(input_image)
    # pad the image for better fft2 performance:
    rowcol_padding = tuple(next_fast_len(size) - size 
                           for size in blur_im.shape)
    rowcol_split_padding = tuple((pad//2, -(-pad//2)) # -(-x//2) == np.ceil(x/2)
                                 for pad in rowcol_padding)
    img = np.pad(blur_im, rowcol_split_padding, 'edge') # now padded
    img = 1 - naive_normalise(img) # normalised and inverted. This is a float
    
    # define function to minimise
    nwells = 4
    fun_to_minimise = lambda x: np.abs(
            img - simulate_wells_lattice(
                    img.shape,
                    x[0],x[1],x[2],
                    nwells=nwells,
                    template_shape = 'square'
                    )
            ).sum()
    # actual minimisation
    # criterion for bounds choice:
    # 1/2n is if well starts at edge, 1/well if there is another half well!
    # bounds are relative to the size of the image (along the y axis)
    # 1/(nwells+1) spacing allows for half an extra well on both side
    # 1/(nwells-1) spacing allows for cut wells at the edges I guess
    bounds = [(1/(2*nwells), 1/nwells),  # x_offset
              (1/(2*nwells), 1/nwells),  # y_offset
              (1/(nwells+1), 1/(nwells-1))]  # spacing
    result = scipy.optimize.differential_evolution(fun_to_minimise, bounds)
    # extract output parameters for spacing grid
    x_offset, y_offset, spacing = result.x.copy()
    # create coordinates for the wells now
    # convert to pixels
    draw_rects(input_image, x_offset, y_offset, spacing)
    def _to_px(rel):
        return rel * input_image.shape[0]
    x_offset_px = _to_px(x_offset)
    y_offset_px = _to_px(y_offset)
    spacing_px = _to_px(spacing)
    # create list of centres and sizes
    # row and column could now come automatically as x and y are ordered
    # but since odd and even channel need to be treated diferently, 
    # leave it to the specialised function
    xyr = np.array([
            (x, y, spacing_px/2)
            for x in np.arange(x_offset_px,input_image.shape[1],spacing_px)[:nwells]
            for y in np.arange(y_offset_px,input_image.shape[0],spacing_px)[:nwells]
            ])
    # write into dataframe
    wells = pd.DataFrame(data=xyr.astype(int), columns=['x','y','r'])
    # now calculate the rest. Don't need all the cleaning-up 
    for d in ['x','y']:
        wells[d+'_min'] = wells[d] - wells['r']
        wells[d+'_max'] = wells[d] + wells['r']





#%%

which_image = 2
desktop = Path.home() / 'Desktop'

if which_image == 1:
    masked_image_file = str(desktop / 'foo/MaskedVideos/pilotdrugs_run1_20191003_161308.22956807/metadata.hdf5')
    img, camera_serial, px2um = calculate_bgnd_from_masked_fulldata(masked_image_file)
elif which_image == 2:
    fname = str(desktop / 'Data_FOVsplitter/short/RawVideos/firstframes/96wpsquare_upright_150ulagar_l1dispensed_1_20190614_105312_firstframes/96wpsquare_upright_150ulagar_l1dispensed_1_20190614_105312.22594548.png')
    img_ = cv2.imread(str(fname))
    img = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    camera_serial = parse_camera_serial(fname)
    px2um = 13.0

full_img = img.copy()
img = get_blur_im(full_img)
    
    
#%%
x_off = 1/8
y_off = 1/8
sp = 1/4
nwells=4
# convert fractions into integers
x_offset = int(x_off*img.shape[0])
y_offset = int(y_off*img.shape[0])
spacing = int(sp*img.shape[0])


tic = time.time()
# create padded canvas from the beginning
padding = img.shape[0]//2
padding_times_2 = padding*2
padded_shape = tuple(s+padding_times_2 for s in img.shape)
padded_canvas = np.zeros(padded_shape)
if nwells is not None:
    padded_canvas[y_offset+padding:y_offset+padding+nwells*spacing:spacing, 
                  x_offset+padding:x_offset+padding+nwells*spacing:spacing] = 1
else:
    padded_canvas[y_offset+padding:padding+img.shape[0]:spacing, 
                  x_offset+padding:padding+img.shape[1]:spacing] = 1

#
print('alt padding time: {}'.format(time.time()-tic) )

tic = time.time()
padding = img.shape[0]//2
padding_times_2 = padding*2
padded_shape = tuple(s+padding_times_2 for s in img.shape)
if nwells is not None:
    r_wells = range(y_offset+padding,
                    y_offset+padding+nwells*spacing,
                    spacing) 
    c_wells = range(x_offset+padding,
                    x_offset+padding+nwells*spacing,
                    spacing)
else:
    r_wells = range(y_offset+padding,
                    padding+img.shape[0],
                    spacing) 
    c_wells = range(x_offset+padding,
                    padding+img.shape[1],
                    spacing)
                        

tmpl_pos_in_padded_canvas = [(r,c) for r in r_wells for c in c_wells]
print('no initial canvas time: {}'.format(time.time()-tic) )


    # make the template for the wells
tic=time.time()
tmpl = make_square_template(n_pxls=spacing, 
                            rel_width=0.7, 
                            blurring=0.1)
# normalise to 0-1 range
tmpl = tmpl.astype(float)/255
print('template time: {}'.format(time.time()-tic))

#%%
plt.close('all')

case = 'simpleplacementfun'

tic = time.time()
if case == 'statusquo':
    # this is what happens now
    # fft_convolve2d only acts on images same size. 
    # So pad the small template to size of padded canvas
    lp = (padded_canvas.shape[1]-spacing)//2  # left padding
    rp = -((-padded_canvas.shape[1]+spacing)//2)  # right padding (==np.ceil)
    tp = (padded_canvas.shape[0]-spacing)//2  # top padding
    bp = -((-padded_canvas.shape[0]+spacing)//2) # bottom padding
    padded_tmpl = np.pad(tmpl,((tp,bp),(lp,rp)),'constant', constant_values=1)
    
    # convolve
    convolved_padded_canvas = fft_convolve2d(padded_canvas, padded_tmpl)
    cutout_canvas = convolved_padded_canvas[padding:padding+canvas.shape[0],
                                            padding:padding+canvas.shape[1]]
    out = 1-naive_normalise(cutout_canvas)
    
elif case == 'scipysignal':
#    foo = fftconvolve(padded_canvas-padded_canvas.mean(), tmpl-tmpl.mean(), mode='same')
    foo = fftconvolve(padded_canvas, tmpl, mode='same')
    foo = foo[padding:padding+img.shape[0],padding:padding+img.shape[1]]
    out = 1-naive_normalise(foo)
    
elif case == 'simpleplacement':
    bar = np.ones(padded_shape)
    for r,c in tmpl_pos_in_padded_canvas:
        bar[r-spacing//2:r-(-spacing//2), c-spacing//2:c-(-spacing//2)] -= (1-tmpl)
#        plt.imshow(bar)
#        plt.show()
#        plt.pause(0.1)
    bar = bar[padding:padding+img.shape[0],padding:padding+img.shape[1]]
    out = 1-naive_normalise(bar)

elif case == 'simpleplacementfun':
    out = simulate_wells_lattice(img.shape, x_off, y_off, sp, nwells=nwells)
    



plt.figure(case)
plt.imshow(out)
print(case, time.time()-tic)



#%%
#
#
#
#
#
#tic = time.time()
#bounds = [(1/8, 1/4), (1/8,1/4), (0.2, 0.28)] # xoff, yoff, spacing (as fraction of short side)
#x = [1/8, 1/8, 1/4]
#now = simulate_wells_lattice(
#                    img.shape,
#                    x[0],x[1],x[2],
#                    nwells=4,
#                    template_shape = 'square'
#                    )
#print('time elapsed: ', time.time() - tic)
