#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:50:47 2019

@author: lferiani
"""

#%% import statements
import numpy as np
import pandas as pd
from numpy.fft import fft2, ifft2, fftshift

#%% constants

WELLS_ATTRIBUTES = ['x','y','r','row','col',
                    'x_min','x_max','y_min','y_max',
                    'well_name', 'is_good_well']

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
               "22956833":'Ch2',
               "22956819":'Ch3',
               "22956827":'Ch4',
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


# this can't be a nice and simple dictionary because people may want to use
# this info in the other direction

CAM2CH_list = [('22956818', 'Ch1', 'Hydra01'), # Hydra01
               ('22956816', 'Ch2', 'Hydra01'),
               ('22956813', 'Ch3', 'Hydra01'),
               ('22956805', 'Ch4', 'Hydra01'),
               ('22956807', 'Ch5', 'Hydra01'),
               ('22956832', 'Ch6', 'Hydra01'),
               ('22956839', 'Ch1', 'Hydra02'), # Hydra02
               ('22956837', 'Ch2', 'Hydra02'),
               ('22956836', 'Ch3', 'Hydra02'),
               ('22956829', 'Ch4', 'Hydra02'),
               ('22956822', 'Ch5', 'Hydra02'),
               ('22956806', 'Ch6', 'Hydra02'),
               ('22956814', 'Ch1', 'Hydra03'), # Hydra03
               ('22956833', 'Ch2', 'Hydra03'),
               ('22956819', 'Ch3', 'Hydra03'),
               ('22956827', 'Ch4', 'Hydra03'),
               ('22956823', 'Ch5', 'Hydra03'),
               ('22956840', 'Ch6', 'Hydra03'),
               ('22956812', 'Ch1', 'Hydra04'), # Hydra04
               ('22956834', 'Ch2', 'Hydra04'),
               ('22956817', 'Ch3', 'Hydra04'),
               ('22956811', 'Ch4', 'Hydra04'),
               ('22956831', 'Ch5', 'Hydra04'),
               ('22956809', 'Ch6', 'Hydra04'),
               ('22594559', 'Ch1', 'Hydra05'), # Hydra05
               ('22594547', 'Ch2', 'Hydra05'),
               ('22594546', 'Ch3', 'Hydra05'),
               ('22436248', 'Ch4', 'Hydra05'),
               ('22594549', 'Ch5', 'Hydra05'),
               ('22594548', 'Ch6', 'Hydra05')]

CAM2CH_df = pd.DataFrame(CAM2CH_list,
                         columns=['camera_serial', 'channel', 'rig'])


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

#%% functions


def get_mwp_map(total_n_wells, whichsideup):
    """
    Given a total number of wells, and whether the multiwell plate
    is upright or upside-down, returns a dataframe with the correct
    channel/row/column -> well_name mapping
    (this works on the Hydra imaging systems - by LoopBio Gmbh - used in Andre
    Brown's lab)
    """
    if total_n_wells==48 and whichsideup=='upright':
        return UPRIGHT_48WP_669999
    elif total_n_wells==96 and whichsideup=='upright':
        return UPRIGHT_96WP
    else:
        raise ValueError('This case has not been coded yet. ' + \
                         'Please contact the devs or open a feature request on GitHub.')


def serial2rigchannel(camera_serial):
    """
    Takes camera serial number, returns a (rig, channel) tuple
    """
    out = CAM2CH_df[CAM2CH_df['camera_serial']==camera_serial]
    if len(out) == 0:
        raise ValueError('{} unknown as camera serial string'.format(camera_serial))
    elif len(out) == 1:
        return tuple(out[['rig','channel']].values[0])
    else:
        raise Exception('Multiple hits for {}. split_fov/helper.py corrupted?'.format(camera_serial))


def serial2channel(camera_serial):
    """
    Takes camera serial number, returns the channel
    """
    return serial2rigchannel(camera_serial)[1]



def parse_camera_serial(filename):
    import re
    regex = r"(?<=20\d{6}\_\d{6}\.)\d{8}"
    camera_serial = re.findall(regex, str(filename).lower())[0]
    return camera_serial


def calculate_bgnd_from_masked_fulldata(masked_image_file):
    """
    - Opens the masked_image_file hdf5 file, reads the /full_data node and
      creates a "background" by taking the maximum value of each pixel over time.
    - Parses the file name to find a camera serial number
    - reads the pixel/um ratio from the masked_image_file
    """
    import numpy as np
    from tierpsy.helper.params import read_unit_conversions

    # read attributes of masked_image_file
    _, (microns_per_pixel, xy_units) , is_light_background = read_unit_conversions(masked_image_file)
    # get "background" and px2um
    with pd.HDFStore(masked_image_file, 'r') as fid:
        assert is_light_background, \
        'MultiWell recognition is only available for brightfield at the moment'
        img = np.max(fid.get_node('/full_data'), axis=0)

    camera_serial = parse_camera_serial(masked_image_file)

    return img, camera_serial, microns_per_pixel


def make_square_template(n_pxls=150, rel_width=0.8, blurring=0.1, dtype_out='float'):
    import numpy as np
    """Function that creates a template that approximates a square well"""
    n_pxls = int(np.round(n_pxls))
    x = np.linspace(-0.5, 0.5, n_pxls)
    y = np.linspace(-0.5, 0.5, n_pxls)
    xx, yy = np.meshgrid(x, y, sparse=False, indexing='ij')

    # inspired by Mark Shattuck's function to make a colloid's template
    zz = (1 - np.tanh( (abs(xx)-rel_width/2)/blurring ))
    zz = zz * (1-np.tanh( (abs(yy)-rel_width/2)/blurring ))
    zz = zz/4

    # add bright border
    edge = int(0.05 * n_pxls)
    zz[:edge,:] = 1
    zz[-edge:,:] = 1
    zz[:,:edge] = 1
    zz[:,-edge:] = 1

    if dtype_out == 'uint8':
        zz *= 255
        zz = zz.astype(np.uint8)
    elif dtype_out == 'float':
        pass
    else:
        raise ValueError("Only 'float' and 'uint8' are valid dtypes for this")


    return zz


# def was_fov_split(timeseries_data):
#     """
#     Check if the FOV was split, looking at timeseries_data
#     """
#     if 'well_name' not in timeseries_data.columns:
#         # for some weird reason, save_feats_stats is being called on an old
#         # featuresN file without calling save_timeseries_feats_table first
#         is_fov_split = False
#     else:
#         # timeseries_data has been updated and now has a well_name column
#         if len(set(timeseries_data['well_name']) - set(['n/a'])) > 0:
#             is_fov_split = True
# #            print('have to split fov by well')
#         else:
#             assert all(timeseries_data['well_name']=='n/a'), \
#                 'Something is wrong with well naming - go check save_feats_stats'
#             is_fov_split = False
#     return is_fov_split

def was_fov_split(fname):
    with pd.HDFStore(fname, 'r') as fid:
        is_fov_tosplit = ('/fov_wells' in fid)
    return is_fov_tosplit


def naive_normalise(img):
    m = img.min()
    M = img.max()
    return (img - m) / (M-m)


def fft_convolve2d(x,y):
    """ 2D convolution, using FFT"""
    fr = fft2(x)
    fr2 = fft2(y)
    cc = np.real(ifft2(fr*fr2))
    cc = fftshift(cc)
    return cc


def simulate_wells_lattice(img_shape, x_off, y_off, sp, nwells=None, template_shape='square'):
    """
    Create mock fov by placing well templates onto a square lattice
    Very simply uses the input parameters and range to define where the wells
    will go, and then places the template in a padded canvas.
    The canvas is then cut to be of img_shape again.
    This simple approach works because the template is created to be exactly
    spacing large, so templates do not overlap
    """

    # convert fractions into integers
    x_offset = int(x_off*img_shape[0])
    y_offset = int(y_off*img_shape[0])
    spacing = int(sp*img_shape[0])

    # create a padded empty canvas
    padding = img_shape[0]//2
    padding_times_2 = padding*2
    padded_shape = tuple(s+padding_times_2 for s in img_shape)
    padded_canvas = np.zeros(padded_shape)

    # determine where the wells wil go in the padded canvas
    if nwells is not None:
        r_wells = range(y_offset+padding,
                        y_offset+padding+nwells*spacing,
                        spacing)
        c_wells = range(x_offset+padding,
                        x_offset+padding+nwells*spacing,
                        spacing)
    else:
        r_wells = range(y_offset+padding,
                        padding+img_shape[0],
                        spacing)
        c_wells = range(x_offset+padding,
                        padding+img_shape[1],
                        spacing)
    tmpl_pos_in_padded_canvas = [(r,c) for r in r_wells for c in c_wells]

    # make the template for the wells
    tmpl = make_square_template(n_pxls=spacing,
                                rel_width=0.7,
                                blurring=0.1,
                                dtype_out='float')
    # invert
    tmpl = 1-tmpl

    # place wells onto canvas
    ts = tmpl.shape[0]
    for r,c in tmpl_pos_in_padded_canvas:
        try:
            padded_canvas[r-ts//2:r-(-ts//2),
                          c-ts//2:c-(-ts//2)] += tmpl
        except Exception as e:
            print(str(e))
            import pdb
            pdb.set_trace()

    cutout_canvas = padded_canvas[padding:padding+img_shape[0],
                                  padding:padding+img_shape[1]]
    cutout_canvas = naive_normalise(cutout_canvas)

    return cutout_canvas


def get_well_color(is_good_well, forCV=False):
    colors = {'undefined': (255, 127, 0),
              'good_well': (77, 220, 74),
              'bad_well': (255, 0, 0)}
    if np.isnan(is_good_well) or is_good_well==-1:
        color = colors['undefined']
    elif is_good_well == True or is_good_well==1:
        color = colors['good_well']
    elif is_good_well == False or is_good_well==0:
        color = colors['bad_well']
    else:
        print('is_good_well not NaN, True, False, -1, 1, 0. Debugging:')
        import pdb; pdb.set_trace()
    if not forCV:
        color = tuple(c/255.0 for c in color)
    return color


if __name__ == '__main__':

    # test that camera serials return the correct channel
    serials_list = [line[0] for line in CAM2CH_list]
#    serials_list.append('22594540') # this raise an exception as it does not exist
    for serial in serials_list:
        print('{} -> {}'.format(serial, serial2channel(serial)))
    # that works as intended!

    # let's now check that the camera name is parsed correctly I guess
    from pathlib import Path
    src_dir = Path('/Users/lferiani/Desktop/Data_FOVsplitter/evgeny/MaskedVideos/20190808')
    masked_fnames = src_dir.rglob('*.hdf5')
    for fname in masked_fnames:
        camera_serial = parse_camera_serial(fname)
        print(fname)
        print(camera_serial)
        print(serial2channel(camera_serial))
        print(' ')
    # this too works perfectly... but I saw wrong data was written in the masked videos
    # so have to check what went wrong there




