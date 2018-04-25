#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 15:56:44 2018

@author: ajaver
"""
import cv2
import tables


fname = '/home/ajaver@cscdom.csc.mrc.ac.uk/Downloads/MaskedVideos/11B_060.hdf5'

with tables.File(fname, 'r') as fid:
    masks = fid.get_node('/mask')
    for ii, img in enumerate(masks):
        cv2.imwrite('{:02}.png'.format(ii), img)


