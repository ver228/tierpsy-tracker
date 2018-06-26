#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 08:41:01 2018

@author: lferiani
"""

import h5py
import cv2
import os

tmp_dir = os.path.expanduser('~/pippo/pluto/')
if os.path.isdir(tmp_dir):
    os.system("rm -rf "+tmp_dir)
# if

os.mkdir(tmp_dir)

hdf5_filename = ('/home/lferiani@cscdom.csc.mrc.ac.uk/Data/'
                 'codec_testing_hqfast_10s_20180612_123152/MaskedVideos/metadata.hdf5')


hf = h5py.File(hdf5_filename,'r')

fs = hf['mask']
fs.shape

fc = 0
for f in fs:
    savename = "%s/%.4d.tiff" % (tmp_dir,fc)
    cv2.imwrite(savename,f)
    fc += 1
# for

#%%
ffmpeg_cmdlist = ["ffmpeg -framerate 25 -pattern_type glob -i",
                  "'"+tmp_dir+"*.tiff'",
                  "-c:v libx264 -pix_fmt yuv420p",
                  hdf5_filename.replace('.hdf5','.mp4')]
ffmpeg_cmd = ' '.join(ffmpeg_cmdlist)
os.system(ffmpeg_cmd)
#%%