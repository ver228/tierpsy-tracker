# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 16:43:32 2015

@author: ajaver
"""
import h5py
import time
import subprocess as sp
import matplotlib.pylab as plt

masked_image_file = '/Volumes/behavgenom$/GeckoVideo/Compressed/20150228/Capture_Ch1_28022015_171254.hdf5'
save_video_file = '/Volumes/behavgenom$/GeckoVideo/Compressed/20150228/Capture_Ch1_28022015_171254.avi'

#root_dir = '/Users/ajaver/Documents/Test_Andre/'
#base_name = 'Capture_Ch1_26022015_161813'

# ffmpeg -i
# /Volumes/Mrc-pc/Full_Resolution/Capture_Ch1_26022015_161813.mjpg -vcodec
# mjpeg -qscale:v 0 -vf, scale=1024:1024
# Capture_Ch1_26022015_161813_full.avi

#video_file = '/Volumes/Mrc-pc/Full_Resolution/Capture_Ch1_26022015_161813.mjpg'
#masked_image_file = root_dir + base_name + '.hdf5'
#save_video_file = root_dir + base_name + '.avi'
#
##'-vf', 'scale=512:512'
command = ['ffmpeg',
           '-y',  # (optional) overwrite output file if it exists
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-s', '2048x2048',  # size of one frame
           '-pix_fmt', 'gray',
           '-r', '25',  # frames per second
           '-i', '-',  # The imput comes from a pipe
           '-an',  # Tells FFMPEG not to expect any audio
           '-vcodec', 'mjpeg',
           '-vf', 'scale=512:512',
           '-threads', '0',
           '-qscale:v', '0',
           save_video_file]

mask_fid = h5py.File(masked_image_file, "r")
I_worms = mask_fid["/mask"]

pipe = sp.Popen(command, stdin=sp.PIPE)

tic = time.time()
for frame_number in range(0, I_worms.shape[0], 25):
    #    if frame_number%25 == 0:
    #            toc = time.time()
    #            print frame_number, toc-tic
    #            tic = toc
    #
    pipe.stdin.write(I_worms[frame_number, :, :].tostring())
pipe.terminate()

mask_fid.close()
