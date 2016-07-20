# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 16:08:07 2015

@author: ajaver
"""

import matplotlib.pylab as plt

import os
import errno
import cv2
import numpy as np
import h5py
import time

fileName = '/Volumes/ajaver$/DinoLite/Videos/Exp5-20150116/A002 - 20150116_140923.wmv'
saveDir = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/'
saveFile = "A002 - 20150116_140923H.hdf5"


BUFF_SIZE = 20
BUFF_DELTA = 900
# percentage in the ordered buffer that correspond to the background (set
# 0.5 for the median)
BGND_ORD = 0.85
BGND_IND = round(BGND_ORD * BUFF_SIZE) - 1

INITIAL_POS = 0  # THIS VALUE IS NOT IN FRAMES, I DON'T KNOW THE UNITS BUT MIGHT BE USEFUL IF ONE WANTS TO SKIP THE BEGGINING OF THE VIDEO

# make the save directory if it didn't exist before
if not os.path.exists(saveDir):
    try:
        os.makedirs(saveDir)
    # throw an exeption if the directory didn't exist
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(saveDir):
            pass

# create a buffer ring to calculate the background


class imRingBuffer:

    def __init__(self, height, width, buff_size, data_type=np.uint8):
        self.buffer = np.zeros((buff_size, height, width), np.uint8)
        self.index = 0
        self.buff_size = buff_size

    def add(self, new_image):
        if self.index >= self.buff_size:
            self.index = 0

        self.buffer[self.index, :, :] = new_image
        self.index += 1


vid = cv2.VideoCapture(fileName)
vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, INITIAL_POS)
im_width = vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
im_height = vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
im_buffer = imRingBuffer(height=im_height, width=im_width, buff_size=BUFF_SIZE)


# this normally does not work very good
tot_frames = vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)


# Initialize the opening of the hdf5 file, and delete the dataset bgnd
# there was some data previously
fidh5 = h5py.File(saveDir + saveFile, "w")
bgnd_set = fidh5.create_dataset("/bgnd", (0, im_height, im_width),
                                dtype="u1",
                                chunks=(1, im_height, im_width),
                                maxshape=(None, im_height, im_width),
                                compression="gzip")
bgnd_set.attrs['buffer_size'] = BUFF_SIZE
bgnd_set.attrs['delta_btw_bgnd'] = BUFF_DELTA
bgnd_set.attrs['bgnd_buffer_order'] = BGND_ORD
bgnd_set.attrs['initial_position'] = INITIAL_POS
bgnd_set.attrs['video_source_file'] = fileName

tot_frames = 0
tot_bgnd = 0
N_chunks = 0

tic_first = time.time()
tic = tic_first

while True:  # N_chunks < TOT_CHUNKS:

    retval, image = vid.read()
    if not retval:
        break

    tot_frames += 1
    if tot_frames % BUFF_DELTA == 1:
        image = cv2.cvtColor(image, cv2.cv.CV_RGB2GRAY).astype(np.double)
        im_buffer.add(image)

        # for kk in range(BUFF_DELTA-1):
        #    retval, image = vid.read()

        if tot_frames / BUFF_DELTA >= BUFF_SIZE:  # N_chunks >=BUFF_SIZE:
            # calculate background only once the buffer had been filled
            sorted_buff = np.sort(im_buffer.buffer, 0)
            bgnd_set.resize(tot_bgnd + 1, axis=0)
            bgnd_set[tot_bgnd, :, :] = sorted_buff[BGND_IND, :, :]
            tot_bgnd += 1

        toc = time.time()
        print tot_frames, toc - tic
        tic = time.time()
        #N_chunks += 1;
        # print N_chunks;
#%%
print 'Total time: %d' % (time.time() - tic_first)

vid.release()
fidh5.close()


# for bgnd in allBgnd:
#    plt.figure();
#    fig = plt.imshow(bgnd);
#    fig.set_cmap('gray');
#    fig.set_interpolation('none');
#%%
# vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES);
#retval, image = vid.read()
#
# plt.imshow(image)

#
#
#
#
#DBName = 'A001_results.db';
#conn = sql.connect(saveDir + DBName)
#
# conn.close()
