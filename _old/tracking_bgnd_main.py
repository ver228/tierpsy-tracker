# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 17:54:54 2015

@author: ajaver
"""

import cv2
import numpy as np
import h5py
from skimage import morphology
#from skimage.morphology import label
from skimage.measure import regionprops, label
import matplotlib.pylab as plt

from sklearn.utils.linear_assignment_ import linear_assignment  # hungarian algorithm
from scipy.spatial.distance import cdist
from collections import defaultdict

MAX_ALLOWED_DIST = 10.0

# def pad_costMatrix(costMatrix, value_to_pad = 1e10):
#    n_pad = costMatrix.shape[1]-costMatrix.shape[0];
#    dum = costMatrix;
#    costMatrix = np.empty((costMatrix.shape[1],costMatrix.shape[1]), costMatrix.dtype);
#    costMatrix.fill(value_to_pad);
#    costMatrix[:-n_pad,:] = dum;
#    return costMatrix


def list_duplicates(seq):
    # get index of replicated elements. It only returns replicates not the
    # first occurence.
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)

    badIndex = []
    for locs in [locs for key, locs in tally.items()
                 if len(locs) > 1]:
        badIndex += locs[1:]
    return badIndex

fileName = '/Volumes/ajaver$/DinoLite/Videos/Exp5-20150116/A002 - 20150116_140923.wmv'

#bgndFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116-2/A002 - 20150116_140923.hdf5'

bgndFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/A002 - 20150116_140923F.hdf5'
bgnd_fid = h5py.File(bgndFile, "r")
bgnd = bgnd_fid["/bgnd"]
BUFF_SIZE = bgnd.attrs['buffer_size']
BUFF_DELTA = bgnd.attrs['delta_btw_bgnd']
INITIAL_FRAME = bgnd.attrs['initial_frame']

maskFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/A002 - 20150116_140923D_mask-2a.hdf5'
mask_fid = h5py.File(maskFile, "w")

maskDB = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/A002 - 20150116_140923D_mask-2a.db'

import sqlite3
conn = sqlite3.connect(maskDB)
cur = conn.cursor()

cur.execute('''DROP TABLE IF EXISTS plate_worms''')
cur.execute('''CREATE TABLE plate_worms
             (worm_index INT NOT NULL,
             frame_number INT NOT NULL,
             coord_x REAL,
             coord_y REAL,
             area REAL,
             perimenter REAL,
             major_axis REAL,
             minor_axis REAL,
             eccentricity REAL,
             compactness REAL,
             orientation REAL,
             speed REAL,
             PRIMARY KEY (worm_index, frame_number)
             )''')  # solidity is quite slow to calculate, unless I have a good reason to do it, I do not see the point


#ind_frame =  5e5;

# open video file
vid = cv2.VideoCapture(fileName)
im_width = vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
im_height = vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)


# create hdf5 file for the masks
if "mask" in mask_fid:
    del mask_fid["mask"]
mask_set = mask_fid.create_dataset("/mask", (0, im_height, im_width),
                                   dtype="u1",
                                   chunks=(1, im_height, im_width),
                                   maxshape=(None, im_height, im_width),
                                   compression="gzip")


nChunk_prev = -1
coord_prev = np.empty([0])
totWorms = 0
indexListPrev = np.empty([0])
# put in in the correct initial frame
vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, INITIAL_FRAME)

max_frame = int(BUFF_DELTA * (bgnd.shape[0] + np.ceil(BUFF_SIZE / 2)))

results_list = []
for ind_frame in range(5):
    # if ind_frame % BUFF_DELTA == 0:
    print ind_frame
    retval, image = vid.read()
    if not retval:
        break
    image = cv2.cvtColor(image, cv2.cv.CV_RGB2GRAY)

    nChunk = np.floor(ind_frame / BUFF_DELTA)

    if nChunk_prev != nChunk:
        bgndInd = nChunk - np.ceil(BUFF_SIZE / 2)
        if bgndInd < 0:
            bgndInd = 0
        elif bgndInd > bgnd.shape[0]:
            bgndInd = bgnd.shape - 1

        mask_bgnd = cv2.adaptiveThreshold(
            bgnd[
                bgndInd,
                :,
                :],
            1,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            61,
            35)
    nChunk_prev = nChunk

    im_diff = np.abs(bgnd[bgndInd, :, :] - image.astype(np.double))
    mask = im_diff > 20
    #mask2 = cv2.adaptiveThreshold(image.astype(np.uint8),1,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,61,35)
    mask2 = cv2.adaptiveThreshold(im_diff.astype(
        np.uint8), 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, -10)

    L = label(mask2 | mask_bgnd)
    L = morphology.remove_small_objects(L, min_size=10, connectivity=2)
    props = regionprops(L)

    im_original = image.copy()
    image[L == 0] = 0
    mask_set.resize(ind_frame + 1, axis=0)
    mask_set[ind_frame, :, :] = image

    if not props:
        coord_prev = np.empty([0])
        continue

    coord = np.array([x.centroid for x in props])
    area = np.array([x.area for x in props])
    perimeter = np.array([x.perimeter for x in props])
    major_axis = np.array([x.major_axis_length for x in props])
    minor_axis = np.array([x.minor_axis_length for x in props])
    eccentricity = np.array([x.eccentricity for x in props])
    compactness = np.array([x.perimeter**2 / x.area for x in props])
    orientation = np.array([x.orientation for x in props])
    # solidity = np.array([x.solidity for x in props]); #solidity is quite
    # slow to calculate, unless I have a good reason to do it, I do not see
    # the point

    if coord_prev.size != 0:
        costMatrix = cdist(coord_prev, coord)

#        if costMatrix.shape[0] < costMatrix.shape[1]:
#            costMatrix = pad_costMatrix(costMatrix);
#
#        elif costMatrix.shape[0] > costMatrix.shape[1]:
#            costMatrix = costMatrix.T;
#            costMatrix = pad_costMatrix(costMatrix);
#            costMatrix = costMatrix.T;
#
        #m = Munkres()
        #assigment = m.compute(costMatrix.copy());
        assigment = linear_assignment(costMatrix)

        indexList = np.zeros(coord.shape[0])
        speed = np.zeros(coord.shape[0])

        for row, column in assigment:  # ll = 1:numel(indexList)
            if costMatrix[row, column] < MAX_ALLOWED_DIST:
                indexList[column] = indexListPrev[row]
                speed[column] = costMatrix[row][column]
            elif column < coord.shape[0]:
                totWorms = totWorms + 1
                indexList[column] = totWorms

        for rep_ind in list_duplicates(indexList):
            totWorms = totWorms + 1  # assign new worm_index to joined trajectories
            indexList[rep_ind] = totWorms

    else:
        indexList = totWorms + np.arange(1, coord.shape[0] + 1)
        totWorms = indexList[-1]
        speed = totWorms * [None]
        # print costMatrix[-1,-1]

    frames = np.empty(indexList.size)
    frames.fill(ind_frame + 1)

    results_list = zip(*(indexList,
                         frames,
                         coord[:,
                               0],
                         coord[:,
                               1],
                         area,
                         perimeter,
                         major_axis,
                         minor_axis,
                         eccentricity,
                         compactness,
                         orientation,
                         speed))

    cur.executemany(
        'INSERT INTO plate_worms VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
        results_list)
    if ind_frame % 1200 == 1:

        results_list = []
        conn.commit()
    coord_prev = coord
    indexListPrev = indexList

mask_fid.close()
vid.release()
conn.commit()
conn.close()

#
# plt.figure()
#plt.imshow(im_original, cmap = 'gray', interpolation='none')
#
# plt.figure()
#plt.imshow(L, cmap = 'gray', interpolation='none')
#
#im_dum = image.copy();
#im_dum[mask==0] = 0
# plt.figure()
#plt.imshow(im_dum, cmap = 'gray', interpolation='none')
#
#im_dum = image.copy();
#im_dum[L==0] = 0
# plt.figure()
#plt.imshow(im_dum, cmap = 'gray', interpolation='none')
#
# plt.figure()
# plt.imshow((, cmap = 'gray', interpolation='none')

# plt.figure()
#plt.imshow((bgnd[bgndInd,:,:]-image), cmap = 'gray', interpolation='none')
#
# plt.figure()
#plt.imshow(image, cmap = 'gray', interpolation='none')
# plt.figure()
#plt.imshow(~mask, cmap = 'gray', interpolation='none')
