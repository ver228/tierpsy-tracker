# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:44:22 2015

@author: ajaver
"""

import h5py
import tables
import sys
import numpy as np
sys.path.append('./image_difference/')
from image_difference import image_difference
sys.path.append('../videoCompression/')
from parallelProcHelper import timeCounterStr


import matplotlib.pylab as plt


masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/CaptureTest_90pc_Ch1_02022015_141431.hdf5';
save_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/CaptureTest_90pc_Ch1_02022015_141431_diff.hdf5';

#masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Capture_Ch1_23032015_111907.hdf5';
#save_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/Capture_Ch1_23032015_111907_diff.hdf5';

#masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Capture_Ch4_23032015_111907.hdf5';
#save_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/Capture_Ch4_23032015_111907_diff.hdf5';


mask_fid = h5py.File(masked_image_file, 'r');
mask_dataset = mask_fid["/mask"]


#im_diff_fid = h5py.File(save_file, 'w');


im_diff_fid = tables.File(save_file, 'w');

im_diff_table = im_diff_fid.createTable('/', 'im_diff', {
'frame_number':    tables.Int32Col(),
'im_diff'         : tables.Float32Col(),
'im_diff_ts'       : tables.Float32Col(),
})



tot_img = mask_dataset.shape[0]
#im_diff = im_diff_fid.create_dataset("/img_diff" , (tot_img, 1), 
#                                        dtype = np.float, maxshape = (tot_img, 1), 
#                                        chunks = True,
#                                        compression="gzip", shuffle=True);
#im_diff_ts = im_diff_fid.create_dataset("/img_diff_timestamp" , (tot_img, 1), 
#                                        dtype = np.float, maxshape = (tot_img, 1), 
#                                        chunks = True,
#                                        compression="gzip", shuffle=True);

progressTime = timeCounterStr('Calculating...')

Iprev = np.zeros([]);
Ilabelprev = np.zeros([]);
for frame_number in range(0,tot_img):
    I = mask_dataset[frame_number,:,:];
    Ilabel = I[0:15,0:479].copy();
    I[0:15,0:479] = 0;
    
    if Iprev.shape:
        im_diff = image_difference(I,Iprev)
        im_diff_ts = image_difference(Ilabel,Ilabelprev)
        im_diff_table.append([(frame_number, im_diff,im_diff_ts)])
    Iprev = I.copy();
    Ilabelprev = Ilabel.copy();
    
    if frame_number % 500 == 0:
        #calculate the progress and put it in a string            
        print(progressTime.getStr(frame_number))

plt.figure()
plt.plot(im_diff_fid.get_node('/im_diff').col('im_diff'))

plt.figure()
plt.plot(im_diff_fid.get_node('/im_diff').col('im_diff_ts'))
im_diff_fid.close()
mask_fid.close()
#%%




