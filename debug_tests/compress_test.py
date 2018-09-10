# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import json
import time
import numpy as np
import h5py

from imgstore import new_for_filename
from pprint import pprint


#from tierpsy.analysis.compress.compressVideo import compressVideo
from tierpsy.analysis.compress.processVideo import processVideo
from tierpsy.analysis.compress.selectVideoReader import selectVideoReader

video_file_list = ["/home/lferiani@cscdom.csc.mrc.ac.uk/Data/codec_testing_hqfast_10s_20180612_123152/metadata.yaml"]#,
#                  "/home/lferiani@cscdom.csc.mrc.ac.uk/Data/codec_testing_hqfast_20s_20180612_123119/metadata.yaml",
#                  "/home/lferiani@cscdom.csc.mrc.ac.uk/Data/codec_testing_hqfast_30s_20180612_123038/metadata.yaml",
#                  "/home/lferiani@cscdom.csc.mrc.ac.uk/Data/codec_testing_hqfast_60s_20180612_122923/metadata.yaml",
#                  "/home/lferiani@cscdom.csc.mrc.ac.uk/Data/codec_testing_hqfast_long_20180612_134936/metadata.yaml"]
json_param_file = "/home/lferiani@cscdom.csc.mrc.ac.uk/Data/_my_TEST_loopbio.json"

#video_file_list = ["/home/lferiani@cscdom.csc.mrc.ac.uk/Tierpsy/tierpsy-tracker/tests/data/GECKO_VIDEOS/RawVideos/GECKO_VIDEOS.mjpg"]
#json_param_file = "/home/lferiani@cscdom.csc.mrc.ac.uk/Tierpsy/tierpsy-tracker/tierpsy/extras/param_files/_AEX_RIG.json"

#video_file_list = ["/home/lferiani@cscdom.csc.mrc.ac.uk/Data/Fluo_test/recording 3.1.avi"]
#json_param_file = "/home/lferiani@cscdom.csc.mrc.ac.uk/Data/Fluo_test/fluotest.json"

# read parameters
with open(json_param_file) as data_file:    
    json_param = json.load(data_file)
#pprint(json_param)

# mask_param has more parameters than needed, and they have different names as well
mask_param_f = ['mask_min_area', 'mask_max_area', 'thresh_block_size', 
        'thresh_C', 'dilation_size', 'keep_border_data', 'is_light_background']
mask_param = {x.replace('mask_', ''):json_param[x] for x in mask_param_f}

if "mask_bgnd_buff_size" in json_param:
    bgnd_param_mask_f = ['mask_bgnd_buff_size', 'mask_bgnd_frame_gap', 'is_light_background']
    bgnd_param_mask = {x.replace('mask_bgnd_', ''):json_param[x] for x in bgnd_param_mask_f}
else:
    bgnd_param_mask = {}
# if
    
if "save_full_interval" not in json_param: json_param['save_full_interval'] = -1
if "is_extract_timestamp" not in json_param: json_param['is_extract_timestamp'] = False

# put parameters together for processVideo.py
compress_vid_param = {
        'buffer_size': json_param['compression_buff'],
        'save_full_interval': json_param['save_full_interval'],
        'mask_param': mask_param,
        'bgnd_param': bgnd_param_mask,
        'expected_fps': json_param['expected_fps'],
        'microns_per_pixel' : json_param['microns_per_pixel'],
        'is_extract_timestamp': json_param['is_extract_timestamp']
    }                   
                   

buffer_sizes = np.array([32])
Nbsz = len(buffer_sizes)
tictoc_results = np.zeros([len(video_file_list),Nbsz])
compression_fps = np.zeros([len(video_file_list),Nbsz])
n_videoframes = np.zeros(len(video_file_list))

# disable background subtraction
#compress_vid_param['bgnd_param'] = {}

# set some parameters for background subtraction
#compress_vid_param['bgnd_param'] = {'buff_size': 10, 'frame_gap': 10, 'is_light_background': False}

      #%%             
vc = 0;

for video_file in video_file_list:
    
    # set path variables 
    #video_dir = "/home/lferiani@cscdom.csc.mrc.ac.uk/Data/codec_testing_hqfast_10s_20180612_123152/"
    video_dir, video_name = os.path.split(video_file)
    
    if video_name.endswith('yaml'):
        
        masked_video_dir = os.path.join(video_dir,"MaskedVideos")
        masked_image_name = video_name.replace('.yaml','.hdf5')
        
    elif video_name.endswith('mjpg'):
        
        masked_video_dir = video_dir.replace('RawVideos','MaskedVideos')
        masked_image_name = video_name.replace('.mjpg','.hdf5')
        
    elif video_name.endswith('avi'):
        masked_video_dir = video_dir
        masked_image_name = video_name.replace('.avi','.hdf5')
    #if
    
    masked_image_file =  os.path.join(masked_video_dir, masked_image_name)

    
    # try Nbsz sizes    
    for bi in range(0, Nbsz):
        
        # clean output folders
        if not os.path.isdir(masked_video_dir):
            os.mkdir(masked_video_dir)
        
        if os.path.isfile(masked_image_file):
            os.remove(masked_image_file)
        
        
#        compress_vid_param['buffer_size'] = buffer_sizes[bi]
        
        
        
        tic = time.time()
        
        processVideo(video_file, masked_image_file, compress_vid_param)
        #compressVideo(video_file, masked_image_file, mask_param,  expected_fps=25,
        #                  microns_per_pixel=None, bgnd_param = bgnd_param_mask, buffer_size=-1,
        #                  save_full_interval=-1, max_frame=1e32, is_extract_timestamp=False)
        
        toc = time.time()
        elapsed = toc - tic
        print("Elapsed: %s" % elapsed)
        
        # now store result
        tictoc_results[vc,bi] = elapsed
        
        # read number of frames
        hf = h5py.File(masked_image_file, 'r')
        n_videoframes[vc] = hf['mask'].shape[0];
       
        # calculate compression fps
        compression_fps[vc,bi] = n_videoframes[vc] / elapsed;
        print("Compression fps: %s" % compression_fps)
        
    # for bi
    vc += 1
# for video_dir
    
#%%
    
"""    
import cv2
import time
import numpy as np
from scipy.ndimage.filters import median_filter

c    
tic = time.time()    

nloops = 20
for cc in range(0, nloops):
    median_filter(img_test,5)
# for

print('scipy')
print((time.time() - tic)/nloops)


tic = time.time()    

nloops = 20
for cc in range(0, nloops):
    cv2.medianBlur(img_test,5)
# for

print('opencv')
print((time.time() - tic)/nloops)

#%%
medscipy = median_filter(img_test,5,mode='nearest')
medcv2 = cv2.medianBlur(img_test,5)
"""
#%%
"""
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

elapsed = np.zeros(len(np.arange(6,12)))
elapsed_2 = np.zeros(len(np.arange(6,12)))

for isz, sz in enumerate(2**np.arange(6,12)):
    img_test = np.random.randint(0, high=256, size=(sz,sz), dtype=np.uint8)
    tic = time.time();
    for i in range(0,20):
        cv2.medianBlur(img_test,5)
    #for
    elapsed[isz] = (time.time() - tic)/20
    img_test2 = img_test[:-3,:-3].copy()
    tic = time.time();
    for i in range(0,20):
        cv2.medianBlur(img_test2,5)
    #for
    elapsed_2[isz] = (time.time() - tic)/20
#for


plt.plot(2**np.arange(6,12), elapsed)
plt.plot(2**np.arange(6,12), elapsed_2)
"""
#%%
"""
import time
import numpy as np

img_test = np.random.randint(0, high=256, size=(2048,2048), dtype=np.uint8)
img_test = np.clip(img_test, 1,255)

img_out = img_test.copy();

tic = time.time();
for i in range(0,400):
    img_out = np.clip(img_test, 1,255)
#for
print((time.time() - tic)/400)
"""




