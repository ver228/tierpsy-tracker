# -*- coding: utf-8 -*-
"""
Created on Fri May 15 17:56:15 2015

@author: ajaver
"""
import sys
import h5py
import numpy as np
import os
from image_difference_mask import image_difference_mask


from parallelProcHelper import timeCounterStr

if __name__ == '__main__':
    
    masked_image_file = sys.argv[1]
    base_name = os.path.splitext(os.path.split(masked_image_file)[-1])[0];
    
    
    mask_fid = h5py.File(masked_image_file, "r+");
    if 'im_diff' in mask_fid.keys():
        print(base_name + ' im_diff field already exists. Nothing to do here.')
    else:
    
        mask_dataset = mask_fid['/mask']
        
            
        tot_frames = mask_dataset.shape[0]
        im_diff_set = mask_fid.create_dataset('/im_diff', (tot_frames,), 
                                              dtype = 'f4', maxshape = (tot_frames,), 
                                            chunks = True, compression = "gzip", compression_opts=9, shuffle = True)
                                            
        image_prev = np.zeros([]);
        
        progressTime = timeCounterStr('Calculating image difference.');
        for frame_number in range(tot_frames):
            image = mask_dataset[frame_number,:,:]                                   
            image[0:15,0:479] = 0; #remove timestamp before calculation
            if image_prev.shape:
                im_diff_set[frame_number] = image_difference_mask(image,image_prev)
            image_prev = image.copy();
            
            if frame_number%500 == 0:
                #calculate the progress and put it in a string
                progress_str = progressTime.getStr(frame_number)
                print(base_name + ' ' + progress_str);
        
        print(base_name + 'Finished');
        mask_fid.close()
    