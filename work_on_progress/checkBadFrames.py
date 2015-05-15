# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:35:09 2015

@author: ajaver
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

import h5py

masked_image_file =  r'/Volumes/behavgenom$/GeckoVideo/Compressed/20150512/Capture_Ch1_12052015_194303.hdf5'
mask_fid = h5py.File(masked_image_file, 'r');


#mask_fid.close()