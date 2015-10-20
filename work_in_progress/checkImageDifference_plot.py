# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:16:43 2015

@author: ajaver
"""
import glob
import matplotlib.pylab as plt
import h5py
from PIL import Image
import os


save_dir_root = r'/Users/ajaver/Desktop/Gecko_compressed/test/'

movie_dir = '/Volumes/behavgenom$/GeckoVideo/Compressed/20150515/';
fileList = glob.glob(movie_dir + '*.hdf5')

for ff in fileList[-1:]:
    print ff
    fid = h5py.File(ff, 'r')
#    yy = fid['/im_diff'][:]
#    plt.figure()    
#    plt.plot(yy)

    base_name = os.path.split(ff)[-1][:-5]
    save_dir = save_dir_root + base_name + os.sep
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for frame in range(618800, 619100):
        result = Image.fromarray(fid['/mask'][frame,:,:])
        save_name = '%s%i.bmp' % (save_dir, frame)
        result.save(save_name)
    
    fid.close()

result.save('out.bmp')    