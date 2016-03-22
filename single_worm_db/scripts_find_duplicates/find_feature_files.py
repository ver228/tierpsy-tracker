# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 18:00:09 2016

@author: ajaver
"""

import os
import shutil

def walkdir(root_dir):
    all_files = []
    for dpath, dnames, fnames in os.walk(root_dir):
        for fname in fnames:
            fullfilename = os.path.abspath(os.path.join(dpath, fname))
            all_files.append(fullfilename)
        print(len(all_files))
    return all_files

if __name__ == '__main__':
    
    #root_dirs = ['/Volumes/behavgenom$/Andre/results-12-05-10/Laura Grundy/', 
    #             '/Volumes/behavgenom$/Andre/results-12-05-10/wild-isolates/']
    #andre_files = []
    #for root_dir in root_dirs:
    #    andre_files += walkdir(root_dir)
    
    #andre_files = walkdir(root_dir)
#    feature_files = [x for x in andre_files if x.endswith('_features.mat')]
#    with open('segworm_feat_files.txt', 'w') as fid:
#        for x in feature_files:
#            fid.write(x + '\n')
    
    with open('segworm_feat_files.txt', 'r') as fid:
        feature_files = [x for x in fid.read().split('\n') if x]    
    feat_dict = {os.path.split(x)[1].rpartition('_features.')[0]:x for x in feature_files}

    for dpart in ['agar_1', 'agar_2', 'agar_goa']:
        mask_dir = '/Users/ajaver/Desktop/Videos/single_worm/%s/MaskedVideos/' % dpart
        feat_dir = mask_dir.replace('/MaskedVideos/', '/Features/')
        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)
        base_names = set(fname.split('.')[0] for fname in os.listdir(mask_dir) if fname.endswith('.hdf5'))  
        
        for bn in base_names:
            if bn in feat_dict:
                print(feat_dict[bn])
        
                shutil.copy(feat_dict[bn], feat_dir)