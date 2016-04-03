# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 17:58:38 2016

@author: ajaver
"""
import random
import os
import shutil

def read_file(fname):
    with open(fname, 'r') as fid:
        flines = fid.read().split('\n')
        flines = [x for x in flines if x]
    return flines

def write_file(fname, flines):
    with open(fname, 'w') as fid:
        for x in flines:
            fid.write(x + '\n')

def select_random_videos(fname, n_random, ori_video_dir, ori_mask_dir, str_filt = ''):
    video_files = read_file(fname)
    video_files = [x for x in video_files if str_filt in x]
    video_files = random.sample(video_files, n_random)
    video_files = [x.replace(ori_video_dir, ori_mask_dir) for x in video_files]    
    video_files = [x.replace('.avi', '.hdf5') for x in video_files]
    
    
    fname_rand = fname.replace('all_', 'random_' if 'all' in fname else 'random_' + fname)
    if str_filt:
        fname_rand = fname_rand.replace('random_', 'random_' + str_filt +  '_')
    
    write_file(fname_rand, video_files)


def copy_files(flist, ori_root_dir, final_root_dir):
    if not os.path.exists(final_root_dir): os.makedirs(final_root_dir)
    for ii, src in enumerate(flist):
        if os.path.exists(src):
            #dst = src.replace(ori_root_dir, final_root_dir)                    
            #dstdir, dstf = os.path.split(src)            
            #if not os.path.exists(dstdir): os.makedirs(dstdir)
                
            _, dstf = os.path.split(src)   
            dst = os.path.join(final_root_dir, dstf)
            
            print(ii+1, dst)
            shutil.copyfile(src, dst)


if __name__ == '__main__':
    ori_video_dir = '/Volumes/behavgenom_archive$/thecus/'
    ori_mask_dir = '/Volumes/behavgenom_archive$/MaskedVideos/'
    
    
    
    #select_random_videos('all_swimming.txt', 50, ori_video_dir, ori_mask_dir)
    #swimming_files = read_file('random_swimming.txt')
    #final_root_dir = '/Users/ajaver/Desktop/Videos/single_worm/swimming/MaskedVideos/'
    #copy_files(swimming_files, ori_mask_dir, final_root_dir)
    
    #select_random_videos('all_agar.txt', 50, ori_video_dir, ori_mask_dir)
    #agar_files = read_file('random_agar.txt')
    #final_root_dir = '/Users/ajaver/Desktop/Videos/single_worm/agar/MaskedVideos/'
    #copy_files(agar_files, ori_mask_dir, final_root_dir)
    
    select_random_videos('all_agar.txt', 55, ori_video_dir, ori_mask_dir, str_filt='goa')
    agar_files = read_file('random_goa_agar.txt')
    final_root_dir = '/Users/ajaver/Desktop/Videos/single_worm/agar_goa/MaskedVideos/'
    agar_files = [x for x in agar_files if '/goa' in x]    
    copy_files(agar_files, ori_mask_dir, final_root_dir)
    
    