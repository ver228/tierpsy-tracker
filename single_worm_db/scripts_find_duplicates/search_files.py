# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:10:03 2016

@author: ajaver
"""
import os

#correct experiments
root_dir = '/Volumes/behavgenom_archive$/thecus/'
root_dir = os.path.abspath(root_dir)
assert os.path.exists(root_dir)
	
tot = 0;
with open('single_worm_movies.txt', 'w') as fid_mov, \
open('stage_move_csv.txt', 'w') as fid_csv, \
open('video_info_xml.txt', 'w') as fid_xml:
    valid_files = []
    for dpath, dnames, fnames in os.walk(root_dir):
        for fname in fnames:
            fullfilename = os.path.abspath(os.path.join(dpath, fname))        
            if fname.endswith('.avi'):
                fid_mov.write(fullfilename + '\n')
                tot += 1
                print(tot)
            elif fname.endswith('.info.xml'):
                fid_xml.write(fullfilename + '\n')
            elif fname.endswith('.log.csv'):
                fid_csv.write(fullfilename + '\n')
                
                
                
            