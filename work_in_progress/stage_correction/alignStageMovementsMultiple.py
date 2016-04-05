# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 22:01:59 2016

@author: ajaver
"""
import sys




if __name__ == '__main__':
    print(sys.argv)
    videos_list = sys.argv[1]
    with open(videos_list, 'r') as fid:
        valid_files = fid.read().split('\n')
    valid_files = [x for x in valid_files if x]
    

    print(valid_files)
