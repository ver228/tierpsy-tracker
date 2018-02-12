#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:19:07 2017

@author: ajaver
"""
import os
import cv2
import sys
import glob
import threading
from functools import partial

main_dir = '/Volumes/behavgenom_archive$/Celine/raw/'
fnames = glob.glob(os.path.join(main_dir, '**', '*.avi'))
fnames = [x for x in fnames if not x.endswith('_seg.avi')]
fnames = sorted(fnames)

def get_and_release(video_file):
    original = sys.stderr
    f = open(os.devnull, 'w')
    sys.stderr = f
    print('here')
    vid = cv2.VideoCapture(video_file)
    vid.release()
    
    sys.stderr = original
    
    return vid

all_threads = []
for ii, video_file in enumerate(fnames):
    print(ii, video_file)
    vid = cv2.VideoCapture(video_file)
    vid.release()
    t = threading.Thread(target = partial(get_and_release, video_file))
    t.start()
    all_threads.append((video_file, t))
    
    
    