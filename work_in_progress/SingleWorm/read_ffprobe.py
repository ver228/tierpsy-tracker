# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:57:54 2016

@author: ajaver
"""
import numpy as np
import json

import os
import subprocess as sp
from collections import OrderedDict


def dict2recarray(dat):
    '''convert into recarray (pytables friendly)'''
    dtype = [(kk, dat[kk].dtype) for kk in dat]
    N = len(dat[dtype[0][0]])

    recarray = np.recarray(N, dtype)
    for kk in dat: recarray[kk] = dat[kk]
    
    return recarray

def ffprobeMetadata(video_file):
    command = ['ffprobe', '-v', 'error', '-show_frames', '-print_format', 'json', video_file]
    FNULL = open(os.devnull, 'w')
    pipe = sp.Popen(command, stdout = sp.PIPE, stderr = sp.PIPE)
    buff = pipe.stdout.read()
    buff_err = pipe.stderr.read()
    
    
    dat = json.loads(buff.decode('utf-8'))    
    if not dat:
        raise Exception(buff_err)
    
    video_metadata = OrderedDict()
    for field in dat['frames'][0].keys():
        video_metadata[field] = [frame[field] for frame in dat['frames']]
        try: #if possible convert the data into float
            video_metadata[field] = [float(dd) for dd in video_metadata[field]]
        except:
            pass
        video_metadata[field] = np.asarray(video_metadata[field])
    
    video_metadata = dict2recarray(video_metadata)
    return video_metadata


    

if __name__ == '__main__':
    #video_file = "/Users/ajaver/Desktop/Videos/Check_Align_samples/Videos/npr-13 (tm1504)V on food L_2010_01_25__11_56_02___4___2.avi"
    video_file = "/Volumes/D/Bertie/13_1/CSTCTest_Ch1_13012016_112004.mjpg"
    dat = ffprobeMetadata(video_file)
    dat = dict2recarray(dat)
    
    import matplotlib.pylab as plt
    plt.plot(np.diff(dat['best_effort_timestamp']))