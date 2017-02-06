# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:57:54 2016

@author: ajaver
"""
import numpy as np
import json
import tables

import os
import subprocess as sp
from collections import OrderedDict

import warnings
from MWTracker.helper.timeCounterStr import timeCounterStr
from MWTracker.helper.misc import print_flush, ReadEnqueue, FFPROBE_CMD

def dict2recarray(dat):
    '''convert into recarray (pytables friendly)'''
    dtype = [(kk, dat[kk].dtype) for kk in dat]
    N = len(dat[dtype[0][0]])

    recarray = np.recarray(N, dtype)
    for kk in dat:
        recarray[kk] = dat[kk]

    return recarray


def get_ffprobe_metadata(video_file):
    if not os.path.exists(video_file):
        raise FileNotFoundError(video_file)

    if not FFPROBE_CMD:
        warnings.warn('ffprobe do not found. Raw video metadata will not be extracted.')
        return np.zeros(0)

    command = [
        FFPROBE_CMD,
        '-v',
        'error',
        '-show_frames',
        '-print_format',
        'json',
        video_file]
    
    base_name = video_file.rpartition('.')[0].rpartition(os.sep)[-1]
    progressTime = timeCounterStr(base_name + ' Extracting video metadata.')
    
    #from MWTracker.helper.runMultiCMD import cmdlist2str
    #print( cmdlist2str(command))
    
    
    frame_number = 0
    buff = []
    buff_err = []
    proc = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
    buf_reader = ReadEnqueue(proc.stdout, timeout=1)
    buf_reader_err = ReadEnqueue(proc.stderr)

    while proc.poll() is None:
        # read line without blocking
        line = buf_reader.read()
        if line is None:
            print('cannot read')
        else:
            buff.append(line)
            if "media_type" in line: #i use the filed "media_type" as a proxy for frame number (just in case the media does not have frame number)
                frame_number += 1
                if frame_number % 500 == 0:
                    print_flush(progressTime.getStr(frame_number))
        
        line = buf_reader_err.read()
        if line is not None:
            buff_err.append(None)


    buff = ''.join(buff)
    #print(buff)

    try:
        dat = json.loads(buff)
    except json.decoder.JSONDecodeError as e:
        return np.zeros(0) 

    if not dat:
        print(buff_err)
        print(base_name + ' Could not read the metadata.')
        return np.zeros(0)
    
    # use the first frame as reference
    frame_fields = list(dat['frames'][0].keys())

    # consider data where the best_effort_timestamp was not calculated
    valid_frames = [
        x for x in dat['frames'] if all(
            ff in x for ff in frame_fields)]

    # store data into numpy arrays
    video_metadata = OrderedDict()
    for field in frame_fields:
        video_metadata[field] = [frame[field] for frame in valid_frames]
        

        try:  # if possible convert the data into float
            video_metadata[field] = [float(dd) for dd in video_metadata[field]]
        except (ValueError, TypeError):
            # pytables does not support unicode strings (python3)
            #the str before is to convert a possible dictionary into a string before converting it to bytes
            video_metadata[field] = [bytes(str(dd), 'utf-8')
                                     for dd in video_metadata[field]]

        video_metadata[field] = np.asarray(video_metadata[field])

    video_metadata = dict2recarray(video_metadata)
    return video_metadata


def store_meta_data(video_file, masked_image_file):

    video_metadata = get_ffprobe_metadata(video_file)
    expected_frames = len(video_metadata)

    if expected_frames == 0:  # nothing to do here. return a dum number of frames
        return 1

    with tables.File(masked_image_file, 'r+') as mask_fid:
        if '/video_metadata' in mask_fid:
            mask_fid.remove_node('/', 'video_metadata')
        mask_fid.create_table('/', 'video_metadata', obj=video_metadata)

    return expected_frames


def correct_timestamp(best_effort_timestamp, best_effort_timestamp_time):
    # delta from the best effort indexes
    xx = np.diff(best_effort_timestamp)
    good_N = (xx != 0)  # & ~np.isnan(xx)
    delta_x = np.median(xx[good_N])

    # delta from the best effort times
    xx_t = np.diff(best_effort_timestamp_time)
    good_Nt = xx_t != 0  # & ~np.isnan(xx)
    delta_t = np.median(xx_t[good_Nt])

    # test that the zero delta from the index and time are the same
    assert np.all(good_N == good_Nt)

    # check that the normalization factors make sense
    xx_N = np.round(xx / delta_x).astype(np.int)
    assert np.all(xx_N == np.round(xx_t / delta_t).astype(np.int))

    # get the indexes with a valid frame
    # add one to consider compensate for the np.diff
    timestamp = np.arange(1, len(xx_N) + 1)
    timestamp = timestamp[good_N] + xx_N[good_N] - 1
    timestamp = np.hstack((0, timestamp))

    timestamp_time = timestamp * delta_t

    return timestamp, timestamp_time


def get_timestamp(masked_image_file):
    '''
    Read the timestamp from the video_metadata, if this field does not exists return an array of nan
    '''

    with tables.File(masked_image_file, 'r') as mask_fid:
        # get the total number of frmes previously processed
        tot_frames = mask_fid.get_node("/mask").shape[0]

        if '/video_metadata' in mask_fid:
            # try to read data from video_metadata
            dd = [(row['best_effort_timestamp'], row['best_effort_timestamp_time'])
                  for row in mask_fid.get_node('/video_metadata/')]

            timestamp, timestamp_time = list(zip(*dd))
            timestamp = np.asarray(timestamp)
            timestamp_time = np.asarray(timestamp_time)

            assert timestamp.size == timestamp_time.size
            timestamp, timestamp_time = correct_timestamp(timestamp, timestamp_time)

        else:
            timestamp = np.full(tot_frames, np.nan)
            timestamp_time = np.full(tot_frames, np.nan)

        assert timestamp.size == timestamp_time.size
        return timestamp, timestamp_time


def read_and_save_timestamp(masked_image_file, dst_file=''):
    '''
        Read and save timestamp data in to the dst_file, if there is not a dst_file save it into the masked_image_file
    '''
    if not dst_file:
        dst_file = masked_image_file

    # read timestamps from the masked_image_file
    timestamp, timestamp_time = get_timestamp(masked_image_file)
    with tables.File(masked_image_file, 'r') as mask_fid:
        tot_frames = mask_fid.get_node("/mask").shape[0]

    if tot_frames > timestamp.size:
        # pad with nan the extra space
        N = tot_frames - timestamp.size
        timestamp = np.hstack((timestamp, np.full(N, np.nan)))
        timestamp_time = np.hstack((timestamp_time, np.full(N, np.nan)))
        assert tot_frames == timestamp.size

    # save timestamp into the dst_file
    with tables.File(dst_file, 'r+') as dst_file:
        dst_file.create_group('/', 'timestamp')
        dst_file.create_carray('/timestamp', 'raw', obj=np.asarray(timestamp))
        dst_file.create_carray(
            '/timestamp',
            'time',
            obj=np.asarray(timestamp_time))


if __name__ == '__main__':
    video_file = "/Users/ajaver/Desktop/Videos/Check_Align_samples/Videos/npr-13 (tm1504)V on food L_2010_01_25__11_56_02___4___2.avi"
    masked_file = "/Users/ajaver/Desktop/Videos/Check_Align_samples/MaskedVideos/npr-13 (tm1504)V on food L_2010_01_25__11_56_02___4___2.hdf5"

    dat = get_ffprobe_metadata(video_file)

    store_meta_data(video_file, masked_file)
    import matplotlib.pylab as plt
    plt.plot(np.diff(dat['best_effort_timestamp']))
