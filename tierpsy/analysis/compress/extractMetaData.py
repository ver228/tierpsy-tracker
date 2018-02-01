# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:57:54 2016

@author: ajaver
"""
import json
import os
import subprocess as sp
import warnings
from collections import OrderedDict

import numpy as np
import tables

from tierpsy.helper.misc import TimeCounter, print_flush, ReadEnqueue, FFPROBE_CMD


def dict2recarray(dat):
    '''convert into recarray (pytables friendly)'''

    if len(dat) > 0:
        dtype = [(kk, dat[kk].dtype) for kk in dat]
        N = len(dat[dtype[0][0]])
        recarray = np.recarray(N, dtype)
        for kk in dat:
            recarray[kk] = dat[kk]
        return recarray
    else:
        return np.array([])

    


def get_ffprobe_metadata(video_file):
    if not os.path.exists(video_file):
        raise FileNotFoundError(video_file)

    if not os.path.exists(FFPROBE_CMD):
        raise FileNotFoundError('ffprobe do not found.')
        
    command = [
        FFPROBE_CMD,
        '-v',
        'error',
        '-show_frames',
        '-print_format',
        'compact',
        video_file]
    
    base_name = video_file.rpartition('.')[0].rpartition(os.sep)[-1]
    progressTime = TimeCounter(base_name + ' Extracting video metadata.')
    
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
                    print_flush(progressTime.get_str(frame_number))
        
        line = buf_reader_err.read()
        if line is not None:
            buff_err.append(None)


    #the buff is in the shape
    # frame|feat1=val1|feat2=val2|feat3=val3\n 
    # I want to store each property as a vector
    dat = [[d.split('=') for d in x.split('|')] for x in ''.join(buff).split('\n')]
    
    # use the first frame as reference
    frame_fields = [x[0] for x in dat[0] if len(x) == 2]
    
    # store data into numpy arrays
    video_metadata = OrderedDict()
    for row in dat:
        for dd in row:
            if (len(dd) != 2) or (not dd[0] in frame_fields):
                continue
            field, value = dd

            if not field in video_metadata:
                video_metadata[field] = []

            try:  # if possible convert the data into float
                value = float(value)
            except (ValueError, TypeError):
                if value == 'N/A':
                    value = np.nan
                else:
                    # pytables does not support unicode strings (python3)
                    #the str before is to convert a possible dictionary into a string before converting it to bytes
                    value = bytes(str(value), 'utf-8')

            video_metadata[field].append(value)


    #convert all the lists into numpy arrays
    video_metadata = {field:np.asarray(values) for field,values in video_metadata.items()}
    
    #convert data into a recarray to store in pytables
    video_metadata = dict2recarray(video_metadata)

    #sometimes the last frame throws a nan in the timestamp. I want to remove it
    if video_metadata.size > 0:
        if np.isnan(video_metadata[-1]['best_effort_timestamp']):
            video_metadata = video_metadata[:-1]

        #if there is still nan's raise an error
        if np.any(np.isnan(video_metadata['best_effort_timestamp'])):
            raise ValueError('The timestamp contains nan values')
    return video_metadata


def store_meta_data(video_file, masked_image_file):
    try:
        video_metadata = get_ffprobe_metadata(video_file)
        if len(video_metadata) == 0:  # nothing to do here. return a dum number of frames
            raise ValueError('Metadata is empty.')
    except (json.decoder.JSONDecodeError, ValueError, FileNotFoundError):
            raise Exception('I could not extract the meta data. Set is_extract_timestamp to False in the json_file parameters file if you do not want to execute this step.')
    
    expected_frames = len(video_metadata)
    with tables.File(masked_image_file, 'r+') as mask_fid:
        if '/video_metadata' in mask_fid:
            mask_fid.remove_node('/', 'video_metadata')
        mask_fid.create_table('/', 'video_metadata', obj=video_metadata)

    return expected_frames


def _correct_timestamp(best_effort_timestamp, best_effort_timestamp_time):
    timestamp = best_effort_timestamp.astype(np.int)
    timestamp_time = best_effort_timestamp_time
    
    if len(timestamp) > 1:
        deli = np.diff(best_effort_timestamp)
        good = deli>0
        deli_min = np.min(deli[good])

        if deli_min != 1:
            timestamp = timestamp/deli_min
    
    return timestamp, timestamp_time



def get_timestamp(masked_file):
    '''
    Read the timestamp from the video_metadata, if this field does not exists return an array of nan
    '''
    with tables.File(masked_file, 'r') as mask_fid:
        # get the total number of frmes previously processed
        tot_frames = mask_fid.get_node("/mask").shape[0]
        
        try:        
            # try to read data from video_metadata
            dd = [(row['best_effort_timestamp'], row['best_effort_timestamp_time'])
                  for row in mask_fid.get_node('/video_metadata/')]
            
            if abs(tot_frames - len(dd)) > 2: 
                warnings.warn('The total number of frames is {}, but the timestamp is {}. Either you are using a list of images or there is something weird with the timestamps.'.format(tot_frames, len(dd)))
                raise ValueError
            
            best_effort_timestamp, best_effort_timestamp_time = list(map(np.asarray, zip(*dd)))
            assert best_effort_timestamp.size == best_effort_timestamp_time.size
            
            timestamp, timestamp_time = _correct_timestamp(best_effort_timestamp, best_effort_timestamp_time)
        except (tables.exceptions.NoSuchNodeError, ValueError):
            #no metadata return empty frames
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
        # pad with the same value the missing values
        N = tot_frames - timestamp.size
        timestamp = np.pad(timestamp, (0, N), 'edge')
        timestamp_time = np.pad(timestamp_time, (0, N), 'edge')
        assert tot_frames == timestamp.size

    # save timestamp into the dst_file
    with tables.File(dst_file, 'r+') as dst_fid:
        if '/timestamp' in dst_fid:
            dst_fid.remove_node('/timestamp', recursive=True)

        dst_fid.create_group('/', 'timestamp')
        dst_fid.create_carray('/timestamp', 'raw', obj=np.asarray(timestamp))
        dst_fid.create_carray(
            '/timestamp',
            'time',
            obj=np.asarray(timestamp_time))

    return timestamp, timestamp_time


if __name__ == '__main__':
    masked_file = '/Users/ajaver/Tmp/MaskedVideos/Laura-phase2/tracker 3/Laura-phase2/other labs/06-11-15/N2 con_2016_01_13__12_43_02___3___2.hdf5'
    # video_file = "/Users/ajaver/Desktop/Videos/Check_Align_samples/Videos/npr-13 (tm1504)V on food L_2010_01_25__11_56_02___4___2.avi"
    # masked_file = "/Users/ajaver/Desktop/Videos/Check_Align_samples/MaskedVideos/npr-13 (tm1504)V on food L_2010_01_25__11_56_02___4___2.hdf5"

    # dat = get_ffprobe_metadata(video_file)

    # store_meta_data(video_file, masked_file)
    # import matplotlib.pylab as plt
    # plt.plot(np.diff(dat['best_effort_timestamp']))
