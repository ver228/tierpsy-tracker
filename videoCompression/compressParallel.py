# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 15:17:44 2015

@author: Avelino
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 15:06:13 2015

@author: ajaver
"""

import os, sys, getopt
import datetime, time

from wormCompression import compressVideo
from writeDownsampledVideo import writeDownsampledVideo
from writeFullFramesTiff import writeFullFramesTiff
from parallelProcHelper import sendQueueOrPrint, parallelizeTask


def videoProcessingWorker(video_dir, save_dir, base_name='', video_ext ='.mjpg', status_queue=''):
    '''
    Worker function used to process several videos in parallel.
    Compress the video into a hdf5, create a downsampled version for visualization, and save the full frames as tiff stacks.
    '''
    initial_time = time.time();
    
    video_file = video_dir + base_name + video_ext
    masked_image_file = save_dir + base_name + '.hdf5'
    masked_image_file = masked_image_file.replace(' ', '_') #spaces are evil get ride of them
    
    try:
        compressVideo(video_file, masked_image_file, useVideoCapture = True, \
        base_name = base_name, status_queue = status_queue, max_frame = 1e32);
    except:
        sendQueueOrPrint(status_queue, 'Video Conversion failed', base_name)
        raise
    
    try:
        writeDownsampledVideo(masked_image_file, base_name = base_name, status_queue = status_queue);
    except:
        sendQueueOrPrint(status_queue, 'Video Downsampling failed', base_name)
        raise
        
    try:
        writeFullFramesTiff(masked_image_file, base_name = base_name, status_queue = status_queue);
    except:
        sendQueueOrPrint(status_queue, 'Tiff writing failed', base_name)
        raise
    
    time_str = str(datetime.timedelta(seconds=round(time.time()-initial_time)))
    progress_str = 'Processing Done. Total time = %s' % time_str;
    sendQueueOrPrint(status_queue, progress_str, base_name)
    

if __name__ == '__main__':    

    '''process in parallel each of the .mjpg files in video_dir and save the output in save_dir'''
    
#    video_dir = '/Volumes/behavgenom$/syngenta/RawData/data_20150114'
#    save_dir = '/Volumes/behavgenom$/syngenta/Compressed/data_20150114_dum'
#    video_ext = 'mpjg'

    video_dir = r'/Users/ajaver/Desktop/sygenta/RawData/data_20150114/'
    save_dir = r'/Users/ajaver/Desktop/sygenta/Compressed/data_20150114/'
    video_ext = 'avi'

    #obtain input from the command line
    print sys.argv[1:]
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:o:e::",["ifile=","ofile=", "ext="])
    except getopt.GetoptError:
        print 'compress_worm_videos.py -i <inputfile> -o <outputfile> -e <videoextension>'
        sys.exit(2)
    
    
    for opt, arg in opts:
        print opt, arg
        if opt == '-h':
            print 'compress_worm_videos.py -i <inputfile> -o <outputfile> -e <videoextension>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            video_dir = arg
        elif opt in ("-o", "--ofile"):
            save_dir = arg
        elif opt in ('--ext'):
            video_ext = arg
    
    if video_ext[0] != '.':
        video_ext = '.' + video_ext
    
    if save_dir[-1] != os.sep:
        save_dir += os.sep

    if video_dir[-1] != os.sep:
        video_dir += os.sep
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    
    #start the parallizeTask object, obtain the queue where the progress status is stored
    task = parallelizeTask(6);
    
    #get a list 
    file_list = os.listdir(video_dir);
    base_name_list = [os.path.splitext(x)[0] for x in file_list if (video_ext in x)]
    
    #get a list of arguments for each child
    workers_arg = {};
    for base_name in base_name_list:
        workers_arg[base_name] = (video_dir, save_dir, base_name, video_ext, task.status_queue)
    
    task.start(videoProcessingWorker, workers_arg)

 

    