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
import numpy as np
import cv2
import subprocess as sp
import time
import os
import re
import h5py
import datetime
import matplotlib.pylab as plt
from skimage.io._plugins import freeimage_plugin as fi
import sys, collections
import multiprocessing as mp

#import skimage.m
class ReadVideoffmpeg:
    def __init__(self, fileName, width = -1, height = -1):
        if os.name == 'nt':
            ffmpeg_cmd = 'ffmpeg.exe'
        else:
            ffmpeg_cmd = 'ffmpeg'
        
        
        if width<=0 or height <=0:
            try:
                command = [ffmpeg_cmd, '-i', fileName, '-']
                pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
                buff = pipe.stderr.read()
                pipe.terminate()
                dd = buff.partition('Video: ')[2].split(',')[2]
                dd = re.findall(r'\d*x\d*', dd)[0].split('x')
                self.width = int(dd[0])
                self.height = int(dd[1])
                
            except:
                print 'I could not determine the frame size from ffmpeg, introduce the values manually'
                print buff
                raise
        else:
            self.width = width
            self.height = height
                
        self.tot_pix = self.width*self.width
        
        command = [ffmpeg_cmd, 
           '-i', fileName,
           '-f', 'image2pipe',
           '-threads', '0',
           '-vcodec', 'rawvideo', '-']
        devnull = open(os.devnull, 'w')
        self.pipe = sp.Popen(command, stdout = sp.PIPE, \
        bufsize = self.tot_pix, stderr=devnull) #use a buffer size as small as possible, makes things faster
    
    def read(self):
        raw_image = self.pipe.stdout.read(self.tot_pix)
        if len(raw_image) < self.tot_pix:
            return (0, []);
        
        image = np.fromstring(raw_image, dtype='uint8')
        image = image.reshape(self.width,self.height)
        return (1, image)
    
    def release(self):
        self.pipe.stdout.flush()
        self.pipe.terminate()

def writeDownsampledVideo(masked_image_file, base_name = '', save_video_file = -1, 
                          final_frame_size = (512, 512), n_frames_jumped = 25):
    if save_video_file == -1:
        save_video_file = os.path.splitext(masked_image_file)[0] +  '_downsampled.avi';
        
    mask_fid = h5py.File(masked_image_file, "r");
    I_worms = mask_fid["/mask"]
        
    command = [ 'ffmpeg',
            '-y', # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-s', '%ix%i' % I_worms.shape[1:3], # size of one frame
            '-pix_fmt', 'gray',
            '-r', '25', # frames per second
            '-i', '-', # The imput comes from a pipe
            '-an', # Tells FFMPEG not to expect any audio
            '-vcodec', 'mjpeg',
            '-vf', 'scale=%i:%i' % final_frame_size,
            '-threads', '0',
            '-qscale:v', '0',
            save_video_file]
    
    devnull = open(os.devnull, 'w')
    pipe = sp.Popen(command, stdin=sp.PIPE, stderr=devnull)
    
    tot_frames = float(I_worms.shape[0])
    
    initial_time = fps_time = time.time()
    last_frame = 0;
    for frame_number in range(0,I_worms.shape[0],n_frames_jumped):
        pipe.stdin.write(I_worms[frame_number,:,:].tostring() )
        
        if frame_number%1000 == 0:
            time_str = str(datetime.timedelta(seconds=round(time.time()-initial_time)))
            fps = (frame_number-last_frame+1)/(time.time()-fps_time)
            progress_str = 'Downsampling video. Total time = %s, fps = %2.1f; %3.2f%% '\
                % (time_str, fps, frame_number/tot_frames*100)
            status.put([base_name, progress_str]) 
            
            fps_time = time.time()
            last_frame = frame_number;

    status.put([base_name, 'Downsampled video done.']) 
    pipe.terminate()
    mask_fid.close()

def writeFullFramesTiff(masked_image_file, tiff_file = -1, reduce_fractor = 8, base_name = ''):
    if tiff_file == -1:
        tiff_file = os.path.splitext(masked_image_file)[0] + '_full.tiff';
    
    mask_fid = h5py.File(masked_image_file, "r");

    expected_size = int(np.floor(mask_fid["/mask"].shape[0]/float(mask_fid["/full_data"].attrs['save_interval']) + 1));
    if expected_size > mask_fid["/full_data"].shape[0]: 
        expected_size = mask_fid["/full_data"].shape[0]
    
    im_size = tuple(np.array(mask_fid["/full_data"].shape[1:])/reduce_fractor)
    
    I_worms = np.zeros((expected_size, im_size[0],im_size[1]), dtype = np.uint8)
    
    
    status.put([base_name, 'Reading for data the tiff file...']) 
    for frame_number in range(expected_size):
        I_worms[frame_number, :,:] = cv2.resize(mask_fid["/full_data"][frame_number,:,:], im_size);
    
    status.put([base_name, 'Writing tiff file...']) 
    fi.write_multipage(I_worms, tiff_file, fi.IO_FLAGS.TIFF_LZW)
    
    status.put([base_name, 'Tiff file done.']) 

def getImageROI(image):
    #if it is needed to keep the original image then use "image=getImageROI(np.copy(image))"
    STRUCT_ELEMENT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)) 
    IM_LIMX = image.shape[0]-2
    IM_LIMY = image.shape[1]-2
    MAX_AREA = 5000
    MIN_AREA = 100

    mask = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,61,15)

    [contours, hierarchy]= cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    goodIndex = []
    for ii, contour in enumerate(contours):
        if np.all(contour!=1) and np.all(contour[:,:,0] !=  IM_LIMX)\
        and np.all(contour[:,:,1] != IM_LIMY):
            area = cv2.contourArea(contour)
            if (area>=MIN_AREA) and (area<=MAX_AREA):
                goodIndex.append(ii)
    
    #typically there are more bad contours therefore it is cheaper to draw only the valid contours
    mask = np.zeros(image.shape, dtype=image.dtype)
    for ii in goodIndex:
        cv2.drawContours(mask, contours, ii, 255, cv2.cv.CV_FILLED)
        
    mask[0,:] = 0; mask[:,0] = 0; mask[-1,:] = 0; mask[:,-1]=0;
    mask = cv2.dilate(mask, STRUCT_ELEMENT, iterations = 3)
    cv2.rectangle(mask, (0,0), (479,15), 255, thickness=-1) 

    return mask
   

def compressVideo(video_file, masked_image_file, SAVE_FULL_INTERVAL = 5000, MAX_FRAMES = 1e32, base_name = ''):
    #Compressed video in "video_file" by selecting ROI and making the rest of 
    #the image zero (creating a large amount of redundant data)
    #the final images are saving in the file given by "masked_image_file" 
    #as hdf5 with gzip compression

    # MAX_N_PROCESSES : number of processes using during image processing, if -1, this value is set to the number of cpu is using
    # SAVE_FULL_INTERVAL :  Full frame is saved every 'SAVE_FULL_INTERVAL' in '/full_data'
    # MAX_FRAMES : maximum number of frames to be analyzed. Set this value to a large value to compress all the video
    
    vid = ReadVideoffmpeg(video_file);
    im_height = vid.height;
    im_width = vid.width;
    
    mask_fid = h5py.File(masked_image_file, "w");
    #maked images are save in the dataset /mask
    mask_dataset = mask_fid.create_dataset("/mask", (0, im_height, im_width), 
                                    dtype = "u1", maxshape = (None, im_height, im_width), 
                                    chunks = (1, im_height, im_height),#chunks = (1, im_height, im_width), 
                                    compression="gzip", 
                                    compression_opts=4,
                                    shuffle=True);
    
    #full frames are saved in "/full_data" every SAVE_FULL_INTERVAL frames
    full_dataset = mask_fid.create_dataset("/full_data", (0, im_height, im_width), 
                                    dtype = "u1", maxshape = (None, im_height, im_width), 
                                    chunks = (1, im_height, im_height),#chunks = (1, im_height, im_width), 
                                    compression="gzip", 
                                    compression_opts=9,
                                    shuffle=True);
    full_dataset.attrs['save_interval'] = SAVE_FULL_INTERVAL

    #proc_queue = Queue.Queue()
    frame_number = 0;
    full_frame_number = 0;
    
    initial_time = fps_time = time.time()
    last_frame = 0;    
    while frame_number < MAX_FRAMES:
        ret, image = vid.read()
        if ret == 0:
            break
        frame_number += 1;
        
        #Resize mask array every 1000 frames
        if (frame_number)%1000 == 1:
            mask_dataset.resize(frame_number + 1000, axis=0); 

        #Add a full frame every SAVE_FULL_INTERVAL
        if frame_number % SAVE_FULL_INTERVAL== 1:
            
            full_dataset.resize(full_frame_number+1, axis=0); 
            assert(np.floor(frame_number/SAVE_FULL_INTERVAL) == full_frame_number)
            
            full_dataset[full_frame_number,:,:] = image
            full_frame_number += 1;

        N_BUFFER = 25;
        #collect every N_BUFFER
        if frame_number % N_BUFFER == 1:
            Ibuff = np.zeros((N_BUFFER, vid.width, vid.height), dtype = np.uint8)
        
        ind_buff = (frame_number-1) % N_BUFFER
        Ibuff[ind_buff, :, :] = image
        
        if ind_buff == N_BUFFER-1:
            
            mask = getImageROI(np.min(Ibuff, axis=0))
            
            for ii in range(Ibuff.shape[0]):
                #create a reference copy
                im = Ibuff[ii,:,:]; 
                #bitwise_and by reference (keep values having 255 in the mask)
                cv2.bitwise_and(im,mask, im); 
            

            #add buffer to the hdf5 file
            mask_dataset[(frame_number-N_BUFFER):frame_number,:,:] = Ibuff
        
    
        if frame_number%25 == 0:
            time_str = str(datetime.timedelta(seconds=round(time.time()-initial_time)))
            fps = (frame_number-last_frame+1)/(time.time()-fps_time)
            progress_str = 'Compressing video. Total time = %s, fps = %2.1f; Frame %i '\
                % (time_str, fps, frame_number)
                
            status.put([base_name, progress_str]) 
            
            fps_time = time.time()
            last_frame = frame_number;

    vid.release() 
    mask_fid.close()
    status.put([base_name, 'Compressed video done.'])


def print_progress(progress):
    sys.stdout.write('\033[2J')
    sys.stdout.write('\033[H')
    for filename, progress_str in progress.items():
        print filename, progress_str

    sys.stdout.flush()


def video_process(video_dir, save_dir, base_name):
    initial_time = time.time();
    
    video_file = video_dir + base_name + '.mjpg'
    masked_image_file = save_dir + base_name + '.hdf5'
    
    try:
        compressVideo(video_file, masked_image_file, base_name = base_name);
    except:
        status.put([base_name, 'Video Conversion failed'])
        raise

    
    try:
        writeDownsampledVideo(masked_image_file, base_name = base_name);
    except:
        status.put([base_name, 'Video Downsampling failed'])
        raise
        
    try:
        writeFullFramesTiff(masked_image_file, base_name = base_name);
    except:
        status.put([base_name, 'Tiff writing failed'])
        raise
    
    time_str = str(datetime.timedelta(seconds=round(time.time()-initial_time)))
    progress_str = 'Processing Done. Total time = %s' % time_str;
    status.put([base_name, progress_str])


if __name__ == '__main__':    
    video_dir = '/Volumes/Mrc-pc/20150309/'
    save_dir = '/Volumes/behavgenom$/GeckoVideo/Compressed/20150309/'
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    workers = []
    status = mp.Queue()
    progress = collections.OrderedDict()

    file_list = os.listdir(video_dir);
    base_name_list = [os.path.splitext(x)[0] for x in file_list if ('mjpg' in x)]
    
    for base_name in base_name_list:
        child = mp.Process(target=video_process, args=(video_dir, save_dir, base_name))
        child.start()
        workers.append(child)
        progress[base_name] = ''
    
    while any(i.is_alive() for i in workers):
        time.sleep(0.2)
        while not status.empty():
            filename, percent = status.get()
            progress[filename] = percent
            #print progress
            print_progress(progress)

    