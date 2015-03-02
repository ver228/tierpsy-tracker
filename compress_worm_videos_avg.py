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
import multiprocessing as mp
import Queue
import matplotlib.pylab as plt

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
           '-vcodec', 'rawvideo', '-']
        self.pipe = sp.Popen(command, stdout = sp.PIPE, bufsize = self.tot_pix) #use a buffer size as small as possible, makes things faster
    
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

def getImageROI(image):
    #if it is needed to keep the original image then use "image=getImageROI(np.copy(image))"
    STRUCT_ELEMENT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)) 
    IM_LIMX = image.shape[0]-2
    IM_LIMY = image.shape[1]-2
    MAX_AREA = 5000
    MIN_AREA = 100

    mask = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,61,15)

    [contours, hierarchy]= cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    badIndex = []
#    for ii, contour in enumerate(contours):
#        if np.any(contour==1) or np.any(contour[:,:,0] ==  IM_LIMX)\
#        or np.any(contour[:,:,1] == IM_LIMY):
#            badIndex.append(ii) 
#        else:
#            area = cv2.contourArea(contour)
#            if area<MIN_AREA or area>MAX_AREA:
#                badIndex.append(ii)
#    for ii in badIndex:
#        cv2.drawContours(mask, contours, ii, 0, cv2.cv.CV_FILLED)
    
    
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
    
    
    #image[mask==0] = 0
    return mask
    
def proccessWorker(conn, frame_number, image):
    conn.send({'frame':frame_number, 'image':getImageROI(image)})
    conn.close()



def compressVideo(video_file, masked_image_file, SAVE_FULL_INTERVAL = 5000, MAX_FRAMES = 1e32):
    #Compressed video in "video_file" by selecting ROI and making the rest of 
    #the image zero (creating a large amount of redundant data)
    #the final images are saving in the file given by "masked_image_file" 
    #as hdf5 with gzip compression

    # MAX_N_PROCESSES : number of processes using during image processing, if -1, this value is set to the number of cpu is using
    # SAVE_FULL_INTERVAL :  Full frame is saved every 'SAVE_FULL_INTERVAL' in '/full_data'
    # MAX_FRAMES : maximum number of frames to be analyzed. Set this value to a large value to compress all the video
    
#    if MAX_N_PROCESSES == -1:
#        MAX_N_PROCESSES = mp.cpu_count()

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
    tic_first = time.time()
    tic = tic_first
    
    while frame_number < MAX_FRAMES:
        ret, image = vid.read()
        if ret == 0:
            break
        frame_number += 1;
        
        if frame_number%25 == 0:
            toc = time.time()
            print frame_number, toc-tic
            tic = toc
        
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

#        
#        #mask_dataset[frame_number-1,:,:] = image_dum
#        #mask_dataset[frame_number-1,:,:] = getImageROI(image)
#        
#        parent_conn, child_conn = mp.Pipe();
#        p = mp.Process(target = proccessWorker, args=(parent_conn, frame_number, image));
#        
#        p.start();
#        proc_queue.put((child_conn, p))
#        
#        if proc_queue.qsize() >= MAX_N_PROCESSES:
#            dd = proc_queue.get();
#            data = dd[0].recv() #read pipe
#            dd[1].join() #wait for the proccess to be completed
#            
#            mask_dataset[data['frame']-1,:,:] = data['image'];
            
#    if mask_dataset != frame_number:
#        mask_dataset.resize(frame_number, axis=0); 
#    
#    for x in range(proc_queue.qsize()):
#        dd = proc_queue.get();
#        data = dd[0].recv()
#        dd[1].join()
#        mask_dataset[data['frame']-1,:,:] = data['image'];

    
    
    vid.release() 
    mask_fid.close()
    
    print 'TOTAL TIME: ', time.time()-tic_first 
    
if __name__ == '__main__':
    
    #fileName = '/Volumes/Mrc-pc/GeckoVideo/CaptureTest_90pc_Ch2_16022015_174636.mjpg';
    #maskFile = '/Volumes/ajaver$/GeckoVideo/Compressed/CaptureTest_90pc_Ch2_16022015_174636.hdf5';
    
    #fileName = '/Volumes/H/GeckoVideo/20150218/CaptureTest_90pc_Ch4_18022015_230213.mjpg'
    #maskFile = '/Volumes/ajaver$/GeckoVideo/Compressed/CaptureTest_90pc_Ch4_18022015_230213.hdf5';
    
    #fileName = '/Volumes/H/GeckoVideo/20150218/CaptureTest_90pc_Ch2_18022015_230108.mjpg'
    #maskFile = '/Volumes/ajaver$/GeckoVideo/Compressed/CaptureTest_90pc_Ch2_18022015_230108.hdf5';
    
#    video_file = '/Volumes/behavgenom$/GeckoVideo/20150220/CaptureTest_90pc_Ch3_20022015_183607.mjpg';
#    masked_image_file = '/Volumes/ajaver$/GeckoVideo/Compressed/CaptureTest_90pc_Ch3_20022015_183607.hdf5';

    #fileName = '/Volumes/Mrc-pc/GeckoVideo/CaptureTest_90pc_Ch4_16022015_174636.mjpg';
    #maskFile = '/Volumes/ajaver$/GeckoVideo/Compressed/CaptureTest_90pc_Ch4_16022015_174636.hdf5';
    
    #fileName = r'G:\GeckoVideo\CaptureTest_90pc_Ch4_16022015_174636.mjpg';
    #maskFile = r'Z:\GeckoVideo\Compressed\CaptureTest_90pc_Ch4_16022015_174636.hdf5';
    
#    video_file = '/Volumes/behavgenom$/GeckoVideo/20150221/CaptureTest_90pc_Ch4_21022015_210020.mjpg';
#    masked_image_file = '/Volumes/ajaver$/GeckoVideo/Compressed/aCaptureTest_90pc_Ch4_21022015_210020.hdf5';

#    video_file = '/Volumes/Mrc-pc/Full_Resolution/Capture_Ch3_26022015_161548.mjpg'
#    masked_image_file = '/Users/ajaver/Documents/Test_Andre/Capture_Ch3_26022015_161548.hdf5'
    
#    video_file = '/Volumes/behavgenom$/GeckoVideo/20150223/CaptureTest_90pc_Ch4_23022015_192449.mjpg';
#    #masked_image_file = '/Volumes/ajaver$/GeckoVideo/Compressed/CaptureTest_90pc_Ch4_23022015_192449.hdf5';
#    masked_image_file = '/Volumes/behavgenom$/GeckoVideo/Compressed/20150223/CaptureTest_90pc_Ch4_23022015_192449.hdf5';
 
#    video_file = '/Volumes/behavgenom$/GeckoVideo/20150224/CaptureTest_90pc_Ch2_24022015_222017.mjpg';
#    masked_image_file = '/Volumes/behavgenom$/GeckoVideo/Compressed/20150224/CaptureTest_90pc_Ch2_24022015_222017.hdf5';

#    video_file = '/Volumes/behavgenom$/GeckoVideo/20150224/CaptureTest_90pc_Ch2_24022015_222017.mjpg';
    video_file = '/Volumes/Mrc-pc/20150228/Capture_Ch1_28022015_171254.mjpg'
    masked_image_file = '/Volumes/behavgenom$/GeckoVideo/Compressed/20150228/Capture_Ch1_28022015_171254.hdf5';
    
    compressVideo(video_file, masked_image_file)
#%%
    mask_fid = h5py.File(masked_image_file, "r");   
    plt.figure()
    plt.imshow(mask_fid['/mask'][0,:,:], interpolation = 'none', cmap = 'gray')
    print mask_fid['/full_data'].shape
    mask_fid.close()
    