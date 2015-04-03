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

import sys, collections
import multiprocessing as mp

import getopt

class readVideoffmpeg:
    '''
    Read video frame using ffmpeg. Assumes 8bits gray video.
    Requires that ffmpeg is installed in the computer.
    This class is an alternative of the captureframe of opencv since:
    -> it can be a pain to compile opencv with ffmpeg compability. 
    -> this funciton is a bit faster (less overhead).
    '''
    def __init__(self, fileName, width = -1, height = -1):
        #requires the fileName, and optionally the frame width and heigth.
        if os.name == 'nt':
            ffmpeg_cmd = 'ffmpeg.exe'
        else:
            ffmpeg_cmd = 'ffmpeg'
        
        #try to open the file and determine the frame size. Raise an exception otherwise.
        if width<=0 or height <=0:
            try:
                command = [ffmpeg_cmd, '-i', fileName, '-']
                pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
                buff = pipe.stderr.read()
                pipe.terminate()
                #the frame size is somewhere printed at the beggining by ffmpeg
                dd = buff.partition('Video: ')[2].split(',')[2]
                dd = re.findall(r'\d*x\d*', dd)[0].split('x')
                self.height = int(dd[0])
                self.width = int(dd[1])
            except:
                print buff
                raise
        else:
            self.width = width
            self.height = height
                
        self.tot_pix = self.height*self.width
        
        command = [ffmpeg_cmd, 
           '-i', fileName,
           '-f', 'image2pipe',
           '-threads', '0',
           '-vcodec', 'rawvideo', '-']
        devnull = open(os.devnull, 'w') #use devnull to avoid printing the ffmpeg command output in the screen
        self.pipe = sp.Popen(command, stdout = sp.PIPE, \
        bufsize = self.tot_pix, stderr=devnull) 
        #use a buffer size as small as possible (frame size), makes things faster
        
    
    def read(self):
        #retrieve an image as numpy array 
        raw_image = self.pipe.stdout.read(self.tot_pix)
        if len(raw_image) < self.tot_pix:
            return (0, []);
        
        image = np.fromstring(raw_image, dtype='uint8')
        #print len(image), self.width, self.height
        image = image.reshape(self.width,self.height)
        return (1, image)
    
    def release(self):
        #close the buffer
        self.pipe.stdout.flush()
        self.pipe.terminate()



def getROIMask(image,  min_area = 100, max_area = 5000):
    '''
    Calculate a binary mask to mark areas where it is possible to find worms.
    Objects with less than min_area or more than max_area pixels are rejected.
    '''
    #Objects that touch the limit of the image are removed. I use -2 because openCV findCountours remove the border pixels
    IM_LIMX = image.shape[0]-2
    IM_LIMY = image.shape[1]-2
    
    #adaptative threshold is the best way to find possible worms. I setup the parameters manually, they seems to work fine if there is no condensation in the sample
    mask = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,61,15)

    #find the contour of the connected objects (much faster than labeled images)
    [contours, hierarchy]= cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #find good contours: between max_area and min_area, and do not touch the image border
    goodIndex = []
    for ii, contour in enumerate(contours):
        if np.all(contour!=1) and np.all(contour[:,:,0] !=  IM_LIMX)\
        and np.all(contour[:,:,1] != IM_LIMY):
            area = cv2.contourArea(contour)
            if (area>=min_area) and (area<=max_area):
                goodIndex.append(ii)
    
    #typically there are more bad contours therefore it is cheaper to draw only the valid contours
    mask = np.zeros(image.shape, dtype=image.dtype)
    for ii in goodIndex:
        cv2.drawContours(mask, contours, ii, 255, cv2.cv.CV_FILLED)
    
    #drawContours left an extra line if the blob touches the border. It is necessary to remove it
    mask[0,:] = 0; mask[:,0] = 0; mask[-1,:] = 0; mask[:,-1]=0;
    
    #dilate the elements to increase the ROI, in case we are missing something important
    struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)) 
    mask = cv2.dilate(mask, struct_element, iterations = 3)
    
    #the gecko images have a time stamp in the image border
    cv2.rectangle(mask, (0,0), (479,15), 255, thickness=-1) 

    return mask
   
def compressVideo(video_file, masked_image_file, buffer_size = 25, \
save_full_interval = 5000, max_frame = 1e32, base_name = '', \
status_queue = '', check_empty = True,  useVideoCapture = True):
   
    '''
    Compressed video in "video_file" by selecting ROI and making the rest of 
    the image zero (creating a large amount of redundant data)
    the final images are saving in the file given by "masked_image_file" 
    as hdf5 with gzip compression
    To reduce the processing load buffer_size images are collected, and a minimum filter 
    applied over the stack. In this way the pixels corresponding to the worms are 
    preserved as black pixels in the min-average image, and only in this 
     image the the binary mask with possible worms is calculated.

     MAX_N_PROCESSES -- number of processes using during image processing, if -1, this value is set to the number of cpu is using
     save_full_interval --  Full frame is saved every 'save_full_interval' in '/full_data'
     max_frame -- maximum number of frames to be analyzed. Set this value to a large value to compress all the video    
     status_queue -- queue were the status is sended. Only used in multiprocessing case 
     base_name -- processes identifier. Only used in the multiprocessing case.
    '''
    
    #max_frame = 1000
    #open video to read
    if not useVideoCapture:
        vid = readVideoffmpeg(video_file);
        im_height = vid.height;
        im_width = vid.width;
    else:
        vid = cv2.VideoCapture(video_file);
        im_width= vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        im_height= vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    
    if check_empty:
        rand_ind = np.random.randint(0, im_height*im_width-1, (200)); #select 200 random index use to check if the image is empty
    
    
    #open hdf5 to store the processed data
    mask_fid = h5py.File(masked_image_file, "w");
    #open node to store the compressed (masked) data
    mask_dataset = mask_fid.create_dataset("/mask", (0, im_width, im_height), 
                                    dtype = "u1", maxshape = (None, im_width, im_height), 
                                    chunks = (1, im_width, im_height),
                                    compression="gzip", 
                                    compression_opts=4,
                                    shuffle=True);
    
    #full frames are saved in "/full_data" every save_full_interval frames
    full_dataset = mask_fid.create_dataset("/full_data", (0, im_width, im_height), 
                                    dtype = "u1", maxshape = (None, im_width, im_height), 
                                    chunks = (1, im_width, im_height),
                                    compression="gzip", 
                                    compression_opts=9,
                                    shuffle=True);
    full_dataset.attrs['save_interval'] = save_full_interval

    
    #intialize frame number
    frame_number = 0;
    full_frame_number = 0;
    
    #initialize timers
    initial_time = fps_time = time.time()
    last_frame = 0;    
    
    while frame_number < max_frame:
        ret, image = vid.read() #get video frame, stop program when no frame is retrive (end of file)
        if ret == 0:
            break
        
        if useVideoCapture:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).T
            
        #check if the 200 randomly selected index are equal (much faster that check the whole image)
        if check_empty and np.all(image.flat[rand_ind]-np.mean(image.flat[rand_ind])==0):
            continue
        
        frame_number += 1;
        
        #if frame_number%skip_n_frames+1 != 1:
        #    continue
        
        #import matplotlib.pylab as plt
        #plt.figure()
        #plt.imshow(image)
        
        #Resize mask array every 1000 frames (doing this every frame does not impact much the performance)
        if (frame_number)%1000 == 1:
            mask_dataset.resize(frame_number + 1000, axis=0); 

        #Add a full frame every save_full_interval
        if frame_number % save_full_interval== 1:
            
            full_dataset.resize(full_frame_number+1, axis=0); 
            assert(np.floor(frame_number/save_full_interval) == full_frame_number) #just to be sure that the index we are saving in is what we what we are expecting
            full_dataset[full_frame_number,:,:] = image
            full_frame_number += 1;

        
        ind_buff = (frame_number-1) % buffer_size #buffer index
        
        #initialize the buffer when the index correspond to 0
        if ind_buff == 0:
            Ibuff = np.zeros((buffer_size, im_width, im_height), dtype = np.uint8)

        #add image to the buffer
        Ibuff[ind_buff, :, :] = image
        
        if ind_buff == buffer_size-1:
            #calculate the mask only when the buffer is full
            mask = getROIMask(np.min(Ibuff, axis=0))
            
            #mask all the images in the buffer
            for ii in range(Ibuff.shape[0]):
                #create a reference copy
                im = Ibuff[ii,:,:]; 
                #bitwise_and by reference (keep values having 255 in the mask)
                cv2.bitwise_and(im,mask, im); 

            #add buffer to the hdf5 file
            mask_dataset[(frame_number-buffer_size):frame_number,:,:] = Ibuff
        
        
        if frame_number%25 == 0:
            #calculate the progress and put it in a string
            time_str = str(datetime.timedelta(seconds=round(time.time()-initial_time)))
            fps = (frame_number-last_frame+1)/(time.time()-fps_time)
            progress_str = 'Compressing video. Total time = %s, fps = %2.1f; Frame %i '\
                % (time_str, fps, frame_number)
            
            sendQueueOrPrint(status_queue, progress_str, base_name);
            
            fps_time = time.time()
            last_frame = frame_number;
    
    if mask_dataset.shape[0] != frame_number:
        mask_dataset.resize(frame_number, axis=0);
    if full_dataset.shape[0] != full_frame_number:
        full_dataset.resize(full_frame_number, axis=0);
    #close the video and hdf5 files
    vid.release() 
    mask_fid.close()
    sendQueueOrPrint(status_queue, 'Compressed video done.', base_name);

    
    
def writeDownsampledVideo(masked_image_file, base_name = '', status_queue = '', save_video_file = -1, 
                          final_frame_size = 0, n_frames_jumped = 25):
    '''
    Write a downsampled video for visualization purposes from the processed hdf5 file. 
    The downsampling is both in scale (final_frame_size), and in time (n_frames_jumped)
    This function requires ffmpeg
    '''
                     
    #if no save_video_file is given, the save name will be derived from masked_image_file                         
    if save_video_file == -1:
        save_video_file = os.path.splitext(masked_image_file)[0] +  '_downsampled.avi';
    
    #open the hdf5 with masked data
    mask_fid = h5py.File(masked_image_file, "r");
    I_worms = mask_fid["/mask"]
    
    if final_frame_size == 0:
        final_frame_size = (I_worms.shape[-1]/4, I_worms.shape[-2]/4)
    
    #parameters to minimize the video using ffmpeg
    command = [ 'ffmpeg',
            '-y', # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-s', '%ix%i' % (I_worms.shape[-1], I_worms.shape[-2]), # size of one frame
            '-pix_fmt', 'gray',
            '-r', '25', # frames per second
            '-i', '-', # The imput comes from a pipe
            '-an', # Tells FFMPEG not to expect any audio
            '-vcodec', 'mjpeg',
            '-vf', 'scale=%i:%i' % final_frame_size,
            '-threads', '0',
            '-qscale:v', '0',
            save_video_file]
    
    
    devnull = open(os.devnull, 'w')  #use devnull to avoid printing the ffmpeg command output in the screen
    pipe = sp.Popen(command, stdin=sp.PIPE, stderr=devnull)
    
    #total number of frames.
    tot_frames = float(I_worms.shape[0])
    initial_time = fps_time = time.time()
    last_frame = 0;
    
    
    for frame_number in range(0,I_worms.shape[0],n_frames_jumped):
        pipe.stdin.write(I_worms[frame_number,:,:].tostring()) #write frame 
        
        #calculate progress
        if frame_number%1000 == 0:
            time_str = str(datetime.timedelta(seconds=round(time.time()-initial_time)))
            fps = (frame_number-last_frame+1)/(time.time()-fps_time)
            progress_str = 'Downsampling video. Total time = %s, fps = %2.1f; %3.2f%% '\
                % (time_str, fps, frame_number/tot_frames*100)
            
            sendQueueOrPrint(status_queue, progress_str, base_name);
            fps_time = time.time()
            last_frame = frame_number;

    #close files
    pipe.terminate()
    mask_fid.close()
    sendQueueOrPrint(status_queue, 'Downsampled video done.', base_name);

def writeFullFramesTiff(masked_image_file, tiff_file = -1, reduce_fractor = 8, base_name = '', status_queue = ''):
    '''
    write scale down the saved full frames, and put them into a tiff stack.
    requires either install skimage with the freeimage library plugin, or the tifffile module.
    '''
    #if no tiff_file is given, the save name will be derived from masked_image_file   
    if tiff_file == -1:
        tiff_file = os.path.splitext(masked_image_file)[0] + '_full.tiff';
    
    
    mask_fid = h5py.File(masked_image_file, "r");

    #determine the expected size of the full_data size with respect to the 
    #number of frames in full_data and the same interval (I only do that 
    #to correct for a bug in previously saved files)
    expected_size = int(np.floor(mask_fid["/mask"].shape[0]/float(mask_fid["/full_data"].attrs['save_interval']) + 1));
    if expected_size > mask_fid["/full_data"].shape[0]: 
        expected_size = mask_fid["/full_data"].shape[0]
    
    #initialized reduced array  
    im_size = tuple(np.array(mask_fid["/full_data"].shape)[1:]/reduce_fractor)
    reduce_factor = (im_size[-1], im_size[-2])
    I_worms = np.zeros((expected_size, im_size[0],im_size[1]), dtype = np.uint8)
    
    sendQueueOrPrint(status_queue, 'Reading for data the tiff file...', base_name);
    
    for frame_number in range(expected_size):
        I_worms[frame_number, :,:] = cv2.resize(mask_fid["/full_data"][frame_number,:,:], reduce_factor);
    
    sendQueueOrPrint(status_queue, 'Writing tiff file...', base_name);
    
    try: 
        #Requires the installation of freeimage library. 
        #On mac is trivial using brew (brew install freeimage), 
        #but i am not sure how to do it on windows
        from skimage.io._plugins import freeimage_plugin as fi
        fi.write_multipage(I_worms, tiff_file, fi.IO_FLAGS.TIFF_LZW) #the best way I found to write lzw compression on python
    
    except:
        import tifffile #pip intall tifffile
        #For some reason gzip compression appears as an inverted image in 
        #preview (both windows and mac), but it is read correctly in ImageJ
        tifffile.imsave(tiff_file, I_worms, compress=4) 
    
    sendQueueOrPrint(status_queue, 'Tiff file done.', base_name);
    

def videoProcessingWorker(video_dir, save_dir, base_name='', video_ext ='.mjpg', status_queue=''):
    '''
    Worker function used to process several videos in parallel.
    Compress the video into a hdf5, create a downsampled version for visualization, and save the full frames as tiff stacks.
    '''
    initial_time = time.time();
    
    video_file = video_dir + base_name + video_ext
    masked_image_file = save_dir + base_name + '.hdf5'
    
    try:
        #pass
        compressVideo(video_file, masked_image_file, base_name = base_name, status_queue = status_queue);
    except:
        sendQueueOrPrint(status_queue, 'Video Conversion failed', base_name)
        raise
    
    try:
        writeDownsampledVideo(masked_image_file, base_name = base_name, status_queue = status_queue);
    except:
        sendQueueOrPrint(status_queue, 'Video Downsampling failed', base_name)
        #status_queue.put([base_name, 'Video Downsampling failed'])
        raise
        
    try:
        writeFullFramesTiff(masked_image_file, base_name = base_name, status_queue = status_queue);
    except:
        sendQueueOrPrint(status_queue, 'Tiff writing failed', base_name)
        #status_queue.put([base_name, 'Tiff writing failed'])
        raise
    
    time_str = str(datetime.timedelta(seconds=round(time.time()-initial_time)))
    progress_str = 'Processing Done. Total time = %s' % time_str;
    status_queue.put([base_name, progress_str])

def sendQueueOrPrint(status_queue, progress_str, base_name):
    '''small code to decide if the progress must be send to a queue or printed in the screen'''
    if type(status_queue).__name__ == 'Queue':
        status_queue.put([base_name, progress_str]) 
    else:
        print (progress_str) 

def printProgress(progress):
    '''useful function to write the progress status in multiprocesses mode'''
    sys.stdout.write('\033[2J')
    sys.stdout.write('\033[H')
    for filename, progress_str in progress.items():
        print filename, progress_str

    sys.stdout.flush()

if __name__ == '__main__':    

    '''process in parallel each of the .mjpg files in video_dir and save the output in save_dir'''
    
    video_dir = '/Users/ajaver/Desktop/Gecko_compressed/pruebaa/' #'/Volumes/Mrc-pc/20150309/'
    save_dir = '/Users/ajaver/Desktop/Gecko_compressed/pruebaa/' #'/Volumes/behavgenom$/GeckoVideo/Compressed/20150309/'
#    video_dir = '/Volumes/behavgenom$/Camille Recordings/test1/'
#    save_dir = '/Volumes/behavgenom$/Camille Recordings/test1/'
#    
    video_ext = '.mjpg'#'.mjpg'
    
    #obtain input from the command line
    try:
        
        opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile=", "ext="])
    except getopt.GetoptError:
        print 'compress_worm_videos.py -i <inputfile> -o <outputfile>'
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
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    workers = []
    status_queue = mp.Queue() #queue where progress status of each process if save
    progress = collections.OrderedDict()

    #get a list 
    file_list = os.listdir(video_dir);
    base_name_list = [os.path.splitext(x)[0] for x in file_list if (video_ext in x)]
    
    max_num_process = 6;
    for nChunk in range(0, len(base_name_list), max_num_process):  
    
        for ii in range(max_num_process):
            if nChunk + ii < len(base_name_list):
                base_name = base_name_list[nChunk + ii]
                #start a process for each .mjpeg file in save_dir
                child = mp.Process(target = videoProcessingWorker, args=(video_dir, save_dir, base_name, video_ext, status_queue))
                child.start()
                workers.append(child)
                progress[base_name] = 'idle'
        
    
        #update the progress status as long as there is a worker alive
        while any(i.is_alive() for i in workers):
            time.sleep(10.0) # I made this value larger because I can only save the output in a file on a schedule task with "at". Refreshing it too frequently will produce a huge file.
            while not status_queue.empty():
                filename, percent = status_queue.get()
                progress[filename] = percent
                printProgress(progress)

    