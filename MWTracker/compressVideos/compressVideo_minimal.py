# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:19:58 2015

@author: ajaver
"""
import numpy as np
import cv2
import h5py
import os
import sys
import time
import datetime

#class used to calculate time progress
class timeCounterStr:
    def __init__(self, task_str):
        self.initial_time = time.time();
        self.last_frame = 0;
        self.task_str = task_str;
        self.fps_time = float('nan');
        
    def getStr(self, frame_number):
        #calculate the progress and put it in a string
        time_str = str(datetime.timedelta(seconds = round(time.time()-self.initial_time)))
        fps = (frame_number-self.last_frame+1)/(time.time()-self.fps_time)
        progress_str = '%s Total time = %s, fps = %2.1f; Frame %i '\
            % (self.task_str, time_str, fps, frame_number)
        self.fps_time = time.time()
        self.last_frame = frame_number;
        return progress_str;
    
    def getTimeStr(self):
        return  str(datetime.timedelta(seconds = round(time.time()-self.initial_time)))

DEFAULT_MASK_PARAM = {'min_area':50, 'max_area':1e10, 'has_timestamp':True, 
'thresh_block_size':61, 'thresh_C':15, 'dilation_size': 9, 'keep_border_data': False}

def getROIMask(image,  min_area = DEFAULT_MASK_PARAM['min_area'], max_area = DEFAULT_MASK_PARAM['max_area'], 
    has_timestamp = DEFAULT_MASK_PARAM['has_timestamp'], thresh_block_size = DEFAULT_MASK_PARAM['thresh_block_size'], 
    thresh_C = DEFAULT_MASK_PARAM['thresh_C'], dilation_size = DEFAULT_MASK_PARAM['dilation_size'], 
    keep_border_data = DEFAULT_MASK_PARAM['keep_border_data']):
    '''
    Calculate a binary mask to mark areas where it is possible to find worms.
    Objects with less than min_area or more than max_area pixels are rejected.
    '''

    #Objects that touch the limit of the image are removed. I use -2 because openCV findCountours remove the border pixels
    IM_LIMX = image.shape[0]-2
    IM_LIMY = image.shape[1]-2
    
    if thresh_block_size%2==0:
        thresh_block_size+=1 #this value must be odd
    
    #adaptative threshold is the best way to find possible worms. I setup the parameters manually, they seems to work fine if there is no condensation in the sample
    mask = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, thresh_block_size, thresh_C)

    #find the contour of the connected objects (much faster than labeled images)
    _, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    #find good contours: between max_area and min_area, and do not touch the image border
    goodIndex = []
    for ii, contour in enumerate(contours):

        if not keep_border_data:
            #eliminate blobs that touch a border
            keep = not np.any(contour ==1) and \
            not np.any(contour[:,:,0] ==  IM_LIMY)\
            and not np.any(contour[:,:,1] == IM_LIMX)
        else:
            keep = True

        if keep:
            area = cv2.contourArea(contour)
            if (area >= min_area) and (area <= max_area):
                goodIndex.append(ii)
    
    #typically there are more bad contours therefore it is cheaper to draw only the valid contours
    mask = np.zeros(image.shape, dtype=image.dtype)
    for ii in goodIndex:
        cv2.drawContours(mask, contours, ii, 1, cv2.FILLED)
    
    #drawContours left an extra line if the blob touches the border. It is necessary to remove it
    mask[0,:] = 0; mask[:,0] = 0; mask[-1,:] = 0; mask[:,-1]=0;
    
    #dilate the elements to increase the ROI, in case we are missing something important
    struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size,dilation_size)) 
    mask = cv2.dilate(mask, struct_element, iterations = 3)
    
    if has_timestamp:
        #the gecko images have a time stamp in the image border
        cv2.rectangle(mask, (0,0), (479,15), 1, thickness=-1) 

    return mask


def compressVideo(video_file, masked_image_file, buffer_size = 25, \
save_full_interval = 5000, max_frame = 1e32, mask_param = DEFAULT_MASK_PARAM):

    '''
    Compresses video by selecting pixels that are likely to have worms on it and making the rest of 
    the image zero. By creating a large amount of redundant data, any lossless compression
    algorithm will dramatically increase its efficiency. The masked images are saved as hdf5 with gzip compression.
    The mask is calculated over a minimum projection of an image stack. This projection preserve darker regions
    where the worm has more probability to be located. Additionally it has the advantage of reducing 
    the processing load by only requiring to calculate the mask once per image stack.
     video_file --  original video file
     masked_image_file -- 
     buffer_size -- size of the image stack used to calculate the minimal projection and the mask
     save_full_interval -- have often a full image is saved
     max_frame -- last frame saved (default a very large number, so it goes until the end of the video)
     mask_param -- parameters used to calculate the mask:
        > min_area -- minimum blob area to be considered in the mask
        > max_area -- max blob area to be considered in the mask
        > thresh_C -- threshold used by openCV adaptiveThreshold
        > thresh_block_size -- block size used by openCV adaptiveThreshold
        > has_timestamp -- (bool) indicates if the timestamp stamp in Gecko images is going to be conserved
        > dilation_size -- size of the structure element to dilate the mask
        > keep_border_data -- (bool) if false it will reject any blob that touches the image border 
    '''

    expected_frames = 10000

    #processes identifier.
    base_name = masked_image_file.rpartition('.')[0].rpartition(os.sep)[-1]

    #use opencv VideoCapture to read video
    vid = cv2.VideoCapture(video_file);
    im_width= vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    im_height= vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    if im_width == 0 or im_height == 0:
        raise(RuntimeError('Cannot read the video file correctly. Dimensions w=%i h=%i' % (im_width, im_height)))

    #open hdf5 to store the processed data
    with h5py.File(masked_image_file, "w") as mask_fid:
        #open node to store the compressed (masked) data
        mask_dataset = mask_fid.create_dataset("/mask", (expected_frames, im_height,im_width), 
                                        dtype = "u1", maxshape = (None, im_height,im_width), 
                                        chunks = (1, im_height,im_width),
                                        compression="gzip", 
                                        compression_opts=4,
                                        shuffle=True, fletcher32=True);

        #attribute labels to make the group compatible with the standard image definition in hdf5
        mask_dataset.attrs["CLASS"] = np.string_("IMAGE")
        mask_dataset.attrs["IMAGE_SUBCLASS"] = np.string_("IMAGE_GRAYSCALE")
        mask_dataset.attrs["IMAGE_WHITE_IS_ZERO"] = np.array(0, dtype="uint8")
        mask_dataset.attrs["DISPLAY_ORIGIN"] = np.string_("UL") # not rotated
        mask_dataset.attrs["IMAGE_VERSION"] = np.string_("1.2")

        #flag to store the parameters using in the mask calculation
        for key in DEFAULT_MASK_PARAM:
            if key in mask_param:
                mask_dataset.attrs[key] = int(mask_param[key])
            else:
                mask_dataset.attrs[key] = int(DEFAULT_MASK_PARAM[key])
        
        #flag to indicate that the conversion finished succesfully
        mask_dataset.attrs['has_finished'] = 0

        #full frames are saved in "/full_data" every save_full_interval frames
        full_dataset = mask_fid.create_dataset("/full_data", (expected_frames//save_full_interval, im_height,im_width), 
                                        dtype = "u1", maxshape = (None, im_height,im_width), 
                                        chunks = (1, im_height,im_width),
                                        compression="gzip", 
                                        compression_opts=4,
                                        shuffle=True, fletcher32=True);
        full_dataset.attrs['save_interval'] = save_full_interval
        
        #attribute labels to make the group compatible with the standard image definition in hdf5
        full_dataset.attrs["CLASS"] = np.string_("IMAGE")
        full_dataset.attrs["IMAGE_SUBCLASS"] = np.string_("IMAGE_GRAYSCALE")
        full_dataset.attrs["IMAGE_WHITE_IS_ZERO"] = np.array(0, dtype="uint8")
        full_dataset.attrs["DISPLAY_ORIGIN"] = np.string_("UL") # not rotated
        full_dataset.attrs["IMAGE_VERSION"] = np.string_("1.2")

        #intialize frame number
        frame_number = 0;
        full_frame_number = 0;
        image_prev = np.zeros([]);
        
        vid_frame_pos = []
        vid_time_pos = []

        #initialize timers
        progressTime = timeCounterStr('Compressing video.');

        while frame_number < max_frame:
            #get the current frame timestamp (opencv might not give the correct value, it is better to use ffprobe)
            vid_frame_pos.append(int(vid.get(cv2.CAP_PROP_POS_FRAMES)))
            vid_time_pos.append(vid.get(cv2.CAP_PROP_POS_MSEC))

            ret, image = vid.read() #get video frame, stop program when no frame is retrive (end of file)
            if ret == 0:
                break
            
            #opencv can give an artificial rgb image. Let's get it back to gray scale.
            if image.ndim==3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            #increase frame number
            frame_number += 1;
            
            #Increase the size of the hdf5 masked array every 1000 frames (hdf5 resizing does does not impact much the performance, probably i could get away with doing it every frame)
            if mask_dataset.shape[0] <= frame_number + 1: 
                mask_dataset.resize(frame_number + 1000, axis=0);

            #Add a full frame every save_full_interval
            if frame_number % save_full_interval == 1:
                if full_dataset.shape[0] <= full_frame_number:
                    full_dataset.resize(full_frame_number+1, axis=0); 
                    assert(frame_number//save_full_interval == full_frame_number) #just to be sure that the index we are saving in is what we what we are expecting
                full_dataset[full_frame_number,:,:] = image.copy()
                full_frame_number += 1;

            
            #calculate the buffer index from the frame_number
            ind_buff = (frame_number-1) % buffer_size 
            
            #clear (initialize) the buffer when the index correspond to 0
            if ind_buff == 0:
                Ibuff = np.zeros((buffer_size, im_height, im_width), dtype = np.uint8)

            #add image to the buffer
            Ibuff[ind_buff, :, :] = image.copy()
            
            #HERE IS THE MASK CALCULATION AND SAVE DATA INTO HDF5
            if ind_buff == buffer_size-1:
                #calculate the mask only when the buffer is full. We use the minimal projection of the buffer.
                mask = getROIMask(np.min(Ibuff, axis=0), **mask_param)
                
                #mask all the images in the buffer
                Ibuff *= mask
                
                #add buffer to the hdf5 file
                mask_dataset[(frame_number-buffer_size):frame_number,:,:] = Ibuff
            
            #calculate the progress every 500 frames and put it in a string
            if frame_number%500 == 0:
                progress_str = progressTime.getStr(frame_number)
                print(base_name + ' ' + progress_str);
                sys.stdout.flush()
            
        
        #once we finished to read the whole video, we need to make sure that the hdf5 array sizes are correct.
        if mask_dataset.shape[0] != frame_number:
            mask_dataset.resize(frame_number, axis=0);
            
        if full_dataset.shape[0] != full_frame_number:
            full_dataset.resize(full_frame_number, axis=0);
        
        #attribute to indicate the program finished correctly
        mask_dataset.attrs['has_finished'] = 1
            
        #close the video
        vid.release() 

        #save the timestamps into separate arrays
        mask_fid.create_dataset("/vid_frame_pos", data = np.asarray(vid_frame_pos));
        mask_fid.create_dataset("/vid_time_pos", data = np.asarray(vid_time_pos));


        #close the hdf5 files
        mask_fid.close()

        #completion message
        print(base_name + ' Compressed video done.');
        sys.stdout.flush()

if __name__ == '__main__':
    video_file = '/Users/ajaver/Desktop/Videos/Check_Align_samples/Videos/npr-9 (tm1652) on food R_2010_01_25__16_24_16___4___13.avi'
    masked_image_file = '/Users/ajaver/Desktop/Videos/test.hdf5'
    compressVideo(video_file, masked_image_file, buffer_size = 25)
