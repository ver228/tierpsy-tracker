# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:39:00 2015

@author: ajaver
"""
import matplotlib.pylab as plt
video_file = '/Volumes/behavgenom$/syngenta/RawData/data_20150114/compound k repeat 3 thurs 11th dec.avi'
masked_image_file = '/Volumes/behavgenom$/syngenta/Compressed/data_20150114/compound k repeat 3 thurs 11th dec.hdf5'


if __name__ == '__main__':  
    buffer_size = 25; save_full_interval = 5000; max_frame = 10;#e32; 
    base_name = ''; status_queue = ''; check_empty = True;
    useVideoCapture = True
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
        plt.figure()
        plt.imshow(image)
        
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
