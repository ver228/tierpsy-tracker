# -*- coding: utf-8 -*-
import numpy as np
import cv2
import tables
from  tierpsy.analysis.compress.selectVideoReader import selectVideoReader

class BackgroundSubtractor():
    def __init__(self, video_file, buff_size = 100, frame_gap = 10, is_light_background=True):
        '''
        Object to subtract background
        '''
        #input parameters
        self.video_file = video_file
        self.buff_size = buff_size
        self.frame_gap = frame_gap
        self.is_light_background = is_light_background

        #internal variables
        self.bgnd = None
        self._buffer = None
        self._buffer_ind = None
        
        self.reduce_func = np.max if self.is_light_background else np.min

        self._init_buffer()
        self._calculate_bgnd()

    def _get_buf_ind(self, frame_n):
        return (frame_n//self.frame_gap)

    def _is_update_bgnd(self,frame_n):
        return (frame_n >= 0 and (frame_n % self.frame_gap) == 0)

    
    def _init_buffer(self):
        ret = 1
        frame_n = 0

        vid = selectVideoReader(self.video_file)


        d_info = np.iinfo(vid.dtype)
        if self.is_light_background:
            init_value = d_info.min
        else:
            init_value = d_info.max


        self._buffer = np.full((self.buff_size, vid.height, vid.width), init_value, vid.dtype)
        

        max_frames = self.buff_size*self.frame_gap
        while frame_n < max_frames:
            ret, image = vid.read()

            #if not valid frame is returned return.
            if ret == 0:
                break

            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            if self._is_update_bgnd(frame_n):
                if image.sum() == 0.:
                    #here i am assuming that if there is an old black image the video finished
                    break
                
                self._buffer_ind = self._get_buf_ind(frame_n)
                self._buffer[self._buffer_ind] = image

            frame_n += 1
        vid.release()

        self.last_frame = frame_n - 1

        if self._buffer_ind is None:
            #no valid frames read
            self._buffer = None
        
        elif self._buffer_ind<(self._buffer_ind-1):
            #not many frames red
            self._buffer = self._buffer[:(self._buffer_ind+1)]

    def _calculate_bgnd(self):

        self.bgnd = self.reduce_func(self._buffer, axis=0)
        self.bgnd = self.bgnd.astype(np.int32)

    def _update_background(self, image, frame_n):
        self.last_frame = frame_n
        self.buff_ind = self._get_buf_ind(frame_n)
        self.buff_ind += 1
        if self.buff_ind >= self._buffer.shape[0]:
            self.buff_ind=0
        self._buffer[self.buff_ind] = image

        self._calculate_bgnd()

    def update_background(self, image, frame_n):
        #update background only in the proper frames
        if frame_n > self.last_frame and self._is_update_bgnd(frame_n):
            # if it is a dark background I do not want to accept an all black image because 
            #it will mess up the background calculation
            if (not self.is_light_background) and (image.sum() == 0):
                return 

            self._update_background(image, frame_n)

    def subtract_bgnd(self, image):
        # new method using bitwise not
        def _remove_func(_img, _func, _bg):
            #the reason to use opencv2 instead of numpy is to avoid buffer overflow
            #https://stackoverflow.com/questions/45817037/opencv-image-subtraction-vs-numpy-subtraction/45817868
            new_img = np.zeros_like(_img); #maybe can do this in place
            if image.ndim == 2:
                _func(_img, _bg, new_img)
            else:
                for ii, this_frame in enumerate(_img):
                    _func(this_frame, _bg, new_img[ii])
            return new_img
        
        bg = ~self.bgnd.astype(np.uint8)
        if self.is_light_background:
            notbg = ~bg # should check if necessary at all to have self.bgnd as int32
            ss = _remove_func(image, cv2.add, notbg)
        else: # fluorescence
            ss = _remove_func(image, cv2.subtract, bg)
            
        ss = np.clip( ss ,1,255);
        
        return ss

    def apply(self, image, last_frame=-1):
        if last_frame > 0:
            #we have to decide if we are dealing with a single image or a buffer or images
            if image.ndim == 2:
                self.update_background(image, last_frame)
            else:
                first_frame = last_frame - image.shape[0]+1
                for ii, frame_n in enumerate(range(first_frame, last_frame+1)):
                    self.update_background(image[ii], frame_n)
        
        return self.subtract_bgnd(image)
            

class ReadFullImage():
    def __init__(self, video_file):
        self.video_file = video_file
        self.current_index = 0
        self.next_frame = 0
        self.full_interval = None
        self.full_img = None

        self._update(0)
    
    def _update(self, frame_n):
        if frame_n >= self.next_frame:
            with tables.File(self.video_file, 'r') as fid:            
                self.full_img = fid.get_node('/full_data')[self.current_index]
                if self.full_interval is None:
                    self.full_interval = int(fid.get_node('/full_data')._v_attrs['save_interval'])
            
            self.current_index += 1
            self.next_frame = self.full_interval*(self.current_index + 1)

    def  __getitem__(self, frame_n):
        self._update(frame_n)
        return self.full_img

class BackgroundSubtractorMasked(BackgroundSubtractor):
    '''
    Experimental...
    Object to subtract the background from a masked image. 
    '''
    def __init__(self):
        super().__init__()
        self._full_img = None
        self._full_img_reader = None

    def _init_buffer(self):    
        #we only accept masked files
        assert self.video_file.endswith('hdf5')
        
        with tables.File(self.video_file, 'r') as fid:
            masks = fid.get_node('/mask')
            
            n_frames = self.buff_size*self.frame_gap
            self._buffer = masks[:n_frames:self.frame_gap]
            self._buffer_ind = self._buffer.size 
            self.last_frame = self._buffer_ind*self.frame_gap

            if '/full_data' in fid:
                self.full_img_reader = ReadFullImage(self.video_file)
                self.full_img = self.full_img_reader[0]
        
    def _calculate_bgnd(self):
        self.super()._calculate_bgnd()

        if self.full_img is not None:
            dd = (self.bgnd, self.full_img)
            self.bgnd = self.reduce_func(dd, axis=0)
            
    def _update_background(self, image, frame_n):
        self.super()._update_background()
        if self.full_img_reader is not None:
            self.full_img = self.full_img_reader[frame_n]

        self._calculate_bgnd()

#%%
if __name__ == '__main__':
    video_file = '/Users/ajaver/OneDrive - Imperial College London/paper_tierpsy_tracker/benchmarks/CeLeST/samples/RawVideos/Sample01/frame001.jpg'
    bngd_subtr = BackgroundSubtractor(video_file, buff_size = 5, frame_gap = 1000)
    assert (bngd_subtr.bgnd).sum() > 0
    print(np.mean(bngd_subtr.bgnd))

    bngd_subtr.apply(bngd_subtr._buffer)

