# -*- coding: utf-8 -*-
import numpy as np
import numpy.ma as ma
import cv2
import tables
from  tierpsy.analysis.compress.selectVideoReader import selectVideoReader

class BackgroundSubtractorBase():
    def __init__(self, 
                 video_file, 
                 buff_size = -1, 
                 frame_gap = -1,
                 is_light_background = True):
        
        #input parameters
        self.video_file = video_file
        self.buff_size = buff_size
        self.frame_gap = frame_gap
        self.is_light_background = is_light_background
        
        self.bgnd = None
        
    def init_buffer(self):
        '''Initilize the buffer. As the reading of the video progress, i will update this buffer.'''
        pass
    
    def subtract_bgnd(self, imgage):
        '''Function that deals how to do the background subtraction'''
        pass
    
    def is_update_frame(self, current_frame):
        '''Test if the current frame is valid to update the background'''
        pass
    
    
    def update_background(self, image, current_frame):
        '''Update background on base of the current image and frame'''
        pass
    
    
    def _apply_single(self, image, current_frame):
        #substract background from a single to a single frame
        if self.is_update_frame(current_frame):
            self.update_background(image, current_frame)
        return self.subtract_bgnd(image)
    
    def apply(self, image, current_frame = np.nan):
        #substract background from a single to a single frame and deal with a batch of several images
        if image.ndim == 2:
            return self._apply_single(image, current_frame)
        elif image.ndim == 3:
            out = [self._apply_single(img, current_frame + ii) for ii, img in enumerate(image)]
            return np.array(out)
        else:
            raise ValueError
    
    def _subtract_bgnd_from_mask(self, img):
        ss = np.zeros_like(img)
        
        if self.is_light_background:
            cv2.subtract(self.bgnd, img, ss)
        else:
            cv2.subtract(img, self.bgnd, ss)
        
        ss[img==0] = 0
        
        return ss
        
class BackgroundSubtractorStream(BackgroundSubtractorBase):
    def __init__(self,  
                 *args,
                 **argkws
                 ):
        
        super().__init__(*args, **argkws)
        
        #make sure this values are correct
        assert self.buff_size > 0
        assert self.frame_gap > 0

        self.reduce_func = np.max if self.is_light_background else np.min
        
        #internal variables
        self.last_frame = -1
        self.buffer = None
        self.buffer_ind = -1
        
        self.init_buffer()
        self.calculate_bgnd()
        
    
    def is_update_frame(self, current_frame):
        '''Test if the current frame is valid to update the background'''
        #update only if the frame gap is larger between the current frame and the last frame used
        _is_good = ((self.last_frame < 0) | (current_frame - self.last_frame >= self.frame_gap))
        return  _is_good
    
    def update_background(self, image, current_frame):
         self.update_buffer(image, current_frame)
         self.calculate_bgnd()
        
    def calculate_bgnd(self):
        '''Calculate the background from the buffer.'''
        pass
    
    def update_buffer(self, image, current_frame):
        '''Add new frame to the buffer.'''
        
        if image.sum() == 0:
            #this is a black image that sometimes occurs, ignore it...
            return
        
        self.last_frame = current_frame
        #treat it as a circular buffer (if it exceds if size return to the beginning)
        self.buffer_ind += 1
        if self.buffer_ind >= self.buffer.shape[0]:
            self.buffer_ind = 0
        
        self.buffer[self.buffer_ind] = image
    

class BackgroundSubtractorVideo(BackgroundSubtractorStream):
    
    def __init__(self, *args, **argkws):
        super().__init__(*args, **argkws)
        
    def init_buffer(self):
        ret = 1
        current_frame = 0

        vid = selectVideoReader(self.video_file)
        
        d_info = np.iinfo(vid.dtype)
        if self.is_light_background:
            init_value = d_info.min
        else:
            init_value = d_info.max


        self.buffer = np.full((self.buff_size, vid.height, vid.width), init_value, vid.dtype)
        self.buffer_ind = -1

        max_frames = self.buff_size*self.frame_gap

        if vid.__class__.__name__ != 'readLoopBio':
            # for non-loopbio videos
            while current_frame < max_frames:
                ret, image = vid.read()
                #if not valid frame is returned return.
                if ret == 0:
                    break
    
                if image.ndim == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
                if self.is_update_frame(current_frame):
                    self.update_buffer(image, current_frame)
                
                current_frame += 1
            self.last_frame = current_frame - 1
                
        else:
            # loopbio videos:
            for fc in range(self.buff_size):
                frame_to_read = fc * self.frame_gap
                ret, image = vid.read_frame(frame_to_read)
                # if not valid frame is returned return.
                if ret == 0:
                    break
                self.update_buffer(image, frame_to_read)
            self.last_frame = frame_to_read - self.frame_gap

        vid.release()

        if self.buffer_ind < 0:
            #no valid frames read
            self.buffer = None
        
        elif self.buffer_ind < (self.buff_size - 1):
            #not enough frames to fill the buffer, reduce its size
            self.buffer = self.buffer[:(self.buffer_ind+1)]
    
    def calculate_bgnd(self):
        '''Calculate the background from the buffer.'''
        self.bgnd = self.reduce_func(self.buffer, axis=0)
        self.bgnd = self.bgnd.astype(np.int32)
    
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
        
        bg = self.bgnd.astype(np.uint8)
        if self.is_light_background:
            notbg = ~bg
            ss = _remove_func(image, cv2.add, notbg)
        else: # fluorescence
            ss = _remove_func(image, cv2.subtract, bg)
            
        ss = np.clip( ss ,1,255);
        
        return ss

class BackgroundSubtractorMasked(BackgroundSubtractorStream):
    '''
    Object to subtract the background from a masked image. 
    '''
    
    def __init__(self, *args, **argkws):
        self.full_img = None
        super().__init__(*args, **argkws)
        

    def init_buffer(self):    
        #we only accept masked files
        assert self.video_file.endswith('hdf5')
        
        with tables.File(self.video_file, 'r') as fid:
            masks = fid.get_node('/mask')
            
            #here i am using a masked numpy array to deal with the zeroed background
            last_frame = self.buff_size*self.frame_gap - 1
            self.buffer = masks[:last_frame + 1:self.frame_gap]
            self.buffer = ma.masked_array(self.buffer, self.buffer==0)
            
            self.buffer_ind = self.buffer.shape[0] - 1
            self.last_frame = last_frame

            if '/full_data' in fid:
                #here i am using as the canonical background the results of using the reducing function in all the full frames
                full_data = fid.get_node('/full_data')
                self.full_img = self.reduce_func(full_data, axis=0)
                
             
            
    def calculate_bgnd(self):
        fill_value = 0 if self.is_light_background else 255
        self.bgnd = self.reduce_func(self.buffer, axis=0).filled(fill_value)
        if self.full_img is not None:
            dd = (self.bgnd, self.full_img)
            self.bgnd = self.reduce_func(dd, axis=0)
        
    
    def _update_background(self, image, frame_n):
        super()._update_background(image, frame_n)
        dd = self.buffer[self.buffer_ind]
        self.buffer[self.buffer_ind] = ma.masked_array(dd, dd ==0)
    
    def subtract_bgnd(self, image):
        return self._subtract_bgnd_from_mask(image)
#%%
class BackgroundSubtractorPrecalculated(BackgroundSubtractorBase):
    def __init__(self, *args,  **argkws):
        
        self.save_interval = -1
        self.precalculated_bgnd = None
        self.full_img = None
        
        super().__init__(*args, **argkws)
        self.init_buffer()
    
    def init_buffer(self):    
        #we only accept masked files
        assert self.video_file.endswith('hdf5')
        with tables.File(self.video_file, 'r') as fid:
            _bgnd = fid.get_node('/bgnd')
            self.save_interval = int(_bgnd._v_attrs['save_interval'])  
            self.precalculated_bgnd = _bgnd[:]
            
        self.last_frame = 0
        self.bgnd = self.precalculated_bgnd[0]
    
    def subtract_bgnd(self, image):
        return self._subtract_bgnd_from_mask(image)
        
    def is_update_frame(self, current_frame):
        '''Here the update is trivial so i can do it in every frame'''
        return True
    
    def update_background(self, image, current_frame):
        self.bgnd = self.precalculated_bgnd[current_frame//self.save_interval]
    
                
#%%
if __name__ == '__main__':
    import matplotlib.pylab as plt
    import tqdm
    
    #video_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/CX11254_Ch1_05092017_075253.hdf5'
    #video_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/CX11314_Ch2_01072017_093003.hdf5'
    
    video_file = '/Users/avelinojaver/OneDrive - Nexus365/worms/Bertie_movies/CX11314_Ch1_04072017_103259.hdf5'
    
    
    
    #%%
    #video_file = '/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_examples/tierpsy_test_data/different_animals/worm_motel/MaskedVideos/Position6_Ch2_12012017_102810_s.hdf5'
    
    #_sub = BackgroundSubtractorMasked(video_file, buff_size = 5, frame_gap = 100)
    _sub = BackgroundSubtractorPrecalculated(video_file)
    
    with tables.File(video_file, 'r') as fid:
        masks = fid.get_node('/mask')
        
        
        tot = min(1000000, masks.shape[0])
        for frame_number in tqdm.tqdm(range(0, tot, 1000)):
            img = masks[frame_number]
        
            img_s = _sub.apply(img, current_frame = frame_number)
            
            
            fig, axs = plt.subplots(1,3, sharex = True, sharey=True)
            axs[0].imshow(img)
            axs[1].imshow(_sub.bgnd)
            axs[2].imshow(img_s)

    #%%
    
#    from pathlib import Path
#    video_file = Path.home () / 'OneDrive - Imperial College London/documents/papers_in_progress/paper_tierpsy_tracker/figures_data/different_setups/CeLeST/RawVideos/Sample01/frame001.jpg'
#    video_file = str(video_file)
#    bngd_subtr = BackgroundSubtractorVideo(video_file, buff_size = 10, frame_gap = 50, is_light_background=False)
#    assert (bngd_subtr.bgnd).sum() > 0
#    print(np.mean(bngd_subtr.bgnd))
#    
#    img_s = bngd_subtr.apply(bngd_subtr.buffer)
#    
#    fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
#    axs[0].imshow(bngd_subtr.buffer[0])
#    axs[1].imshow(img_s[0])
    
