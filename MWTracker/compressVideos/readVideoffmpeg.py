# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:15:59 2015

@author: ajaver
"""

import os
import re
import subprocess as sp
import numpy as np
from threading  import Thread
from queue import Queue, Empty


def enqueue_error(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close

class readVideoffmpeg:
    '''
    Read video frame using ffmpeg. Assumes 8bits gray video.
    Requires that ffmpeg is installed in the computer.
    This class is an alternative of the captureframe of opencv since:
    -> it can be a pain to compile opencv with ffmpeg compability. 
    -> this funciton is a bit faster (less overhead), but only works with gecko's mjpeg 
    '''
    def __init__(self, fileName):
        #get the correct path for ffmpeg. First we look in the auxFiles directory, otherwise we look in the system path.
        aux_file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'auxFiles')
        if os.name == 'nt':
            ffmpeg_cmd = os.path.join(aux_file_dir, 'ffmpeg.exe')
            if not os.path.exists(ffmpeg_cmd):
                ffmpeg_cmd = 'ffmpeg.exe'
        else:
            ffmpeg_cmd = os.path.join(aux_file_dir, 'ffmpeg22')
            if not os.path.exists(ffmpeg_cmd):
                ffmpeg_cmd = '/usr/local/bin/ffmpeg22'

        #try to open the file and determine the frame size. Raise an exception otherwise.
        
        try:
            command = [ffmpeg_cmd, '-i', fileName, '-']
            pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
            buff = pipe.stderr.read()
            pipe.terminate()
            #the frame size is somewhere printed at the beggining by ffmpeg
            dd = str(buff).partition('Video: ')[2].split(',')[2]
            dd = re.findall(r'\d*x\d*', dd)[0].split('x')
            self.height = int(dd[1])
            self.width = int(dd[0])
        except:
            raise Exception('Error while getting the width and height using ffmpeg. Buffer output:', buff)
    
        self.tot_pix = self.height*self.width
        
        command = [ffmpeg_cmd, 
           '-i', fileName,
           '-f', 'image2pipe',
           '-vsync', 'drop', #avoid repeating frames due to changes in the time stamp, it is better to solve those situations manually after
           '-threads', '0',
           '-vf', 'showinfo',
           '-vcodec', 'rawvideo', '-']
        
        self.vid_frame_pos = []
        self.vid_time_pos = []
        
        #devnull = open(os.devnull, 'w') #use devnull to avoid printing the ffmpeg command output in the screen
        self.pipe = sp.Popen(command, stdout = sp.PIPE, \
        bufsize = self.tot_pix, stderr=sp.PIPE)

        self.queue = Queue()
        self.thread = Thread(target = enqueue_error, args = (self.pipe.stderr, self.queue))
        self.thread.start()

        
        #use a buffer size as small as possible (frame size), makes things faster
    
    def get_timestamp(self):
        while 1:    
            # read line without blocking
            try: 
                line = self.queue.get_nowait().decode("utf-8")
                #self.err_out.append(line)

                frame_N = line.partition(' n:')[-1].partition(' ')[0]
                timestamp = line.partition(' pts_time:')[-1].partition(' ')[0]
                
                if frame_N and timestamp:
                    self.vid_frame_pos.append(int(frame_N))
                    self.vid_time_pos.append(float(timestamp))
            except Empty: break

    
    def read(self):
        #retrieve an image as numpy array 
        raw_image = self.pipe.stdout.read(self.tot_pix)
        if len(raw_image) < self.tot_pix:
            return (0, []);
        
        image = np.fromstring(raw_image, dtype='uint8')
        image = image.reshape(self.height, self.width)

        # i need to read this here because otherwise the err buff will get full.
        self.get_timestamp()

        return (1, image)
    
    def release(self):
        #close the buffer
        self.pipe.stdout.flush()
        self.pipe.stderr.flush()
        self.get_timestamp()
        
        self.pipe.terminate()
        self.pipe.stdout.close()
        self.pipe.wait()
        