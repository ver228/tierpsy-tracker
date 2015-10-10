# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:15:59 2015

@author: ajaver
"""

import os
import re
import subprocess as sp
import numpy as np

class readVideoffmpeg:
    '''
    Read video frame using ffmpeg. Assumes 8bits gray video.
    Requires that ffmpeg is installed in the computer.
    This class is an alternative of the captureframe of opencv since:
    -> it can be a pain to compile opencv with ffmpeg compability. 
    -> this funciton is a bit faster (less overhead), but only works with gecko's mjpeg 
    '''
    def __init__(self, fileName, width = -1, height = -1):
        #requires the fileName, and optionally the frame width and heigth.
        if os.name == 'nt':
            ffmpeg_cmd = 'ffmpeg.exe'
        else:
            ffmpeg_cmd = 'ffmpeg263' #this version reads the Gecko files
        
        #try to open the file and determine the frame size. Raise an exception otherwise.
        if width<=0 or height <=0:
            try:
                command = [ffmpeg_cmd, '-i', fileName, '-']
                pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
                buff = pipe.stderr.read()
                pipe.terminate()
                #the frame size is somewhere printed at the beggining by ffmpeg
                dd = str(buff).partition('Video: ')[2].split(',')[2]
                dd = re.findall(r'\d*x\d*', dd)[0].split('x')
                self.height = int(dd[0])
                self.width = int(dd[1])
            except:
                print(buff)
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
