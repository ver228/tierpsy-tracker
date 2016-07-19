#!/usr/bin/env python
# encoding: utf-8
"""
TiffCapture.py - a capture class 

This guy provides a capture interface to multi-part-tiffs for 
use with OpenCV. It uses PIL to open the images and then reads
them frame-by-frame as needed, avoiding memory borking.

Example usage:
    import tiffcapture as tc
    import cv2
    tiff = tc.opentiff(filename) #open img
    _, first_img = tiff.retrieve() 
    cv2.namedWindow('video')
    for f,img in tiff:
        tempimg = cv2.absdiff(first_img, img) # bkgnd sub
        _, tempimg = cv2.threshold(tempimg, 5, 255, 
            cv2.cv.CV_THRESH_BINARY) # convert to binary
        cv2.imshow('video', tempimg)
        cv2.waitKey(80)
    cv2.destroyWindow('video')

Created by Dave Williams on 2013-07-15
"""

try:
    from PIL import Image
    import numpy as np
except ImportError:
    raise Exception("You'll need both numpy and PIL installed (and to be able to import 'Image' from it) for TiffCapture to work")


def opentiff(filename=None):
    """Open a tiff with TiffCapture and return the capture object"""
    return TiffCapture(filename)

class TiffCapture(object):
    
    """Feed me a filename, I'll give you a tiff's capture object. 
    It should be noted that some of the method names aren't 
    intuitive, this is not a choice but is to retain compatibility 
    with OpenCV VideoCapture objects.
    """
    
    def __init__(self, filename=None):
        """Initialize and return the capture object"""
        self._is_open = False #set true upon opening
        self.open(filename) #open and set class variables
    
    def __iter__(self):
        return self
    
    def _count_frames(self):
        """Return the number of frames in the tiff, takes a bit
        Honestly, there should be a faster way to do this.
        """
        try:
            self.tiff.seek(10**1000) #gonna assume that's long enough
        except EOFError:
            length = self.tiff.tell()
            self.tiff.seek(0)
            return length
    
    def open(self, filename):
        """Open a multi-stack tiff for video capturing.
        Open also sets class variables to allow dummy TiffCapture
        objects to be initialized without all the attendant content.
        A filename of None results in nothing being opened. 
        Takes:
            filename - the full path to the tiff stack
        Gives:
            isOpened - True if opened, False otherwise
        """
        if filename is not None:
            self.filename = filename
            self.tiff = Image.open(filename)
            self._is_open = True
            self.length = self._count_frames()
            self.shape = self.tiff.size
            self._curr = 0
        return self.isOpened()
    
    def next(self):
        """Grab and read next frame, stopping iteration on reaching 
        the end of the file.
        """
        f, img = self.read()
        if f is True:
            return img
        else:
            raise StopIteration()
    
    def __next__(self):
        """Python 3 requires __next__ rather than next"""
        return self.next()
    
    def grab(self):
        """Move to the next stack image, return True for success."""
        try:
            self.tiff.seek(self._curr+1) #updates tiff
            self._curr += 1 #updates accounting
            return True
        except EOFError:
            return False
    
    def retrieve(self):
        """Decode and return the grabbed video frame."""
        return True, np.array(self.tiff) 
    
    def read(self):
        """Grab, decode, and return the next video frame."""
        grabbed = self.grab()
        if grabbed is True:
            return self.retrieve()
        else:
            return False, np.array([])
    
    def find_and_read(self, i):
        """Find and return a specific frame number, i."""
        self.tiff.seek(i)
        try:
            img = np.array(self.tiff)
            self.tiff.seek(self._curr)
        except EOFError:
            img = np.array([])
        return img
    
    def seek(self, i):
        """Set a given location in the tiff stack as our current."""
        self._curr = i
        return
    
    def release(self):
        """Release the open tiff file."""
        if self.isOpened() is True:
            del(self.tiff)
            self._is_open = False
    
    def isOpened(self):
        """Returns true if a video capturing is initialized."""
        return self._is_open

