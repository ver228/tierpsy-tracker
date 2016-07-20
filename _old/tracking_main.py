# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 17:54:54 2015

@author: ajaver
"""

h5File = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/A002 - 20150116_140923D.hdf5'
fidh5 = h5py.File(h5File, "r")
bgnd = fidh5["/bgnd"]

BUFF_SIZE = 20
BUFF_DELTA = 1200
INITIAL_FRAME = 1e7
