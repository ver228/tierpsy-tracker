# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:59:25 2015

@author: ajaver
"""
import sys
from MWTracker.batchProcessing.compressSingleWorker import compressSingleWorker


if __name__ == '__main__':
    if len(sys.argv)==6:
        compressSingleWorker(*sys.argv[1:])
    else:
        print('Wrong number of arguments.')
        print(sys.argv)
    
    
    
