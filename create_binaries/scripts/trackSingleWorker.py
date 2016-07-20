# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:59:25 2015

@author: ajaver
"""
import sys
from MWTracker.batchProcessing.trackSingleWorker import getTrajectoriesWorker, track_worker_parser

if __name__ == '__main__':

    if len(sys.argv) > 1:
        args = track_worker_parser.parse_args()
        getTrajectoriesWorker(**vars(args))
    else:
        print('Bad', sys.argv)
