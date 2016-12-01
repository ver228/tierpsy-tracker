# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:12:48 2015

@author: ajaver
"""
from MWTracker.batch_processing.processMultipleFilesFun import trackMultipleFilesFun
from MWTracker.batch_processing.ProcessMultipleFilesParser import TrackMultipleFilesParser

if __name__ == '__main__':
	args = TrackMultipleFilesParser().parse_args()
	trackMultipleFilesFun(**vars(args))
