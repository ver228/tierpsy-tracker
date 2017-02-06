# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:12:48 2015

@author: ajaver
"""
from MWTracker.processing.processMultipleFilesFun import processMultipleFilesFun
from MWTracker.processing.ProcessMultipleFilesParser import ProcessMultipleFilesParser

if __name__ == '__main__':
	args = ProcessMultipleFilesParser().parse_args()
	processMultipleFilesFun(**vars(args))
