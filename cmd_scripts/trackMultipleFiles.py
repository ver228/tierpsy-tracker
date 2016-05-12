# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:12:48 2015

@author: ajaver
"""


from MWTracker.batchProcessing.trackMultipleFilesFun import trackMultipleFilesFun, parser

if __name__ == '__main__':
	args = parser.parse_args()
	trackMultipleFilesFun(**vars(args))