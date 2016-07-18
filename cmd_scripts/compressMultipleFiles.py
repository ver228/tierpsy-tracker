# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:12:48 2015

@author: ajaver
"""

from MWTracker.batchProcessing.compressMultipleFilesFun import compressMultipleFilesFun, compress_parser

if __name__ == '__main__':
	args = compress_parser.parse_args()
	compressMultipleFilesFun(**vars(args))