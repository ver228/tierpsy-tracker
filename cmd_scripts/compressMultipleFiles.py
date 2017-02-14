# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:12:48 2015

@author: ajaver
"""

from tierpsy.processing.processMultipleFilesFun import compressMultipleFilesFun
from tierpsy.processing.ProcessMultipleFilesParser import CompressMultipleFilesParser

if __name__ == '__main__':

    args = CompressMultipleFilesParser().parse_args()
    print(args)
    compressMultipleFilesFun(**vars(args))
