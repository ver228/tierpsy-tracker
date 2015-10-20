import numpy as np
import cv2
import sys
import tables

a=np.arange(10)
print(a)
print(cv2.__version__)
inputFiles="test.h5"
with tables.File(inputFiles, 'w') as inputFileOpen:
	pass