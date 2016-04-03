import numpy as np
import sys

#import matplotlib
#matplotlib.use("Qt5Agg")
#import matplotlib.pylab as plt
import pandas as pd
import cv2
import h5py
import tables

from PyQt5.QtWidgets import QApplication, QWidget


if __name__ == '__main__':
    
    app = QApplication(sys.argv)

    w = QWidget()
    w.resize(250, 150)
    w.move(300, 300)
    w.setWindowTitle('Simple')
    w.show()
    
    print('HOLA')
    a = np.arange(10)
    print(a)
       

    print(cv2.__version__)

    inputFiles = "test.h5"
    with h5py.File(inputFiles, 'w') as inputFileOpen:
        print('good h5py')
    
    with tables.File(inputFiles, 'w') as inputFileOpen:
        print('good tables')
    
    sys.exit(app.exec_()) 
    
    