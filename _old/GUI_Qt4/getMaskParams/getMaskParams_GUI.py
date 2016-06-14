import json
import h5py
import os
import numpy as np
import sys
import cv2

import sys
from PyQt4.QtGui import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt4.QtCore import QDir, QTimer, Qt, QPointF
from PyQt4.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QPolygonF, QPen

from MWTracker.helperFunctions.tracker_param import tracker_param
from MWTracker.GUI_Qt4.getMaskParams.getMaskParams_ui import Ui_MainWindow
from MWTracker.compressVideos.compressVideo import getROIMask, selectVideoReader
from MWTracker.batchProcessing.compressSingleWorker import compressSingleWorker
from MWTracker.batchProcessing.trackSingleWorker import getTrajectoriesWorker

import psutil, os

def kill_proc_tree(pid, including_parent=True):  
    '''http://stackoverflow.com/questions/22291434/pyqt-application-closes-successfully-but-process-is-not-killed'''
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    if including_parent:
        parent.kill()

class getMaskParams_GUI(QMainWindow):
    def __init__(self, default_videos_dir = '', scripts_dir = ''):
        super().__init__()
        
        # Set up the user interface from Designer.
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.dial_min_area.valueChanged.connect(self.ui.spinBox_min_area.setValue)
        self.ui.dial_max_area.valueChanged.connect(self.ui.spinBox_max_area.setValue)
        self.ui.dial_block_size.valueChanged.connect(self.ui.spinBox_block_size.setValue)
        self.ui.dial_thresh_C.valueChanged.connect(self.ui.spinBox_thresh_C.setValue)

        self.ui.spinBox_max_area.valueChanged.connect(self.updateMaxArea)
        self.ui.spinBox_min_area.valueChanged.connect(self.updateMinArea)
        self.ui.spinBox_block_size.valueChanged.connect(self.updateBlockSize)
        self.ui.spinBox_thresh_C.valueChanged.connect(self.updateThreshC)
        self.ui.spinBox_dilation_size.valueChanged.connect(self.updateDilationSize)

        self.ui.spinBox_buff_size.valueChanged.connect(self.updateBuffSize)

        self.ui.checkBox_hasTimestamp.stateChanged.connect(self.updateMask)
        self.ui.checkBox_keepBorderData.stateChanged.connect(self.updateMask)

        
        self.ui.pushButton_video.clicked.connect(self.getVideoFile)
        self.ui.pushButton_results.clicked.connect(self.updateResultsDir)
        self.ui.pushButton_mask.clicked.connect(self.updateMasksDir)
        
        self.ui.pushButton_next.clicked.connect(self.getNextChunk)
        self.ui.pushButton_start.clicked.connect(self.startAnalysis)

        

        self.videos_dir = default_videos_dir
        if not os.path.exists(self.videos_dir): self.videos_dir = ''

        self.mask_files_dir = ''
        self.results_dir = ''
        
        self.video_file = ''
        self.json_file = ''
        
        self.Ifull = np.zeros(0)
        self.vid = 0


        self.ui.lineEdit_mask.setText(self.mask_files_dir)
        self.ui.lineEdit_results.setText(self.results_dir)
        self.updateBuffSize()
        self.read_json()

    
    #file dialog to the the hdf5 file
    def getVideoFile(self):
        video_file = QFileDialog.getOpenFileName(self, "Find video file", 
        self.videos_dir, "All files (*)")

        if video_file:
            self.video_file = video_file
            if os.path.exists(self.video_file):
                
                self.json_file = self.video_file.rpartition('.')[0] + '.json'
                self.read_json()


                self.ui.label_full.clear()
                self.Ifull = np.zeros(0)

                self.videos_dir = self.video_file.rpartition(os.sep)[0] + os.sep
                
                self.ui.lineEdit_video.setText(self.video_file)
                self.vid, self.im_width, self.im_height, self.reader_type = selectVideoReader(video_file)
                if self.im_width == 0 or self.im_height == 0:
                     QMessageBox.critical(self, 'Cannot read video file.', "Cannot read video file. Try another file",
                    QMessageBox.Ok)
                     self.vid = 0
                     return

                #replace Worm_Videos or add a directory for the Results and MaskedVideos directories
                if 'Worm_Videos' in self.videos_dir:
                    self.mask_files_dir = self.videos_dir.replace('Worm_Videos', 'MaskedVideos')
                    self.results_dir = self.videos_dir.replace('Worm_Videos', 'Results')
                else:
                    self.mask_files_dir = os.path.join(self.videos_dir, 'MaskedVideos')
                    self.results_dir = os.path.join(self.videos_dir, 'Results')
                
                self.ui.lineEdit_mask.setText(self.mask_files_dir)
                self.ui.lineEdit_results.setText(self.results_dir)

                self.getNextChunk()

    def getNextChunk(self):
        if self.vid:
            Ibuff = np.zeros((self.buffer_size, self.im_height, self.im_width), dtype = np.uint8)

            tot = 0;
            for ii in range(self.buffer_size):    
                ret, image = self.vid.read() #get video frame, stop program when no frame is retrive (end of file)
                
                if ret == 0:
                    break
                if image.ndim==3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                Ibuff[ii] = image
                tot += 1
            if tot < self.buffer_size:
                return

            self.Imin = np.min(Ibuff, axis=0)
            self.Ifull = Ibuff[0]
        
            self.updateMask()


    def updateImage(self):
        if self.Ifull.size == 0:
            return

        #plot full image and masked image in order to compare them
        label_size = min(self.ui.label_full.height(), self.ui.label_full.width())
        for ori_image, label in [(self.Ifull,self.ui.label_full), 
                                        (self.Imask, self.ui.label_mask)]:
            
            qimage = QImage(ori_image.data, self.im_width, self.im_height, 
                ori_image.strides[0], QImage.Format_Indexed8)
            qimage = qimage.scaled(label_size, label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.img_w_ratio = qimage.size().width()/self.im_width;
            self.img_h_ratio = qimage.size().height()/self.im_height;
        
            pixmap = QPixmap.fromImage(qimage)
            label.setPixmap(pixmap);

    def updateMaxArea(self):
        self.ui.dial_max_area.setValue(self.ui.spinBox_max_area.value())
        self.updateMask()

    def updateMinArea(self):
        self.ui.dial_min_area.setValue(self.ui.spinBox_min_area.value())
        self.updateMask()

    def updateBlockSize(self):
        self.ui.dial_block_size.setValue(self.ui.spinBox_block_size.value())
        self.updateMask()

    def updateThreshC(self):
        self.ui.dial_thresh_C.setValue(self.ui.spinBox_thresh_C.value())
        self.updateMask()

    def updateDilationSize(self):
        self.updateMask()

    def updateBuffSize(self):
        self.buffer_size = int(np.round(self.ui.spinBox_buff_size.value()))
    
    def updateMask(self):
        if self.Ifull.size == 0:
            return
        #read parameters used to calculate the mask
        self.mask_param = {
        'max_area': self.ui.spinBox_max_area.value(),
        'min_area' : self.ui.spinBox_min_area.value(), 
        'thresh_block_size' : self.ui.spinBox_block_size.value(),
        'thresh_C' : self.ui.spinBox_thresh_C.value(),
        'dilation_size' : self.ui.spinBox_dilation_size.value(),
        'has_timestamp' : self.ui.checkBox_hasTimestamp.isChecked(),
        'keep_border_data' : self.ui.checkBox_keepBorderData.isChecked()
        }
        
        mask = getROIMask(self.Imin.copy(), **self.mask_param)
        self.Imask =  mask*self.Ifull

        self.updateImage()

    #update image if the GUI is resized event
    def resizeEvent(self, event):
        self.updateImage()

    def updateResultsDir(self):
        results_dir = QFileDialog.getExistingDirectory(self, "Selects the directory where the analysis results will be stored", 
        self.results_dir)
        if results_dir:
            self.results_dir = results_dir + os.sep
            self.ui.lineEdit_results.setText(self.results_dir)


    def updateMasksDir(self):
        mask_files_dir = QFileDialog.getExistingDirectory(self, "Selects the directory where the hdf5 video will be stored", 
        self.mask_files_dir)
        if mask_files_dir:
            self.mask_files_dir = mask_files_dir + os.sep
            self.ui.lineEdit_mask.setText(self.mask_files_dir)

    def startAnalysis(self):
        if self.video_file == '' or self.Ifull.size == 0:
            QMessageBox.critical(self, 'No valid video file selected.', "No valid video file selected.", QMessageBox.Ok)
            return

        self.close()

        self.mask_param['fps'] = self.ui.spinBox_fps.value()
        self.mask_param['resampling_N'] = self.ui.spinBox_skelSeg.value()
        self.mask_param['compression_buff'] = self.ui.spinBox_buff_size.value()
        
        with open(self.json_file, 'w') as fid:
            json.dump(self.mask_param, fid)

        base_name = os.path.splitext(os.path.split(self.video_file)[-1])[0]
        self.masked_image_file = os.path.join(self.mask_files_dir, base_name + '.hdf5')
        
        os.system(['clear','cls'][os.name == 'nt'])
        compressSingleWorker(self.video_file, self.mask_files_dir, json_file = self.json_file)
        getTrajectoriesWorker(self.masked_image_file, self.results_dir, json_file = self.json_file)

        

    def read_json(self):
        if not self.json_file:
            param = tracker_param();
            param.get_param()

            self.mask_param = param.mask_param
            self.mask_param['fps'] = param.compress_vid_param['expected_fps']
            self.mask_param['resampling_N'] = param.skeletons_param['resampling_N']
            self.mask_param['compression_buff'] = param.compress_vid_param['buffer_size']
        
        elif os.path.exists(self.json_file): 
            with open(self.json_file, 'r') as fid:
                json_str = fid.read()
                self.mask_param = json.loads(json_str)
        else:
            self.mask_param = {}

        param2fun_gui = {
            'max_area':self.ui.spinBox_max_area.setValue,
            'min_area':self.ui.spinBox_min_area.setValue,
            'thresh_block_size':self.ui.spinBox_block_size.setValue,
            'thresh_C':self.ui.spinBox_thresh_C.setValue,
            'dilation_size':self.ui.spinBox_dilation_size.setValue,
            'has_timestamp':self.ui.checkBox_hasTimestamp.setCheckState,
            'keep_border_data':self.ui.checkBox_keepBorderData.setCheckState,
            'fps':self.ui.spinBox_fps.setValue,
            'resampling_N':self.ui.spinBox_skelSeg.setValue,
            'compression_buff':self.ui.spinBox_buff_size.setValue
        }

        for key, fun in param2fun_gui.items():
            if key in self.mask_param:
                fun(self.mask_param[key])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    ui = getMaskParams_GUI()
    ui.show()
    
    #sys.exit(app.exec_())
    app.exec_()
    me = os.getpid()
    kill_proc_tree(me)
