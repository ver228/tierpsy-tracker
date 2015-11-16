import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QDir, QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage
from MWTracker.GUI.HDF5videoViewer.HDF5videoViewer_ui import Ui_ImageViewer

import tables
import os
import numpy as np

class HDF5videoViewer_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set up the user interface from Designer.
        self.ui = Ui_ImageViewer()
        self.ui.setupUi(self)

        self.isPlay = False
        self.fid = -1
        self.image_group = -1
        #self.videos_dir =  r"/Volumes/behavgenom$/GeckoVideo/Results/20150521_1115/"
        self.videos_dir =  os.path.expanduser("~") + os.sep + 'Downloads' + os.sep + 'wetransfer-cf3818' + os.sep
        
        #self.ui.centralWidget.setChildrenFocusPolicy(Qt.NoFocus)

        self.h5path = self.ui.comboBox_h5path.itemText(0)
        
        self.ui.fileButton.clicked.connect(self.getVideoFile)
        
        self.ui.playButton.clicked.connect(self.playVideo)
        self.ui.imageSlider.sliderPressed.connect(self.imSldPressed)
        self.ui.imageSlider.sliderReleased.connect(self.imSldReleased)
        
        self.ui.spinBox_frame.valueChanged.connect(self.updateFrameNumber)
        self.ui.doubleSpinBox_fps.valueChanged.connect(self.updateFPS)
        self.ui.spinBox_step.valueChanged.connect(self.updateFrameStep)
        
        self.ui.spinBox_step.valueChanged.connect(self.updateFrameStep)

        self.ui.comboBox_h5path.activated.connect(self.getImGroup)

        self.ui.pushButton_h5groups.clicked.connect(self.updateGroupNames)

        self.updateFPS()
        self.updateFrameStep()
        
        # SET UP RECURRING EVENTS
        self.timer = QTimer()
        self.timer.timeout.connect(self.getNextImage)
        
        
    #Scroller
    def imSldPressed(self):
        self.ui.imageSlider.setCursor(Qt.ClosedHandCursor)
    
    def imSldReleased(self):
        self.ui.imageSlider.setCursor(Qt.OpenHandCursor)
        if self.image_group != -1:
            self.frame_number = round((self.tot_frames-1)*self.ui.imageSlider.value()/100)
            self.updateImage()
    
    #frame spin box
    def updateFrameNumber(self):
        self.frame_number = self.ui.spinBox_frame.value()
        self.updateImage()

    #fps spin box
    def updateFPS(self):
        self.fps = self.ui.doubleSpinBox_fps.value()

    #frame steps spin box
    def updateFrameStep(self):
        self.frame_step = self.ui.spinBox_step.value()

    #Play Button
    def playVideo(self):
        if self.image_group == -1:
            return
        if not self.isPlay:
            self.startPlay()
        else:
            self.stopPlay()
    
    def startPlay(self):
        self.timer.start(round(1000/self.fps))
        self.isPlay = True
        self.ui.playButton.setText('Stop')
        self.ui.doubleSpinBox_fps.setEnabled(False)

    def stopPlay(self):
        self.timer.stop()
        self.isPlay = False
        self.ui.playButton.setText('Play')
        self.ui.doubleSpinBox_fps.setEnabled(True)

    #Function to get the new valid frame during video play
    def getNextImage(self):
        self.frame_number += self.frame_step
        if self.frame_number >= self.tot_frames:
            self.stopPlay()
            return
        self.updateImage()

    #update image: get the next frame_number, and resize it to fix in the GUI area
    def updateImage(self):
        if self.image_group == -1:
            return

        self.ui.spinBox_frame.setValue(self.frame_number)
        
        self.label_height = self.ui.imageCanvas.height()
        self.label_width = self.ui.imageCanvas.width()
        
        self.original_image = self.image_group[self.frame_number,:,:];
        
        #equalize and cast if it is not uint8
        if self.original_image.dtype != np.uint8:
            top = np.max(self.original_image)
            bot = np.min(self.original_image)

            self.original_image = (self.original_image-bot)*255./(top-bot)

            self.original_image = np.round(self.original_image).astype(np.uint8)
            #print(self.original_image)
        image = QImage(self.original_image.data, 
            self.image_height, self.image_width, self.original_image.strides[0], QImage.Format_Indexed8)
        image = image.scaled(self.label_width, self.label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.ui.imageCanvas.setPixmap(QPixmap.fromImage(image));
        
        progress = round(100*self.frame_number/self.tot_frames)
        if progress != self.ui.imageSlider.value():
            self.ui.imageSlider.setValue(progress)

    #file dialog to the the hdf5 file
    def getVideoFile(self):
        vfilename, _ = QFileDialog.getOpenFileName(self, "Find HDF5 video file", 
        self.videos_dir, "HDF5 files (*.hdf5);; All files (*)")

        if vfilename:
            if self.fid != -1:
                self.fid.close()
                self.ui.imageCanvas.clear()

            self.vfilename = vfilename
            self.updateVideoFile()
    
    def updateVideoFile(self):
        if not os.path.exists(self.vfilename):
            QMessageBox.critical(self, 'The hdf5 video file does not exists', "The hdf5 video file does not exists. Please select a valid file",
                    QMessageBox.Ok)
            return
        
        self.ui.lineEdit.setText(self.vfilename)
        self.videos_dir = self.vfilename.rpartition(os.sep)[0] + os.sep
        self.fid = tables.File(self.vfilename, 'r')
        
        self.updateImGroup()

    def updateGroupNames(self):
        valid_groups = []
        for group in self.fid.walk_groups("/"):
            for array in self.fid.list_nodes(group, classname='Array'):
                if array.ndim == 3:
                    valid_groups.append(array._v_pathname)
        
        if not valid_groups:
            QMessageBox.critical(self, '', "No valid video groups were found. Dataset with three dimensions and uint8 data type.",
                    QMessageBox.Ok)
            return

        self.ui.comboBox_h5path.clear()
        for kk in valid_groups:
            self.ui.comboBox_h5path.addItem(kk)
        self.getImGroup(0)
        self.updateImage()

    def getImGroup(self, index):
        self.h5path = self.ui.comboBox_h5path.itemText(index)
        self.updateImGroup()

    #read a valid groupset from the hdf5
    def updateImGroup(self):
        if self.fid == -1:
            return

        #self.h5path = self.ui.comboBox_h5path.text()
        if not self.h5path in self.fid:
            self.ui.imageCanvas.clear()
            self.image_group == -1
            QMessageBox.critical(self, 'The groupset path does not exists', "The groupset path does not exists. You must specify a valid groupset path",
                    QMessageBox.Ok)
            return

        self.image_group = self.fid.get_node(self.h5path)
        if len(self.image_group.shape) != 3:
            self.ui.imageCanvas.clear()
            self.image_group == -1
            QMessageBox.critical(self, 'Invalid groupset', "Invalid groupset. The groupset must have three dimensions",
                    QMessageBox.Ok)

        self.tot_frames = self.image_group.shape[0]
        self.image_height = self.image_group.shape[2]
        self.image_width = self.image_group.shape[1]
            
        self.ui.spinBox_frame.setMaximum(self.tot_frames-1)

        self.frame_number = 0
        self.ui.spinBox_frame.setValue(self.frame_number)
        self.updateImage()


    def setFileName(self, filename):
        self.filename = filename
        self.ui.lineEdit.setText(filename)

    
    def resizeEvent(self, event):
        if self.fid != -1:
            self.updateImage()
    
    def keyPressEvent(self, event):
        #print(event.key())
        if self.fid == -1:
            return

        #Move backwards when < or , are pressed
        if event.key() == 44 or event.key() == 60:
            self.frame_number -= self.frame_step
            if self.frame_number<0:
                self.frame_number = 0
            self.updateImage()
        
        #Move forward when  > or . are pressed
        elif event.key() == 46 or event.key() == 62:
            self.frame_number += self.frame_step
            if self.frame_number >= self.tot_frames:
                self.frame_number = self.tot_frames-1
            self.updateImage()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    ui = HDF5videoViewer()
    ui.show()
    
    sys.exit(app.exec_())