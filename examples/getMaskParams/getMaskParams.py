import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QDir, QTimer, Qt, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QPolygonF, QPen
from getMaskParams_ui import Ui_MainWindow

import json
import h5py
import os
import numpy as np
import sys
import cv2
import matplotlib.pylab as plt

import sys
#sys.path.append('../..')
#sys.path.append(os.path.join(os.path.expanduser("~") , 'Documents', 'GitHub', 'Multiworm_Tracking'))
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking')

from MWTracker.compressVideos.compressVideo import getROIMask, selectVideoReader
from MWTracker.helperFunctions.compressVideoWorkerL import compressVideoWorkerL
from MWTracker.helperFunctions.getTrajectoriesWorkerL import getTrajectoriesWorkerL

class getMaskParams(QMainWindow):
	def __init__(self):
		super().__init__()
		
		# Set up the user interface from Designer.
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)

		self.ui.dial_min_area.valueChanged.connect(self.ui.spinBox_min_area.setValue)
		self.ui.dial_max_area.valueChanged.connect(self.ui.spinBox_max_area.setValue)
		self.ui.dial_block_size.valueChanged.connect(self.ui.spinBox_block_size.setValue)
		self.ui.dial_thresh_C.valueChanged.connect(self.ui.spinBox_thresh_C.setValue)
		self.ui.dial_buff_size.valueChanged.connect(self.ui.spinBox_buff_size.setValue)

		self.ui.spinBox_max_area.valueChanged.connect(self.updateMaxArea)
		self.ui.spinBox_min_area.valueChanged.connect(self.updateMinArea)
		self.ui.spinBox_block_size.valueChanged.connect(self.updateBlockSize)
		self.ui.spinBox_thresh_C.valueChanged.connect(self.updateThreshC)
		self.ui.spinBox_buff_size.valueChanged.connect(self.ui.updateBuffSize)

		self.ui.checkBox_hasTimestamp.stateChanged.connect(self.updateMask)

		self.mask_files_dir = '/Users/ajaver/Desktop/Pratheeban_videos/MaskedVideos/'
		self.results_dir = '/Users/ajaver/Desktop/Pratheeban_videos/Results/'
		self.video_file = ''

		#self.videos_dir = '/Users/ajaver/Google Drive/MWTracker_Example/Worm_Videos'
		self.videos_dir = '/Volumes/behavgenom$/Andre/shige-oda/Worm_Videos/'
		#self.videos_dir = '/Users/ajaver/Desktop/Pratheeban_videos/Worm_Videos/'
		
		
		self.Ifull = np.zeros(0)
		self.vid = 0

		if not os.path.exists(self.mask_files_dir):
			self.mask_files_dir = ''
		
		if not os.path.exists(self.results_dir):
			self.results_dir = ''
					

		self.ui.lineEdit_mask.setText(self.mask_files_dir)
		self.ui.lineEdit_results.setText(self.results_dir)

		self.ui.pushButton_video.clicked.connect(self.getVideoFile)
		self.ui.pushButton_results.clicked.connect(self.updateResultsDir)
		self.ui.pushButton_mask.clicked.connect(self.updateMasksDir)
		
		self.ui.pushButton_next.clicked.connect(self.getNextChunk)
		self.ui.pushButton_start.clicked.connect(self.startAnalysis)

		self.ui.spinBox_fps.valueChanged.connect(self.updateFPS)
		self.updateFPS()

		
	def updateFPS(self):
		self.buffer_size = int(np.round(self.ui.spinBox_fps.value()))

	#file dialog to the the hdf5 file
	def getVideoFile(self):
		#print(self.videos_dir)
		video_file, _ = QFileDialog.getOpenFileName(self, "Find video file", 
		self.videos_dir, "All files (*)")

		if video_file:
			self.video_file = video_file
			if os.path.exists(self.video_file):
				self.ui.label_full.clear()
				self.Ifull = np.zeros(0)

				self.videos_dir = self.video_file.rpartition(os.sep)[0] + os.sep
				

				self.ui.lineEdit_video.setText(self.video_file)
				self.vid, self.im_width, self.im_height = selectVideoReader(video_file)

				ret, image = self.vid.read()
				if not ret:
					QMessageBox.critical(self, 'Cannot read video file.', "Cannot read video file. Try another file",
					QMessageBox.Ok)
					self.vid = 0
					return

				self.im_width= image.shape[1]
				self.im_height= image.shape[0]
				
				print('H', self.im_height)
				print('W', self.im_width)
				
				if self.im_width == 0 or self.im_height == 0:
					 QMessageBox.critical(self, 'Cannot read video file.', "Cannot read video file. Try another file",
					QMessageBox.Ok)
					 self.vid = 0
					 return

				if 'Worm_Videos' in self.videos_dir:
					self.results_dir = self.videos_dir.replace('Worm_Videos', 'Results')
					self.mask_files_dir = self.videos_dir.replace('Worm_Videos', 'MaskedVideos')
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

		self.full_size = min(self.ui.label_full.height(), self.ui.label_full.width())
		
		image = QImage(self.Ifull.data, 
			self.im_width, self.im_height, self.Ifull.strides[0], QImage.Format_Indexed8)
		
		image = image.scaled(self.full_size, self.full_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
		self.img_w_ratio = image.size().width()/self.im_width;
		self.img_h_ratio = image.size().height()/self.im_height;
		
		pixmap = QPixmap.fromImage(image)
		self.ui.label_full.setPixmap(pixmap);

		self.mask_size = self.full_size

		mask = QImage(self.Imask.data, 
			self.im_width, self.im_height, self.Imask.strides[0], QImage.Format_Indexed8)
		
		mask = mask.scaled(self.mask_size, self.mask_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
		
		self.img_w_ratio = mask.size().width()/self.im_width;
		self.img_h_ratio = mask.size().height()/self.im_height;
		
		pixmap = QPixmap.fromImage(mask)
		self.ui.label_mask.setPixmap(pixmap);

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

	def updateMask(self):
		if self.Ifull.size == 0:
			return
		
		self.mask_param = {'max_area': self.ui.spinBox_max_area.value(),
		'min_area' : self.ui.spinBox_min_area.value(), 
		'thresh_block_size' : self.ui.spinBox_block_size.value(),
		'thresh_C' : self.ui.spinBox_thresh_C.value(),
		'has_timestamp':self.ui.checkBox_hasTimestamp.isChecked()
		}
		
		mask = getROIMask(self.Imin.copy(), **self.mask_param)
		self.Imask =  mask*self.Ifull

		self.updateImage()
	
	#update image if the GUI is resized event
	def resizeEvent(self, event):
		self.updateImage()

	def startAnalysis(self):
		if self.video_file == '' or self.Ifull.size == 0:
			QMessageBox.critical(self, 'No valid video file selected.', "No valid video file selected.", QMessageBox.Ok)
			return

		self.mask_param['fps'] = self.ui.spinBox_fps.value()
		self.mask_param['resampling_N'] = self.ui.spinBox_skelSeg.value()
		self.close()

		json_file = self.video_file.rpartition('.')[0] + '.json'
		with open(json_file, 'w') as fid:
			json.dump(self.mask_param, fid)

		masked_image_file = compressVideoWorkerL(self.video_file, self.mask_files_dir, param_file = json_file)
		getTrajectoriesWorkerL(masked_image_file, self.results_dir, param_file = json_file)

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

if __name__ == '__main__':
	app = QApplication(sys.argv)
	
	ui = getMaskParams()
	ui.show()
	app.exec_()
	#sys.exit()

