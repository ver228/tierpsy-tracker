import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QDir, QTimer, Qt, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QPolygonF, QPen
from imageviewer_ui import Ui_ImageViewer

import h5py
import os
import pandas as pd
import numpy as np
import sys

from functools import partial

#sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking')
sys.path.append('../..')

from MWTracker.trackWorms.getSkeletonsTables import getWormROI

import matplotlib.pylab as plt

class ImageViewer(QMainWindow):
	def __init__(self):
		super().__init__()
		
		# Set up the user interface from Designer.
		self.ui = Ui_ImageViewer()
		self.ui.setupUi(self)

		self.isPlay = False
		self.fid = -1
		self.image_group = -1
		self.trajectories_data = -1

		self.worm_index_roi1 = 1
		self.worm_index_roi2 = 1
		self.frame_data = -1
		self.h5path = self.ui.comboBox_h5path.itemText(0)
		
		
		#self.videos_dir = "/Users/ajaver/Google Drive/MWTracker_Example/MaskedVideos/18062015/"
		self.videos_dir = r"/Volumes/behavgenom$/GeckoVideo/"
		#"/Users/ajaver/Desktop/Gecko_compressed/MaskedVideos/"
		self.results_dir = ''
		self.skel_file = ''

		
		self.ui.pushButton_skel.clicked.connect(self.getSkelFile)
		self.ui.pushButton_video.clicked.connect(self.getVideoFile)
		
		self.ui.playButton.clicked.connect(self.playVideo)
		self.ui.imageSlider.sliderPressed.connect(self.imSldPressed)
		self.ui.imageSlider.sliderReleased.connect(self.imSldReleased)
		
		self.ui.spinBox_frame.valueChanged.connect(self.updateFrameNumber)
		self.ui.doubleSpinBox_fps.valueChanged.connect(self.updateFPS)
		self.ui.spinBox_step.valueChanged.connect(self.updateFrameStep)
		
		self.ui.comboBox_h5path.activated.connect(self.getImGroup)

		self.ui.comboBox_ROI1.activated.connect(self.updateROI1)
		self.ui.comboBox_ROI2.activated.connect(self.updateROI2)

		self.ui.checkBox_ROI1.stateChanged.connect(partial(self.updateROIcanvasN, 1))
		self.ui.checkBox_ROI2.stateChanged.connect(partial(self.updateROIcanvasN, 2))
		self.ui.checkBox_label.stateChanged.connect(self.updateImage)

		self.ui.imageCanvas.mousePressEvent = self.getPos


		self.updateFPS()
		self.updateFrameStep()
		
		# SET UP RECURRING EVENTS
		self.timer = QTimer()
		self.timer.timeout.connect(self.getNextImage)
		

	def getPos(self , event):
		x = event.pos().x()
		y = event.pos().y() 
		
		if isinstance(self.frame_data, pd.DataFrame):
			x /= self.img_w_ratio
			y /= self.img_h_ratio

			R = (x-self.frame_data['coord_x'])**2 + (y-self.frame_data['coord_y'])**2
			
			ind = R.idxmin()
			good_row = self.frame_data.loc[ind]
			if np.sqrt(R.loc[ind]) < good_row['roi_size']:
				worm_ind = int(good_row['worm_index_joined'])
				
				if self.ui.radioButton_ROI1.isChecked():
					self.worm_index_roi1 = worm_ind
					self.updateROIcanvasN(1)
				elif self.ui.radioButton_ROI2.isChecked():
					self.worm_index_roi2 = worm_ind
					self.updateROIcanvasN(2)
	
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
		
		self.ui.lineEdit_video.setText(self.vfilename)
		self.videos_dir = self.vfilename.rpartition(os.sep)[0] + os.sep
		dum = self.videos_dir.replace('MaskedVideos', 'Results')
		if os.path.exists(dum):
			self.results_dir = dum
			self.basename = self.vfilename.rpartition(os.sep)[-1].rpartition('.')[0]
			self.skel_file = self.results_dir + os.sep + self.basename + '_skeletons.hdf5'
			if not os.path.exists(self.skel_file):
				self.skel_file = ''
			self.ui.lineEdit_skel.setText(self.skel_file)

		self.fid = h5py.File(self.vfilename, 'r')
		self.updateImGroup()
		self.updateSkelFile()

	def getSkelFile(self):
		self.skel_file, _ = QFileDialog.getOpenFileName(self, 'Select file with the worm skeletons', 
			self.results_dir,  "Skeletons files (*_skeletons.hdf5);; All files (*)")
		self.ui.lineEdit_skel.setText(self.skel_file)

		if self.fid != -1:
			self.updateSkelFile()

	def updateSkelFile(self):
		if not self.skel_file or self.fid == -1:
			self.ske_file_id = -1
			return

		with pd.HDFStore(self.skel_file, 'r') as ske_file_id:
			trajectories_df = ske_file_id['/trajectories_data']
			self.trajectories_data = trajectories_df.groupby('frame_number')
			del(trajectories_df)

		self.ske_file_id = h5py.File(self.skel_file, 'r')
		
		self.skel_dat = {}
		self.skel_dat['skeleton'] = self.ske_file_id['/skeleton']
		self.skel_dat['contour_side1'] = self.ske_file_id['/contour_side1']
		self.skel_dat['contour_side2'] = self.ske_file_id['/contour_side2']

	def getImGroup(self, index):
		self.h5path = self.ui.comboBox_h5path.itemText(index)
		self.updateImGroup()

	#read a valid groupset from the hdf5
	def updateImGroup(self):
		if self.fid == -1:
			return

		if not self.h5path in self.fid:
			self.ui.imageCanvas.clear()
			self.image_group == -1
			QMessageBox.critical(self, 'The groupset path does not exists', "The groupset path does not exists. You must specify a valid groupset path",
					QMessageBox.Ok)
			return

		self.image_group = self.fid[self.h5path]
		if len(self.image_group.shape) != 3:
			self.ui.imageCanvas.clear()
			self.image_group == -1
			QMessageBox.critical(self, 'Invalid groupset', "Invalid groupset. The groupset must have three dimensions",
					QMessageBox.Ok)

		self.tot_frames = self.image_group.shape[0]
		self.image_height = self.image_group.shape[1]
		self.image_width = self.image_group.shape[2]
			
		self.ui.spinBox_frame.setMaximum(self.tot_frames-1)

		self.frame_number = 0
		self.ui.spinBox_frame.setValue(self.frame_number)

		if self.h5path == '/mask':
			self.updateSkelFile()

		self.updateImage()

	#update image if the GUI is resized event
	def resizeEvent(self, event):
		if self.fid != -1:
			self.updateImage()

	#Scroller
	def imSldPressed(self):
		self.ui.imageSlider.setCursor(Qt.ClosedHandCursor)
	
	def imSldReleased(self):
		self.ui.imageSlider.setCursor(Qt.OpenHandCursor)
		if self.image_group != -1:
			self.frame_number = round((self.tot_frames-1)*self.ui.imageSlider.value()/100)
			self.ui.spinBox_frame.setValue(self.frame_number)
	
	#frame spin box, important since I would update the image by changing the spinBox value
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

	#start to play video
	def startPlay(self):
		self.timer.start(round(1000/self.fps))
		self.isPlay = True
		self.ui.playButton.setText('Stop')
		self.ui.doubleSpinBox_fps.setEnabled(False)
		self.ui.comboBox_ROI1.setEnabled(False)
		self.ui.comboBox_ROI2.setEnabled(False)

	#stop to play video
	def stopPlay(self):
		self.timer.stop()
		self.isPlay = False
		self.ui.playButton.setText('Play')
		self.ui.doubleSpinBox_fps.setEnabled(True)
		self.ui.comboBox_ROI1.setEnabled(True)
		self.ui.comboBox_ROI2.setEnabled(True)

	#Function to get the new valid frame during video play
	def getNextImage(self):
		self.frame_number += self.frame_step
		if self.frame_number >= self.tot_frames:
			self.stopPlay()
			return
		self.ui.spinBox_frame.setValue(self.frame_number)

	
	#change frame number using the keys
	def keyPressEvent(self, event):
		#I couldn't make Qt to recognize the key arrows, so I am using <> or ,. instead
		if self.fid == -1:
			return

		#Move backwards when < or , are pressed
		if event.key() == 44 or event.key() == 60:
			self.frame_number -= self.frame_step
			if self.frame_number<0:
				self.frame_number = 0
			self.ui.spinBox_frame.setValue(self.frame_number)
		
		#Move forward when  > or . are pressed
		elif event.key() == 46 or event.key() == 62:
			self.frame_number += self.frame_step
			if self.frame_number >= self.tot_frames:
				self.frame_number = self.tot_frames-1
			self.ui.spinBox_frame.setValue(self.frame_number)

	#update image
	def updateImage(self):
		if self.image_group == -1:
			return

		self.ui.spinBox_frame.setValue(self.frame_number)
		
		self.label_height = self.ui.imageCanvas.height()
		self.label_width = self.ui.imageCanvas.width()

		self.original_image = self.image_group[self.frame_number];
		
		image = QImage(self.original_image.data, 
			self.image_height, self.image_width, self.original_image.strides[0], QImage.Format_Indexed8)
		image = image.scaled(self.label_width, self.label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
		self.img_w_ratio = image.size().width()/self.image_width;
		self.img_h_ratio = image.size().height()/self.image_height;
		
		if self.trajectories_data != -1:
			try:
				self.frame_data = self.trajectories_data.get_group(self.frame_number)
			except KeyError:
				self.frame_data = -1

			if isinstance(self.frame_data, pd.DataFrame) and self.ui.checkBox_label.isChecked():
			
				painter = QPainter()
				painter.begin(image)
				painter.setPen(QColor(205,205,205))
				painter.setFont(QFont('Decorative', 10))
				for row_id, row_data in self.frame_data.iterrows():
					x = int(row_data['coord_x']*self.img_w_ratio)
					y = int(row_data['coord_y']*self.img_h_ratio)
					painter.drawText(x, y, str(int(row_data['worm_index_joined'])))

					bb = row_data['roi_size']*self.img_w_ratio
					painter.drawRect(x-bb/2, y-bb/2, bb, bb);
				painter.end()
		
		pixmap = QPixmap.fromImage(image)
		self.ui.imageCanvas.setPixmap(pixmap);
		
		if self.ske_file_id != -1:
			self.updateROIcanvasN(1)
			self.updateROIcanvasN(2)
			
		progress = round(100*self.frame_number/self.tot_frames)
		if progress != self.ui.imageSlider.value():
			self.ui.imageSlider.setValue(progress)

	#update zoomed ROI
	def updateROI1(self, index):
		self.worm_index_roi1 = int(self.ui.comboBox_ROI1.itemText(index))
		self.updateROIcanvasN(1)

	def updateROI2(self, index):
		self.worm_index_roi2 = int(self.ui.comboBox_ROI2.itemText(index))
		self.updateROIcanvasN(2)

	def updateROIcanvasN(self, n_canvas):
		if n_canvas == 1:
			self.updateROIcanvas(self.ui.wormCanvas1, self.worm_index_roi1, self.ui.comboBox_ROI1, self.ui.checkBox_ROI1.isChecked())
		elif n_canvas == 2:
			self.updateROIcanvas(self.ui.wormCanvas2, self.worm_index_roi2, self.ui.comboBox_ROI2, self.ui.checkBox_ROI2.isChecked())


	#function that generalized the updating of the ROI
	def updateROIcanvas(self, wormCanvas, worm_index_roi, comboBox_ROI, isDrawSkel):
		if not isinstance(self.frame_data, pd.DataFrame):
			wormCanvas.clear()
			return
		
		#update valid index for the comboBox
		comboBox_ROI.clear()
		comboBox_ROI.addItem(str(worm_index_roi))
		
		for ind in self.frame_data['worm_index_joined'].data:
			comboBox_ROI.addItem(str(ind))
		
		canvas_size = min(wormCanvas.height(),wormCanvas.width())
		
		#extract individual worm ROI
		good = self.frame_data['worm_index_joined'] == worm_index_roi
		roi_data = self.frame_data.loc[good].squeeze()

		if roi_data.size == 0:
			wormCanvas.clear()
			return

		worm_roi, roi_corner = getWormROI(self.original_image, roi_data['coord_x'], roi_data['coord_y'], roi_data['roi_size'])
		worm_roi = np.ascontiguousarray(worm_roi)
		worm_img = QImage(worm_roi.data, worm_roi.shape[0], worm_roi.shape[1], worm_roi.strides[0], QImage.Format_Indexed8)
		worm_img = worm_img.scaled(canvas_size,canvas_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
		

		if isDrawSkel and roi_data['has_skeleton']:
			c_ratio = canvas_size/roi_data['roi_size'];
			skel_id = int(roi_data['skeleton_id'])

			qPlg = {}
			for tt in ['skeleton', 'contour_side1', 'contour_side2']:
				dat = (self.skel_dat[tt][skel_id] - roi_corner)*c_ratio
				qPlg[tt] = QPolygonF()
				for p in dat:
					qPlg[tt].append(QPointF(*p))
			
			self.skel_colors = {'skeleton':(27, 158, 119 ), 
			'contour_side1':(217, 95, 2), 'contour_side2':(231, 41, 138)}
			
			pen = QPen()
			pen.setWidth(2)
			
			painter = QPainter()
			painter.begin(worm_img)
		
			for tt, color in self.skel_colors.items():
				pen.setColor(QColor(*color))
				painter.setPen(pen)
				painter.drawPolyline(qPlg[tt])
			
			pen.setColor(Qt.black)
			painter.setBrush(Qt.white)
			painter.setPen(pen)
		
			radius = 1.5*c_ratio
			painter.drawEllipse(qPlg['skeleton'][0], radius, radius)

			painter.end()
		
		
		pixmap = QPixmap.fromImage(worm_img)
		wormCanvas.setPixmap(pixmap);
		
if __name__ == '__main__':
	app = QApplication(sys.argv)
	
	ui = ImageViewer()
	ui.show()
	
	sys.exit(app.exec_())