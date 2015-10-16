import h5py
import os
import pandas as pd
import numpy as np
import sys
import cv2
import tables
from functools import partial

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QFrame
from PyQt5.QtCore import QDir, QTimer, Qt, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QPolygonF, QPen

from MWTracker.GUI.MWTrackerViewer.MWTrackerViewer_ui import Ui_ImageViewer

from MWTracker.trackWorms.getSkeletonsTables import getWormROI, getWormMask
from MWTracker.trackWorms.segWormPython.mainSegworm import binaryMask2Contour
from MWTracker.FeaturesAnalysis.obtainFeatures_N import getWormFeaturesLab

class MWTrackerViewer_GUI(QMainWindow):
	def __init__(self):
		super().__init__()
		# Set up the user interface from Designer.
		self.ui = Ui_ImageViewer()
		self.ui.setupUi(self)

		self.lastKey = ''
		self.isPlay = False
		self.fid = -1
		self.image_group = -1
		self.trajectories_data = -1

		self.worm_index_roi1 = 1
		self.worm_index_roi2 = 1
		self.frame_data = -1
		self.h5path = self.ui.comboBox_h5path.itemText(0)
		
		self.wlab = {'U':0, 'WORM':1, 'WORMS':2, 'BAD':3}
		#self.ui.comboBox_labels.clear()
		#for lab in ('Undefined', 'Single Worm', 'Worm Cluster', 'Bad'):
		#	self.ui.comboBox_labels.addItem(lab)

		self.wlabC = {self.wlab['U']:Qt.white, self.wlab['WORM']:Qt.green, self.wlab['WORMS']:Qt.blue, self.wlab['BAD']:Qt.darkRed}
		
		#self.videos_dir = '/Volumes/behavgenom$/Andre/shige-oda/MaskedVideos/'
		#self.videos_dir = "/Users/ajaver/Google Drive/MWTracker_Example/MaskedVideos/"
		self.videos_dir = r"/Volumes/behavgenom$/GeckoVideo/MaskedVideos/"
		#self.videos_dir = "/Users/ajaver/Desktop/Gecko_compressed/MaskedVideos/"
		
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
		self.ui.checkBox_showLabel.stateChanged.connect(self.updateImage)

		self.ui.pushButton_h5groups.clicked.connect(self.updateGroupNames)

		#flags for RW and FF
		self.RW = 1
		self.FF = 2
		self.ui.pushButton_ROI1_RW.clicked.connect(partial(self.roiRWFF, 1, self.RW))
		self.ui.pushButton_ROI1_FF.clicked.connect(partial(self.roiRWFF, 1, self.FF))
		self.ui.pushButton_ROI2_RW.clicked.connect(partial(self.roiRWFF, 2, self.RW))
		self.ui.pushButton_ROI2_FF.clicked.connect(partial(self.roiRWFF, 2, self.FF))

		self.ui.pushButton_U.clicked.connect(partial(self.tagWorm, self.wlab['U']))
		self.ui.pushButton_W.clicked.connect(partial(self.tagWorm, self.wlab['WORM']))
		self.ui.pushButton_WS.clicked.connect(partial(self.tagWorm, self.wlab['WORMS']))
		self.ui.pushButton_B.clicked.connect(partial(self.tagWorm, self.wlab['BAD']))

		self.ui.pushButton_save.clicked.connect(self.saveData)

		self.ui.radioButton_ROI1.toggled.connect(self.makeEditableROI1)
		self.ui.radioButton_ROI2.toggled.connect(self.makeEditableROI2)

		self.ui.imageCanvas.mousePressEvent = self.getPos

		#self.ui.pushButton_bad.clicked.connect(self.tagAsBad)
		#self.ui.pushButton_wormS.clicked.connect(self.tagAsWormS)
		#self.ui.pushButton_worm.clicked.connect(self.tagAsWorm)

		self.ui.pushButton_join.clicked.connect(self.joinTraj)
		self.ui.pushButton_split.clicked.connect(self.splitTraj)
		
		self.ui.pushButton_feats.clicked.connect(self.calcIndFeat)

		self.updateFPS()
		self.updateFrameStep()
		
		# SET UP RECURRING EVENTS
		self.timer = QTimer()
		self.timer.timeout.connect(self.getNextImage)
	
	def makeEditableROI1(self, state):
		self.ui.comboBox_ROI1.setEditable(state)
	def makeEditableROI2(self, state):
		self.ui.comboBox_ROI2.setEditable(state)


	def calcIndFeat(self):
		if self.image_group == -1:
			return

		#save data
		self.saveData()
		#close GUI
		self.close()

		#start the analysis
		trajectories_data = self.trajectories_data[self.trajectories_data['worm_label']==1]
		worm_indexes = trajectories_data['worm_index_N'].unique()

		features_file = self.results_dir + os.sep + self.basename + '_feat_ind.hdf5'
		getWormFeaturesLab(self.skel_file, features_file, worm_indexes)

	def saveData(self):
		
		#pytables saving format is more convenient than pandas
		
		#convert data into a rec array to save into pytables
		trajectories_recarray = self.trajectories_data.to_records(index=False)

		with tables.File(self.skel_file, "r+") as ske_file_id:
			#pytables filters.
			table_filters = tables.Filters(complevel=5, complib='zlib', shuffle=True, fletcher32=True)
			
			newT = ske_file_id.create_table('/', 'trajectories_data_d', obj = trajectories_recarray, filters=table_filters)
			
			ske_file_id.remove_node('/', 'trajectories_data')
			newT.rename('trajectories_data')
		
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
				worm_ind = int(good_row['worm_index_N'])
				
				if self.ui.radioButton_ROI1.isChecked():
					self.worm_index_roi1 = worm_ind
					self.updateROIcanvasN(1)
					#self.ui.radioButton_ROI2.setChecked(True)

				elif self.ui.radioButton_ROI2.isChecked():
					self.worm_index_roi2 = worm_ind
					self.updateROIcanvasN(2)
					#self.ui.radioButton_ROI1.setChecked(True)

				#if self.ui.checkBox_labelROI.isChecked():
				#	label_ind = self.ui.comboBox_labels.currentIndex()
				#	good = self.trajectories_data['worm_index_N'] == worm_ind
				#	self.trajectories_data.loc[good, 'worm_label'] = label_ind
				#	self.updateImage()
				
				#if self.lastKey == 81: #q
				#	self.ui.spinBox_join1.setValue(worm_ind)
				#elif self.lastKey == 87: #w
				#	self.ui.spinBox_join2.setValue(worm_ind)
	
	def tagWorm(self, label_ind):

		if self.ui.radioButton_ROI1.isChecked():
			worm_ind = self.worm_index_roi1
		elif self.ui.radioButton_ROI2.isChecked():
			worm_ind = self.worm_index_roi2
		
		if not isinstance(self.frame_data, pd.DataFrame):
			return

		if not worm_ind in self.frame_data['worm_index_N'].values:
			QMessageBox.critical(self, 'The selected worm is not in this frame.', 'Select a worm in the current frame to label.',
					QMessageBox.Ok)
			return

		good = self.trajectories_data['worm_index_N'] == worm_ind
		self.trajectories_data.loc[good, 'worm_label'] = label_ind
		self.updateImage()

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

	def updateGroupNames(self):
		valid_groups = []
		def geth5name(name, dat):
			if isinstance(dat, h5py.Dataset) and len(dat.shape) == 3 and dat.dtype == np.uint8:
		 		valid_groups.append('/' + name)
		self.fid.visititems(geth5name)
		
		if not valid_groups:
			QMessageBox.critical(self, '', "No valid video groups were found. Dataset with three dimensions and uint8 data type.",
					QMessageBox.Ok)
			return

		self.ui.comboBox_h5path.clear()
		for kk in valid_groups:
			self.ui.comboBox_h5path.addItem(kk)

		self.getImGroup(0)
		self.self.updateImage()

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
			self.trajectories_data = ske_file_id['/trajectories_data']

			if not 'worm_index_N' in self.trajectories_data.columns:
				self.trajectories_data['worm_label'] = self.wlab['U']
				self.trajectories_data['worm_index_N'] = self.trajectories_data['worm_index_joined']

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
		
		if event.key() == 16777235: #q
			self.ui.radioButton_ROI1.setChecked(True)
		elif event.key() == 16777237: #w
			self.ui.radioButton_ROI2.setChecked(True)

		#Duplicate the frame step size when pressed  > or .: 
		if event.key() == 46 or event.key() == 62:
			self.frame_step *= 2
			self.ui.spinBox_step.setValue(self.frame_step)

		#Half the frame step size when pressed: < or ,
		if event.key() == 44 or event.key() == 60:
			self.frame_step //=2
			if self.frame_step<1:
				self.frame_step = 1
			self.ui.spinBox_step.setValue(self.frame_step)

		#I couldn't make Qt to recognize the key arrows, so I am using <> or ,. instead
		if self.fid == -1:
			return

		#Move backwards when  are pressed
		if event.key() == 16777234:
			self.frame_number -= self.frame_step
			if self.frame_number<0:
				self.frame_number = 0
			self.ui.spinBox_frame.setValue(self.frame_number)
		
		#Move forward when  are pressed
		elif event.key() == 16777236:
			self.frame_number += self.frame_step
			if self.frame_number >= self.tot_frames:
				self.frame_number = self.tot_frames-1
			self.ui.spinBox_frame.setValue(self.frame_number)

		#undefined: u
		if event.key() == 85:
			self.tagWorm(self.wlab['U'])
		#worm: w
		elif event.key() == 87:
			self.tagWorm(self.wlab['WORM'])
		#worm cluster: c
		elif event.key() ==  67:
			self.tagWorm(self.wlab['WORMS'])
		#bad: b
		elif event.key() ==  66:
			self.tagWorm(self.wlab['BAD'])
		#s
		elif event.key() == 74:
			self.joinTraj()
		#j
		elif event.key() == 83:
			self.splitTraj()

		elif event.key() == 91:
			current_roi = 1 if self.ui.radioButton_ROI1.isChecked() else 2
			self.roiRWFF(current_roi, self.RW)
		#[ <<
		elif event.key() == 93:
			current_roi = 1 if self.ui.radioButton_ROI1.isChecked() else 2
			self.roiRWFF(current_roi, self.FF)
		#] >>

	#move to the first or the last frames of a trajectory
	def roiRWFF(self, n_roi, rwff):

		if not isinstance(self.frame_data, pd.DataFrame):
			return

		if n_roi == 1:
			worm_ind = self.worm_index_roi1
		else:
			worm_ind = self.worm_index_roi2

		
		#use 1 for rewind RW or 2 of fast forward
		good = self.trajectories_data['worm_index_N'] == worm_ind
		frames = self.trajectories_data.loc[good, 'frame_number']

		if frames.size == 0:
			return

		if rwff == 1: 
			self.frame_number =  frames.min()
		elif rwff == 2:
			self.frame_number =  frames.max()

		self.updateImage()

	#update image
	def updateImage(self):
		if self.image_group == -1:
			return

		self.ui.spinBox_frame.setValue(self.frame_number)
		
		self.label_height = self.ui.imageCanvas.height()
		self.label_width = self.ui.imageCanvas.width()

		self.original_image = self.image_group[self.frame_number];

		image = QImage(self.original_image.data, 
			self.image_width, self.image_height, self.original_image.strides[0], QImage.Format_Indexed8)
		image = image.convertToFormat(QImage.Format_RGB32, Qt.AutoColor)
		image = image.scaled(self.label_width, self.label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
		
		self.img_h_ratio = image.height()/self.image_height;
		self.img_w_ratio = image.width()/self.image_width;
		
		if isinstance(self.trajectories_data, pd.DataFrame): 
			try:
				self.frame_data = self.trajectories_data[self.trajectories_data['frame_number'] == self.frame_number]#self.trajectories_data.get_group(self.frame_number)
			except KeyError:
				self.frame_data = -1

			if isinstance(self.frame_data, pd.DataFrame) and self.ui.checkBox_showLabel.isChecked():
				
				painter = QPainter()
				painter.begin(image)
				for row_id, row_data in self.frame_data.iterrows():
					x = int(row_data['coord_x']*self.img_h_ratio)
					y = int(row_data['coord_y']*self.img_w_ratio)
					
					c = self.wlabC[int(row_data['worm_label'])]
					
					painter.setPen(c)
					painter.setFont(QFont('Decorative', 10))
				
					painter.drawText(x, y, str(int(row_data['worm_index_N'])))

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
		try:
			self.worm_index_roi1 = int(self.ui.comboBox_ROI1.itemText(index))
		except ValueError:
			self.worm_index_roi1 = -1

		self.updateROIcanvasN(1)

	def updateROI2(self, index):
		try:
			self.worm_index_roi2 = int(self.ui.comboBox_ROI2.itemText(index))
		except ValueError:
			self.worm_index_roi2 = -1
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
		
		for ind in self.frame_data['worm_index_N'].data:
			comboBox_ROI.addItem(str(ind))
		
		canvas_size = min(wormCanvas.height(),wormCanvas.width())
		
		#extract individual worm ROI
		good = self.frame_data['worm_index_N'] == worm_index_roi
		roi_data = self.frame_data.loc[good].squeeze()

		if roi_data.size == 0:
			wormCanvas.clear()
			return

		worm_roi, roi_corner = getWormROI(self.original_image, roi_data['coord_x'], roi_data['coord_y'], roi_data['roi_size'])
		worm_roi = np.ascontiguousarray(worm_roi)
		
		#worm_roi = cv2.cvtColor(worm_img, cv2.COLOR_GRAY2RGB);
		worm_img = QImage(worm_roi.data, worm_roi.shape[0], worm_roi.shape[1], worm_roi.strides[0], QImage.Format_Indexed8)
		worm_img = worm_img.scaled(canvas_size,canvas_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

		if isDrawSkel:
			if roi_data['has_skeleton']:
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
			else:
				worm_mask = getWormMask(worm_roi, roi_data['threshold'])
				worm_cnt = binaryMask2Contour(worm_mask)
				worm_mask = np.zeros_like(worm_mask)
				cv2.drawContours(worm_mask, [worm_cnt.astype(np.int32)], 0, 1, -1)

				worm_mask = QImage(worm_mask.data, worm_mask.shape[0], 
					worm_mask.shape[1], worm_mask.strides[0], QImage.Format_Indexed8)
				worm_mask = worm_mask.scaled(canvas_size,canvas_size, 
					Qt.KeepAspectRatio, Qt.SmoothTransformation)
				worm_mask = QPixmap.fromImage(worm_mask)

				worm_mask = worm_mask.createMaskFromColor(Qt.black)
				p = QPainter(worm_img)
				p.setPen(QColor(0,204,102))
				p.drawPixmap(worm_img.rect(), worm_mask, worm_mask.rect())
				p.end()

		
		
		pixmap = QPixmap.fromImage(worm_img)
		wormCanvas.setPixmap(pixmap);
	
	

	def joinTraj(self):
		worm_ind1 = self.worm_index_roi1#self.ui.spinBox_join1.value()
		worm_ind2 = self.worm_index_roi2#self.ui.spinBox_join2.value()
		
		if worm_ind1 == worm_ind2:
			QMessageBox.critical(self, 'Cannot join the same trajectory with itself', 'Cannot join the same trajectory with itself.',
					QMessageBox.Ok)
			return

		index1 = (self.trajectories_data['worm_index_N'] == worm_ind1).values
		index2 = (self.trajectories_data['worm_index_N'] == worm_ind2).values
		
		#if the trajectories do not overlap they shouldn't have frame_number indexes in commun
		frame_number = self.trajectories_data.loc[index1|index2, 'frame_number']
		
		if frame_number.size != np.unique(frame_number).size:
			QMessageBox.critical(self, 'Cannot join overlaping trajectories', 'Cannot join overlaping trajectories.',
					QMessageBox.Ok)
			return


		if not (worm_ind1 in self.frame_data['worm_index_N'].values or worm_ind2 in self.frame_data['worm_index_N'].values):
			reply = QMessageBox.question(self, 'Message',
            "The none of the selected worms to join is not in this frame. Are you sure to continue?",
             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
			
			if reply == QMessageBox.No:
				return

		#get the first row for each segment to extract some data
		first_row1 = self.trajectories_data.loc[index1, :].iloc[0]
		first_row2 = self.trajectories_data.loc[index2, :].iloc[0]

		#assign the largest ROI size to both trajectories
		#roi_size = max(first_row1['roi_size_N'], first_row2['roi_size'])
		#self.trajectories_data.loc[index1, 'roi_size_N'] = roi_size
		#self.trajectories_data.loc[index2, 'roi_size_N'] = roi_size

		#join trajectories
		self.trajectories_data.loc[index2, 'worm_label'] = first_row1['worm_label']
		self.trajectories_data.loc[index2, 'worm_index_N'] = worm_ind1
		
		self.worm_index_roi1 = worm_ind1
		self.worm_index_roi2 = worm_ind1
		self.updateImage()

		self.ui.spinBox_join1.setValue(worm_ind1)
		self.ui.spinBox_join2.setValue(worm_ind1)


	def splitTraj(self):
		if self.ui.radioButton_ROI1.isChecked():
			worm_ind = self.worm_index_roi1#self.ui.spinBox_join1.value()
		elif self.ui.radioButton_ROI2.isChecked():
			worm_ind = self.worm_index_roi2

		if not worm_ind in self.frame_data['worm_index_N'].data:
			QMessageBox.critical(self, 'Worm index is not in the current frame.', 'Worm index is not in the current frame. Select a valid index.',
					QMessageBox.Ok)
			return

		last_index = self.trajectories_data['worm_index_N'].max()
		
		new_ind1 = last_index+1
		new_ind2 = last_index+2

		good = self.trajectories_data['worm_index_N'] == worm_ind
		frames = self.trajectories_data.loc[good, 'frame_number']
		frames = frames.sort(inplace=False)

		good = frames<self.frame_number
		index1 = frames[good].index
		index2 = frames[~good].index
		self.trajectories_data.ix[index1, 'worm_index_N']=new_ind1
		self.trajectories_data.ix[index2, 'worm_index_N']=new_ind2
		
		self.worm_index_roi1 = new_ind1
		self.worm_index_roi2 = new_ind2
		self.updateImage()
		
		self.ui.spinBox_join1.setValue(new_ind1)
		self.ui.spinBox_join2.setValue(new_ind2)

if __name__ == '__main__':
	app = QApplication(sys.argv)
	
	ui = MWTrackerViewer_GUI()
	ui.show()
	
	sys.exit(app.exec_())