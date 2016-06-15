from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap, QImage, QPainter, QFont, QPen
from PyQt5.QtCore import Qt, QEvent

import h5py
import os
import pandas as pd
import numpy as np
import sys
import cv2
import json
import tables
from functools import partial

import sys

from MWTracker.GUI_Qt5.MWTrackerViewer.MWTrackerViewer_ui import Ui_ImageViewer
from MWTracker.GUI_Qt5.MWTrackerViewerSingle.MWTrackerViewerSingle_GUI import MWTrackerViewerSingle_GUI

from MWTracker.trackWorms.getSkeletonsTables import getWormROI, getWormMask, binaryMask2Contour
from MWTracker.featuresAnalysis.obtainFeatures import getWormFeaturesFilt
from MWTracker.helperFunctions.trackProvenance import getGitCommitHash, execThisPoint
from MWTracker.helperFunctions.tracker_param import tracker_param

class MWTrackerViewer_GUI(MWTrackerViewerSingle_GUI):
	def __init__(self, ui='', argv=''):
		if not ui:
			super().__init__(Ui_ImageViewer())
		else:
			super().__init__(ui)

		self.vfilename = '' if len(argv)<=1 else argv[1]
		
		self.lastKey = ''
		self.isPlay = False
		self.fid = -1
		self.image_group = -1
		self.trajectories_data = -1

		self.worm_index_roi1 = 1
		self.worm_index_roi2 = 1
		self.frame_data = -1
		self.h5path = self.ui.comboBox_h5path.itemText(0)
		
		self.wlab = {'U':0, 'WORM':1, 'WORMS':2, 'BAD':3, 'GOOD_SKE':4}
		self.wlabC = {self.wlab['U']:Qt.white, self.wlab['WORM']:Qt.green, self.wlab['WORMS']:Qt.blue, self.wlab['BAD']:Qt.darkRed, self.wlab['GOOD_SKE']:Qt.darkCyan}
		
		self.videos_dir = r"/Volumes/behavgenom$/GeckoVideo/MaskedVideos/"
		self.results_dir = ''
		self.skeletons_file = ''
		self.worm_index_type = 'worm_index_manual'
		self.label_type = 'worm_label'

		self.ui.comboBox_ROI1.activated.connect(self.selectROI1)
		self.ui.comboBox_ROI2.activated.connect(self.selectROI2)
		self.ui.checkBox_ROI1.stateChanged.connect(partial(self.updateROIcanvasN, 1))
		self.ui.checkBox_ROI2.stateChanged.connect(partial(self.updateROIcanvasN, 2))
		
		self.ui.comboBox_labelType.currentIndexChanged.connect(self.selectWormIndexType)
		
		self.ui.pushButton_feats.clicked.connect(self.getManualFeatures)

		#flags for RW and FF
		self.RW, self.FF = 1, 2
		self.ui.pushButton_ROI1_RW.clicked.connect(partial(self.roiRWFF, 1, self.RW))
		self.ui.pushButton_ROI1_FF.clicked.connect(partial(self.roiRWFF, 1, self.FF))
		self.ui.pushButton_ROI2_RW.clicked.connect(partial(self.roiRWFF, 2, self.RW))
		self.ui.pushButton_ROI2_FF.clicked.connect(partial(self.roiRWFF, 2, self.FF))

		self.ui.pushButton_U.clicked.connect(partial(self.tagWorm, self.wlab['U']))
		self.ui.pushButton_W.clicked.connect(partial(self.tagWorm, self.wlab['WORM']))
		self.ui.pushButton_WS.clicked.connect(partial(self.tagWorm, self.wlab['WORMS']))
		self.ui.pushButton_B.clicked.connect(partial(self.tagWorm, self.wlab['BAD']))

		self.ui.pushButton_save.clicked.connect(self.saveData)

		self.ui.pushButton_join.clicked.connect(self.joinTraj)
		self.ui.pushButton_split.clicked.connect(self.splitTraj)
		
		#select worm ROI when doubleclick a worm
		self.ui.imageCanvas.mouseDoubleClickEvent = self.selectWorm

		
	def getManualFeatures(self):
		self.saveData()
		self.close()

		self.feat_manual_file = self.skeletons_file.replace('_skeletons.hdf5', '_feat_manual.hdf5')
		self.commit_hash = getGitCommitHash()
		
		
		with h5py.File(self.vfilename, 'r') as mask_fid, \
			h5py.File(self.skeletons_file, 'r') as skel_fid:
			
			has_expected_fps = 'expected_fps' in mask_fid['/mask'].attrs
			has_prov_skel_filt = '/provenance_tracking/SKE_FILT' in skel_fid
			
			#if any of this fields is missing load the default parameters
			if not has_expected_fps or not has_prov_skel_filt:
				param_default = tracker_param()
				param_default.get_param()

			if has_expected_fps:
				expected_fps = mask_fid['/mask'].attrs['expected_fps']
			else:
				expected_fps = param_default.expected_fps

			if has_prov_skel_filt:
				ss = skel_fid['/provenance_tracking/SKE_FILT'].value
				ss = json.loads(ss.decode("utf-8"))
				feat_filt_param = json.loads(ss['func_arguments'])
				use_skel_filter = feat_filt_param['use_skel_filter']
				del feat_filt_param['use_skel_filter']
				del feat_filt_param['skeletons_file']
			else:
				use_skel_filter = True
				feat_filt_param = param_default.feat_filt_param

		points_parameters = { 'func':getWormFeaturesFilt,
            	'argkws':{
            	'skeletons_file':self.skeletons_file, 
            	'features_file':self.feat_manual_file,  
                'expected_fps': expected_fps, 'is_single_worm':False, 
                'use_skel_filter':use_skel_filter, 'use_manual_join':True,
                'feat_filt_param':feat_filt_param
                },
        	    'output_file':self.feat_manual_file
        	}

		execThisPoint('FEAT_MANUAL_CREATE', **points_parameters, 
                    commit_hash=self.commit_hash, cmd_original='')
	
	def selectWormIndexType(self):
		#select between automatic and manual worm indexing and label
		if self.ui.comboBox_labelType.currentIndex() == 0:
			self.label_type = 'worm_label'
			self.worm_index_type = 'worm_index_manual' 
			self.ui.pushButton_U.setEnabled(True)
			self.ui.pushButton_W.setEnabled(True)
			self.ui.pushButton_WS.setEnabled(True)
			self.ui.pushButton_B.setEnabled(True)
			self.ui.pushButton_join.setEnabled(True)
			self.ui.pushButton_split.setEnabled(True)
			
		else:
			self.label_type = 'auto_label'
			self.worm_index_type = 'worm_index_auto'
			self.ui.pushButton_U.setEnabled(False)
			self.ui.pushButton_W.setEnabled(False)
			self.ui.pushButton_WS.setEnabled(False)
			self.ui.pushButton_B.setEnabled(False)
			self.ui.pushButton_join.setEnabled(False)
			self.ui.pushButton_split.setEnabled(False)
			
		self.updateImage()

	def saveData(self):
		'''save data from manual labelling. pytables saving format is more convenient than pandas'''
		
		#convert data into a rec array to save into pytables
		trajectories_recarray = self.trajectories_data.to_records(index=False)

		with tables.File(self.skeletons_file, "r+") as ske_file_id:
			#pytables filters.
			table_filters = tables.Filters(complevel=5, complib='zlib', shuffle=True, fletcher32=True)
			
			newT = ske_file_id.create_table('/', 'trajectories_data_d', obj = trajectories_recarray, filters=table_filters)
			
			ske_file_id.remove_node('/', 'trajectories_data')
			newT.rename('trajectories_data')

		self.updateSkelFile()

	def updateSkelFile(self):
		super().updateSkelFile()
		if not self.skeletons_file or not isinstance(self.trajectories_data, pd.DataFrame):
			return

		if 'worm_index_N' in self.trajectories_data:
			self.trajectories_data.rename(columns={'worm_index_N':'worm_index_manual'})

		if not 'worm_index_manual' in self.trajectories_data:
			self.trajectories_data['worm_label'] = self.wlab['U']
			self.trajectories_data['worm_index_manual'] = self.trajectories_data['worm_index_joined']
			self.updateImage()

	#update image
	def updateImage(self):
		if self.image_group == -1:
			return

		super(MWTrackerViewerSingle_GUI,self).readImage()
		self.img_h_ratio = self.frame_qimg.height()/self.image_height;
		self.img_w_ratio = self.frame_qimg.width()/self.image_width;
		
		#read the data of the particles that exists in the frame
		if isinstance(self.traj_time_grouped,  pd.core.groupby.DataFrameGroupBy):
			try:
				self.frame_data = self.traj_time_grouped.get_group(self.frame_number)
				self.frame_data = self.frame_data[self.frame_data[self.worm_index_type] >= 0]
			except KeyError:
				self.frame_data = -1

		#draw the boxes in each of the trajectories found		
		if self.ui.checkBox_showLabel.isChecked():
			self.drawROIBoxes(self.frame_qimg)

		#create the pixmap for the label
		pixmap = QPixmap.fromImage(self.frame_qimg)
		self.ui.imageCanvas.setPixmap(pixmap);
		
		self.updateROIcanvasN(1)
		self.updateROIcanvasN(2)

	def drawROIBoxes(self, image):
		#print(self.frame_data)
		if not isinstance(self.frame_data, pd.DataFrame) or len(self.frame_data) == 0 \
		or not self.label_type in self.frame_data:
			return 
		 			
		self.img_h_ratio = image.height()/self.image_height;
		self.img_w_ratio = image.width()/self.image_width;

		fontsize = max(1, max(image.height(), image.width())//60)
		penwidth = max(1, max(image.height(), image.width())//600)
		

		painter = QPainter()
		painter.begin(image)
		for row_id, row_data in self.frame_data.iterrows():
			x = row_data['coord_x']*self.img_h_ratio
			y = row_data['coord_y']*self.img_w_ratio
			#check if the coordinates are nan
			if not (x == x) or  not (y == y): 
				continue

			x = int(x)
			y = int(y)
			pen = QPen(self.wlabC[int(row_data[self.label_type])])
			pen.setWidth(penwidth)
			painter.setPen(pen)

			painter.setFont(QFont('Decorative', fontsize))
			
			painter.drawText(x, y, str(int(row_data[self.worm_index_type])))

			bb = row_data['roi_size']*self.img_w_ratio
			painter.drawRect(x-bb/2, y-bb/2, bb, bb);
		painter.end()

	#update zoomed ROI
	def selectROI1(self, index):
		try:
			self.worm_index_roi1 = int(self.ui.comboBox_ROI1.itemText(index))
		except ValueError:
			self.worm_index_roi1 = -1

		self.updateROIcanvasN(1)

	def selectROI2(self, index):
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
			#no trajectories data presented, nothing to do here
			wormCanvas.clear()
			return
		
		#update valid index for the comboBox
		comboBox_ROI.clear()
		comboBox_ROI.addItem(str(worm_index_roi))
		#add the indexes of the current frame into the roi combo box
		for ind in self.frame_data[self.worm_index_type].data:
			comboBox_ROI.addItem(str(ind))
		
		
		#extract individual worm ROI
		good = self.frame_data[self.worm_index_type] == worm_index_roi
		row_data = self.frame_data.loc[good].squeeze()

		if row_data.size == 0 or np.isnan(row_data['coord_x']) or np.isnan(row_data['coord_y']):
			#invalid data nothing to do here
			wormCanvas.clear()
			return
		
		worm_img, roi_corner = getWormROI(self.frame_img, row_data['coord_x'], row_data['coord_y'], row_data['roi_size'])
		#roi_corner = roi_corner+1
		
		roi_ori_size = worm_img.shape
		
		worm_img = np.ascontiguousarray(worm_img)
		worm_qimg = QImage(worm_img.data, worm_img.shape[1], worm_img.shape[0], worm_img.strides[0], QImage.Format_Indexed8)
		worm_qimg = worm_qimg.convertToFormat(QImage.Format_RGB32, Qt.AutoColor)

		canvas_size = min(wormCanvas.height(),wormCanvas.width())
		worm_qimg = worm_qimg.scaled(canvas_size,canvas_size, Qt.KeepAspectRatio)
		
		if isDrawSkel:
			if row_data['has_skeleton'] == 1:
				self.drawSkel(worm_img, worm_qimg, row_data, roi_corner = roi_corner)
			elif row_data['has_skeleton'] == 0:
				self.drawThreshMask(worm_img, worm_qimg, row_data, read_center=False)


		pixmap = QPixmap.fromImage(worm_qimg)
		wormCanvas.setPixmap(pixmap);
	


	def selectWorm(self , event):

		x = event.pos().x()
		y = event.pos().y() 
		
		if not isinstance(self.frame_data, pd.DataFrame) or len(self.frame_data) == 0:
			return 
		
		x /= self.img_w_ratio
		y /= self.img_h_ratio
		R = (x-self.frame_data['coord_x'])**2 + (y-self.frame_data['coord_y'])**2

		ind = R.idxmin()

		good_row = self.frame_data.loc[ind]
		if np.sqrt(R.loc[ind]) < good_row['roi_size']:
			worm_ind = int(good_row[self.worm_index_type])
			
			if self.ui.radioButton_ROI1.isChecked():
				self.worm_index_roi1 = worm_ind
				self.updateROIcanvasN(1)

			elif self.ui.radioButton_ROI2.isChecked():
				self.worm_index_roi2 = worm_ind
				self.updateROIcanvasN(2)



	def tagWorm(self, label_ind):
		if not self.worm_index_type == 'worm_index_manual':
					return
		if self.ui.radioButton_ROI1.isChecked():
			worm_ind = self.worm_index_roi1
		elif self.ui.radioButton_ROI2.isChecked():
			worm_ind = self.worm_index_roi2
		
		if not isinstance(self.frame_data, pd.DataFrame):
			return

		if not worm_ind in self.frame_data['worm_index_manual'].values:
			QMessageBox.critical(self, 'The selected worm is not in this frame.', 'Select a worm in the current frame to label.',
					QMessageBox.Ok)
			return

		good = self.trajectories_data['worm_index_manual'] == worm_ind
		self.trajectories_data.loc[good, 'worm_label'] = label_ind
		self.updateImage()

	#move to the first or the last frames of a trajectory
	def roiRWFF(self, n_roi, rwff):

		if not isinstance(self.frame_data, pd.DataFrame):
			return

		if n_roi == 1:
			worm_ind = self.worm_index_roi1
		else:
			worm_ind = self.worm_index_roi2
		
		#use 1 for rewind RW or 2 of fast forward
		good = self.trajectories_data[self.worm_index_type] == worm_ind
		frames = self.trajectories_data.loc[good, 'frame_number']

		if frames.size == 0:
			return

		if rwff == 1: 
			self.frame_number =  frames.min()
		elif rwff == 2:
			self.frame_number =  frames.max()
		
		self.ui.spinBox_frame.setValue(self.frame_number)

	def joinTraj(self):
		if not self.worm_index_type == 'worm_index_manual':
			return

		worm_ind1 = self.worm_index_roi1
		worm_ind2 = self.worm_index_roi2
		
		if worm_ind1 == worm_ind2:
			QMessageBox.critical(self, 'Cannot join the same trajectory with itself', 'Cannot join the same trajectory with itself.',
					QMessageBox.Ok)
			return

		index1 = (self.trajectories_data['worm_index_manual'] == worm_ind1).values
		index2 = (self.trajectories_data['worm_index_manual'] == worm_ind2).values
		
		#if the trajectories do not overlap they shouldn't have frame_number indexes in commun
		frame_number = self.trajectories_data.loc[index1|index2, 'frame_number']
		
		if frame_number.size != np.unique(frame_number).size:
			QMessageBox.critical(self, 'Cannot join overlaping trajectories', 'Cannot join overlaping trajectories.',
					QMessageBox.Ok)
			return


		if not (worm_ind1 in self.frame_data['worm_index_manual'].values or worm_ind2 in self.frame_data['worm_index_manual'].values):
			reply = QMessageBox.question(self, 'Message',
            "The none of the selected worms to join is not in this frame. Are you sure to continue?",
             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
			
			if reply == QMessageBox.No:
				return

		#get the first row for each segment to extract some data
		first_row1 = self.trajectories_data.loc[index1, :].iloc[0]
		first_row2 = self.trajectories_data.loc[index2, :].iloc[0]

		#join trajectories
		self.trajectories_data.loc[index2, 'worm_label'] = first_row1['worm_label']
		self.trajectories_data.loc[index2, 'worm_index_manual'] = worm_ind1
		
		self.worm_index_roi1 = worm_ind1
		self.worm_index_roi2 = worm_ind1
		self.updateImage()

		self.ui.spinBox_join1.setValue(worm_ind1)
		self.ui.spinBox_join2.setValue(worm_ind1)


	def splitTraj(self):
		if not self.worm_index_type == 'worm_index_manual':
			return

		if self.ui.radioButton_ROI1.isChecked():
			worm_ind = self.worm_index_roi1#self.ui.spinBox_join1.value()
		elif self.ui.radioButton_ROI2.isChecked():
			worm_ind = self.worm_index_roi2

		if not worm_ind in self.frame_data['worm_index_manual'].data:
			QMessageBox.critical(self, 'Worm index is not in the current frame.', 'Worm index is not in the current frame. Select a valid index.',
					QMessageBox.Ok)
			return

		last_index = self.trajectories_data['worm_index_manual'].max()
		
		new_ind1 = last_index+1
		new_ind2 = last_index+2

		good = self.trajectories_data['worm_index_manual'] == worm_ind
		frames = self.trajectories_data.loc[good, 'frame_number']
		frames = frames.sort_values(inplace=False)

		good = frames<self.frame_number
		index1 = frames[good].index
		index2 = frames[~good].index
		self.trajectories_data.ix[index1, 'worm_index_manual']=new_ind1
		self.trajectories_data.ix[index2, 'worm_index_manual']=new_ind2
		
		self.worm_index_roi1 = new_ind1
		self.worm_index_roi2 = new_ind2
		self.updateImage()
		
		self.ui.spinBox_join1.setValue(new_ind1)
		self.ui.spinBox_join2.setValue(new_ind2)

	#change frame number using the keys
	def keyPressEvent(self, event):
		#select uchange the radio button pression the up and down keys
		if event.key() == Qt.Key_Up:
			self.ui.radioButton_ROI1.setChecked(True)
		elif event.key() == Qt.Key_Down:
			self.ui.radioButton_ROI2.setChecked(True)
		
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

		#go the the start of end of a trajectory	
		elif event.key() == 91:
			current_roi = 1 if self.ui.radioButton_ROI1.isChecked() else 2
			self.roiRWFF(current_roi, self.RW)
		#[ <<
		elif event.key() == 93:
			current_roi = 1 if self.ui.radioButton_ROI1.isChecked() else 2
			self.roiRWFF(current_roi, self.FF)

		super().keyPressEvent(event)
	

if __name__ == '__main__':
	app = QApplication(sys.argv)
	
	
	ui = MWTrackerViewer_GUI(argv=sys.argv)
	ui.show()
	app.exec_()
	sys.exit()