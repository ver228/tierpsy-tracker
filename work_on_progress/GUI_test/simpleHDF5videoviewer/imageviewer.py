import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QDir, QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage
from imageviewer_ui import Ui_ImageViewer

import h5py

class ImageViewer(QMainWindow):
	def __init__(self):
		super().__init__()
		
		# Set up the user interface from Designer.
		self.ui = Ui_ImageViewer()
		self.ui.setupUi(self)

		self.isPlay = False
		self.fid = -1
		self.image_group = -1

		#self.ui.centralWidget.setChildrenFocusPolicy(Qt.NoFocus)

		self.ui.fileButton.clicked.connect(self.getFilePath)
		
		self.ui.playButton.clicked.connect(self.playVideo)
		self.ui.imageSlider.sliderPressed.connect(self.imSldPressed)
		self.ui.imageSlider.sliderReleased.connect(self.imSldReleased)
		
		self.ui.spinBox_frame.valueChanged.connect(self.updateFrameNumber)
		self.ui.doubleSpinBox_fps.valueChanged.connect(self.updateFPS)
		self.ui.spinBox_step.valueChanged.connect(self.updateFrameStep)
		
		self.ui.spinBox_step.valueChanged.connect(self.updateFrameStep)

		self.ui.lineEdit_h5path.returnPressed.connect(self.updateImGroup)

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
		image = QImage(self.image_group[self.frame_number].data, 
			self.image_height, self.image_width, QImage.Format_Indexed8) 
		image = image.scaled(self.label_width, self.label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
		
		self.ui.imageCanvas.setPixmap(QPixmap.fromImage(image));
		
		progress = round(100*self.frame_number/self.tot_frames)
		if progress != self.ui.imageSlider.value():
			self.ui.imageSlider.setValue(progress)

	#file dialog to the the hdf5 file
	def getFilePath(self):
		filename, _ = QFileDialog.getOpenFileName(self, "Find HDF5 file", 
		"/Users/ajaver/Desktop/Gecko_compressed/MaskedVideos", "HDF5 files (*.hdf5);; All files (*)")

		if filename:
			if self.fid != -1:
				self.fid.close()
				self.ui.imageCanvas.clear()

			self.setFileName(filename)
			self.fid = h5py.File(self.filename, 'r')
			self.updateImGroup()
	
	#read a valid groupset from the hdf5
	def updateImGroup(self):
		if self.fid == -1:
			return

		self.h5path = self.ui.lineEdit_h5path.text()
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
	
	ui = ImageViewer()
	ui.show()
	
	sys.exit(app.exec_())