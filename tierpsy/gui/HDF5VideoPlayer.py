from tierpsy.gui.HDF5VideoPlayer_ui import Ui_HDF5VideoPlayer

import sys
import tables
import os
from pathlib import Path
import numpy as np
from functools import partial

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

try:
    import imgstore
except ModuleNotFoundError:
    pass


def setChildrenFocusPolicy(obj, policy):
    # recursively change the focus policy of all the objects in the widgets
    def recursiveSetChildFocusPolicy(parentQWidget):
        for childQWidget in parentQWidget.findChildren(QtWidgets.QWidget):
            childQWidget.setFocusPolicy(policy)
            recursiveSetChildFocusPolicy(childQWidget)
    recursiveSetChildFocusPolicy(obj)

class LineEditDragDrop():
    def __init__(self, main_obj, update_fun, test_file_fun):
        self.update_fun = update_fun
        self.test_file_fun = test_file_fun

        self.main_obj = main_obj
        if isinstance(self.main_obj, QtWidgets.QLineEdit):
            self.line_edit_obj = self.main_obj
        else:
            self.line_edit_obj = self.main_obj.lineEdit()
            

        self.main_obj.setAcceptDrops(True)
        self.main_obj.dragEnterEvent = self.dragEnterEvent
        self.main_obj.dropEvent = self.dropEvent
        self.line_edit_obj.returnPressed.connect(self.returnPressedFun)


    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            vfilename = url.toLocalFile()
            if self.test_file_fun(vfilename):
                self.update_fun(vfilename)

    def returnPressedFun(self):
        vfilename = self.line_edit_obj.text()
        if self.test_file_fun(vfilename):
            self.update_fun(vfilename)

class ViewsWithZoom():

    def __init__(self, view):
        self._view = view
        self._scene = QtWidgets.QGraphicsScene(self._view)
        self._view.setScene(self._scene)
        self._canvas = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._canvas)

        self._zoom = 0
        self._view.wheelEvent = self.zoomWheelEvent

    # zoom wheel
    def zoomWheelEvent(self, event):
        if not self._canvas.pixmap().isNull():
            numPixels = event.pixelDelta()
            numDegrees = event.angleDelta() / 8

            delta = numPixels if not numPixels.isNull() else numDegrees
            self.zoom(delta.y())

    def zoom(self, zoom_direction):
        if zoom_direction > 0:
            factor = 1.25
            self._zoom += 1
        else:
            factor = 0.8
            self._zoom -= 1
        if self._zoom > 0:
            self._view.scale(factor, factor)
        elif self._zoom == 0:
            self.zoomFitInView()
        else:
            self._zoom = 0

    def zoomFitInView(self):
        rect = QtCore.QRectF(self._canvas.pixmap().rect())
        if not rect.isNull():
            unity = self._view.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
            self._view.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self._view.viewport().rect()
            scenerect = self._view.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(),
                         viewrect.height() / scenerect.height())
            self._view.scale(factor, factor)
            self._view.centerOn(rect.center())
            self._zoom = 0

    def cleanCanvas(self):
        self._canvas.setPixmap(QtGui.QPixmap())

    def setPixmap(self, frame_qimg=None):
        if frame_qimg is None:
            return

        pixmap = QtGui.QPixmap.fromImage(frame_qimg)
        self._canvas.setPixmap(pixmap)

class SimplePlayer(QtWidgets.QMainWindow):
    def __init__(self, ui):
        super().__init__()
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.getNextImage)
        self.ui = ui
        self.isPlay = False
        self.image_group = None
        self.frame_number = None

    def keyPressEvent(self, event):
        #HOT KEYS
        key = event.key()

        # Duplicate the frame step size (speed) when pressed  > or .:
        if key == Qt.Key_Greater or key == Qt.Key_Period:
            self.frame_step *= 2
            self.ui.spinBox_step.setValue(self.frame_step)
            

        # Half the frame step size (speed) when pressed: < or ,
        elif key == Qt.Key_Less or key == Qt.Key_Comma:
            self.frame_step //= 2
            if self.frame_step < 1:
                self.frame_step = 1
            self.ui.spinBox_step.setValue(self.frame_step)
            

        # Move backwards when  are pressed
        elif key == Qt.Key_Left:
            self.frame_number -= self.frame_step
            if self.frame_number < 0:
                self.frame_number = 0
            self.ui.spinBox_frame.setValue(self.frame_number)
            

        # Move forward when  are pressed
        elif key == Qt.Key_Right:
            self.frame_number += self.frame_step
            if self.frame_number >= self.tot_frames:
                self.frame_number = self.tot_frames - 1
            self.ui.spinBox_frame.setValue(self.frame_number)
            
        #super().keyPressEvent(event)

    def playVideo(self):
        if self.is_video_opened is None:
            return
        if not self.isPlay:
            self.startPlay()
        else:
            self.stopPlay()

    def startPlay(self):
        self.timer.start(round(1000 / self.fps))
        self.isPlay = True
        self.ui.playButton.setText('Stop')
        self.ui.doubleSpinBox_fps.setEnabled(False)

    def stopPlay(self):
        self.timer.stop()
        self.isPlay = False
        self.ui.playButton.setText('Play')
        self.ui.doubleSpinBox_fps.setEnabled(True)

    # Function to get the new valid frame during video play
    def getNextImage(self):
        self.frame_number += self.frame_step
        if self.frame_number >= self.tot_frames:
            self.frame_number = self.tot_frames - 1
            self.stopPlay()
        self.ui.spinBox_frame.setValue(self.frame_number)

    @property
    def fps(self):
        return self.ui.doubleSpinBox_fps.value()
    @fps.setter
    def fps(self, value):
        return self.ui.doubleSpinBox_fps.setValue(value)

    @property
    def frame_step(self):
        return self.ui.spinBox_step.value()

    @frame_step.setter
    def frame_step(self, value):
        return self.ui.spinBox_step.setValue(value)


class _LoopBioReader():
    def __init__(self, src_file):
        self.frame_save_interval = 1
        self.is_light_background = 1
        self.video_data = None
        self.groupnames = ['None']

        self.src_file = src_file

        self.video_data = imgstore.new_for_filename(str(src_file))
        self._ini_frame = self.video_data.frame_min

        img, (frame_number, frame_timestamp) = self.video_data.get_next_image()
        self.height, self.width = img.shape

    def close(self):
        pass

    def update_data(self, new_path):
        pass

    def update_groupnames(self):
        pass

    def __getitem__(self, frame_number):
        img = self.video_data.get_image(frame_number + self._ini_frame)[0]
        return img

    def __len__(self):
        return self.video_data.frame_count


class _HDF5Reader():
    def __init__(self, src_file):
        self.frame_save_interval = 1
        self.is_light_background = 1
        self.video_data = None
        self.groupnames = ["/mask", "/full_data"]

        self.src_file = src_file
        try:
            self.fid = tables.File(str(self.src_file), 'r')
        except (IOError, tables.exceptions.HDF5ExtError):
            raise IOError

        self.update_data(self.groupnames[0])

    def __getitem__(self, frame_number):
        if self.video_data is None:
            return
        img = self.video_data[frame_number, :, :]
        return img

    def __len__(self):
        if self.video_data is None:
            return 0

        return self.video_data.shape[0]

    @property
    def width(self):
        if self.video_data is not None:
            return self.video_data.shape[2]

    @property
    def height(self):
        if self.video_data is not None:
            return self.video_data.shape[1]

    def update_data(self, new_path):
        if not new_path in self.fid:
            self.video_data = None
            return

        self.video_data = self.fid.get_node(new_path)
        if len(self.video_data.shape) != 3:
            raise ValueError(f'Invalid data dimensions {new_path}')

        try:
            self.is_light_background = self.video_reader.video_data._v_attrs['is_light_background']
        except:
            self.is_light_background = 1


    def update_groupnames(self):
        self.groupnames = []
        for group in self.fid.walk_groups("/"):
            for array in self.fid.list_nodes(group, classname='Array'):
                if array.ndim == 3:
                    self.groupnames.append(array._v_pathname)

    def close(self):
        self.fid.close()


class HDF5VideoPlayerGUI(SimplePlayer):

    def __init__(self, ui=None):
        if ui is None:
            ui = Ui_HDF5VideoPlayer()
        
        super().__init__(ui)

        # Set up the user interface from Designer.
        self.ui.setupUi(self)

        self.video_reader = None
        self.isPlay = False
        self.videos_dir = ''
        self.frame_img = None
        self.frame_qimg = None

        self.ui.pushButton_video.clicked.connect(self.getVideoFile)
        self.ui.playButton.clicked.connect(self.playVideo)

        # set scroller
        sld_pressed = partial(self.ui.imageSlider.setCursor, QtCore.Qt.ClosedHandCursor)
        sld_released = partial(self.ui.imageSlider.setCursor, QtCore.Qt.OpenHandCursor)
        
        self.ui.imageSlider.sliderPressed.connect(sld_pressed)
        self.ui.imageSlider.sliderReleased.connect(sld_released)
        self.ui.imageSlider.valueChanged.connect(self.ui.spinBox_frame.setValue)
        #eliminate ticks, they will be a problem since I make the maximum size of the slider tot_frames
        self.ui.imageSlider.setTickPosition(QtWidgets.QSlider.NoTicks)

        #%%
        self.ui.spinBox_frame.valueChanged.connect(self.updateFrameNumber)
        self.ui.comboBox_h5path.activated.connect(self.getImGroup)
        self.ui.pushButton_h5groups.clicked.connect(self.updateGroupNames)

        
        # setup image view as a zoom
        self.mainImage = ViewsWithZoom(self.ui.mainGraphicsView)

        # let drag and drop a file into the video file line edit
        LineEditDragDrop(
            self.ui.lineEdit_video,
            self.updateVideoFile,
            os.path.isfile)

        # make sure the childrenfocus policy is none in order to be able to use
        # the arrow keys
        setChildrenFocusPolicy(self, QtCore.Qt.ClickFocus)
    
    def keyPressEvent(self, event):
        #HOT KEYS

        if self.video_reader is None:
            # break no file open, nothing to do here
            return

        key = event.key()
        if key == Qt.Key_Minus:
            self.mainImage.zoom(-1)
        elif key == Qt.Key_Plus:
            self.mainImage.zoom(1)

        super().keyPressEvent(event)

    # frame spin box
    def updateFrameNumber(self):
        self.frame_number = self.ui.spinBox_frame.value()
        self.ui.imageSlider.setValue(self.frame_number)
        self.updateImage()

    # update image: get the next frame_number, and resize it to fix in the GUI
    # area
    def updateImage(self):
        self.readCurrentFrame()
        self.mainImage.setPixmap(self.frame_qimg)

    def readCurrentFrame(self):
        if self.is_video_opened:
            self.frame_img = self.video_reader[self.frame_number]
            self._normalizeImage()
        

    def _normalizeImage(self):
        if self.frame_img is None:
            return 

        dd = self.ui.mainGraphicsView.size()
        self.label_height = dd.height()
        self.label_width = dd.width()

        # equalize and cast if it is not uint8
        if self.frame_img.dtype != np.uint8:
            top = np.max(self.frame_img)
            bot = np.min(self.frame_img)

            self.frame_img = (self.frame_img - bot) * 255. / (top - bot)
            self.frame_img = np.round(self.frame_img).astype(np.uint8)

        self.frame_qimg = self._convert2Qimg(self.frame_img)


    def _convert2Qimg(self, img):
        qimg = QtGui.QImage(
            img.data,
            img.shape[1],
            img.shape[0],
            img.strides[0],
            QtGui.QImage.Format_Indexed8)
        qimg = qimg.convertToFormat(
            QtGui.QImage.Format_RGB32, QtCore.Qt.AutoColor)

        return qimg

    # file dialog to the the hdf5 file
    def getVideoFile(self):
        vfilename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Find HDF5 video file", self.videos_dir, "HDF5 files (*.hdf5);; LoopBio files (metadata.yaml);; All files (*)")

        self.updateVideoFile(vfilename)

    def updateVideoFile(self, vfilename):
        # close the if there was another file opened before.
        if self.video_reader is not None:
            self.video_reader.close()
            self.mainImage.cleanCanvas()
            self.video_reader = None

        self.vfilename = Path(vfilename)
        self.ui.lineEdit_video.setText(str(self.vfilename))
        self.videos_dir = self.vfilename.parent

        try:
            if self.vfilename.name == 'metadata.yaml':
                self.video_reader = _LoopBioReader(self.vfilename)
            else:
                self.video_reader = _HDF5Reader(self.vfilename)


        except (IOError, tables.exceptions.HDF5ExtError):
            self.reader = None
            QtWidgets.QMessageBox.critical(
                self,
                '',
                "The selected file is not a valid .hdf5. Please select a valid file",
                QtWidgets.QMessageBox.Ok)
            return

        self.ui.comboBox_h5path.clear()
        for kk in self.video_reader.groupnames:
            self.ui.comboBox_h5path.addItem(kk)

        self.getImGroup(0)

    def updateGroupNames(self):
        if self.video_reader is None:
            return

        self.video_reader.update_groupnames()
        if not self.video_reader.groupnames:
            QtWidgets.QMessageBox.critical(
                self,
                '',
                "No valid video groups were found. Dataset with three dimensions and uint8 data type. Closing file.",
                QtWidgets.QMessageBox.Ok)
            self.video_reader.close()
            self.mainImage.cleanCanvas()

            return

        self.ui.comboBox_h5path.clear()
        for kk in self.video_reader.groupnames:
            self.ui.comboBox_h5path.addItem(kk)
        self.getImGroup(0)
        self.updateImage()

    def getImGroup(self, index):
        h5path = self.ui.comboBox_h5path.itemText(index)
        self.ui.comboBox_h5path.setCurrentIndex(index)
        self.updateImGroup(h5path)

    # read a valid groupset from the hdf5
    def updateImGroup(self, h5path):
        if self.video_reader is None:
            return
        
        self.h5path = h5path
        try:
            self.video_reader.update_data(self.h5path)
        except ValueError:
            self.mainImage.cleanCanvas()
            QtWidgets.QMessageBox.critical(
                self,
                'Invalid groupset',
                "Invalid groupset.",
                QtWidgets.QMessageBox.Ok)
        
        self.tot_frames = len(self.video_reader)
        self.image_height = self.video_reader.height
        self.image_width = self.video_reader.width

        self.ui.spinBox_frame.setMaximum(self.tot_frames - 1)
        self.ui.imageSlider.setMaximum(self.tot_frames - 1)

        self.frame_number = 0
        self.ui.spinBox_frame.setValue(self.frame_number)
        self.updateImage()
        self.mainImage.zoomFitInView()

    def setFileName(self, filename):
        self.filename = filename
        self.ui.lineEdit.setText(filename)

    def resizeEvent(self, event):
        if self.video_reader is not None:
            self.updateImage()
            self.mainImage.zoomFitInView()

    def closeEvent(self, event):
        if self.video_reader is not None:
            self.video_reader.close()
        super(HDF5VideoPlayerGUI, self).closeEvent(event)

    @property
    def is_video_opened(self):
        if self.video_reader is not None:
            if self.video_reader.video_data is not None:
                return True

        return False
    

def tierpsy_gui_simple():
    app = QApplication(sys.argv)

    ui = HDF5VideoPlayerGUI()
    ui.show()

    app.exec_()

if __name__ == '__main__':
    sys.exit(tierpsy_gui_simple())
