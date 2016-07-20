import json
import h5py
import os
import numpy as np
import sys
import cv2

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QDir, QTimer, Qt, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QPolygonF, QPen

from MWTracker.GUI.getMaskParams.getMaskParams_ui import Ui_MainWindow

from MWTracker.compressVideos.compressVideo import getROIMask, selectVideoReader
from MWTracker.helperFunctions.compressVideoWorkerL import compressVideoWorkerL
from MWTracker.helperFunctions.getTrajectoriesWorkerL import getTrajectoriesWorkerL


def list2cmd(cmd):
    for ii, dd in enumerate(cmd):
        if ii >= 2 and not dd[0] == '-':
            dd = '"' + dd + '"'

        if ii == 0:
            cmd_str = dd
        else:
            cmd_str += ' ' + dd
    return cmd_str


class getMaskParams_GUI(QMainWindow):

    def __init__(self, default_videos_dir='', scripts_dir=''):
        super().__init__()

        scripts_dir = scripts_dir
        self.script_compress = os.path.join(
            scripts_dir, 'compressSingleLocal.py')
        self.script_track = os.path.join(scripts_dir, 'trackSingleLocal.py')

        self.videos_dir = default_videos_dir
        if not os.path.exists(self.videos_dir):
            self.videos_dir = ''

        self.mask_files_dir = ''
        self.results_dir = ''

        self.video_file = ''
        self.Ifull = np.zeros(0)
        self.vid = 0

        # Set up the user interface from Designer.
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.dial_min_area.valueChanged.connect(
            self.ui.spinBox_min_area.setValue)
        self.ui.dial_max_area.valueChanged.connect(
            self.ui.spinBox_max_area.setValue)
        self.ui.dial_block_size.valueChanged.connect(
            self.ui.spinBox_block_size.setValue)
        self.ui.dial_thresh_C.valueChanged.connect(
            self.ui.spinBox_thresh_C.setValue)

        self.ui.spinBox_max_area.valueChanged.connect(self.updateMaxArea)
        self.ui.spinBox_min_area.valueChanged.connect(self.updateMinArea)
        self.ui.spinBox_block_size.valueChanged.connect(self.updateBlockSize)
        self.ui.spinBox_thresh_C.valueChanged.connect(self.updateThreshC)
        self.ui.spinBox_dilation_size.valueChanged.connect(
            self.updateDilationSize)

        self.ui.spinBox_buff_size.valueChanged.connect(self.updateBuffSize)

        self.ui.checkBox_hasTimestamp.stateChanged.connect(self.updateMask)
        self.ui.checkBox_keepBorderData.stateChanged.connect(self.updateMask)

        self.ui.lineEdit_mask.setText(self.mask_files_dir)
        self.ui.lineEdit_results.setText(self.results_dir)

        self.ui.pushButton_video.clicked.connect(self.getVideoFile)
        self.ui.pushButton_results.clicked.connect(self.updateResultsDir)
        self.ui.pushButton_mask.clicked.connect(self.updateMasksDir)

        self.ui.pushButton_next.clicked.connect(self.getNextChunk)
        self.ui.pushButton_start.clicked.connect(self.startAnalysis)

        self.updateBuffSize()

    def updateBuffSize(self):
        self.buffer_size = int(np.round(self.ui.spinBox_buff_size.value()))

    # file dialog to the the hdf5 file
    def getVideoFile(self):
        video_file, _ = QFileDialog.getOpenFileName(
            self, "Find video file", self.videos_dir, "All files (*)")

        if video_file:
            self.video_file = video_file
            if os.path.exists(self.video_file):

                self.json_file = self.video_file.rpartition('.')[0] + '.json'
                if os.path.exists(self.json_file):
                    self.read_json()

                self.ui.label_full.clear()
                self.Ifull = np.zeros(0)

                self.videos_dir = self.video_file.rpartition(os.sep)[
                    0] + os.sep

                self.ui.lineEdit_video.setText(self.video_file)
                self.vid, self.im_width, self.im_height, self.reader_type = selectVideoReader(
                    video_file)
                print(self.reader_type)
                if self.im_width == 0 or self.im_height == 0:
                    QMessageBox.critical(
                        self,
                        'Cannot read video file.',
                        "Cannot read video file. Try another file",
                        QMessageBox.Ok)
                    self.vid = 0
                    return

                if 'Worm_Videos' in self.videos_dir:
                    self.results_dir = self.videos_dir.replace(
                        'Worm_Videos', 'Results')
                    self.mask_files_dir = self.videos_dir.replace(
                        'Worm_Videos', 'MaskedVideos')
                    self.ui.lineEdit_mask.setText(self.mask_files_dir)
                    self.ui.lineEdit_results.setText(self.results_dir)

                self.getNextChunk()

    def getNextChunk(self):
        if self.vid:
            Ibuff = np.zeros(
                (self.buffer_size,
                 self.im_height,
                 self.im_width),
                dtype=np.uint8)

            tot = 0
            for ii in range(self.buffer_size):
                # get video frame, stop program when no frame is retrive (end
                # of file)
                ret, image = self.vid.read()

                if ret == 0:
                    break
                if image.ndim == 3:
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

        self.full_size = min(
            self.ui.label_full.height(),
            self.ui.label_full.width())

        image = QImage(
            self.Ifull.data,
            self.im_width,
            self.im_height,
            self.Ifull.strides[0],
            QImage.Format_Indexed8)

        image = image.scaled(
            self.full_size,
            self.full_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation)
        self.img_w_ratio = image.size().width() / self.im_width
        self.img_h_ratio = image.size().height() / self.im_height

        pixmap = QPixmap.fromImage(image)
        self.ui.label_full.setPixmap(pixmap)

        self.mask_size = self.full_size

        mask = QImage(
            self.Imask.data,
            self.im_width,
            self.im_height,
            self.Imask.strides[0],
            QImage.Format_Indexed8)

        mask = mask.scaled(
            self.mask_size,
            self.mask_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation)

        self.img_w_ratio = mask.size().width() / self.im_width
        self.img_h_ratio = mask.size().height() / self.im_height

        pixmap = QPixmap.fromImage(mask)
        self.ui.label_mask.setPixmap(pixmap)

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

    def updateMask(self):
        if self.Ifull.size == 0:
            return

        self.mask_param = {
            'max_area': self.ui.spinBox_max_area.value(),
            'min_area': self.ui.spinBox_min_area.value(),
            'thresh_block_size': self.ui.spinBox_block_size.value(),
            'thresh_C': self.ui.spinBox_thresh_C.value(),
            'dilation_size': self.ui.spinBox_dilation_size.value(),
            'has_timestamp': self.ui.checkBox_hasTimestamp.isChecked(),
            'keep_border_data': self.ui.checkBox_keepBorderData.isChecked()}

        mask = getROIMask(self.Imin.copy(), **self.mask_param)
        self.Imask = mask * self.Ifull

        self.updateImage()

    # update image if the GUI is resized event
    def resizeEvent(self, event):
        self.updateImage()

    def updateResultsDir(self):
        results_dir = QFileDialog.getExistingDirectory(
            self,
            "Selects the directory where the analysis results will be stored",
            self.results_dir)
        if results_dir:
            self.results_dir = results_dir + os.sep
            self.ui.lineEdit_results.setText(self.results_dir)

    def updateMasksDir(self):
        mask_files_dir = QFileDialog.getExistingDirectory(
            self,
            "Selects the directory where the hdf5 video will be stored",
            self.mask_files_dir)
        if mask_files_dir:
            self.mask_files_dir = mask_files_dir + os.sep
            self.ui.lineEdit_mask.setText(self.mask_files_dir)

    def startAnalysis(self):
        if self.video_file == '' or self.Ifull.size == 0:
            QMessageBox.critical(
                self,
                'No valid video file selected.',
                "No valid video file selected.",
                QMessageBox.Ok)
            return

        self.mask_param['fps'] = self.ui.spinBox_fps.value()
        self.mask_param['resampling_N'] = self.ui.spinBox_skelSeg.value()
        self.mask_param['compression_buff'] = self.ui.spinBox_buff_size.value()
        #self.mask_param['is_single_worm'] = self.ui.checkBox_isSingleWorm.isChecked()

        self.close()

        #self.json_file = self.video_file.rpartition('.')[0] + '.json'
        with open(self.json_file, 'w') as fid:
            json.dump(self.mask_param, fid)

        base_name = self.video_file.rpartition('.')[0].rpartition(os.sep)[-1]
        self.masked_image_file = self.mask_files_dir + base_name + '.hdf5'

        arg_compress = [
            'python3',
            self.script_compress,
            self.video_file,
            self.mask_files_dir]
        for arg_str in ['json_file']:
            if getattr(self, arg_str):
                arg_compress += ['--' + arg_str, getattr(self, arg_str)]

        print(arg_compress)
        self.cmd_compress = list2cmd(arg_compress)

        arg_track = [
            'python3',
            self.script_track,
            self.masked_image_file,
            self.results_dir]
        for arg_str in ['json_file']:
            if getattr(self, arg_str):
                arg_compress += ['--' + arg_str, getattr(self, arg_str)]

        self.cmd_track = list2cmd(arg_track)

        print(self.cmd_compress)
        os.system(self.cmd_compress)

        print(self.cmd_track)
        os.system(self.cmd_track)

        #masked_image_file = compressVideoWorkerL(self.video_file, self.mask_files_dir, param_file = self.json_file)
        #getTrajectoriesWorkerL(masked_image_file, self.results_dir, param_file = self.json_file)

    def read_json(self):

        with open(self.json_file, 'r') as fid:
            json_str = fid.read()
            self.mask_param = json.loads(json_str)

            if 'max_area' in self.mask_param.keys():
                self.ui.spinBox_max_area.setValue(self.mask_param['max_area'])
            if 'min_area' in self.mask_param.keys():
                self.ui.spinBox_min_area.setValue(self.mask_param['min_area'])
            if 'thresh_block_size' in self.mask_param.keys():
                self.ui.spinBox_block_size.setValue(
                    self.mask_param['thresh_block_size'])
            if 'thresh_C' in self.mask_param.keys():
                self.ui.spinBox_thresh_C.setValue(self.mask_param['thresh_C'])
            if 'dilation_size' in self.mask_param.keys():
                self.ui.spinBox_dilation_size.setValue(
                    self.mask_param['dilation_size'])

            if 'has_timestamp' in self.mask_param.keys():
                self.ui.checkBox_hasTimestamp.setCheckState(
                    self.mask_param['has_timestamp'])
            if 'keep_border_data' in self.mask_param.keys():
                self.ui.checkBox_keepBorderData.setCheckState(
                    self.mask_param['keep_border_data'])

            if 'fps' in self.mask_param.keys():
                self.ui.spinBox_fps.setValue(self.mask_param['fps'])
            if 'resampling_N' in self.mask_param.keys():
                self.ui.spinBox_skelSeg.setValue(
                    self.mask_param['resampling_N'])
            if 'compression_buff' in self.mask_param.keys():
                self.ui.spinBox_buff_size.setValue(
                    self.mask_param['compression_buff'])


if __name__ == '__main__':
    app = QApplication(sys.argv)

    ui = getMaskParams_GUI()
    ui.show()
    app.exec_()
    # sys.exit()
