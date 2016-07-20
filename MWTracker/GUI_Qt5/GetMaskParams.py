import json
import h5py
import os
import numpy as np
import cv2
from functools import partial

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QGraphicsView, \
    QCheckBox, QDoubleSpinBox, QSpinBox
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QPixmap, QImage

from MWTracker.GUI_Qt5.GetMaskParams_ui import Ui_GetMaskParams
from MWTracker.GUI_Qt5.AnalysisProgress import WorkerFunQt, AnalysisProgress
from MWTracker.GUI_Qt5.HDF5VideoPlayer import lineEditDragDrop, ViewsWithZoom, setChildrenFocusPolicy
from MWTracker.GUI_Qt5.GetAllParameters import GetAllParameters


from MWTracker.helperFunctions.tracker_param import tracker_param
from MWTracker.compressVideos.compressVideo import getROIMask, selectVideoReader
from MWTracker.batchProcessing.compressSingleWorker import compressSingleWorker
from MWTracker.batchProcessing.trackSingleWorker import getTrajectoriesWorker

class twoViewsWithZoom():

    def __init__(self, view_full, view_mask):
        self.view_full = ViewsWithZoom(view_full)
        self.view_mask = ViewsWithZoom(view_mask)

        self.view_full._view.wheelEvent = self.zoomWheelEvent
        self.view_mask._view.wheelEvent = self.zoomWheelEvent

        # link scrollbars in the graphics view so if one changes the other
        # changes too
        self.hscroll_full = self.view_full._view.horizontalScrollBar()
        self.vscroll_full = self.view_full._view.verticalScrollBar()
        self.hscroll_mask = self.view_mask._view.horizontalScrollBar()
        self.vscroll_mask = self.view_mask._view.verticalScrollBar()
        dd = partial(self.chgScroll, self.hscroll_mask, self.hscroll_full)
        self.hscroll_mask.valueChanged.connect(dd)
        dd = partial(self.chgScroll, self.vscroll_mask, self.vscroll_full)
        self.vscroll_mask.valueChanged.connect(dd)
        dd = partial(self.chgScroll, self.hscroll_full, self.hscroll_mask)
        self.hscroll_full.valueChanged.connect(dd)
        dd = partial(self.chgScroll, self.vscroll_full, self.vscroll_mask)
        self.vscroll_full.valueChanged.connect(dd)

    def chgScroll(self, scroll_changed, scroll2change):
        scroll2change.setValue(scroll_changed.value())

    def zoomWheelEvent(self, event):
        self.view_full.zoomWheelEvent(event)
        self.view_mask.zoomWheelEvent(event)

    def zoom(self, zoom_direction):
        self.view_full.zoom(zoom_direction)
        self.view_mask.zoom(zoom_direction)

    def zoomFitInView(self):
        self.view_full.zoomFitInView()
        self.view_mask.zoomFitInView()

    def setPixmap(self, qimage_full, qimage_mask):
        # for both maps to have the same size
        v_height = min(
            self.view_full._view.height(),
            self.view_mask._view.height())
        v_width = min(
            self.view_full._view.width(),
            self.view_mask._view.width())

        self.view_full._view.resize(v_width, v_height)
        self.view_mask._view.resize(v_width, v_height)

        self.view_full.setPixmap(qimage_full)
        self.view_mask.setPixmap(qimage_mask)

    def cleanCanvas(self):
        self.view_full.cleanCanvas()
        self.view_mask.cleanCanvas()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._mousePressed = True
            self._dragPos = event.pos()
            event.accept()

            self.view_full._view.setCursor(Qt.ClosedHandCursor)
            self.view_mask._view.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._mousePressed:
            newPos = event.pos()
            diff = newPos - self._dragPos
            self._dragPos = newPos

            dx = self.view_full._view.horizontalScrollBar().value() - diff.x()
            dy = self.view_full._view.verticalScrollBar().value() - diff.y()
            self.view_full._view.horizontalScrollBar().setValue(dx)
            self.view_full._view.verticalScrollBar().setValue(dy)

            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.view_full._view.setCursor(Qt.OpenHandCursor)
            self.view_mask._view.setCursor(Qt.OpenHandCursor)
            self._mousePressed = False


class ParamWidgetMapper():
    # let's map a parameter name into a widget

    def __init__(self, param2widget_dict):
        self.param2widget = param2widget_dict

    def set(self, param_name, value):
        widget = self.param2widget[param_name]
        if isinstance(widget, QCheckBox):
            return widget.setChecked(value)
        else:
            return widget.setValue(value)

    def get(self, param_name):
        widget = self.param2widget[param_name]
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        else:
            return widget.value()


class GetMaskParams_GUI(QMainWindow):

    def __init__(self, default_videos_dir='', scripts_dir=''):
        super(GetMaskParams_GUI, self).__init__()

        # Set up the user interface from Designer.
        self.ui = Ui_GetMaskParams()
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

        self.ui.checkBox_keepBorderData.stateChanged.connect(self.updateMask)
        self.ui.checkBox_isFluorescence.stateChanged.connect(self.updateMask)

        self.ui.pushButton_video.clicked.connect(self.getVideoFile)
        self.ui.pushButton_results.clicked.connect(self.getResultsDir)
        self.ui.pushButton_mask.clicked.connect(self.getMasksDir)
        self.ui.pushButton_paramFile.clicked.connect(self.getParamFile)

        self.ui.pushButton_saveParam.clicked.connect(self.saveParamFile)

        self.ui.pushButton_next.clicked.connect(self.getNextChunk)
        self.ui.pushButton_start.clicked.connect(self.startAnalysis)

        self.ui.pushButton_moreParams.clicked.connect(self.getMoreParams)
        self.mask_files_dir = ''
        self.results_dir = ''

        self.video_file = ''
        self.json_file = ''
        self.json_param = {}

        self.Ifull = np.zeros(0)
        self.vid = 0

        self.mapper = ParamWidgetMapper({
            'max_area': self.ui.spinBox_max_area,
            'min_area': self.ui.spinBox_min_area,
            'thresh_block_size': self.ui.spinBox_block_size,
            'thresh_C': self.ui.spinBox_thresh_C,
            'dilation_size': self.ui.spinBox_dilation_size,
            'keep_border_data': self.ui.checkBox_keepBorderData,
            'is_invert_thresh': self.ui.checkBox_isFluorescence,
            'fps': self.ui.spinBox_fps,
            'compression_buff': self.ui.spinBox_buff_size
        })

        self.videos_dir = default_videos_dir
        if not os.path.exists(self.videos_dir):
            self.videos_dir = ''
        self.updateParamFile('')

        self.ui.lineEdit_mask.setText(self.mask_files_dir)
        self.ui.lineEdit_results.setText(self.results_dir)
        # self.updateBuffSize()

        # setup image view as a zoom
        self.twoViews = twoViewsWithZoom(
            self.ui.graphicsView_full,
            self.ui.graphicsView_mask)

        # let drag and drop a file into the video file line edit
        lineEditDragDrop(
            self.ui.lineEdit_video,
            self.updateVideoFile,
            os.path.isfile)
        lineEditDragDrop(
            self.ui.lineEdit_results,
            self.updateResultsDir,
            os.path.isdir)
        lineEditDragDrop(
            self.ui.lineEdit_mask,
            self.updateMasksDir,
            os.path.isdir)
        lineEditDragDrop(
            self.ui.lineEdit_paramFile,
            self.updateParamFile,
            os.path.isfile)

        # make sure the childrenfocus policy is none in order to be able to use
        # the arrow keys
        setChildrenFocusPolicy(self, Qt.ClickFocus)

    def closeEvent(self, event):
        if not isinstance(self.vid, int):
            self.vid.release()
        super(GetMaskParams_GUI, self).closeEvent(event)

    # update image if the GUI is resized event
    def resizeEvent(self, event):
        self.updateImage()
        self.twoViews.zoomFitInView()

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

    def getResultsDir(self):
        results_dir = QFileDialog.getExistingDirectory(
            self,
            "Selects the directory where the analysis results will be stored",
            self.results_dir)
        self.updateResultsDir(results_dir)

    def updateResultsDir(self, results_dir):
        if results_dir:
            self.results_dir = results_dir
            self.ui.lineEdit_results.setText(self.results_dir)

    def getMasksDir(self):
        mask_files_dir = QFileDialog.getExistingDirectory(
            self,
            "Selects the directory where the hdf5 video will be stored",
            self.mask_files_dir)
        self.updateMasksDir(mask_files_dir)

    def updateMasksDir(self, mask_files_dir):
        if mask_files_dir:
            self.mask_files_dir = mask_files_dir
            self.ui.lineEdit_mask.setText(self.mask_files_dir)

    # file dialog to the the hdf5 file
    def getParamFile(self):
        json_file, _ = QFileDialog.getOpenFileName(
            self, "Find parameters file", self.videos_dir, "JSON files (*.json);; All (*)")
        if json_file:
            self.updateParamFile(json_file)

    # def readAllParam(self):
    def _setDefaultParam(self):
        param = tracker_param()
        widget_param = param.mask_param
        widget_param['fps'] = param.compress_vid_param['expected_fps']
        widget_param['compression_buff'] = param.compress_vid_param[
            'buffer_size']

        for param_name in widget_param:
            self.mapper.set(param_name, widget_param[param_name])

    def updateParamFile(self, json_file):
        # set the widgets with the default parameters, in case the parameters are not given
        # by the json file.
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as fid:
                    json_str = fid.read()
                    self.json_param = json.loads(json_str)

            except (OSError, UnicodeDecodeError, json.decoder.JSONDecodeError):
                QMessageBox.critical(
                    self,
                    'Cannot read parameters file.',
                    "Cannot read parameters file. Try another file",
                    QMessageBox.Ok)
                return
        else:
            self.json_param = {}

        # put reset to the default paramters in the main application. Any paramter not contain
        # in the json file will be keep with the default value.
        self._setDefaultParam()

        # set correct widgets to the values given in the json file
        for param_name in self.json_param:

            if param_name in self.mapper.param2widget:
                self.mapper.set(param_name, self.json_param[param_name])

        self.json_file = json_file
        self.ui.lineEdit_paramFile.setText(self.json_file)

    def saveParamFile(self):
        if not self.json_file:
            QMessageBox.critical(
                self,
                'No parameter file name given.',
                'No parameter file name given. Please select name using the "Parameters File" button',
                QMessageBox.Ok)
            return

        if os.path.exists(self.json_file):

            reply = QMessageBox.question(
                self,
                'Message',
                '''The parameters file already exists. Do you want to overwrite it?
            If No the parameters in the existing file will be used instead of the values displayed.''',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No)

            if reply == QMessageBox.No:
                return

        # read all the values in the GUI
        for param_name in self.mapper.param2widget:
            param_value = self.mapper.get(param_name)
            self.json_param[param_name] = param_value

        # save data into the json file
        with open(self.json_file, 'w') as fid:
            json.dump(self.json_param, fid)

        # file dialog to the the hdf5 file
    def getVideoFile(self):
        video_file, _ = QFileDialog.getOpenFileName(
            self, "Find video file", self.videos_dir, "All files (*)")
        self.updateVideoFile(video_file)

    def updateVideoFile(self, video_file):
        if video_file and os.path.exists(video_file):
            try:
                vid, im_width, im_height, reader_type = \
                    selectVideoReader(video_file)
                if im_width == 0 or im_height == 0:
                    raise ValueError
                else:
                    if not isinstance(self.vid, int):
                        self.vid.release()
                    self.vid, self.im_width, self.im_height, self.reader_type = \
                        vid, im_width, im_height, reader_type

            except (OSError, ValueError, IOError):
                QMessageBox.critical(
                    self,
                    'Cannot read video file.',
                    "Cannot read video file. Try another file",
                    QMessageBox.Ok)
                return

            # self.twoViews.cleanCanvas()

            self.video_file = video_file
            self.videos_dir = os.path.split(self.video_file)[0]
            self.ui.lineEdit_video.setText(self.video_file)

            # replace Worm_Videos or add a directory for the Results and
            # MaskedVideos directories
            if 'Worm_Videos' in self.videos_dir:
                mask_files_dir = self.videos_dir.replace(
                    'Worm_Videos', 'MaskedVideos')
                results_dir = self.videos_dir.replace('Worm_Videos', 'Results')
            else:
                mask_files_dir = os.path.join(self.videos_dir, 'MaskedVideos')
                results_dir = os.path.join(self.videos_dir, 'Results')

            self.updateResultsDir(results_dir)
            self.updateMasksDir(mask_files_dir)

            # read json file
            json_file = self.video_file.rpartition('.')[0] + '.json'
            if not os.path.exists(json_file):
                json_file = ''

            self.updateParamFile(json_file)

            # get next chuck
            self.getNextChunk()

            # fit the image to the canvas size
            self.twoViews.zoomFitInView()

    def getNextChunk(self):

        if self.vid:
            # read the buffsize before getting the next chunk
            self.updateBuffSize()

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

    def _numpy2qimage(self, im_ori):
        return QImage(im_ori.data, im_ori.shape[1], im_ori.shape[0],
                      im_ori.data.strides[0], QImage.Format_Indexed8)

    def updateImage(self):
        if self.Ifull.size == 0:
            self.twoViews.cleanCanvas()
        else:
            qimage_full = self._numpy2qimage(self.Ifull)
            qimage_mask = self._numpy2qimage(self.Imask)
            self.twoViews.setPixmap(qimage_full, qimage_mask)

    def updateMask(self):
        if self.Ifull.size == 0:
            return
        # read parameters used to calculate the mask
        mask_param = {}
        for param_name in [
            'max_area',
            'min_area',
            'thresh_block_size',
            'thresh_C',
            'dilation_size',
            'keep_border_data',
                'is_invert_thresh']:
            mask_param[param_name] = self.mapper.get(param_name)

        mask = getROIMask(self.Imin.copy(), **mask_param)
        self.Imask = mask * self.Ifull

        self.updateImage()

    def startAnalysis(self):
        if self.video_file == '' or self.Ifull.size == 0:
            QMessageBox.critical(
                self,
                'Message',
                "No valid video file selected.",
                QMessageBox.Ok)
            return

        self.saveParamFile()

        base_name = os.path.splitext(os.path.split(self.video_file)[-1])[0]
        self.masked_image_file = os.path.join(
            self.mask_files_dir, base_name + '.hdf5')

        if os.name == 'nt':
            # I Windows the paths return by QFileDialog use / as the file
            # separation character. We need to correct it.
            for field_name in [
                'video_file',
                'mask_files_dir',
                'masked_image_file',
                    'results_dir']:
                setattr(
                    self, field_name, getattr(
                        self, field_name).replace(
                        '/', os.sep))

        compress_argvs = {
            'video_file': self.video_file,
            'mask_dir': self.mask_files_dir,
            'json_file': self.json_file,
            'is_single_worm': False}

        track_argvs = {
            'masked_image_file': self.masked_image_file,
            'results_dir': self.results_dir,
            'json_file': self.json_file,
            'is_single_worm': False,
            'use_skel_filter': True,
            'use_manual_join': False,
            'cmd_original': 'GUI'}

        def complete_analysis(compress_argvs, track_argvs):
            compressSingleWorker(**compress_argvs)
            getTrajectoriesWorker(**track_argvs)

        analysis_worker = WorkerFunQt(
            complete_analysis, {
                'compress_argvs': compress_argvs, 'track_argvs': track_argvs})
        progress_dialog = AnalysisProgress(analysis_worker)
        progress_dialog.setAttribute(Qt.WA_DeleteOnClose)
        progress_dialog.exec_()

    def keyPressEvent(self, event):
        if self.vid == -1:
            # break no file open, nothing to do here
            return

        key = event.key()
        if key == Qt.Key_Minus:
            self.twoViews.zoom(-1)
        elif key == Qt.Key_Plus:
            self.twoViews.zoom(1)

        else:
            QMainWindow.keyPressEvent(self, event)

    def getMoreParams(self):
        json_file = self.ui.lineEdit_paramFile.text()
        allparamGUI = GetAllParameters(json_file)
        allparamGUI.file_saved.connect(self.updateParamFile)
        allparamGUI.exec_()


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)

    ui = GetMaskParams_GUI()
    ui.show()

    sys.exit(app.exec_())
