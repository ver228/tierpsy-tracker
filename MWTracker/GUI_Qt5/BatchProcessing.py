import os
import json

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt

from MWTracker.batchProcessing.compressMultipleFilesFun import compressMultipleFilesFun, compress_dflt_vals
from MWTracker.batchProcessing.trackMultipleFilesFun import trackMultipleFilesFun, track_dflt_vals, getResultsDir

from MWTracker.GUI_Qt5.AnalysisProgress import AnalysisProgress, WorkerFunQt
from MWTracker.GUI_Qt5.HDF5VideoPlayer import lineEditDragDrop
from MWTracker.GUI_Qt5.BatchProcessing_ui import Ui_BatchProcessing


class BatchProcessing_GUI(QMainWindow):

    def __init__(self):
        super(BatchProcessing_GUI, self).__init__()
        self.ui = Ui_BatchProcessing()
        self.ui.setupUi(self)

        self.ui.checkBox_txtFileList.stateChanged.connect(
            self.enableTxtFileListButton)
        self.ui.checkBox_tmpDir.stateChanged.connect(self.enableTmpDirButton)
        self.ui.checkBox_isCompress.stateChanged.connect(
            self.enableCompressInput)
        self.ui.checkBox_isTrack.stateChanged.connect(self.enableTrackInput)

        self.ui.pushButton_txtFileList.clicked.connect(self.getTxtFileList)
        self.ui.pushButton_paramFile.clicked.connect(self.getParamFile)
        self.ui.pushButton_videosDir.clicked.connect(self.getVideosDir)
        self.ui.pushButton_masksDir.clicked.connect(self.getMasksDir)
        self.ui.pushButton_resultsDir.clicked.connect(self.getResultsDir)
        self.ui.pushButton_tmpDir.clicked.connect(self.getTmpDir)
        self.ui.pushButton_start.clicked.connect(self.startAnalysis)

        # print(compress_dflt_vals)
        # print(track_dflt_vals)

        self.mask_files_dir = ''
        self.results_dir = ''
        self.videos_dir = ''

        self.ui.checkBox_tmpDir.setChecked(False)
        self.enableTxtFileListButton()
        self.ui.checkBox_tmpDir.setChecked(True)
        self.enableTmpDirButton()

        self.ui.checkBox_isCompress.setChecked(True)
        self.enableCompressInput()
        self.ui.checkBox_isTrack.setChecked(True)
        self.enableTrackInput()

        self.ui.spinBox_numMaxProc.setValue(track_dflt_vals['max_num_process'])
        self.ui.lineEdit_patternInComp.setText(
            compress_dflt_vals['pattern_include'])
        self.ui.lineEdit_patternExcComp.setText(
            compress_dflt_vals['pattern_exclude'])
        self.ui.lineEdit_patternInTrack.setText(
            track_dflt_vals['pattern_include'])
        self.ui.lineEdit_patternExcTrack.setText(
            track_dflt_vals['pattern_exclude'])

        assert compress_dflt_vals[
            'max_num_process'] == track_dflt_vals['max_num_process']
        assert compress_dflt_vals[
            'tmp_dir_root'] == track_dflt_vals['tmp_dir_root']

        lineEditDragDrop(
            self.ui.lineEdit_txtFileList,
            self.updateTxtFileList,
            os.path.isfile)
        lineEditDragDrop(
            self.ui.lineEdit_paramFile,
            self.updateParamFile,
            os.path.isfile)
        lineEditDragDrop(
            self.ui.lineEdit_videosDir,
            self.updateVideosDir,
            os.path.isdir)
        lineEditDragDrop(
            self.ui.lineEdit_masksDir,
            self.updateMasksDir,
            os.path.isdir)
        lineEditDragDrop(
            self.ui.lineEdit_resultsDir,
            self.updateResultsDir,
            os.path.isdir)
        lineEditDragDrop(
            self.ui.lineEdit_tmpDir,
            self.updateTmpDir,
            os.path.isdir)

    def enableTxtFileListButton(self):
        is_enable = self.ui.checkBox_txtFileList.isChecked()
        self.ui.pushButton_txtFileList.setEnabled(is_enable)
        self.ui.lineEdit_txtFileList.setEnabled(is_enable)
        self.ui.label_comp.setEnabled(not is_enable)
        self.ui.label_track.setEnabled(not is_enable)
        self.ui.label_patternIn.setEnabled(not is_enable)
        self.ui.label_patternExc.setEnabled(not is_enable)
        self.ui.lineEdit_patternExcComp.setEnabled(not is_enable)
        self.ui.lineEdit_patternInComp.setEnabled(not is_enable)
        self.ui.lineEdit_patternExcTrack.setEnabled(not is_enable)
        self.ui.lineEdit_patternInTrack.setEnabled(not is_enable)

        if not is_enable:
            self.updateTxtFileList('')

    def getTxtFileList(self):
        videos_list, _ = QFileDialog.getOpenFileName(
            self, "Select a text file with a list of files to be analyzed.", '', "Text file (*.txt);;All files (*)")
        if os.path.isfile(videos_list):
            self.updateTxtFileList(videos_list)

    def updateTxtFileList(self, videos_list):
        self.videos_list = videos_list
        self.ui.lineEdit_txtFileList.setText(videos_list)

    def enableTmpDirButton(self):
        is_enable = self.ui.checkBox_tmpDir.isChecked()
        self.ui.pushButton_tmpDir.setEnabled(is_enable)
        self.ui.lineEdit_tmpDir.setEnabled(is_enable)

        tmp_dir_root = track_dflt_vals['tmp_dir_root'] if is_enable else ''
        self.updateTmpDir(tmp_dir_root)

    def getTmpDir(self):
        tmp_dir_root = QFileDialog.getExistingDirectory(
            self,
            "Selects the directory where the hdf5 masked videos will be stored",
            self.tmp_dir_root)
        if tmp_dir_root:
            self.updateTmpDir(tmp_dir_root)

    def updateTmpDir(self, tmp_dir_root):
        self.tmp_dir_root = tmp_dir_root
        self.ui.lineEdit_tmpDir.setText(self.tmp_dir_root)

    def enableCompressInput(self):
        is_enable = self.ui.checkBox_isCompress.isChecked()
        self.ui.pushButton_videosDir.setEnabled(is_enable)
        self.ui.lineEdit_videosDir.setEnabled(is_enable)
        self.ui.label_comp.setEnabled(is_enable)
        self.ui.lineEdit_patternExcComp.setEnabled(is_enable)
        self.ui.lineEdit_patternInComp.setEnabled(is_enable)

    def enableTrackInput(self):
        is_enable = self.ui.checkBox_isTrack.isChecked()
        self.ui.pushButton_resultsDir.setEnabled(is_enable)
        self.ui.lineEdit_resultsDir.setEnabled(is_enable)
        self.ui.label_track.setEnabled(is_enable)
        self.ui.lineEdit_patternExcTrack.setEnabled(is_enable)
        self.ui.lineEdit_patternInTrack.setEnabled(is_enable)

    def getVideosDir(self):
        videos_dir = QFileDialog.getExistingDirectory(
            self,
            "Selects the directory where the original video files are stored.",
            self.videos_dir)
        if videos_dir:
            self.updateVideosDir(videos_dir)

    def updateVideosDir(self, videos_dir):
        self.videos_dir = videos_dir
        self.ui.lineEdit_videosDir.setText(self.videos_dir)

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

    def getResultsDir(self):
        results_dir = QFileDialog.getExistingDirectory(
            self,
            "Selects the directory where the analysis results will be stored",
            self.results_dir)
        if results_dir:
            self.updateResultsDir(results_dir)

    def updateResultsDir(self, results_dir):
        self.results_dir = results_dir
        self.ui.lineEdit_resultsDir.setText(self.results_dir)

    def getMasksDir(self):
        mask_files_dir = QFileDialog.getExistingDirectory(
            self,
            "Selects the directory where the hdf5 masked videos will be stored",
            self.mask_files_dir)
        if mask_files_dir:
            self.updateMasksDir(mask_files_dir)

    def updateMasksDir(self, mask_files_dir):
        self.mask_files_dir = mask_files_dir
        self.ui.lineEdit_masksDir.setText(self.mask_files_dir)

        results_dir = getResultsDir(mask_files_dir)
        self.updateResultsDir(results_dir)

    def getParamFile(self):
        param_file, _ = QFileDialog.getOpenFileName(
            self, "Find parameters file", '', "JSON files (*.json);; All (*)")
        if param_file:
            self.updateParamFile(param_file)

    def updateParamFile(self, param_file):
        # i accept a file if it is empty (the program will call default parameters),
        # but if not I want to check if it is a valid file.
        if param_file:
            try:
                with open(param_file, 'r') as fid:
                    json_str = fid.read()
                    json_param = json.loads(json_str)
                    self.ui.lineEdit_paramFile.setText(param_file)

            except (IOError, OSError, UnicodeDecodeError, json.decoder.JSONDecodeError):
                QMessageBox.critical(
                    self,
                    'Cannot read parameters file.',
                    "Cannot read parameters file. Try another file",
                    QMessageBox.Ok)
                return

        self.param_file = param_file

    def startAnalysis(self):
        is_compress = self.ui.checkBox_isCompress.isChecked()
        is_track = self.ui.checkBox_isTrack.isChecked()

        if not is_compress and not is_track:
            QMessageBox.critical(
                self,
                'Error',
                "Neither compression or selection is selected. No analysis to be done.",
                QMessageBox.Ok)

        compress_args = {}
        if is_compress:
            compress_args = self.readCompressArgs()
            if not os.path.exists(compress_args['video_dir_root']):
                QMessageBox.critical(
                    self,
                    'Error',
                    "The videos directory does not exist. Please select a valid directory.",
                    QMessageBox.Ok)
                return

        track_args = {}
        if is_track:
            track_args = self.readTrackArgs()
            if not is_compress and not os.path.exists(
                    track_args['mask_dir_root']):
                QMessageBox.critical(
                    self,
                    'Error',
                    "The masks directory does not exist. Please select a valid directory.",
                    QMessageBox.Ok)
                return

        def analysis_fun(compress_args, track_args):
            if is_compress:
                print('Compression...')
                compressMultipleFilesFun(**compress_args)
            if is_track:
                print('Tracking...')
                trackMultipleFilesFun(**track_args)

        analysis_worker = WorkerFunQt(
            analysis_fun, {
                'compress_args': compress_args, 'track_args': track_args})
        progress = AnalysisProgress(analysis_worker)
        progress.exec_()

    def readCompressArgs(self):
        compress_vals = compress_dflt_vals.copy()
        compress_vals['video_dir_root'] = self.ui.lineEdit_videosDir.text()
        compress_vals['mask_dir_root'] = self.ui.lineEdit_masksDir.text()
        compress_vals['max_num_process'] = self.ui.spinBox_numMaxProc.value()
        compress_vals['tmp_dir_root'] = self.ui.lineEdit_tmpDir.text()
        compress_vals[
            'is_single_worm'] = self.ui.checkBox_isSingleWorm.isChecked()
        compress_vals['json_file'] = self.ui.lineEdit_paramFile.text()
        compress_vals[
            'is_copy_video'] = self.ui.checkBox_isCopyVideo.isChecked()
        if self.ui.checkBox_txtFileList.isChecked():
            compress_vals['videos_list'] = self.ui.lineEdit_txtFileList.value()
        else:
            compress_vals[
                'pattern_include'] = self.ui.lineEdit_patternInComp.text()
            compress_vals[
                'pattern_exclude'] = self.ui.lineEdit_patternExcComp.text()

        return compress_vals

    def readTrackArgs(self):
        track_vals = track_dflt_vals.copy()
        track_vals['mask_dir_root'] = self.ui.lineEdit_masksDir.text()
        track_vals['results_dir_root'] = self.ui.lineEdit_resultsDir.text()
        track_vals['max_num_process'] = self.ui.spinBox_numMaxProc.value()
        track_vals['tmp_dir_root'] = self.ui.lineEdit_tmpDir.text()
        track_vals['is_single_worm'] = self.ui.checkBox_isSingleWorm.isChecked()
        track_vals['json_file'] = self.ui.lineEdit_paramFile.text()
        if self.ui.checkBox_txtFileList.isChecked():
            track_vals['videos_list'] = self.ui.lineEdit_txtFileList.value()
        else:
            track_vals[
                'pattern_include'] = self.ui.lineEdit_patternInTrack.text()
            track_vals[
                'pattern_exclude'] = self.ui.lineEdit_patternExcTrack.text()
        return track_vals


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    ui = BatchProcessing_GUI()
    ui.show()
    sys.exit(app.exec_())
