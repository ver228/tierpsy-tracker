import os
import json

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt

from tierpsy.processing.processMultipleFilesFun import processMultipleFilesFun
from tierpsy.processing.helper import remove_border_checkpoints, get_results_dir, get_masks_dir 

from tierpsy.gui.AnalysisProgress import AnalysisProgress, WorkerFunQt
from tierpsy.gui.HDF5VideoPlayer import LineEditDragDrop
from tierpsy.gui.BatchProcessing_ui import Ui_BatchProcessing
from tierpsy.gui.GetAllParameters import ParamWidgetMapper

#get default parameters files
from tierpsy import DFLT_PARAMS_FILES
from tierpsy.helper.params import TrackerParams
from tierpsy.helper.params.tracker_param import get_dflt_sequence
from tierpsy.helper.params.docs_process_param import proccess_args_dflt, proccess_args_info


#If a widget is not enabled ParamWidgetMapper will return the value in proccess_args_dflt,
#however for the tmp directory I want to change this behaviour so when it is not enabled it is left empty so it is not used.
TMP_DIR_ROOT = proccess_args_dflt['tmp_dir_root']
proccess_args_dflt_r = proccess_args_dflt.copy()
proccess_args_dflt_r['tmp_dir_root'] = ''

class BatchProcessing_GUI(QMainWindow):

    def __init__(self):
        super(BatchProcessing_GUI, self).__init__()
        self.mask_files_dir = ''
        self.results_dir = ''
        self.videos_dir = ''
        self.tmp_dir_root = ''
        self.analysis_checkpoints = []

        self.ui = Ui_BatchProcessing()
        self.ui.setupUi(self)
        
        # for the moment this option is more confusing that helpful so we hide it
        self.ui.p_unmet_requirements.hide() 

        valid_options = dict(json_file = [''] + DFLT_PARAMS_FILES)
        self.mapper = ParamWidgetMapper(self.ui,
                                        #do not want to pass this values by reference
                                        default_param=proccess_args_dflt_r.copy(), 
                                        info_param=proccess_args_info.copy(), 
                                        valid_options=valid_options)

        self.ui.p_json_file.currentIndexChanged.connect(self.updateCheckpointsChange)
        self.ui.p_force_start_point.currentIndexChanged.connect(self.updateStartPointChange)
        self.updateCheckpointsChange()

        self.ui.checkBox_txtFileList.stateChanged.connect(self.enableTxtFileListButton)
        self.ui.checkBox_tmpDir.stateChanged.connect(self.enableTmpDirButton)
        

        self.ui.pushButton_txtFileList.clicked.connect(self.getTxtFileList)
        self.ui.pushButton_paramFile.clicked.connect(self.getParamFile)
        self.ui.pushButton_videosDir.clicked.connect(self.getVideosDir)
        self.ui.pushButton_masksDir.clicked.connect(self.getMasksDir)
        self.ui.pushButton_resultsDir.clicked.connect(self.getResultsDir)
        self.ui.pushButton_tmpDir.clicked.connect(self.getTmpDir)
        self.ui.pushButton_start.clicked.connect(self.startAnalysis)
        

        self.ui.checkBox_tmpDir.setChecked(False)
        self.enableTxtFileListButton()
        self.ui.checkBox_tmpDir.setChecked(True)
        self.enableTmpDirButton()

        LineEditDragDrop(
            self.ui.p_videos_list,
            self.updateTxtFileList,
            os.path.isfile)
        LineEditDragDrop(
            self.ui.p_video_dir_root,
            self.updateVideosDir,
            os.path.isdir)
        LineEditDragDrop(
            self.ui.p_mask_dir_root,
            self.updateMasksDir,
            os.path.isdir)
        LineEditDragDrop(
            self.ui.p_results_dir_root,
            self.updateResultsDir,
            os.path.isdir)
        LineEditDragDrop(
            self.ui.p_tmp_dir_root,
            self.updateTmpDir,
            os.path.isdir)

        LineEditDragDrop(
            self.ui.p_json_file,
            self.updateParamFile,
            os.path.isfile)

    def enableTxtFileListButton(self):
        is_enable = self.ui.checkBox_txtFileList.isChecked()
        self.ui.pushButton_txtFileList.setEnabled(is_enable)
        self.ui.p_videos_list.setEnabled(is_enable)
        self.ui.label_patternIn.setEnabled(not is_enable)
        self.ui.label_patternExc.setEnabled(not is_enable)
        self.ui.p_pattern_exclude.setEnabled(not is_enable)
        self.ui.p_pattern_include.setEnabled(not is_enable)

        if not is_enable:
            self.updateTxtFileList('')

    def getTxtFileList(self):
        videos_list, _ = QFileDialog.getOpenFileName(
            self, "Select a text file with a list of files to be analyzed.", '', "Text file (*.txt);;All files (*)")
        if os.path.isfile(videos_list):
            self.updateTxtFileList(videos_list)

    def updateTxtFileList(self, videos_list):
        if videos_list:
            #test that it is a valid text file with a list of files inside of it.
            try:
                with open(videos_list, 'r') as fid:
                    first_line = fid.readline().strip()
                    if not os.path.exists(first_line):
                        raise FileNotFoundError
            except:
                QMessageBox.critical(
                        self,
                        "It is not a text file with a valid list of files.",
                        "The selected file does not seem to contain a list of valid files to process.\n"
                        "Plase make sure to select a text file that contains a list of existing files.",
                        QMessageBox.Ok)
                return


        self.videos_list = videos_list
        self.ui.p_videos_list.setText(videos_list)

    def enableTmpDirButton(self):
        is_enable = self.ui.checkBox_tmpDir.isChecked()
        self.ui.pushButton_tmpDir.setEnabled(is_enable)
        self.ui.p_tmp_dir_root.setEnabled(is_enable)

        tmp_dir_root = TMP_DIR_ROOT if is_enable else ''
        self.updateTmpDir(tmp_dir_root)

    def getTmpDir(self):
        tmp_dir_root = QFileDialog.getExistingDirectory(
            self,
            "Selects the directory where the hdf5 masked videos will be stored",
            self.tmp_dir_root)
        if tmp_dir_root:
            self.updateTmpDir(tmp_dir_root)

    def updateTmpDir(self, tmp_dir_root):
        self.mapper['tmp_dir_root'] = tmp_dir_root

    def getVideosDir(self):
        videos_dir = QFileDialog.getExistingDirectory(
            self,
            "Selects the directory where the original video files are stored.",
            self.videos_dir)
        if videos_dir:
            self.updateVideosDir(videos_dir)


    def updateVideosDir(self, videos_dir):
        self.videos_dir = videos_dir
        self.ui.p_video_dir_root.setText(self.videos_dir)

        mask_files_dir = get_masks_dir(self.videos_dir)
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
        self.mapper['results_dir_root'] = self.results_dir

    def getMasksDir(self):
        mask_files_dir = QFileDialog.getExistingDirectory(
            self,
            "Selects the directory where the hdf5 masked videos will be stored",
            self.mask_files_dir)
        if mask_files_dir:
            self.updateMasksDir(mask_files_dir)

    def updateMasksDir(self, mask_files_dir):

        results_dir = get_results_dir(mask_files_dir)

        self.mask_files_dir = mask_files_dir
        self.mapper['mask_dir_root'] = self.mask_files_dir
        
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

                    #find the current index in the combobox, if it preappend it.
                    ind_comb = self.ui.p_json_file.findText(param_file)
                    if ind_comb == -1:
                        self.ui.p_json_file.insertItem(-1, param_file)
                        ind_comb = 0

                    self.ui.p_json_file.setCurrentIndex(ind_comb)


            except (IOError, OSError, UnicodeDecodeError, json.decoder.JSONDecodeError):
                QMessageBox.critical(
                    self,
                    'Cannot read parameters file.',
                    "Cannot read parameters file. Try another file",
                    QMessageBox.Ok)
                return

        self.param_file = param_file

    def updateCheckpointsChange(self, index=0):
        '''
        index - dum variable to be able to connect to currentIndexChanged
        '''

        try:
            param = TrackerParams(self.mapper['json_file'])
        except FileNotFoundError:
            return

        analysis_checkpoints = param.p_dict['analysis_checkpoints'].copy()
        if not analysis_checkpoints:
            analysis_checkpoints = get_dflt_sequence(param.p_dict['analysis_type'])

        if analysis_checkpoints[-1] != 'FEAT_MANUAL_CREATE':
            analysis_checkpoints.append('FEAT_MANUAL_CREATE')

        self.analysis_checkpoints = analysis_checkpoints
        self.ui.p_force_start_point.clear()
        self.ui.p_force_start_point.addItems(self.analysis_checkpoints)
        self.ui.p_force_start_point.setCurrentIndex(0)

    def updateStartPointChange(self, index):
        remaining_points = self.analysis_checkpoints.copy()
        remove_border_checkpoints(remaining_points, 
                                    self.mapper['force_start_point'], 
                                    0)

        #Force to be able to select FEAT_MANUAL_CREATE only from p_force_start_point
        if len(remaining_points) >= 1 and remaining_points[-1] == 'FEAT_MANUAL_CREATE':
            remaining_points = remaining_points[:-1]



        if self.mapper['force_start_point'] == 'COMPRESS':
            self.ui.pushButton_videosDir.setEnabled(True)
            self.ui.p_video_dir_root.setEnabled(True)
            #self.mapper['pattern_include'] = '*.avi'
        else:
            self.ui.pushButton_videosDir.setEnabled(False)
            self.ui.p_video_dir_root.setEnabled(False)
            #if not '.hdf5' in self.mapper['pattern_include']:
            #    self.mapper['pattern_include'] = '*.hdf5'



        self.ui.p_end_point.clear()
        self.ui.p_end_point.addItems(remaining_points)

        nn = self.ui.p_end_point.count()
        self.ui.p_end_point.setCurrentIndex(nn-1)

    def startAnalysis(self):
        process_args = proccess_args_dflt_r.copy()
        #append the root dir if we are using any of the default parameters files. I didn't add the dir before because it is easy to read them in this way.
        process_args['analysis_checkpoints'] = self.analysis_checkpoints
        
        for x in self.mapper:
            process_args[x] = self.mapper[x]

        if 'COMPRESS' == self.mapper['force_start_point']:
            if not os.path.exists(process_args['video_dir_root']):
                QMessageBox.critical(
                    self,
                    'Error',
                    "The videos directory does not exist. Please select a valid directory.",
                    QMessageBox.Ok)
                return
        elif not os.path.exists(process_args['mask_dir_root']):
            QMessageBox.critical(
                self,
                'Error',
                "The masks directory does not exist. Please select a valid directory.",
                QMessageBox.Ok)
            return

        print(process_args)
        analysis_worker = WorkerFunQt(processMultipleFilesFun, process_args)
        progress = AnalysisProgress(analysis_worker)
        progress.exec_()

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    ui = BatchProcessing_GUI()
    ui.show()
    sys.exit(app.exec_())
