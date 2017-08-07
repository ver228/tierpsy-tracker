import json
import os
from functools import partial

import numpy as np
import pandas as pd
import tables
from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QFont, QPen, QPolygonF, QColor, QKeySequence
from PyQt5.QtWidgets import QApplication, QMessageBox, QShortcut

from tierpsy.analysis.feat_create.obtainFeatures import getWormFeaturesFilt
from tierpsy.analysis.ske_create.helperIterROI import getWormROI
from tierpsy.analysis.ske_filt.getFilteredSkels import getValidIndexes
from tierpsy.analysis.ske_filt import get_feat_filt_param

from tierpsy.gui.AnalysisProgress import WorkerFunQt, AnalysisProgress
from tierpsy.gui.MWTrackerViewer_ui import Ui_MWTrackerViewer
from tierpsy.gui.TrackerViewerAux import TrackerViewerAuxGUI

from tierpsy.helper.misc import WLAB, save_modified_table
from tierpsy.helper.params import TrackerParams, read_fps, read_microns_per_pixel

from tierpsy.processing.trackProvenance import getGitCommitHash, execThisPoint


class MWTrackerViewer_GUI(TrackerViewerAuxGUI):

    def __init__(self, ui='', argv=''):
        if not ui:
            super().__init__(Ui_MWTrackerViewer())
        else:
            super().__init__(ui)

        self.setWindowTitle("Multi-Worm Viewer")


        self.vfilename = '' if len(argv) <= 1 else argv[1]
        self.lastKey = ''
        self.traj_for_plot = {}

        self.food_coordinates = None

        self.worm_index_roi1 = 1
        self.worm_index_roi2 = 1

        self.wlab = WLAB
        self.wlabC = {
            self.wlab['U']: Qt.white,
            self.wlab['WORM']: Qt.green,
            self.wlab['WORMS']: Qt.blue,
            self.wlab['BAD']: Qt.darkRed,
            self.wlab['GOOD_SKE']: Qt.darkCyan}

        self.videos_dir = r"/Volumes/behavgenom$/GeckoVideo/MaskedVideos/"
        self.results_dir = ''
        self.skeletons_file = ''
        self.worm_index_type = 'worm_index_manual'
        self.label_type = 'worm_label'
        self.frame_data = None
        self.mean_intensity = None

        self.ui.intensity_label.setStyleSheet('') #avoid displaying color at the start of the programı

        self.ui.comboBox_ROI1.activated.connect(self.selectROI1)
        self.ui.comboBox_ROI2.activated.connect(self.selectROI2)
        self.ui.checkBox_ROI1.stateChanged.connect(
            partial(self.updateROIcanvasN, 1))
        self.ui.checkBox_ROI2.stateChanged.connect(
            partial(self.updateROIcanvasN, 2))

        self.ui.comboBox_labelType.currentIndexChanged.connect(
            self.selectWormIndexType)

        self.ui.checkBox_showFood.stateChanged.connect(self.updateImage)
        self.ui.checkBox_showFood.setEnabled(False)
        self.ui.checkBox_showFood.setChecked(True)


        # flags for RW and FF
        self.RW, self.FF = 1, 2
        self.ui.pushButton_ROI1_RW.clicked.connect(
            partial(self.roiRWFF, self.RW, 1))
        self.ui.pushButton_ROI1_FF.clicked.connect(
            partial(self.roiRWFF, self.FF, 1))
        self.ui.pushButton_ROI2_RW.clicked.connect(
            partial(self.roiRWFF, self.RW, 2))
        self.ui.pushButton_ROI2_FF.clicked.connect(
            partial(self.roiRWFF, self.FF, 2))

        self.ui.pushButton_U.clicked.connect(
            partial(self.tagWorm, self.wlab['U']))
        self.ui.pushButton_W.clicked.connect(
            partial(self.tagWorm, self.wlab['WORM']))
        self.ui.pushButton_WS.clicked.connect(
            partial(self.tagWorm, self.wlab['WORMS']))
        self.ui.pushButton_B.clicked.connect(
            partial(self.tagWorm, self.wlab['BAD']))

        self.ui.pushButton_save.clicked.connect(self.saveData)

        self.ui.pushButton_join.clicked.connect(self.joinTraj)
        self.ui.pushButton_split.clicked.connect(self.splitTraj)

        self.showT = {x: self.ui.comboBox_showLabels.findText(x , flags=Qt.MatchContains) 
                                for x in ['hide', 'all', 'filter']}
        self.ui.comboBox_showLabels.setCurrentIndex(self.showT['all'])
        self.ui.comboBox_showLabels.currentIndexChanged.connect(self.updateImage)        
        
        self.drawT = {x: self.ui.comboBox_drawType.findText(x , flags=Qt.MatchContains) 
                                for x in ['boxes', 'traj']}
        self.ui.comboBox_drawType.currentIndexChanged.connect(self._purge_draw_traj)

        # select worm ROI when doubleclick a worm
        self.mainImage._canvas.mouseDoubleClickEvent = self.selectWorm

        #SHORTCUTS
        self.ui.pushButton_W.setShortcut(QKeySequence(Qt.Key_W))
        self.ui.pushButton_U.setShortcut(QKeySequence(Qt.Key_U))
        self.ui.pushButton_WS.setShortcut(QKeySequence(Qt.Key_C))
        self.ui.pushButton_B.setShortcut(QKeySequence(Qt.Key_B))
        self.ui.radioButton_ROI1.setShortcut(QKeySequence(Qt.Key_Up))
        self.ui.radioButton_ROI2.setShortcut(QKeySequence(Qt.Key_Down))
        self.ui.pushButton_join.setShortcut(QKeySequence(Qt.Key_J))
        self.ui.pushButton_split.setShortcut(QKeySequence(Qt.Key_S))

        
        #This part is broken I think I will remove it from here, and force the user to use BatchProcessing
        
        self.ui.pushButton_feats.hide() 
        #self.ui.pushButton_feats.clicked.connect(self.getManualFeatures)




    def keyPressEvent(self, event):
        #MORE SHORTCUTS
        # go the the start of end of a trajectory
        if event.key() == Qt.Key_BracketLeft:
            current_roi = 1 if self.ui.radioButton_ROI1.isChecked() else 2
            self.roiRWFF(current_roi, self.RW)
        #[ <<
        elif event.key() == Qt.Key_BracketRight:
            current_roi = 1 if self.ui.radioButton_ROI1.isChecked() else 2
            self.roiRWFF(current_roi, self.FF)

        super().keyPressEvent(event)


    def getManualFeatures(self):
        # save the user changes before recalculating anything
        self.saveData()

        #%%
        self.feat_manual_file = self.skeletons_file.replace(
            '_skeletons.hdf5', '_feat_manual.hdf5')
        
        point_parameters = {
            'func': getWormFeaturesFilt,
            'argkws': {
                'skeletons_file': self.skeletons_file,
                'features_file': self.feat_manual_file,
                'is_single_worm': False,
                'use_skel_filter': True,
                'use_manual_join': True,
                'feat_filt_param': self.feat_filt_param,
                'split_traj_time': self.param_default.feats_param['split_traj_time']
                },
                'provenance_file': self.feat_manual_file
            }


        def featManualFun(point_argvs):
            commit_hash = getGitCommitHash()
            execThisPoint('FEAT_MANUAL_CREATE', **point_argvs,
                          commit_hash=commit_hash, cmd_original='GUI')

        trackpoint_worker = WorkerFunQt(
            featManualFun, {
                'point_argvs': point_parameters})
        progress_dialog = AnalysisProgress(trackpoint_worker)
        progress_dialog.setAttribute(Qt.WA_DeleteOnClose)
        progress_dialog.exec_()

    

    def saveData(self):
        '''save data from manual labelling. pytables saving format is more convenient than pandas'''

        if os.name == 'nt':
            # I Windows the paths return by QFileDialog use / as the file
            # separation character. We need to correct it.
            for field_name in ['vfilename', 'skeletons_file']:
                setattr(
                    self, field_name, getattr(
                        self, field_name).replace(
                        '/', os.sep))

        save_modified_table(self.skeletons_file, self.trajectories_data, 'trajectories_data')

        self.updateSkelFile(self.skeletons_file)

    def updateVideoFile(self, vfilename):
        super().updateVideoFile(vfilename)
        
        if self.fid is not None:
            #get mean intensity information.
            #Useful for the optogenetic experiments. 
            try:
                mean_int = self.fid.get_node('/mean_intensity')[:]
                
                #calculate the intensity range and normalize the data. 
                #I am ignoring any value less than 1. The viewer only works with uint8 data.
                
                dd = mean_int[mean_int>=1] 
                if dd.size == 0:
                    raise ValueError

                bot = np.min(dd)
                top = np.max(dd)
                rr = top-bot

                # if the mean value change is less than 1 (likely continous image do nothing)
                if rr <= 1:
                    raise ValueError

                self.mean_intensity = (mean_int-bot)/(rr)

            except (tables.exceptions.NoSuchNodeError, ValueError):
                self.mean_intensity = None
                self.ui.intensity_label.setStyleSheet('')
        
        self.updateImage()

    def updateSkelFile(self, skeletons_file):
        super().updateSkelFile(skeletons_file)
        
        if not self.skeletons_file or self.trajectories_data is None:
            self.food_coordinates = None
            return

        with tables.File(self.skeletons_file, 'r') as fid:
            if not '/food_cnt_coord' in fid:
                self.food_coordinates = None
                self.ui.checkBox_showFood.setEnabled(False)
            else:
                self.food_coordinates = fid.get_node('/food_cnt_coord')[:]
                self.ui.checkBox_showFood.setEnabled(True)

        #correct the index in case it was given before as worm_index_N
        if 'worm_index_N' in self.trajectories_data:
            self.trajectories_data = self.trajectories_data.rename(
                columns={'worm_index_N': 'worm_index_manual'})
        
        if not 'worm_index_manual' in self.trajectories_data:
            self.trajectories_data['worm_label'] = self.wlab['U']
            self.trajectories_data['worm_index_manual'] = self.trajectories_data[
                'worm_index_joined']
        
        self.updateWormIndexTypeMenu()
        
        #read filter skeletons parameters
        with tables.File(self.skeletons_file, 'r') as skel_fid:

            # if any of this fields is missing load the default parameters
            self.param_default = TrackerParams()
            try:
                ss = skel_fid.get_node('/provenance_tracking/ske_filt').read()
                ss = json.loads(ss.decode("utf-8"))
                saved_func_args = json.loads(ss['func_arguments'])

                self.feat_filt_param = {
                x: saved_func_args[x] for x in [
                'min_num_skel',
                'bad_seg_thresh',
                'min_displacement']}
            except (KeyError, tables.exceptions.NoSuchNodeError):
                self.feat_filt_param = get_feat_filt_param(self.param_default.p_dict)

        self.expected_fps = read_fps(self.vfilename)
        
        #TODO: THIS IS NOT REALLY THE INDEX I USE IN THE FEATURES FILES. I NEED A MORE CLEVER WAY TO SEE WHAT I AM REALLY FILTERING.
        dd = {x:self.feat_filt_param[x] for x in ['min_num_skel', 'bad_seg_thresh', 'min_displacement']}
        good_traj_index, _ = getValidIndexes(self.trajectories_data, **dd, worm_index_type=self.worm_index_type)
        self.trajectories_data['is_valid_index'] = self.trajectories_data[self.worm_index_type].isin(good_traj_index)
        
        self.traj_time_grouped = self.trajectories_data.groupby('frame_number')

        self.traj_for_plot = {} #delete previous plotted trajectories
        self.updateImage()

    def updateWormIndexTypeMenu(self):
        possible_indexes = [x.replace('worm_index_', '') for x in self.trajectories_data.columns if x.startswith('worm_index_')]
        assert len(set(possible_indexes)) == len(possible_indexes) #all indexes ending must be different
        
        menu_names = sorted([x + ' index' for x in possible_indexes])
        self.ui.comboBox_labelType.clear()
        self.ui.comboBox_labelType.addItems(menu_names)
        if 'manual' in possible_indexes:
            dd = self.ui.comboBox_labelType.findText('manual index')
            self.ui.comboBox_labelType.setCurrentIndex(dd);

        self.selectWormIndexType()

    def selectWormIndexType(self):
        index_option = self.ui.comboBox_labelType.currentText()
        
        if not index_option:
            return
        assert index_option.endswith(' index')
        self.worm_index_type = 'worm_index_' + index_option.replace(' index', '')
        

        # select between automatic and manual worm indexing and label
        if self.worm_index_type == 'worm_index_manual':
            self.label_type = 'worm_label'
            self.ui.pushButton_U.setEnabled(True)
            self.ui.pushButton_W.setEnabled(True)
            self.ui.pushButton_WS.setEnabled(True)
            self.ui.pushButton_B.setEnabled(True)
            self.ui.pushButton_join.setEnabled(True)
            self.ui.pushButton_split.setEnabled(True)

        else:
            self.label_type = 'auto_label'
            self.ui.pushButton_U.setEnabled(False)
            self.ui.pushButton_W.setEnabled(False)
            self.ui.pushButton_WS.setEnabled(False)
            self.ui.pushButton_B.setEnabled(False)
            self.ui.pushButton_join.setEnabled(False)
            self.ui.pushButton_split.setEnabled(False)

        self.updateImage()


    # update image
    def updateImage(self):
        if self.image_group is None:
            return

        super(TrackerViewerAuxGUI, self).readCurrentFrame()
        self.img_h_ratio = self.frame_qimg.height() / self.image_height
        self.img_w_ratio = self.frame_qimg.width() / self.image_width

        # read the data of the particles that exists in the frame
        self.frame_data = self.getFrameData(self.frame_number)
            
        #draw extra info only if the worm_index_type is valid
        if self.frame_data is not None and \
        self.worm_index_type in self.frame_data:
            #filter any -1 index
            self.frame_data = self.frame_data[self.frame_data[self.worm_index_type]>=0]
            if self.frame_data.size > 0:
                self._draw_worm_markers(self.frame_qimg)
                self._draw_food_contour(self.frame_qimg)

                self.updateROIcanvasN(1)
                self.updateROIcanvasN(2)

                
        
        else:
            self.ui.wormCanvas1.clear() 
            self.ui.wormCanvas2.clear()        
        # create the pixmap for the label
        self.mainImage.setPixmap(self.frame_qimg)

        if self.mean_intensity is not None and self.frame_number < self.mean_intensity.size:
            d = int(self.mean_intensity[self.frame_number]*255)
            self.ui.intensity_label.setStyleSheet('QLabel {background-color: rgb(%i, %i, %i);}' % (0, 0, d))



    def _draw_food_contour(self, image):
        if self.food_coordinates is None or not self.ui.checkBox_showFood.isChecked():
            return

        painter = QPainter()
        painter.begin(image)

        penwidth = max(1, max(image.height(), image.width()) // 800)
        self.img_h_ratio = image.height() / self.image_height
        self.img_w_ratio = image.width() / self.image_width

        col = QColor(255, 0, 0)
        p = QPolygonF()
        for x,y in self.food_coordinates:
            p.append(QPointF(x,y))
            
        pen = QPen()
        pen.setWidth(penwidth)
        pen.setColor(col)
        painter.setPen(pen)

        painter.drawPolyline(p)
        painter.end()

    def _draw_worm_markers(self, image):
        '''
        Draw traj worm trajectory.
        '''
        
        if not self.worm_index_type in self.frame_data or \
        self.ui.comboBox_showLabels.currentIndex() == self.showT['hide']:
            return

        self.img_h_ratio = image.height() / self.image_height
        self.img_w_ratio = image.width() / self.image_width

        
        painter = QPainter()
        painter.begin(image)

        fontsize = max(1, max(image.height(), image.width()) // 120)
        penwidth = max(1, max(image.height(), image.width()) // 800)
        penwidth = penwidth if penwidth % 2 == 1 else penwidth + 1

        if not self.label_type in self.frame_data:
            self.frame_data[self.label_type] = self.wlab['U']

        new_traj = {}
        for row_id, row_data in self.frame_data.iterrows():
            x = row_data['coord_x']
            y = row_data['coord_y']
            # check if the coordinates are nan
            if not (x == x) or not (y == y):
                continue

            #if select between showing filtered index or not
            if self.ui.comboBox_showLabels.currentIndex() == self.showT['filter'] and not row_data['is_valid_index']:
                continue

            traj_ind = int(row_data[self.worm_index_type])
            x = int(round(x * self.img_h_ratio))
            y = int(round(y * self.img_w_ratio))
            label_type = self.wlabC[int(row_data[self.label_type])]
            
            roi_size = row_data['roi_size']

            cb_ind = self.ui.comboBox_drawType.currentIndex()
            if cb_ind == self.drawT['boxes']:
                self._draw_boxes(painter, traj_ind, x, y, roi_size, label_type, penwidth, fontsize)
            elif cb_ind == self.drawT['traj']:
                self._draw_trajectories(painter, new_traj, traj_ind, x, y, penwidth)
            
        painter.end()
        self.traj_for_plot = new_traj

    def _purge_draw_traj(self):
        self.traj_for_plot = {}
        self.updateImage()
        

    def _draw_trajectories(self, painter, new_traj, traj_ind, x, y, penwidth):
        if not traj_ind in self.traj_for_plot:
            new_traj[traj_ind] = {'col':QColor(*np.random.randint(50, 230, 3)),
                        'p':QPolygonF()}
        else:
            new_traj[traj_ind] = self.traj_for_plot[traj_ind]

        new_traj[traj_ind]['p'].append(QPointF(x,y))
            
        col = new_traj[traj_ind]['col']
        pen = QPen()
        pen.setWidth(penwidth)
        pen.setColor(col)
        painter.setPen(pen)

        painter.drawPolyline(new_traj[traj_ind]['p'])
        painter.drawEllipse(x,y, penwidth, penwidth)

    def _draw_boxes(self, painter, traj_ind, x, y, roi_size, label_type, penwidth, fontsize):
        '''
        Draw traj worm trajectory.
        '''
        pen = QPen(label_type)
        pen.setWidth(penwidth)
        painter.setPen(pen)
        painter.setFont(QFont('Decorative', fontsize))

        painter.drawText(x, y, str(traj_ind))

        bb = roi_size * self.img_w_ratio
        painter.drawRect(x - bb / 2, y - bb / 2, bb, bb)


    # update zoomed ROI
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
            self.updateROIcanvas(
                self.ui.wormCanvas1,
                self.worm_index_roi1,
                self.ui.comboBox_ROI1,
                self.ui.checkBox_ROI1.isChecked())
        elif n_canvas == 2:
            self.updateROIcanvas(
                self.ui.wormCanvas2,
                self.worm_index_roi2,
                self.ui.comboBox_ROI2,
                self.ui.checkBox_ROI2.isChecked())

        

                

    # function that generalized the updating of the ROI
    def updateROIcanvas(
            self,
            wormCanvas,
            worm_index_roi,
            comboBox_ROI,
            isDrawSkel):

        if self.frame_data is None:
            # no trajectories data presented, nothing to do here
            wormCanvas.clear()
            return


        # update valid index for the comboBox
        comboBox_ROI.clear()
        comboBox_ROI.addItem(str(worm_index_roi))
        
        for ind in self.frame_data[self.worm_index_type].data:
            comboBox_ROI.addItem(str(ind))

        # extract individual worm ROI
        good = self.frame_data[self.worm_index_type] == worm_index_roi
        row_data = self.frame_data.loc[good].squeeze()

        if row_data.size == 0 or np.isnan(
                row_data['coord_x']) or np.isnan(
                row_data['coord_y']):
            # invalid data nothing to do here
            wormCanvas.clear()
            return

        worm_img, roi_corner = getWormROI(self.frame_img, row_data['coord_x'], row_data[
                                          'coord_y'], row_data['roi_size'])
        
        roi_ori_size = worm_img.shape

        worm_img = np.ascontiguousarray(worm_img)
        worm_qimg = self._convert2Qimg(worm_img)

        canvas_size = min(wormCanvas.height(), wormCanvas.width())
        worm_qimg = worm_qimg.scaled(
            canvas_size, canvas_size, Qt.KeepAspectRatio)

        worm_qimg = self.drawSkelResult(worm_img, worm_qimg, row_data, 
            isDrawSkel, roi_corner, read_center=False)
        
        pixmap = QPixmap.fromImage(worm_qimg)
        wormCanvas.setPixmap(pixmap)

    def selectWorm(self, event):

        x = event.pos().x()
        y = event.pos().y()

        if self.frame_data is None or self.frame_data.size == 0:
            return

        x /= self.img_w_ratio
        y /= self.img_h_ratio
        R = (x - self.frame_data['coord_x'])**2 + \
            (y - self.frame_data['coord_y'])**2

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

        if self.frame_data is None:
            return

        if not worm_ind in self.frame_data['worm_index_manual'].values:
            QMessageBox.critical(
                self,
                'The selected worm is not in this frame.',
                'Select a worm in the current frame to label.',
                QMessageBox.Ok)
            return

        good = self.trajectories_data['worm_index_manual'] == worm_ind
        self.trajectories_data.loc[good, 'worm_label'] = label_ind
        self.updateImage()

    # move to the first or the last frames of a trajectory
    def roiRWFF(self, rwff, n_roi=None):

        if self.frame_data is None:
            return
        if n_roi is None:
            n_roi = 1 if self.ui.radioButton_ROI1.isChecked() else 2
        if n_roi == 1:
            worm_ind = self.worm_index_roi1
        elif n_roi == 2:
            worm_ind = self.worm_index_roi2
        else:
            raise ValueError('Invalid n_roi value : {} '.format(n_roi))

        # use 1 for rewind RW or 2 of fast forward
        good = self.trajectories_data[self.worm_index_type] == worm_ind
        frames = self.trajectories_data.loc[good, 'frame_number']

        if frames.size == 0:
            return

        if rwff == self.RW:
            self.frame_number = frames.min()
        elif rwff == self.FF:
            self.frame_number = frames.max()
        else:
            raise ValueError('Invalid rwff value : {} '.format(rwff))

        self.ui.spinBox_frame.setValue(self.frame_number)

    def joinTraj(self):
        if self.worm_index_type != 'worm_index_manual' \
        or self.frame_data is None:
            return

        worm_ind1 = self.worm_index_roi1
        worm_ind2 = self.worm_index_roi2

        if worm_ind1 == worm_ind2:
            QMessageBox.critical(
                self,
                'Cannot join the same trajectory with itself',
                'Cannot join the same trajectory with itself.',
                QMessageBox.Ok)
            return

        index1 = (self.trajectories_data[
                  'worm_index_manual'] == worm_ind1).values
        index2 = (self.trajectories_data[
                  'worm_index_manual'] == worm_ind2).values

        # if the trajectories do not overlap they shouldn't have frame_number
        # indexes in commun
        frame_number = self.trajectories_data.loc[
            index1 | index2, 'frame_number']

        if frame_number.size != np.unique(frame_number).size:
            QMessageBox.critical(
                self,
                'Cannot join overlaping trajectories',
                'Cannot join overlaping trajectories.',
                QMessageBox.Ok)
            return

        if not (worm_ind1 in self.frame_data[
                'worm_index_manual'].values or worm_ind2 in self.frame_data['worm_index_manual'].values):
            reply = QMessageBox.question(
                self,
                'Message',
                "The none of the selected worms to join is not in this frame. Are you sure to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No)

            if reply == QMessageBox.No:
                return

        # get the first row for each segment to extract some data
        first_row1 = self.trajectories_data.loc[index1, :].iloc[0]
        first_row2 = self.trajectories_data.loc[index2, :].iloc[0]

        # join trajectories
        self.trajectories_data.loc[
            index2, 'worm_label'] = first_row1['worm_label']
        self.trajectories_data.loc[index2, 'worm_index_manual'] = worm_ind1

        self.worm_index_roi1 = worm_ind1
        self.worm_index_roi2 = worm_ind1
        self.updateImage()

        self.ui.spinBox_join1.setValue(worm_ind1)
        self.ui.spinBox_join2.setValue(worm_ind1)

    def splitTraj(self):
        if self.worm_index_type != 'worm_index_manual' \
        or self.frame_data is None:
            return

        if self.ui.radioButton_ROI1.isChecked():
            worm_ind = self.worm_index_roi1  # self.ui.spinBox_join1.value()
        elif self.ui.radioButton_ROI2.isChecked():
            worm_ind = self.worm_index_roi2

        if not worm_ind in self.frame_data['worm_index_manual'].data:
            QMessageBox.critical(
                self,
                'Worm index is not in the current frame.',
                'Worm index is not in the current frame. Select a valid index.',
                QMessageBox.Ok)
            return

        last_index = self.trajectories_data['worm_index_manual'].max()

        new_ind1 = last_index + 1
        new_ind2 = last_index + 2

        good = self.trajectories_data['worm_index_manual'] == worm_ind
        frames = self.trajectories_data.loc[good, 'frame_number']
        frames = frames.sort_values(inplace=False)

        good = frames < self.frame_number
        index1 = frames[good].index
        index2 = frames[~good].index
        self.trajectories_data.ix[index1, 'worm_index_manual'] = new_ind1
        self.trajectories_data.ix[index2, 'worm_index_manual'] = new_ind2

        self.worm_index_roi1 = new_ind1
        self.worm_index_roi2 = new_ind2
        self.updateImage()

        self.ui.spinBox_join1.setValue(new_ind1)
        self.ui.spinBox_join2.setValue(new_ind2)

    

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    ui = MWTrackerViewer_GUI(argv=sys.argv)
    ui.show()
    sys.exit(app.exec_())
