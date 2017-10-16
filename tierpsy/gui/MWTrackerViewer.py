import json
import os
from functools import partial
import numpy as np
import pandas as pd
import tables

import matplotlib

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
from tierpsy.helper.params import TrackerParams, read_fps

from tierpsy.processing.trackProvenance import getGitCommitHash, execThisPoint

class ContourDrawer(TrackerViewerAuxGUI):
    '''
    Dummy class with the contour functions
    '''
    def __init__(self, ui):
        super().__init__(ui)

        self.food_coordinates = None
        self.wlabC = {
            WLAB['U']: Qt.white,
            WLAB['WORM']: Qt.green,
            WLAB['WORMS']: Qt.blue,
            WLAB['BAD']: Qt.darkRed,
            WLAB['GOOD_SKE']: Qt.darkCyan
            }
        self.ui.checkBox_showFood.stateChanged.connect(self.updateImage)
        self.ui.checkBox_showFood.setEnabled(False)
        self.ui.checkBox_showFood.setChecked(True)
        
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
                #change from microns to pixels
                self.food_coordinates = fid.get_node('/food_cnt_coord')[:]
                if self.skel_microns_per_pixel is not None:
                        self.food_coordinates /= self.skel_microns_per_pixel
                
                self.ui.checkBox_showFood.setEnabled(True)

    def draw_food_contour(self, image):
        if self.food_coordinates is None or not self.ui.checkBox_showFood.isChecked():
            return

        painter = QPainter()
        painter.begin(image)

        penwidth = max(1, max(image.height(), image.width()) // 800)
        col = Qt.darkMagenta
        p = QPolygonF()
        for x,y in self.food_coordinates:
            p.append(QPointF(x,y))
            
        pen = QPen()
        pen.setWidth(penwidth)
        pen.setColor(col)
        painter.setPen(pen)

        painter.drawPolyline(p)
        painter.end()

class MarkersDrawer(TrackerViewerAuxGUI):
    def __init__(self, ui):
        super().__init__(ui)
        
        
        self.timeseries_data = None
        self.feat_column = ''
        
        self.traj_colors = {}
        self.n_points_traj = 250
        self.n_colors = 256
        cmap = matplotlib.cm.get_cmap("bwr")
        palette = [cmap(x) for x in np.linspace(0, 1, self.n_colors)]
        #palette = sns.color_palette("RdBu_r", self.n_colors)
        palette = np.round(np.array(palette)*255).astype(np.int)
        self.palette = [QColor(*x) for x in palette]


        self.drawT = {x: self.ui.comboBox_drawType.findText(x , flags=Qt.MatchContains) 
                                for x in ['boxes', 'traj']}
        
        self.showT = {x: self.ui.comboBox_showLabels.findText(x , flags=Qt.MatchContains) 
                                for x in ['hide', 'all', 'filter']}
        
        self.ui.comboBox_showLabels.setCurrentIndex(self.showT['all'])
        
        self.ui.feature_column.currentIndexChanged.connect(self.change_feature)
        self.ui.comboBox_drawType.currentIndexChanged.connect(self.updateImage)
        self.ui.feat_max_value.valueChanged.connect(self.updateImage)        
        self.ui.feat_min_value.valueChanged.connect(self.updateImage)   
        self.ui.is_color_features.stateChanged.connect(self.updateImage)

        self.enable_color_feats(False)

    def updateSkelFile(self, skeletons_file):
        super().updateSkelFile(skeletons_file)
        self.ui.feature_column.clear()
        self.traj_worm_index_grouped = None
        try:
            self.traj_colors = {}
            with pd.HDFStore(self.skeletons_file, 'r') as ske_file_id:
                self.timeseries_data = ske_file_id['/timeseries_data']
            
            if not 'skeleton_id' in self.trajectories_data:
                raise KeyError

            self.enable_color_feats(True)
            index_cols = ['worm_index', 'timestamp']
            columns = [x for x in self.timeseries_data.columns if x not in index_cols]
            self.ui.feature_column.addItems(columns)

        except (AttributeError, IOError, KeyError, tables.exceptions.HDF5ExtError):
            self.timeseries_data = None
            self.enable_color_feats(False)

        self.ui.is_color_features.setChecked(False)

        self._h_find_feat_limits()

    def change_feature(self):
        self._h_find_feat_limits()
        self.updateImage()

    def _h_find_feat_limits(self):
        self.feat_column = str(self.ui.feature_column.currentText())
        
        if self.feat_column and self.timeseries_data is not None:
            f_max = self.timeseries_data[self.feat_column].max()
            f_min = self.timeseries_data[self.feat_column].min()
            q1, q2 = self.timeseries_data[self.feat_column].quantile([0.02, 0.98])
            
        else:
            f_min, f_max, q1, q2  = 0,0,0,0

        self.ui.feat_max_value.setRange(f_min, f_max)
        self.ui.feat_min_value.setRange(f_min, f_max)
        self.ui.feat_min_value.setValue(q1)
        self.ui.feat_max_value.setValue(q2)
        
    def enable_color_feats(self, value):
        self.ui.feature_column.setEnabled(value)
        self.ui.feat_min_value.setEnabled(value)
        self.ui.feat_max_value.setEnabled(value)
        self.ui.is_color_features.setEnabled(value)

    def _h_assign_feat_color(self, skel_id):
        if (skel_id < 0) or (skel_id != skel_id):
            return Qt.black
        
        skel_id = int(skel_id)
        feat_val = self.timeseries_data.loc[skel_id, self.feat_column]
        
        if (feat_val != feat_val):
            return Qt.black
        
        #this function can and should be optimized
        f_min = self.ui.feat_min_value.value()
        f_max = self.ui.feat_max_value.value()
        
        if f_min == f_max: #dummy range in case all the values are the same
            f_min, f_max = -1, 1
        elif f_min > f_max:
            return Qt.black

        nn = np.clip((feat_val - f_min)/(f_max - f_min), 0, 1) 
        ind = int(np.round(nn*(self.n_colors-1)))
        
        col = self.palette[ind]
        return col


    def draw_worm_markers(self, image):
        '''
        Draw traj worm trajectory.
        '''
        if not self.worm_index_type in self.frame_data or \
        self.ui.comboBox_showLabels.currentIndex() == self.showT['hide']:
            return
        
        painter = QPainter()
        painter.begin(image)

        self.fontsize = max(1, max(image.height(), image.width()) // 120)
        
        penwidth = max(1, max(image.height(), image.width()) // 800)
        self.penwidth = penwidth if penwidth % 2 == 1 else penwidth + 1

        if not self.label_type in self.frame_data:
            self.frame_data[self.label_type] = self.wlab['U']

        for row_id, row_data in self.frame_data.iterrows():
            # check if the coordinates are nan
            if np.isnan(row_data['coord_x']) or np.isnan(row_data['coord_y']):
                continue

            #if select between showing filtered index or not
            if self.ui.comboBox_showLabels.currentIndex() == self.showT['filter'] and not row_data['is_valid_index']:
                continue

            cb_ind = self.ui.comboBox_drawType.currentIndex()
            if cb_ind == self.drawT['boxes']:
                self.draw_boxes(painter, row_data)
            elif cb_ind == self.drawT['traj']:
                self.draw_trajectories(painter, row_data)

        painter.end()
    
    def _h_get_trajectory(self, worm_index, current_frame):
        worm_data = self.traj_worm_index_grouped.get_group(worm_index)
        valid_index = worm_data.index[worm_data['frame_number']<= current_frame]

        ini = max(0, valid_index.size - self.frame_step*self.n_points_traj)
        traj_ind = valid_index.values[ini::self.frame_step]
        traj_data = worm_data.loc[traj_ind]
        return traj_data
    

    def draw_trajectories(self, painter, row_data):

        worm_index = int(row_data[self.worm_index_type])
        current_frame = row_data['frame_number']
        traj_data = self._h_get_trajectory(worm_index, current_frame)

        x_v = traj_data['coord_x'].round()
        y_v = traj_data['coord_y'].round()
        points = [QPointF(*map(int, c)) for c in zip(x_v, y_v)]

        if self.ui.is_color_features.isChecked():
            vec_color = [self._h_assign_feat_color(x) for x in traj_data['skeleton_id'].values]
            
            pen = QPen()
            pen.setWidth(self.penwidth)
            for p1, p2, c in zip(points[1:], points[:-1], vec_color):
                pen.setColor(c)
                painter.setPen(pen)
                painter.drawLine(p1, p2)
        else:
            pol = QPolygonF()
            for p in points:
                pol.append(p)

            if not worm_index in self.traj_colors:
                self.traj_colors[worm_index] = QColor(*np.random.randint(50, 230, 3))
            col = self.traj_colors[worm_index]
            
            pen = QPen()
            pen.setWidth(self.penwidth)
            pen.setColor(col)
            painter.setPen(pen)
            painter.drawPolyline(pol)

    def draw_boxes(self, painter, row_data):
        '''
        Draw traj worm trajectory.
        '''
        worm_index = int(row_data[self.worm_index_type])
        x = int(round(row_data['coord_x']))
        y = int(round(row_data['coord_y']))
        
        label_color = self.wlabC[int(row_data[self.label_type])]
        if not self.ui.is_color_features.isChecked():
            label_color = self.wlabC[int(row_data[self.label_type])]
        else:
            skel_id = row_data['skeleton_id']
            label_color = self._h_assign_feat_color(skel_id)

        pen = QPen()
        pen.setColor(label_color)
        pen.setWidth(self.penwidth)
        painter.setPen(pen)
        painter.setFont(QFont('Decorative', self.fontsize))

        painter.drawText(x, y, str(worm_index))

        bb = row_data['roi_size']
        painter.drawRect(x - bb / 2, y - bb / 2, bb, bb)

class BlobLabeler():
    def __init__(self, ui):
        super().__init__(ui)
        self.wlab = WLAB
        self.label_type = 'worm_label'

        self.ui.pushButton_U.clicked.connect(
            partial(self._h_tag_worm, self.wlab['U']))
        self.ui.pushButton_W.clicked.connect(
            partial(self._h_tag_worm, self.wlab['WORM']))
        self.ui.pushButton_WS.clicked.connect(
            partial(self._h_tag_worm, self.wlab['WORMS']))
        self.ui.pushButton_B.clicked.connect(
            partial(self._h_tag_worm, self.wlab['BAD']))

        self.ui.pushButton_W.setShortcut(QKeySequence(Qt.Key_W))
        self.ui.pushButton_U.setShortcut(QKeySequence(Qt.Key_U))
        self.ui.pushButton_WS.setShortcut(QKeySequence(Qt.Key_C))
        self.ui.pushButton_B.setShortcut(QKeySequence(Qt.Key_B))
        

    def enable_label_buttons(self, value):
        self.ui.pushButton_U.setEnabled(value)
        self.ui.pushButton_W.setEnabled(value)
        self.ui.pushButton_WS.setEnabled(value)
        self.ui.pushButton_B.setEnabled(value)
            

    def _h_tag_worm(self, label_ind):
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

class TrajectoryEditor(TrackerViewerAuxGUI):
    def __init__(self, ui):
        super().__init__(ui) 
        self.ui.pushButton_join.clicked.connect(self.joinTraj)
        self.ui.pushButton_split.clicked.connect(self.splitTraj)

        #SHORTCUTS
        self.ui.pushButton_join.setShortcut(QKeySequence(Qt.Key_J))
        self.ui.pushButton_split.setShortcut(QKeySequence(Qt.Key_S))

    def enable_trajectories_buttons(self, value):
        self.ui.pushButton_join.setEnabled(value)
        self.ui.pushButton_split.setEnabled(value)

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

    def splitTraj(self):
        if self.worm_index_type != 'worm_index_manual' \
        or self.frame_data is None:
            return

        if self.ui.radioButton_ROI1.isChecked():
            worm_ind = self.worm_index_roi1 
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

class ROIManager(TrackerViewerAuxGUI):
    def __init__(self, ui):
        super().__init__(ui)

        self.worm_index_roi1 = 1
        self.worm_index_roi2 = 1

        self.ui.comboBox_ROI1.activated.connect(self.selectROI1)
        self.ui.comboBox_ROI2.activated.connect(self.selectROI2)
        self.ui.checkBox_ROI1.stateChanged.connect(
            partial(self.updateROIcanvasN, 1))
        self.ui.checkBox_ROI2.stateChanged.connect(
            partial(self.updateROIcanvasN, 2))

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

        self.ui.radioButton_ROI1.setShortcut(QKeySequence(Qt.Key_Up))
        self.ui.radioButton_ROI2.setShortcut(QKeySequence(Qt.Key_Down))

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

        if row_data.size == 0 or \
        np.isnan(row_data['coord_x']) or \
        np.isnan(row_data['coord_y']):
            # invalid data nothing to do here
            wormCanvas.clear()
            return

        worm_img, roi_corner = getWormROI(self.frame_img, 
                                        row_data['coord_x'], 
                                        row_data['coord_y'], 
                                        row_data['roi_size']
                                        )
        
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

class IntensityLabeler(TrackerViewerAuxGUI):
    def __init__(self, ui):
        super().__init__(ui)

        self.mean_intensity = None
        self.ui.intensity_label.setStyleSheet('') #avoid displaying color at the start of the programı

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

    def display_intensity(self):
        if self.mean_intensity is not None and self.frame_number < self.mean_intensity.size:
            d = int(self.mean_intensity[self.frame_number]*255)
            self.ui.intensity_label.setStyleSheet('QLabel {background-color: rgb(%i, %i, %i);}' % (0, 0, d))

class MWTrackerViewer_GUI(MarkersDrawer, ContourDrawer, BlobLabeler, IntensityLabeler, ROIManager, TrajectoryEditor):

    def __init__(self, ui='', argv=''):
        if not ui:
            super().__init__(Ui_MWTrackerViewer())
        else:
            super().__init__(ui)
        
        self.setWindowTitle("Multi-Worm Viewer")

        self.vfilename = '' if len(argv) <= 1 else argv[1]        
        self.videos_dir = r"/Volumes/behavgenom$/GeckoVideo/MaskedVideos/"
        self.results_dir = ''
        self.skeletons_file = ''
        self.worm_index_type = 'worm_index_manual'
        self.frame_data = None
        
        
        self.ui.comboBox_labelType.currentIndexChanged.connect(
            self.selectWormIndexType)

        self.ui.pushButton_save.clicked.connect(self.saveData)
                        
        # select worm ROI when doubleclick a worm
        self.mainImage._canvas.mouseDoubleClickEvent = self.selectWorm
        
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
        self.updateImage()

    def updateSkelFile(self, skeletons_file):
        super().updateSkelFile(skeletons_file)
        
        #correct the index in case it was given before as worm_index_N
        if 'worm_index_N' in self.trajectories_data:
            self.trajectories_data = self.trajectories_data.rename(
                columns={'worm_index_N': 'worm_index_manual'})

        if not 'worm_index_manual' in self.trajectories_data:
            self.trajectories_data['worm_label'] = self.wlab['U']
            self.trajectories_data['worm_index_manual'] = self.trajectories_data['worm_index_joined']
            

        if not 'has_skeleton' in self.trajectories_data:
            self.trajectories_data['has_skeleton'] = self.trajectories_data['skeleton_id'] >= 0
        
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
        self.traj_worm_index_grouped = self.trajectories_data.groupby(self.worm_index_type)

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
            self.enable_trajectories_buttons(True)
            self.enable_label_buttons(True)
        else:
            self.label_type = 'auto_label'
            self.enable_trajectories_buttons(False)
            self.enable_label_buttons(False)

        #recalculate the grouped indexes
        self.traj_worm_index_grouped = self.trajectories_data.groupby(self.worm_index_type)

        self.updateImage()


    # update image
    def updateImage(self):
        if self.image_group is None:
            return

        super(TrackerViewerAuxGUI, self).readCurrentFrame()

        # read the data of the particles that exists in the frame
        self.frame_data = self.getFrameData(self.frame_number)
            
        #draw extra info only if the worm_index_type is valid
        if self.frame_data is not None and \
        self.worm_index_type in self.frame_data:
            #filter any -1 index
            self.frame_data = self.frame_data[self.frame_data[self.worm_index_type]>=0]
            if self.frame_data.size > 0:
                self.draw_worm_markers(self.frame_qimg)
                self.draw_food_contour(self.frame_qimg)

                self.updateROIcanvasN(1)
                self.updateROIcanvasN(2)

        else:
            self.ui.wormCanvas1.clear() 
            self.ui.wormCanvas2.clear()        
        
        # create the pixmap for the label
        self.mainImage.setPixmap(self.frame_qimg)

        self.display_intensity()

    def selectWorm(self, event):

        x = event.pos().x()
        y = event.pos().y()

        if self.frame_data is None or self.frame_data.size == 0:
            return

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

    

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    ui = MWTrackerViewer_GUI(argv=sys.argv)
    ui.show()
    sys.exit(app.exec_())
