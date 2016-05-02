import sys
from PyQt4.QtGui import QApplication, QMainWindow, QFileDialog, QMessageBox, QFrame
from PyQt4.QtCore import QDir, QTimer, Qt, QPointF
from PyQt4.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QPolygonF, QPen

from MWTracker.GUI_Qt4.MWTrackerViewerSingle.MWTrackerViewerSingle_ui import Ui_ImageViewer
from MWTracker.GUI_Qt4.HDF5videoViewer.HDF5videoViewer_GUI import HDF5videoViewer_GUI
from MWTracker.trackWorms.getSkeletonsTables import getWormMask, binaryMask2Contour

import tables, os
import numpy as np
import pandas as pd
import cv2
import json

class MWTrackerViewerSingle_GUI(HDF5videoViewer_GUI):
    def __init__(self, ui = ''):
        if not ui:
            super().__init__(Ui_ImageViewer())
        else:
            super().__init__(ui)

        self.results_dir = ''
        self.skeletons_file = ''

        self.trajectories_data = -1
        self.traj_time_grouped = -1

        self.ui.pushButton_skel.clicked.connect(self.getSkelFile)
        self.ui.checkBox_showLabel.stateChanged.connect(self.updateImage)
        

    def getSkelFile(self):
        selected_file = QFileDialog.getOpenFileName(self, 'Select file with the worm skeletons', 
            self.results_dir,  "Skeletons files (*_skeletons.hdf5);; All files (*)")
        
        if not os.path.exists(selected_file):
            return

        self.skeletons_file = selected_file
        self.ui.lineEdit_skel.setText(self.skeletons_file)
        if self.fid != -1:
            self.updateSkelFile()

    def updateSkelFile(self):
        if not self.skeletons_file or self.fid == -1:
            self.trajectories_data = -1
            self.traj_time_grouped = -1
            self.skel_dat = {}
        
        try:
            with pd.HDFStore(self.skeletons_file, 'r') as ske_file_id:
                self.trajectories_data = ske_file_id['/trajectories_data']
                self.traj_time_grouped = self.trajectories_data.groupby('frame_number')

                #read the size of the structural element used in to calculate the mask
                if '/provenance_tracking/INT_SKE_ORIENT' in ske_file_id:
                    prov_str = ske_file_id.get_node('/provenance_tracking/SKE_CREATE').read()
                    func_arg_str = json.loads(prov_str.decode("utf-8"))['func_arguments']
                    strel_size = json.loads(func_arg_str)['strel_size']
                    if isinstance(strel_size, (list, tuple)):
                        strel_size = strel_size[0]

                    self.strel_size = strel_size
                else:
                    #use default
                    self.strel_size = 5


        except (IOError, KeyError):
            self.trajectories_data = -1
            self.traj_time_grouped = -1
            self.skel_dat = {}
        
        if self.frame_number == 0:
            self.updateImage()
        else:
            self.ui.spinBox_frame.setValue(0)
        

    def updateVideoFile(self):
        super().updateVideoFile()
        dum = self.videos_dir.replace('MaskedVideos', 'Results')
        if os.path.exists(dum):
            self.results_dir = dum
            self.basename = self.vfilename.rpartition(os.sep)[-1].rpartition('.')[0]
            self.skeletons_file = self.results_dir + os.sep + self.basename + '_skeletons.hdf5'
            if not os.path.exists(self.skeletons_file):
                self.skeletons_file = ''
            self.ui.lineEdit_skel.setText(self.skeletons_file)

        self.updateSkelFile()

    def getRowData(self):
        if not isinstance(self.traj_time_grouped, pd.core.groupby.DataFrameGroupBy):
            return -1
        try:
            row_data = self.traj_time_grouped.get_group(self.frame_number)
            assert len(row_data) > 0
            return row_data.squeeze()
        
        except KeyError:
            return -1

    #function that generalized the updating of the ROI
    def updateImage(self):
        self.readImage()
        self.drawSkelResult()
        self.pixmap = QPixmap.fromImage(self.frame_qimg)
        self.ui.imageCanvas.setPixmap(self.pixmap);

    def drawSkelResult(self):
        row_data = self.getRowData()
        isDrawSkel = self.ui.checkBox_showLabel.isChecked()
        if isDrawSkel and isinstance(row_data, pd.Series):
            if row_data['has_skeleton'] == 1:
                self.drawSkel(self.frame_img, self.frame_qimg, row_data, roi_corner = (0,0))
            else:
                self.drawThreshMask(self.frame_img, self.frame_qimg, row_data, read_center=True)
        


    def drawSkel(self, worm_img, worm_qimg, row_data, roi_corner = (0,0)):
        if not self.skeletons_file or not isinstance(self.trajectories_data, pd.DataFrame):
            return

        c_ratio_y = worm_qimg.width()/worm_img.shape[1];
        c_ratio_x = worm_qimg.height()/worm_img.shape[0];
        
        skel_id = int(row_data['skeleton_id'])

        qPlg = {}

        with tables.File(self.skeletons_file, 'r') as ske_file_id:
            for tt in ['skeleton', 'contour_side1', 'contour_side2']:
                dat = ske_file_id.get_node('/' + tt)[skel_id];
                dat[:,0] = (dat[:,0]-roi_corner[0])*c_ratio_x
                dat[:,1] = (dat[:,1]-roi_corner[1])*c_ratio_y
            
                qPlg[tt] = QPolygonF()
                for p in dat:
                    qPlg[tt].append(QPointF(*p))
        
        if 'is_good_skel' in row_data and row_data['is_good_skel'] == 0:
            self.skel_colors = {'skeleton':(102, 0, 0 ), 
            'contour_side1':(102, 0, 0 ), 'contour_side2':(102, 0, 0 )}
        else:
            self.skel_colors = {'skeleton':(27, 158, 119 ), 
            'contour_side1':(217, 95, 2), 'contour_side2':(231, 41, 138)}

        pen = QPen()
        pen.setWidth(2)
        
        painter = QPainter()
        painter.begin(worm_qimg)
    
        for tt, color in self.skel_colors.items():
            pen.setColor(QColor(*color))
            painter.setPen(pen)
            painter.drawPolyline(qPlg[tt])
        
        pen.setColor(Qt.black)
        painter.setBrush(Qt.white)
        painter.setPen(pen)
    
        radius = 3
        painter.drawEllipse(qPlg['skeleton'][0], radius, radius)

        painter.end()
     
    def drawThreshMask(self, worm_img, worm_qimg, row_data, read_center = True):
        
        min_mask_area = row_data['area']/2
        c1, c2 = (row_data['coord_x'], row_data['coord_y']) if read_center else (-1, -1)

        worm_mask, _ , _ = getWormMask(worm_img, row_data['threshold'], strel_size = self.strel_size, \
            roi_center_x = c1, roi_center_y = c2, min_mask_area = min_mask_area)
        
        #worm_mask = np.zeros_like(worm_mask)
        #cv2.drawContours(worm_mask, [worm_cnt.astype(np.int32)], 0, 1, -1)

        worm_mask = QImage(worm_mask.data, worm_mask.shape[1], 
        worm_mask.shape[0], worm_mask.strides[0], QImage.Format_Indexed8)
        worm_mask = worm_mask.convertToFormat(QImage.Format_RGB32, Qt.AutoColor)
        worm_mask = worm_mask.scaled(worm_qimg.width(),worm_qimg.height(), Qt.KeepAspectRatio)
        worm_mask = QPixmap.fromImage(worm_mask)

        worm_mask = worm_mask.createMaskFromColor(Qt.black)
        p = QPainter(worm_qimg)
        p.setPen(QColor(0,204,102))
        p.drawPixmap(worm_qimg.rect(), worm_mask, worm_mask.rect())
        p.end()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    ui = MWTrackerViewerSingle_GUI()
    ui.show()
    
    sys.exit(app.exec_())

