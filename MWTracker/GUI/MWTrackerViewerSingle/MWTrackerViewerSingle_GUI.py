import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QFrame
from PyQt5.QtCore import QDir, QTimer, Qt, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QPolygonF, QPen

from MWTracker.GUI.MWTrackerViewerSingle.MWTrackerViewerSingle_ui import Ui_ImageViewer
from MWTracker.GUI.HDF5videoViewer.HDF5videoViewer_GUI import HDF5videoViewer
from MWTracker.trackWorms.getSkeletonsTables import getWormMask, binaryMask2Contour

import tables, os
import numpy as np
import pandas as pd
import cv2

class MWTrackerViewerSingle(HDF5videoViewer):
    def __init__(self, ui = ''):
        if not ui:
            super().__init__(Ui_ImageViewer())
        else:
            super().__init__(ui)

        self.results_dir = ''
        self.skel_file = ''

        self.trajectories_data = -1
        self.traj_time_grouped = -1

        self.ui.pushButton_skel.clicked.connect(self.getSkelFile)
        self.ui.checkBox_showLabel.stateChanged.connect(self.updateImage)
        

    def getSkelFile(self):
        self.skel_file, _ = QFileDialog.getOpenFileName(self, 'Select file with the worm skeletons', 
            self.results_dir,  "Skeletons files (*_skeletons.hdf5);; All files (*)")
        
        self.ui.lineEdit_skel.setText(self.skel_file)
        if self.fid != -1:
            self.updateSkelFile()

    def updateSkelFile(self):
        if not self.skel_file or self.fid == -1:
            self.trajectories_data = -1
            self.traj_time_grouped = -1
            self.skel_dat = {}
        
        try:
            with pd.HDFStore(self.skel_file, 'r') as ske_file_id:
                self.trajectories_data = ske_file_id['/trajectories_data']
                self.traj_time_grouped = self.trajectories_data.groupby('frame_number')

        except (IOError, KeyError):
            self.trajectories_data = -1
            self.traj_time_grouped = -1
            self.skel_dat = {}

        self.updateImage()

    def updateVideoFile(self):
        super().updateVideoFile()
        dum = self.videos_dir.replace('MaskedVideos', 'Results')
        if os.path.exists(dum):
            self.results_dir = dum
            self.basename = self.vfilename.rpartition(os.sep)[-1].rpartition('.')[0]
            self.skel_file = self.results_dir + os.sep + self.basename + '_skeletons.hdf5'
            if not os.path.exists(self.skel_file):
                self.skel_file = ''
            self.ui.lineEdit_skel.setText(self.skel_file)

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

        row_data = self.getRowData()

        isDrawSkel = self.ui.checkBox_showLabel.isChecked()
        if isDrawSkel and isinstance(row_data, pd.Series):
            if row_data['has_skeleton'] == 1:
                self.drawSkel(self.frame_img, self.frame_qimg, row_data, roi_corner = (0,0))
            elif row_data['has_skeleton'] == 0:
                self.drawThreshMask(self.frame_img, self.frame_qimg, row_data, read_center=True)
            
        self.pixmap = QPixmap.fromImage(self.frame_qimg)
        self.ui.imageCanvas.setPixmap(self.pixmap);

    def drawSkel(self, worm_img, worm_qimg, row_data, roi_corner = (0,0)):
        
        c_ratio_y = worm_qimg.width()/worm_img.shape[1];
        c_ratio_x = worm_qimg.height()/worm_img.shape[0];
        
        skel_id = int(row_data['skeleton_id'])

        qPlg = {}

        with tables.File(self.skel_file, 'r') as ske_file_id:
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
        worm_mask = getWormMask(worm_img, row_data['threshold'])
        if read_center:
            worm_cnt, _ = binaryMask2Contour(worm_mask, roi_center_x = row_data['coord_x'], roi_center_y = row_data['coord_y'])
        else:
            worm_cnt, _ = binaryMask2Contour(worm_mask)
        worm_mask = np.zeros_like(worm_mask)
        cv2.drawContours(worm_mask, [worm_cnt.astype(np.int32)], 0, 1, -1)

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
    
    ui = MWTrackerViewerSingle()
    ui.show()
    
    sys.exit(app.exec_())

