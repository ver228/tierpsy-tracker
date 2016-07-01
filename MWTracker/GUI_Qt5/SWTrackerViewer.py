from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt

from MWTracker.GUI_Qt5.SWTrackerViewer_ui import Ui_SWTrackerViewer
from MWTracker.GUI_Qt5.TrackerViewerAux import TrackerViewerAux_GUI
from MWTracker.trackWorms.getSkeletonsTables import getWormMask, binaryMask2Contour
from MWTracker.intensityAnalysis.correctHeadTailIntensity import createBlocks, _fuseOverlapingGroups

import sys
import tables, os
import numpy as np
import pandas as pd
import cv2
import json

class SWTrackerViewer_GUI(TrackerViewerAux_GUI):
    def __init__(self, ui = '', mask_file = ''):
        if not ui:
            super().__init__(Ui_SWTrackerViewer())
        else:
            super().__init__(ui)


        self.skel_block = []
        self.skel_block_n = 0
        self.is_stage_move = []

        self.ui.spinBox_skelBlock.valueChanged.connect(self.changeSkelBlock)

        if mask_file:
            self.vfilename = mask_file
            self.updateVideoFile()

    def updateSkelFile(self):
        super().updateSkelFile()
        try:
            
            with tables.File(self.skeletons_file, 'r') as fid:
                if '/provenance_tracking/INT_SKE_ORIENT' in fid:
                    prov_str = fid.get_node('/provenance_tracking/INT_SKE_ORIENT').read()
                    func_arg_str = json.loads(prov_str.decode("utf-8"))['func_arguments']
                    gap_size = json.loads(func_arg_str)['gap_size']

                    good = (self.trajectories_data['int_map_id']>0).values          
                    has_skel_group = createBlocks(good, min_block_size = 0)
                    if len(has_skel_group)>0:
                        self.skel_block = _fuseOverlapingGroups(has_skel_group, gap_size = gap_size)

                if '/stage_movement/stage_vec' in fid:
                    self.is_stage_move = np.isnan(fid.get_node('/stage_movement/stage_vec')[:,0])
                else:
                    self.is_stage_move = []
       
        except IOError:
            self.skel_block = []

        self.ui.spinBox_skelBlock.setMaximum(max(len(self.skel_block)-1,0))
        self.ui.spinBox_skelBlock.setMinimum(0)
        
        if self.skel_block_n != 0:
            self.skel_block_n = 0
            self.ui.spinBox_skelBlock.setValue(0)
        else:
            self.changeSkelBlock(0)
            

    def updateImage(self):
        self.readCurrentFrame()
        self.drawSkelResult()

        if len(self.is_stage_move) > 0 and self.is_stage_move[self.frame_number]:
            painter = QPainter()
            painter.begin(self.frame_qimg)
            pen = QPen()
            pen_width = 3
            pen.setWidth(pen_width)
            pen.setColor(Qt.red)
            painter.setPen(pen)

            painter.drawRect(1, 1, self.frame_qimg.width()-pen_width, self.frame_qimg.height()-pen_width);
            painter.end()
        
        self.mainImage.setPixmap(self.frame_qimg)

    def changeSkelBlock(self, val):
        
        self.skel_block_n = val

        if len(self.skel_block) > 0:
            self.ui.label_skelBlock.setText('Block limits: %i-%i' % (self.skel_block[self.skel_block_n]))
            #move to the frame where the block starts
            self.ui.spinBox_frame.setValue(self.skel_block[self.skel_block_n][0])
        else:
            self.ui.label_skelBlock.setText('')


    #change frame number using the keys
    def keyPressEvent(self, event):
        #go the previous block
        if event.key() == Qt.Key_BracketLeft:
            self.ui.spinBox_skelBlock.setValue(self.skel_block_n-1)
        
        #go to the next block
        elif event.key() == Qt.Key_BracketRight:
            self.ui.spinBox_skelBlock.setValue(self.skel_block_n+1)

        elif event.key() == Qt.Key_Semicolon:
            if self.ui.checkBox_showLabel.isChecked():
                self.ui.checkBox_showLabel.setChecked(0)
            else:
                self.ui.checkBox_showLabel.setChecked(1)

        super().keyPressEvent(event)
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    ui = SWTrackerViewer_GUI()
    ui.show()

    sys.exit(app.exec_())

