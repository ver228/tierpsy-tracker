import json

import numpy as np
import tables
import os
import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QApplication
from tierpsy.gui.SWTrackerViewer_ui import Ui_SWTrackerViewer
from tierpsy.gui.TrackerViewerAux import TrackerViewerAuxGUI
from tierpsy.analysis.int_ske_orient.correctHeadTailIntensity import createBlocks, _fuseOverlapingGroups


class EggWriter():
    def __init__(self):
        self.fname = os.path.expanduser(os.path.join('~', 'Desktop', 'egg_events_raw.txt'))

    def add(self, vfilename, frame_number):
        if vfilename is not None:
            base_name = os.path.splitext(os.path.basename(vfilename))[0]
            line = '\n{}\t{}'.format(base_name, frame_number)
            with open(self.fname, 'a+') as fid:
                fid.write(line)

    def tag_bad(self):
        with open(self.fname, 'a+') as fid:
            fid.write('X')

    def export(self):
        if not os.path.exists(self.fname):
            return

        tab = pd.read_table(self.fname, header=None)
        tab.columns = ['base_name', 'frame_number'] 
        tab_g = tab.groupby('base_name')

        fexport= os.path.expanduser(os.path.join('~', 'Desktop', 'egg_events.tsv'))
        with open(fexport, 'w') as fid:
            for base_name, dat in tab_g:
                frame_numbers = []
                for f in dat['frame_number'].values:
                    try:
                        frame_numbers.append(int(f))
                    except:
                        pass
                if frame_numbers:
                    frame_numbers = sorted(set(frame_numbers))
                    line = '\t'.join([base_name] + list(map(str, frame_numbers))) + '\n'
                    fid.write(line)



           


class SWTrackerViewer_GUI(TrackerViewerAuxGUI):

    def __init__(self, ui='', mask_file=''):
        if not ui:
            super().__init__(Ui_SWTrackerViewer())
        else:
            super().__init__(ui)
        self.setWindowTitle("Single Worm Viewer")

        self.skel_block = []
        self.skel_block_n = 0
        self.is_stage_move = []

        self.ui.spinBox_skelBlock.valueChanged.connect(self.changeSkelBlock)
        self.ui.checkBox_showLabel.stateChanged.connect(self.updateImage)
        

        if mask_file:
            self.vfilename = mask_file
            self.updateVideoFile()

        self.egg_writer = EggWriter()
    
    # change frame number using the keys
    def keyPressEvent(self, event):
        # go the previous block
        if event.key() == Qt.Key_BracketLeft:
            self.ui.spinBox_skelBlock.setValue(self.skel_block_n - 1)

        # go to the next block
        elif event.key() == Qt.Key_BracketRight:
            self.ui.spinBox_skelBlock.setValue(self.skel_block_n + 1)

        elif event.key() == Qt.Key_Semicolon:
            if self.ui.checkBox_showLabel.isChecked():
                self.ui.checkBox_showLabel.setChecked(0)
            else:
                self.ui.checkBox_showLabel.setChecked(1)
        elif event.key() == Qt.Key_E:
            self.egg_writer.add(self.vfilename, self.frame_number)
        elif event.key() == Qt.Key_X:
            self.egg_writer.tag_bad()
        super().keyPressEvent(event)

    def updateSkelFile(self, skel_file):
        super().updateSkelFile(skel_file)
        try:

            with tables.File(self.skeletons_file, 'r') as fid:
                if '/provenance_tracking/int_ske_orient' in fid:
                    prov_str = fid.get_node(
                        '/provenance_tracking/int_ske_orient').read()
                    func_arg_str = json.loads(
                        prov_str.decode("utf-8"))['func_arguments']
                    gap_size = json.loads(func_arg_str)['gap_size']

                    good = (self.trajectories_data['int_map_id'] > 0).values
                    has_skel_group = createBlocks(good, min_block_size=0)
                    if len(has_skel_group) > 0:
                        self.skel_block = _fuseOverlapingGroups(
                            has_skel_group, gap_size=gap_size)

                if '/stage_movement/stage_vec' in fid:
                    self.is_stage_move = np.isnan(
                        fid.get_node('/stage_movement/stage_vec')[:, 0])
                else:
                    self.is_stage_move = []

        except IOError:
            self.skel_block = []

        self.ui.spinBox_skelBlock.setMaximum(max(len(self.skel_block) - 1, 0))
        self.ui.spinBox_skelBlock.setMinimum(0)

        if self.skel_block_n != 0:
            self.skel_block_n = 0
            self.ui.spinBox_skelBlock.setValue(0)
        else:
            self.changeSkelBlock(0)

    def drawSkelSingleWorm(self):
        frame_data = self.getFrameData(self.frame_number)
        if frame_data is None:
            return

        row_data = frame_data.squeeze()
        
        #for this viewer there must be only one particle per frame
        if len(row_data) == 0: 
            return

        isDrawSkel = self.ui.checkBox_showLabel.isChecked()
        self.frame_qimg = self.drawSkelResult(self.frame_img,
                    self.frame_qimg,
                    row_data, isDrawSkel)

        return self.frame_qimg

    def updateImage(self):
        self.readCurrentFrame()
        self.drawSkelSingleWorm()

        #draw stage movement if necessary
        if len(self.is_stage_move) > 0 and self.is_stage_move[self.frame_number]:
            self.frame_qimg = self._drawRect(self.frame_qimg)

        self.mainImage.setPixmap(self.frame_qimg)

    def _drawRect(self, qimg):
        painter = QPainter()
        painter.begin(qimg)
        pen = QPen()
        pen_width = 3
        pen.setWidth(pen_width)
        pen.setColor(Qt.red)
        painter.setPen(pen)

        dw = qimg.width() - pen_width
        dh = qimg.height() - pen_width
        painter.drawRect(
            1,
            1,
            dw,
            dh)
        painter.end()
        return qimg

    def changeSkelBlock(self, val):

        self.skel_block_n = val

        if len(self.skel_block) > 0:
            self.ui.label_skelBlock.setText(
                'Block limits: %i-%i' %
                (self.skel_block[
                    self.skel_block_n]))
            # move to the frame where the block starts
            self.ui.spinBox_frame.setValue(
                self.skel_block[self.skel_block_n][0])
        else:
            self.ui.label_skelBlock.setText('')

    

    def closeEvent(self, event):
        self.egg_writer.export()
        super().closeEvent(event)

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    ui = SWTrackerViewer_GUI()
    ui.show()
    sys.exit(app.exec_())
