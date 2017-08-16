import json

import numpy as np
import tables
import os
import pandas as pd
from functools import partial

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QApplication
from tierpsy.gui.SWTrackerViewer_ui import Ui_SWTrackerViewer
from tierpsy.gui.TrackerViewerAux import TrackerViewerAuxGUI
from tierpsy.analysis.int_ske_orient.correctHeadTailIntensity import createBlocks, _fuseOverlapingGroups
from tierpsy.helper.params import read_microns_per_pixel

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
        self.is_feat_file = False

        self.microns_per_pixels = None
        self.stage_position_pix = None

        self.ui.spinBox_skelBlock.valueChanged.connect(self.changeSkelBlock)
        self.ui.checkBox_showLabel.stateChanged.connect(self.updateImage)

        if mask_file:
            self.vfilename = mask_file
            self.updateVideoFile()

        self.egg_writer = EggWriter()
    
    def updateVideoFile(self, vfilename):
        super().updateVideoFile(vfilename, possible_ext = ['_features.hdf5', '_skeletons.hdf5'])
        self.updateImage()

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

    def updateSkelFile(self, skel_file, dflt_skel_size = 10):
        super().updateSkelFile(skel_file)
        
        self.ui.spinBox_skelBlock.setMaximum(max(len(self.skel_block) - 1, 0))
        self.ui.spinBox_skelBlock.setMinimum(0)

        if self.skel_block_n != 0:
            self.skel_block_n = 0
            self.ui.spinBox_skelBlock.setValue(0)
        else:
            self.changeSkelBlock(0)


        self.skel_block = []
        self.is_stage_move = []
        self.stage_position_pix = None
        self.is_feat_file = False

        VALID_ERRORS = (IOError, KeyError, tables.exceptions.HDF5ExtError, tables.exceptions.NoSuchNodeError)
        #try to read the information from the features file if possible
        if not self.trajectories_data is None:
            try:
                
                with tables.File(self.skeletons_file, 'r') as fid:
                    self.stage_position_pix = fid.get_node('/stage_movement/stage_vec')[:]

                    #only used for skeletons, and to test the head/tail orientation. I leave it but probably should be removed for in the future
                    prov_str = fid.get_node('/provenance_tracking/INT_SKE_ORIENT').read()
                    func_arg_str = json.loads(
                        prov_str.decode("utf-8"))['func_arguments']
                    gap_size = json.loads(func_arg_str)['gap_size']

                    good = (self.trajectories_data['int_map_id'] > 0).values
                    has_skel_group = createBlocks(good, min_block_size=0)
                    if len(has_skel_group) > 0:
                        self.skel_block = _fuseOverlapingGroups(
                            has_skel_group, gap_size=gap_size)
                
            except VALID_ERRORS:
                pass
        else:
            try:

                #load skeletons from _features.hdf5 
                if '/stage_position_pix' in self.fid:
                    self.stage_position_pix = self.fid.get_node('/stage_position_pix')[:]
                else:
                    n_frames = self.fid.get_node('/mask').shape[0]
                    self.stage_position_pix = np.full((n_frames,2), np.nan)
                
                timestamp = self.fid.get_node('/timestamp/raw')[:]
                self.microns_per_pixel = read_microns_per_pixel(self.skeletons_file)
                
                with pd.HDFStore(self.skeletons_file, 'r') as ske_file_id:
                    #this could be better so I do not have to load everything into memory, but this is faster
                    self.trajectories_data = ske_file_id['/features_timeseries']
                    
                    if self.trajectories_data['worm_index'].unique().size !=1:
                        raise ValueError("There is more than one worm index. This file does not seem to have been analyzed with the WT2 option.")

                    good = self.trajectories_data['timestamp'].isin(timestamp)
                    self.trajectories_data = self.trajectories_data[good]
                    self.trajectories_data.sort_values(by='timestamp', inplace=True)
                    
                    if np.any(self.trajectories_data['timestamp'] < 0) or np.any(self.trajectories_data['timestamp'].isnull()):
                        raise ValueError('There are invalid values in the timestamp. I cannot get the stage movement information.')

                    first_frame = np.where(timestamp==self.trajectories_data['timestamp'].min())[0][0]
                    last_frame = np.where(timestamp==self.trajectories_data['timestamp'].max())[0][0]

                    self.trajectories_data['frame_number'] = np.arange(first_frame, last_frame+1, dtype=np.int)
                    self.trajectories_data['skeleton_id'] = self.trajectories_data.index
                    self.traj_time_grouped = self.trajectories_data.groupby('frame_number')

                self.is_feat_file = True

            except VALID_ERRORS:
                self.trajectories_data = None
                self.traj_time_grouped = None
                self.is_feat_file = False

            if self.stage_position_pix is not None:
                self.is_stage_move = np.isnan(self.stage_position_pix[:, 0])
        self.updateImage()

    def drawSkelSingleWorm(self):
        frame_data = self.getFrameData(self.frame_number)
        if frame_data is None:
            return

        row_data = frame_data.squeeze()
        
        #for this viewer there must be only one particle per frame
        if len(row_data) == 0: 
            return

        isDrawSkel = self.ui.checkBox_showLabel.isChecked()
        skel_id = int(row_data['skeleton_id'])

        if not isDrawSkel or skel_id < 0:
            return self.frame_qimg

        elif self.is_feat_file:
            #read skeletons from the features file
            with tables.File(self.skeletons_file, 'r') as ske_file_id:
                fields = {
                    'dorsal_contours':'contour_side1', 
                    'skeletons':'skeleton', 
                    'ventral_contours':'contour_side2'
                }

                skel_dat = {}
                for ff, tt in fields.items():
                    field = '/coordinates/' + ff
                    if field in ske_file_id:
                        dat = ske_file_id.get_node(field)[skel_id]
                        dat = dat/self.microns_per_pixel - self.stage_position_pix[self.frame_number]

                        print(dat)
                    else:
                        dat = np.full((1,2), np.nan)

                    skel_dat[tt] = dat
            super()._drawSkel(self.frame_qimg, skel_dat)

        else:
            self.frame_qimg = self.drawSkelResult(self.frame_img, self.frame_qimg, row_data, isDrawSkel)

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
