import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + os.sep + '..')
from MWTracker.GUI.getMaskParams.getMaskParams_GUI import getMaskParams_GUI
from PyQt5.QtWidgets import QApplication
#from PyQt4.QtGui import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # '/Volumes/behavgenom$/Camille/Worm_Videos/Camille_12102015_1530_1730/'
    default_videos_dir = '/Users/ajaver/Desktop/Videos/copied_from_pc207-13/Videos/'

    ui = getMaskParams_GUI(
        default_videos_dir=default_videos_dir,
        scripts_dir=os.path.dirname(
            os.path.realpath(__file__)))
    ui.show()
    app.exec_()
    # sys.exit()
