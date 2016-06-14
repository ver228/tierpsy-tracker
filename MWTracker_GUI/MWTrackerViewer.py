import os, sys

from MWTracker.GUI_Qt5.MWTrackerViewer.MWTrackerViewer_GUI import MWTrackerViewer_GUI
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    ui = MWTrackerViewer_GUI()
    ui.show()
    
    sys.exit(app.exec_())