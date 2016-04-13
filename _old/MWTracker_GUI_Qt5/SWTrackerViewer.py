import os, sys


from MWTracker.GUI.SWTrackerViewer.SWTrackerViewer_GUI import SWTrackerViewer_GUI
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    ui = SWTrackerViewer_GUI()
    ui.show()
    
    sys.exit(app.exec_())