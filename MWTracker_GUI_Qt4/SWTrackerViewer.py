import os, sys


from MWTracker.GUI_Qt4.SWTrackerViewer.SWTrackerViewer_GUI import SWTrackerViewer_GUI
from PyQt4.QtGui import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    ui = SWTrackerViewer_GUI()
    ui.show()
    
    sys.exit(app.exec_())