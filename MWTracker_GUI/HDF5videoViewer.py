import os, sys


from MWTracker.GUI_Qt4.HDF5videoViewer.HDF5videoViewer_GUI import HDF5videoViewer_GUI
from PyQt4.QtGui import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    ui = HDF5videoViewer_GUI()
    ui.show()
    
    sys.exit(app.exec_())