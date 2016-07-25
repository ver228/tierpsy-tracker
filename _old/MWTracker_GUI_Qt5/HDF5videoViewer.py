import os
import sys


from MWTracker.GUI.HDF5videoViewer.HDF5videoViewer_GUI import HDF5videoViewer_GUI
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)

    ui = HDF5videoViewer_GUI()
    ui.show()

    sys.exit(app.exec_())
