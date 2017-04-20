import os
import sys


from tierpsy.gui.HDF5VideoPlayer import HDF5VideoPlayerGUI
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)

    ui = HDF5VideoPlayerGUI()
    ui.show()

    sys.exit(app.exec_())
