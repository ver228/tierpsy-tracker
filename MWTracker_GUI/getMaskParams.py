import os
import sys

from MWTracker.GUI_Qt5.GetMaskParams import GetMaskParams_GUI
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)

    ui = GetMaskParams_GUI()
    ui.show()

    sys.exit(app.exec_())
