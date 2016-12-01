import os
import sys

from MWTracker.gui.SelectApp import SelectApp
from PyQt5.QtWidgets import QApplication

from MWTracker import AUX_FILES_DIR

if __name__ == '__main__':
    app = QApplication(sys.argv)

    ui = SelectApp()
    ui.show()

    sys.exit(app.exec_())
