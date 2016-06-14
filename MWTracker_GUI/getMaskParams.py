import os, sys

from MWTracker.GUI_Qt5.getMaskParams.getMaskParams_GUI import getMaskParams_GUI
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    ui = getMaskParams_GUI()
    ui.show()
    
    sys.exit(app.exec_())