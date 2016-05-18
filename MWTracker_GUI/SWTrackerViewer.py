import os, sys


from MWTracker.GUI_Qt4.SWTrackerViewer.SWTrackerViewer_GUI import SWTrackerViewer_GUI
from PyQt4.QtGui import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    if len(sys.argv)>1:
    	mask_file = sys.argv[1]
    else:
    	mask_file =''
    ui = SWTrackerViewer_GUI(mask_file=mask_file)
    ui.show()
    
    sys.exit(app.exec_())