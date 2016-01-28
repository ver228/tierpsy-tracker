import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + os.sep + '..')
from MWTracker.GUI.HDF5videoViewer.HDF5videoViewer_GUI import HDF5videoViewer_GUI

from PyQt5.QtWidgets import QApplication
#from PyQt4.QtGui import QApplication

if __name__ == '__main__':
	app = QApplication(sys.argv)
	
	ui = HDF5videoViewer_GUI()
	ui.show()
	app.exec_()
	#sys.exit()

