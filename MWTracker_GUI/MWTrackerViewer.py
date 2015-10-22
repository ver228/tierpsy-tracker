import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + os.sep + '..')

from MWTracker.GUI.MWTrackerViewer.MWTrackerViewer_GUI import MWTrackerViewer_GUI
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
	app = QApplication(sys.argv)
	
	ui = MWTrackerViewer_GUI()
	ui.show()
	app.exec_()
	#sys.exit()