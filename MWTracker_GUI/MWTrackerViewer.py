import sys
sys.path.append('..')
from MWTracker.GUI.MWTrackerViewer.MWTrackerViewer_GUI import MWTrackerViewer_GUI
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
	app = QApplication(sys.argv)
	
	ui = MWTrackerViewer_GUI()
	ui.show()
	app.exec_()
	#sys.exit()

