import sys
sys.path.append('..')
from MWTracker.GUI.getMaskParams.getMaskParams_GUI import getMaskParams_GUI
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
	app = QApplication(sys.argv)
	
	ui = getMaskParams_GUI()
	ui.show()
	app.exec_()
	#sys.exit()

