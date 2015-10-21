import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + os.sep + '..')
from MWTracker.GUI.getMaskParams.getMaskParams_GUI import getMaskParams_GUI
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
	app = QApplication(sys.argv)
	
	videos_dir = '/Volumes/behavgenom$/Camille/Worm_Videos/Camille_12102015_1530_1730/'
	
	ui = getMaskParams_GUI(videos_dir=videos_dir)
	ui.show()
	app.exec_()
	#sys.exit()

