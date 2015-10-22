import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + os.sep + '..')
from MWTracker.GUI.getMaskParams.getMaskParams_GUI import getMaskParams_GUI
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
	app = QApplication(sys.argv)
	
	default_videos_dir = '/Users/ajaver/Desktop/SingleWormData/Worm_Videos/03-03-11/'#'/Volumes/behavgenom$/Camille/Worm_Videos/Camille_12102015_1530_1730/'
	
	ui = getMaskParams_GUI(default_videos_dir = default_videos_dir, scripts_dir = os.path.dirname(os.path.realpath(__file__)))
	ui.show()
	app.exec_()
	#sys.exit()

