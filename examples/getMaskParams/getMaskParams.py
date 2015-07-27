import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QDir, QTimer, Qt, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QPolygonF, QPen
from getMaskParams_ui import Ui_MainWindow

import h5py
import os
import pandas as pd
import numpy as np
import sys



class getMaskParams(QMainWindow):
	def __init__(self):
		super().__init__()
		
		# Set up the user interface from Designer.
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)

		self.ui.dial_min_area.valueChanged.connect(self.ui.spinBox_min_area.setValue)
		self.ui.dial_max_area.valueChanged.connect(self.ui.spinBox_max_area.setValue)
		self.ui.dial_block_size.valueChanged.connect(self.ui.spinBox_block_size.setValue)
		self.ui.dial_thresh_C.valueChanged.connect(self.ui.spinBox_thresh_C.setValue)

if __name__ == '__main__':
	app = QApplication(sys.argv)
	
	ui = getMaskParams()
	ui.show()
	
	sys.exit(app.exec_())