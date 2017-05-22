
from functools import partial
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt
#from PyQt5 import QtCore, QtGui, QtWidgets

#from tierpsy.gui.SelectApp_ui import Ui_SelectApp
from tierpsy.gui.GetMaskParams import GetMaskParams_GUI
from tierpsy.gui.MWTrackerViewer import MWTrackerViewer_GUI
from tierpsy.gui.SWTrackerViewer import SWTrackerViewer_GUI
from tierpsy.gui.BatchProcessing import BatchProcessing_GUI

# class Ui_SelectApp(object):
#     def setupUi(self, SelectApp):
        
        

widget_lists = {
    'get_params':(GetMaskParams_GUI,"Set Parameters"),
    'batch_processing':(BatchProcessing_GUI,"Batch Processing Multiple Files"),
    'mwtracker':(MWTrackerViewer_GUI,"Multi-Worm Tracker Viewer"),
    'swtracker':(SWTrackerViewer_GUI,"Single-Worm Tracker Viewer")
}


class SelectApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(304, 249)
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)

        self.verticalLayout_C = QVBoxLayout(self.centralwidget)
        self.verticalLayout_C.setObjectName("verticalLayout_C")

        #this second layout makes the buttons closer
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_C.addLayout(self.verticalLayout)
        
        self.buttons = {}
        for name, (func_name, label) in widget_lists.items():
            self.buttons[name] = QPushButton(self.centralwidget)
            self.buttons[name].setObjectName(name)
            self.verticalLayout.addWidget(self.buttons[name])
            self.buttons[name].setText(label)
            fun_d = partial(self.appCall, func_name)
            self.buttons[name].clicked.connect(fun_d)

    def appCall(self, appFun):
        ui = appFun()
        ui.show()
        ui.setAttribute(Qt.WA_DeleteOnClose)

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    ui = SelectApp()
    ui.show()
    sys.exit(app.exec_())
