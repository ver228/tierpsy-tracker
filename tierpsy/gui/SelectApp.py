
from functools import partial
from collections import OrderedDict

from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt

import tierpsy
from tierpsy.gui.GetMaskParams import GetMaskParams_GUI
from tierpsy.gui.MWTrackerViewer import MWTrackerViewer_GUI
from tierpsy.gui.SWTrackerViewer import SWTrackerViewer_GUI
from tierpsy.gui.BatchProcessing import BatchProcessing_GUI
        

dd = [('get_params', (GetMaskParams_GUI,"Set Parameters")),
    ('batch_processing', (BatchProcessing_GUI,"Batch Processing Multiple Files")),
    ('mwtracker', (MWTrackerViewer_GUI, "Tierpsy Tracker Viewer")),
    ('swtracker', (SWTrackerViewer_GUI, "Worm Tracker 2.0 Viewer"))
    ]
widget_lists = OrderedDict(dd)


class SelectApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(304, 249)
        self.setWindowTitle('Tierpsy Tracker ' + tierpsy.__version__)

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
            fun_d = partial(self.appCall, name)
            self.buttons[name].clicked.connect(fun_d)

    def appCall(self, name):
        appFun, label = widget_lists[name]
        ui = appFun()
        ui.setWindowTitle(label)
        ui.show()
        ui.setAttribute(Qt.WA_DeleteOnClose)
        
        

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    ui = SelectApp()
    ui.show()
    sys.exit(app.exec_())
