import tierpsy
from tierpsy.gui.GetMaskParams import GetMaskParams_GUI
from tierpsy.gui.MWTrackerViewer import MWTrackerViewer_GUI
from tierpsy.gui.Summarizer import Summarizer_GUI
from tierpsy.gui.BatchProcessing import BatchProcessing_GUI

from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt

import sys
from functools import partial
from collections import OrderedDict
        

dd = [('get_params', (GetMaskParams_GUI,"Set Parameters")),
    ('batch_processing', (BatchProcessing_GUI,"Batch Processing Multiple Files")),
    ('mwtracker', (MWTrackerViewer_GUI, "Tierpsy Tracker Viewer")),
    ('summarizer', (Summarizer_GUI, "Get Features Summary"))
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
        
 
def main():
    app = QApplication(sys.argv)

    ui = SelectApp()
    ui.show()

    app.exec_()



if __name__ == '__main__':
   main()