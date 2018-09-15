import tierpsy
from tierpsy.gui.GetMaskParams import GetMaskParams_GUI
from tierpsy.gui.MWTrackerViewer import MWTrackerViewer_GUI
from tierpsy.gui.Summarizer import Summarizer_GUI
from tierpsy.gui.BatchProcessing import BatchProcessing_GUI

from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt

import os
import sys
import stat
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
        
 
def _create_desktop_command():
    #currently only works for OSX...
    if sys.platform == 'darwin':
        if 'CONDA_DEFAULT_ENV' in os.environ:
            act_str = 'source activate ' + os.environ['CONDA_DEFAULT_ENV']
            source_cmd = os.path.join(os.environ['CONDA_PREFIX'], 'bin', 'tierpsy_gui')
        else:
            act_str = ''
            source_cmd = os.path.join(os.path.dirname(__file__), 'tierpsy_gui')

        if not os.path.exists(source_cmd):
            script_name = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts' , 'tierpsy_gui.py')
            script_name = os.path.realpath(script_name)
            source_cmd = '{} {}'.format(sys.executable, script_name)
        
        cmd = '\n'.join([act_str, source_cmd])
    
        link_file = os.path.join(os.path.expanduser('~'), 'Desktop', 'tierpsy_gui.command')
        with open(link_file, 'w') as fid:
            fid.write(cmd)

        
        st = os.stat(link_file)
        os.chmod(link_file, st.st_mode | stat.S_IEXEC)


def tierpsy_gui():
    _create_desktop_command()
    app = QApplication(sys.argv)

    ui = SelectApp()
    ui.show()

    app.exec_()



if __name__ == '__main__':
   tierpsy_gui()