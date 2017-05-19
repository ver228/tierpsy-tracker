import sys
import os
import json
from collections import OrderedDict

from PyQt5.QtWidgets import QDialog, QApplication, QGridLayout, QLabel, \
    QSpinBox,  QDoubleSpinBox, QCheckBox, QPushButton, QLineEdit, QSizePolicy, \
    QMessageBox, QSpacerItem, QFileDialog, QComboBox
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot


from tierpsy.helper.params.tracker_param import TrackerParams, info_param, default_param, dflt_param_list, valid_options
from tierpsy.gui.HDF5VideoPlayer import LineEditDragDrop
from tierpsy import DFLT_FILTER_FILES


def save_params_json(json_file, param4file):

    # save data into the json file
    with open(json_file, 'w') as fid:
        json.dump(param4file, fid, indent=4)

class ParamWidget():
    def __init__(self, name, value=None, widget=None):
        self.name = name
        if widget is not None:
            self.widget = widget
        else:
            assert value is not None
            self.widget = self._create(name, value)

        if isinstance(self.widget, (QDoubleSpinBox, QSpinBox)):
            self.widget.setMinimum(int(-1e10))
            self.widget.setMaximum(int(1e10))

        elif isinstance(self.widget, QComboBox):
            if name in valid_options:
                self.widget.addItems(valid_options[name])
            elif name == 'filter_model_name':
                self.widget.addItems([''] + DFLT_FILTER_FILES)
                self.widget.setEditable(True)


        if value is not None:
            self.write(value)


    def _create(self, name, value):
        value_type = type(value)
        
        if name in valid_options or name == 'filter_model_name':
            widget = QComboBox()
        elif value_type is bool:
            widget = QCheckBox(name)

        elif value_type is int:
            widget = QSpinBox()

        elif value_type is float:
            widget = QDoubleSpinBox()

        elif value_type is str:
            widget = QLineEdit(value)
        
        elif value_type is list or value_type is tuple:
            widget = QGridLayout()

            for icol, val in enumerate(value):
                spinbox = QSpinBox() if value_type is int else QDoubleSpinBox()
                spinbox.setMinimum(int(-1e10))
                spinbox.setMaximum(int(1e10))
                spinbox.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
                widget.addWidget(spinbox, 1, icol, 1, 2)

        else:
            raise ValueError('unknown type {}'.format(value_type))
        return widget

    def read(self):
        if isinstance(self.widget, (QDoubleSpinBox, QSpinBox)):
            return self.widget.value()
        elif isinstance(self.widget, QCheckBox):
            return self.widget.isChecked()
        elif isinstance(self.widget, QLineEdit):
            return self.widget.text()
        elif isinstance(self.widget, QGridLayout):
            return [self.widget.itemAt(ii).widget().value() for ii in range(self.widget.count())]
        elif isinstance(self.widget, QComboBox):
            return self.widget.currentText()

        else:
            raise ValueError('unknown type {}'.format(type(self.widget)))

    def write(self, value):
        if isinstance(self.widget, (QDoubleSpinBox, QSpinBox)):
            self.widget.setValue(value)
        elif isinstance(self.widget, QCheckBox):
            self.widget.setChecked(value)
        elif isinstance(self.widget, QLineEdit):
            self.widget.setText(value)
        elif isinstance(self.widget, QGridLayout):
            for ii, val in enumerate(value):
                self.widget.itemAt(ii).widget().setValue(val)
        elif isinstance(self.widget, QComboBox):
            index = self.widget.findText(value)
            self.widget.setCurrentIndex(index)
        else:
            raise ValueError('unknown type {}'.format(type(self.widget)))
        


class GetAllParameters(QDialog):
    file_saved = pyqtSignal(str)

    def __init__(self, param_file='', param_per_row=5):
        super(GetAllParameters, self).__init__()
        self.param_file = param_file
        self.param_per_row = param_per_row
        self.initUI()

        self.updateParamFile(param_file)

        self.pushbutton_save.clicked.connect(self.saveParamFile)
        self.pushbutton_file.clicked.connect(self.getParamFile)

    def closeEvent(self, event):
        currentparams = self._readParams()
        if self.lastreadparams != currentparams:
            reply = QMessageBox.question(
                self,
                'Message',
                '''You select new parameters. Do you want to save them?''',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No)
            if reply == QMessageBox.Yes:
                if not self.saveParamFile():
                    return
        super(GetAllParameters, self).closeEvent(event)

    def initUI(self):
        

        grid = QGridLayout()
        self.setLayout(grid)

        self.widgetlabels = {}
        for ii, (name, value, info) in enumerate(dflt_param_list):
            row = ii // self.param_per_row * 2
            col = (ii % self.param_per_row)
            

            w = ParamWidget(name, value=value)
            self.widgetlabels[name] = w

            if isinstance(w.widget, QCheckBox):
                grid.addWidget(w.widget, row, col, 2, 1)
                w.widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
            else:
                label = QLabel(name)
                label.setWhatsThis(info)
                grid.addWidget(label, row, col, 1, 1)
                label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
                if isinstance(w.widget, QGridLayout):
                    grid.addLayout(w.widget, row+1, col, 1, 1)
                else:
                    grid.addWidget(w.widget, row+1, col, 1, 1)
                    w.widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        #assert all(x for x in self.widgetlabels)
        
        spacer = QSpacerItem(
            40,
            20,
            QSizePolicy.Preferred,
            QSizePolicy.Preferred)

        self.pushbutton_save = QPushButton('Save')
        self.pushbutton_file = QPushButton('Select File')
        self.lineEdit_file = QLineEdit(self.param_file)

        last_row = len(dflt_param_list) // self.param_per_row * 2 + 3
        last_col = max(self.param_per_row - 1, 3)
        grid.addWidget(self.pushbutton_save, last_row, 0, 1, 1)
        grid.addWidget(self.pushbutton_file, last_row, last_col, 1, 1)
        grid.addWidget(self.lineEdit_file, last_row, 1, 1, last_col - 1)
        grid.addItem(spacer, last_row - 1, 0, 1, 1)

        LineEditDragDrop(
            self.lineEdit_file,
            self.updateParamFile,
            os.path.isfile)

        # used to find if anything was modified.
        self.lastreadparams = self._readParams()

        self.show()
        self.setAttribute(Qt.WA_DeleteOnClose)

    # file dialog to the the hdf5 file
    def getParamFile(self):
        json_file, _ = QFileDialog.getOpenFileName(
            self, "Find parameters file", '', "JSON files (*.json);; All (*)")
        if json_file:
            self.updateParamFile(json_file)

    def updateParamFile(self, json_file):
        # set the widgets with the default parameters, in case the parameters are not given
        # by the json file.
        if os.path.exists(json_file):
            try:
                params = TrackerParams(json_file)
                json_param = params.p_dict

            except (OSError, UnicodeDecodeError, json.decoder.JSONDecodeError):
                QMessageBox.critical(
                    self,
                    'Cannot read parameters file.',
                    "Cannot read parameters file. Try another file",
                    QMessageBox.Ok)
                return
        else:
            json_param = {}

        for param_name in json_param:
            if param_name not in self.widgetlabels:
                QMessageBox.critical(
                    self, '', "'%s' is not a valid variable. Please correct the parameters file" %
                    param_name, QMessageBox.Ok)
                return

        # Set the parameters in the correct widget. Any paramter not contained
        # in the json file will be keep with the default value.
        for name in self.widgetlabels:
            w = self.widgetlabels[name]
            if name in json_param:
                value = json_param[name]  
            else:
                value = default_param[name]

            w.write(value)
            
        self.lineEdit_file.setText(json_file)
        # used to find if anything was modified.
        self.lastreadparams = self._readParams()

    def _readParams(self):
        # read all the values in the GUI
        parameters = {}
        for name in self.widgetlabels:
            parameters[name] = self.widgetlabels[name].read()
        return parameters

    @pyqtSlot()
    def saveParamFile(self):
        json_file = self.lineEdit_file.text()
        if not json_file:
            QMessageBox.critical(
                self,
                'No parameter file name given.',
                'No parameter file name given. Please select name using the "Parameters File" button',
                QMessageBox.Ok)
            return 0

        if os.path.exists(json_file):

            reply = QMessageBox.question(
                self,
                'Message',
                '''The parameters file already exists. Do you want to overwrite it?''',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No)

            if reply == QMessageBox.No:
                return 0

        # read all the values in the GUI
        param4file = self._readParams()
        self.lastreadparams = param4file
        save_params_json(json_file, param4file)

        self.file_saved.emit(json_file)
        return 1


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GetAllParameters()
    sys.exit(app.exec_())
