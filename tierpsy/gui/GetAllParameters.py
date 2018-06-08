import sys
import json

from PyQt5.QtWidgets import QDialog, QApplication, QGridLayout, QLabel, \
    QSpinBox,  QDoubleSpinBox, QCheckBox, QPushButton, QLineEdit, QSizePolicy, \
    QMessageBox, QSpacerItem, QComboBox, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal


from tierpsy.helper.params.tracker_param import info_param, default_param, valid_options

# i cannot really add this since i do not have a good way to get the data
_feats2ignore = ['analysis_checkpoints']

def save_params_json(json_file, param4file):
    # save data into the json file
    with open(json_file, 'w') as fid:
        json.dump(param4file, fid, indent=4)

class ParamWidget():
    def __init__(self, name, value=None, widget=None, 
        info_param=info_param, valid_options=valid_options):
        self.name = name

        if widget is not None:
            self.widget = widget
        else:
            assert value is not None
            self.widget = self._create(name, value)

        if isinstance(self.widget, (QDoubleSpinBox, QSpinBox)):
            # In windows 7 it seems this value is int16 so keep it smaller than that
            self.widget.setMinimum(int(-1e9))
            self.widget.setMaximum(int(1e9))

        elif isinstance(self.widget, QComboBox):
            if name in valid_options:
                self.widget.addItems(valid_options[name])
            
        if not isinstance(self.widget, QGridLayout):
            self.widget.setToolTip(info_param[name])
        else:
            for n in range(self.widget.count()):
                self.widget.itemAt(n).widget().setToolTip(info_param[name])

        if value is not None:
            self.write(value)


    def _create(self, name, value):
        value_type = type(value)
        
        if name in valid_options:
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


class ParamWidgetMapper():
    '''
    Class used to read/write data in different inputs. 
    The corresponding widget must an element in the form p_(agument_name)

    '''
    def __init__(self, 
                central_widget, 
                default_param=default_param,
                info_param=info_param, 
                valid_options=valid_options
                ):
        self.params_widgets = {}
        self.default_param=default_param

        for attr_name in dir(central_widget):
            if attr_name.startswith('p_'):
                param_name = attr_name[2:]

                widget = getattr(central_widget, attr_name)
                w = ParamWidget(param_name, 
                                widget=widget, 
                                value=default_param[param_name],
                                info_param=info_param, 
                                valid_options=valid_options
                                )
                self.params_widgets[param_name] = w

    def __setitem__(self, param_name, value):
        assert param_name in self.params_widgets
        if value is None:
            return None
        else:
            self.params_widgets[param_name].write(value)

    def __getitem__(self, param_name):
        w = self.params_widgets[param_name]
        if w.widget.isEnabled():
            return w.read()
        else:
            return self.default_param[param_name]

    def __iter__(self):
        self.remaining_names = list(self.params_widgets.keys())
        return self

    def __next__(self):
        if len(self.remaining_names)==0:
            raise StopIteration

        return self.remaining_names.pop(0)        

class GetAllParameters(QDialog):
    file_saved = pyqtSignal(str)

    def __init__(self, parent_params = {}, param_per_row=5):
        super(GetAllParameters, self).__init__()
        self.param_per_row = param_per_row
        
        # could accept the gui mapper and try to change values in real time put it is a bit complicated 
        self.parent_params = parent_params


        self.initUI()
        self.updateAllParams()

        self.pushbutton_OK.pressed.connect(self.saveAndClose)
        self.pushbutton_cancel.pressed.connect(self.close)

        
    def initUI(self):
        
        grid = QGridLayout()
        
        self.widgetlabels = {}
        for ii, (name, value) in enumerate(default_param.items()):
            row = ii // self.param_per_row * 2
            col = (ii % self.param_per_row)
            

            w = ParamWidget(name, value=value)
            self.widgetlabels[name] = w

            if isinstance(w.widget, QCheckBox):
                grid.addWidget(w.widget, row, col, 2, 1)
                w.widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
            else:
                label = QLabel(name)
                grid.addWidget(label, row, col, 1, 1)
                label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
                if isinstance(w.widget, QGridLayout):
                    grid.addLayout(w.widget, row+1, col, 1, 1)
                else:
                    grid.addWidget(w.widget, row+1, col, 1, 1)
                    w.widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        
        spacer = QSpacerItem(
            40,
            20,
            QSizePolicy.Preferred,
            QSizePolicy.Preferred)


        #add buttons at the end
        self.pushbutton_cancel = QPushButton('Cancel')
        self.pushbutton_OK = QPushButton('OK')
        
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.pushbutton_OK)
        hbox.addWidget(self.pushbutton_cancel)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(grid)
        vbox.addLayout(hbox)
                
        self.setLayout(vbox)

        #set modal so the other window is blocked
        self.setModal(True)
        self.show()
        self.setAttribute(Qt.WA_DeleteOnClose)

    def updateAllParams(self):
        for param_name in self.parent_params:
            if param_name not in self.widgetlabels:
                QMessageBox.critical(
                    self, '', "'%s' is not a valid variable. Please correct the parameters file" %
                    param_name, QMessageBox.Ok)
                return

        # Set the parameters in the correct widget. Any paramter not contained
        # in the json file will be keep with the default value.
        for name in self.widgetlabels:
            if name in _feats2ignore:
                continue

            w = self.widgetlabels[name]
            if name in self.parent_params:
                value = self.parent_params[name]  
            else:
                value = default_param[name]

            w.write(value)
    
    def saveAndClose(self):
        for name in self.widgetlabels:
            if name in _feats2ignore:
                continue
            self.parent_params[name] = self.widgetlabels[name].read()

        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GetAllParameters()
    sys.exit(app.exec_())
