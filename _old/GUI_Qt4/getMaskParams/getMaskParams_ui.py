# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8

    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1161, 746)
        self.centralWidget = QtGui.QWidget(MainWindow)
        self.centralWidget.setObjectName(_fromUtf8("centralWidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralWidget)
        self.gridLayout.setMargin(11)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.verticalLayout_4 = QtGui.QVBoxLayout()
        self.verticalLayout_4.setMargin(11)
        self.verticalLayout_4.setSpacing(6)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.label_2 = QtGui.QLabel(self.centralWidget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.verticalLayout_4.addWidget(self.label_2)
        self.spinBox_thresh_C = QtGui.QSpinBox(self.centralWidget)
        self.spinBox_thresh_C.setMinimum(-100)
        self.spinBox_thresh_C.setMaximum(100)
        self.spinBox_thresh_C.setProperty("value", 15)
        self.spinBox_thresh_C.setObjectName(_fromUtf8("spinBox_thresh_C"))
        self.verticalLayout_4.addWidget(self.spinBox_thresh_C)
        self.gridLayout.addLayout(self.verticalLayout_4, 3, 0, 1, 1)
        self.lineEdit_results = QtGui.QLineEdit(self.centralWidget)
        self.lineEdit_results.setObjectName(_fromUtf8("lineEdit_results"))
        self.gridLayout.addWidget(self.lineEdit_results, 18, 3, 1, 2)
        self.dial_min_area = QtGui.QDial(self.centralWidget)
        self.dial_min_area.setMaximum(10000)
        self.dial_min_area.setProperty("value", 100)
        self.dial_min_area.setObjectName(_fromUtf8("dial_min_area"))
        self.gridLayout.addWidget(self.dial_min_area, 0, 1, 1, 1)
        self.dial_max_area = QtGui.QDial(self.centralWidget)
        self.dial_max_area.setMinimum(100)
        self.dial_max_area.setMaximum(100000)
        self.dial_max_area.setProperty("value", 5000)
        self.dial_max_area.setObjectName(_fromUtf8("dial_max_area"))
        self.gridLayout.addWidget(self.dial_max_area, 1, 1, 1, 1)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setMargin(11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.label = QtGui.QLabel(self.centralWidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout.addWidget(self.label)
        self.spinBox_min_area = QtGui.QSpinBox(self.centralWidget)
        self.spinBox_min_area.setMaximum(10000000)
        self.spinBox_min_area.setProperty("value", 100)
        self.spinBox_min_area.setObjectName(_fromUtf8("spinBox_min_area"))
        self.verticalLayout.addWidget(self.spinBox_min_area)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.dial_thresh_C = QtGui.QDial(self.centralWidget)
        self.dial_thresh_C.setMinimum(5)
        self.dial_thresh_C.setMaximum(100)
        self.dial_thresh_C.setProperty("value", 15)
        self.dial_thresh_C.setObjectName(_fromUtf8("dial_thresh_C"))
        self.gridLayout.addWidget(self.dial_thresh_C, 3, 1, 1, 1)
        self.dial_block_size = QtGui.QDial(self.centralWidget)
        self.dial_block_size.setMinimum(5)
        self.dial_block_size.setMaximum(200)
        self.dial_block_size.setProperty("value", 61)
        self.dial_block_size.setObjectName(_fromUtf8("dial_block_size"))
        self.gridLayout.addWidget(self.dial_block_size, 6, 1, 1, 1)
        self.pushButton_mask = QtGui.QPushButton(self.centralWidget)
        self.pushButton_mask.setObjectName(_fromUtf8("pushButton_mask"))
        self.gridLayout.addWidget(self.pushButton_mask, 19, 2, 1, 1)
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setMargin(11)
        self.verticalLayout_3.setSpacing(6)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.label_3 = QtGui.QLabel(self.centralWidget)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.verticalLayout_3.addWidget(self.label_3)
        self.spinBox_block_size = QtGui.QSpinBox(self.centralWidget)
        self.spinBox_block_size.setMinimum(5)
        self.spinBox_block_size.setMaximum(1000000)
        self.spinBox_block_size.setProperty("value", 61)
        self.spinBox_block_size.setObjectName(_fromUtf8("spinBox_block_size"))
        self.verticalLayout_3.addWidget(self.spinBox_block_size)
        self.gridLayout.addLayout(self.verticalLayout_3, 6, 0, 1, 1)
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setMargin(11)
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.label_4 = QtGui.QLabel(self.centralWidget)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.verticalLayout_2.addWidget(self.label_4)
        self.spinBox_max_area = QtGui.QSpinBox(self.centralWidget)
        self.spinBox_max_area.setMinimum(100)
        self.spinBox_max_area.setMaximum(10000000)
        self.spinBox_max_area.setProperty("value", 5000)
        self.spinBox_max_area.setObjectName(_fromUtf8("spinBox_max_area"))
        self.verticalLayout_2.addWidget(self.spinBox_max_area)
        self.gridLayout.addLayout(self.verticalLayout_2, 1, 0, 1, 1)
        self.pushButton_start = QtGui.QPushButton(self.centralWidget)
        font = QtGui.QFont()
        font.setPointSize(24)
        self.pushButton_start.setFont(font)
        self.pushButton_start.setMouseTracking(False)
        self.pushButton_start.setAutoFillBackground(False)
        self.pushButton_start.setCheckable(False)
        self.pushButton_start.setObjectName(_fromUtf8("pushButton_start"))
        self.gridLayout.addWidget(self.pushButton_start, 19, 0, 1, 2)
        self.pushButton_video = QtGui.QPushButton(self.centralWidget)
        self.pushButton_video.setObjectName(_fromUtf8("pushButton_video"))
        self.gridLayout.addWidget(self.pushButton_video, 17, 2, 1, 1)
        self.pushButton_results = QtGui.QPushButton(self.centralWidget)
        self.pushButton_results.setObjectName(_fromUtf8("pushButton_results"))
        self.gridLayout.addWidget(self.pushButton_results, 18, 2, 1, 1)
        self.lineEdit_mask = QtGui.QLineEdit(self.centralWidget)
        self.lineEdit_mask.setReadOnly(True)
        self.lineEdit_mask.setObjectName(_fromUtf8("lineEdit_mask"))
        self.gridLayout.addWidget(self.lineEdit_mask, 19, 3, 1, 2)
        self.lineEdit_video = QtGui.QLineEdit(self.centralWidget)
        self.lineEdit_video.setObjectName(_fromUtf8("lineEdit_video"))
        self.gridLayout.addWidget(self.lineEdit_video, 17, 3, 1, 2)
        spacerItem = QtGui.QSpacerItem(
            40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 16, 4, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(
            40,
            20,
            QtGui.QSizePolicy.MinimumExpanding,
            QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 16, 3, 1, 1)
        self.pushButton_next = QtGui.QPushButton(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Maximum,
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_next.sizePolicy().hasHeightForWidth())
        self.pushButton_next.setSizePolicy(sizePolicy)
        self.pushButton_next.setObjectName(_fromUtf8("pushButton_next"))
        self.gridLayout.addWidget(self.pushButton_next, 16, 5, 1, 1)
        self.label_mask = QtGui.QLabel(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Ignored,
            QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.label_mask.sizePolicy().hasHeightForWidth())
        self.label_mask.setSizePolicy(sizePolicy)
        self.label_mask.setFrameShape(QtGui.QFrame.Box)
        self.label_mask.setFrameShadow(QtGui.QFrame.Raised)
        self.label_mask.setText(_fromUtf8(""))
        self.label_mask.setObjectName(_fromUtf8("label_mask"))
        self.gridLayout.addWidget(self.label_mask, 0, 4, 13, 2)
        self.label_full = QtGui.QLabel(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Ignored,
            QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.label_full.sizePolicy().hasHeightForWidth())
        self.label_full.setSizePolicy(sizePolicy)
        self.label_full.setFrameShape(QtGui.QFrame.Box)
        self.label_full.setFrameShadow(QtGui.QFrame.Raised)
        self.label_full.setText(_fromUtf8(""))
        self.label_full.setWordWrap(False)
        self.label_full.setObjectName(_fromUtf8("label_full"))
        self.gridLayout.addWidget(self.label_full, 0, 2, 13, 2)
        self.checkBox_hasTimestamp = QtGui.QCheckBox(self.centralWidget)
        self.checkBox_hasTimestamp.setChecked(True)
        self.checkBox_hasTimestamp.setObjectName(
            _fromUtf8("checkBox_hasTimestamp"))
        self.gridLayout.addWidget(self.checkBox_hasTimestamp, 9, 0, 1, 1)
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setMargin(11)
        self.gridLayout_3.setSpacing(6)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.label_6 = QtGui.QLabel(self.centralWidget)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.gridLayout_3.addWidget(self.label_6, 0, 0, 1, 1)
        self.label_skelSeg = QtGui.QLabel(self.centralWidget)
        self.label_skelSeg.setObjectName(_fromUtf8("label_skelSeg"))
        self.gridLayout_3.addWidget(self.label_skelSeg, 1, 0, 1, 1)
        self.spinBox_fps = QtGui.QDoubleSpinBox(self.centralWidget)
        self.spinBox_fps.setProperty("value", 25.0)
        self.spinBox_fps.setObjectName(_fromUtf8("spinBox_fps"))
        self.gridLayout_3.addWidget(self.spinBox_fps, 2, 1, 1, 1)
        self.label_fps = QtGui.QLabel(self.centralWidget)
        self.label_fps.setObjectName(_fromUtf8("label_fps"))
        self.gridLayout_3.addWidget(self.label_fps, 2, 0, 1, 1)
        self.spinBox_buff_size = QtGui.QSpinBox(self.centralWidget)
        self.spinBox_buff_size.setMinimum(1)
        self.spinBox_buff_size.setMaximum(1000)
        self.spinBox_buff_size.setProperty("value", 25)
        self.spinBox_buff_size.setObjectName(_fromUtf8("spinBox_buff_size"))
        self.gridLayout_3.addWidget(self.spinBox_buff_size, 3, 1, 1, 1)
        self.label_5 = QtGui.QLabel(self.centralWidget)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout_3.addWidget(self.label_5, 3, 0, 1, 1)
        self.spinBox_skelSeg = QtGui.QSpinBox(self.centralWidget)
        self.spinBox_skelSeg.setProperty("value", 49)
        self.spinBox_skelSeg.setObjectName(_fromUtf8("spinBox_skelSeg"))
        self.gridLayout_3.addWidget(self.spinBox_skelSeg, 1, 1, 1, 1)
        self.spinBox_dilation_size = QtGui.QSpinBox(self.centralWidget)
        self.spinBox_dilation_size.setMinimum(1)
        self.spinBox_dilation_size.setMaximum(999)
        self.spinBox_dilation_size.setProperty("value", 9)
        self.spinBox_dilation_size.setObjectName(
            _fromUtf8("spinBox_dilation_size"))
        self.gridLayout_3.addWidget(self.spinBox_dilation_size, 0, 1, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_3, 15, 0, 4, 2)
        self.checkBox_keepBorderData = QtGui.QCheckBox(self.centralWidget)
        self.checkBox_keepBorderData.setObjectName(
            _fromUtf8("checkBox_keepBorderData"))
        self.gridLayout.addWidget(self.checkBox_keepBorderData, 9, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtGui.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1161, 22))
        self.menuBar.setObjectName(_fromUtf8("menuBar"))
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtGui.QToolBar(MainWindow)
        self.mainToolBar.setObjectName(_fromUtf8("mainToolBar"))
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtGui.QStatusBar(MainWindow)
        self.statusBar.setObjectName(_fromUtf8("statusBar"))
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.label_2.setText(_translate("MainWindow", "Thresh_C", None))
        self.label.setText(_translate("MainWindow", "Min Area", None))
        self.pushButton_mask.setText(
            _translate("MainWindow", "Mask Dir", None))
        self.label_3.setText(_translate("MainWindow", "Block Size", None))
        self.label_4.setText(_translate("MainWindow", "Max Area", None))
        self.pushButton_start.setText(_translate("MainWindow", "Start", None))
        self.pushButton_video.setText(
            _translate("MainWindow", "Video File", None))
        self.pushButton_results.setText(
            _translate("MainWindow", "Results Dir", None))
        self.pushButton_next.setText(
            _translate(
                "MainWindow",
                "Next Chunk",
                None))
        self.checkBox_hasTimestamp.setText(
            _translate("MainWindow", "has time stamp?", None))
        self.label_6.setText(_translate("MainWindow", "Dilation", None))
        self.label_skelSeg.setText(
            _translate(
                "MainWindow",
                "Skeleton Segments",
                None))
        self.label_fps.setText(_translate("MainWindow", "FPS", None))
        self.label_5.setText(
            _translate(
                "MainWindow",
                "Frames to average",
                None))
        self.checkBox_keepBorderData.setText(
            _translate("MainWindow", "keep border data?", None))
