# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'imageviewer.ui'
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


class Ui_ImageViewer(object):

    def setupUi(self, ImageViewer):
        ImageViewer.setObjectName(_fromUtf8("ImageViewer"))
        ImageViewer.resize(642, 760)
        self.centralWidget = QtGui.QWidget(ImageViewer)
        self.centralWidget.setEnabled(True)
        self.centralWidget.setObjectName(_fromUtf8("centralWidget"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.centralWidget)
        self.verticalLayout_2.setMargin(11)
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setMargin(11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.scrollArea = QtGui.QScrollArea(self.centralWidget)
        self.scrollArea.setFrameShadow(QtGui.QFrame.Sunken)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName(_fromUtf8("scrollArea"))
        self.scrollAreaWidgetContents = QtGui.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 614, 485))
        self.scrollAreaWidgetContents.setObjectName(
            _fromUtf8("scrollAreaWidgetContents"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(
            self.scrollAreaWidgetContents)
        self.verticalLayout_3.setMargin(11)
        self.verticalLayout_3.setSpacing(6)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.imageCanvas = QtGui.QLabel(self.scrollAreaWidgetContents)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Ignored,
            QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.imageCanvas.sizePolicy().hasHeightForWidth())
        self.imageCanvas.setSizePolicy(sizePolicy)
        self.imageCanvas.setFrameShape(QtGui.QFrame.Box)
        self.imageCanvas.setFrameShadow(QtGui.QFrame.Sunken)
        self.imageCanvas.setText(_fromUtf8(""))
        self.imageCanvas.setObjectName(_fromUtf8("imageCanvas"))
        self.verticalLayout_3.addWidget(self.imageCanvas)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout.addWidget(self.scrollArea)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setMargin(11)
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.spinBox_frame = QtGui.QSpinBox(self.centralWidget)
        self.spinBox_frame.setMaximum(999999999)
        self.spinBox_frame.setObjectName(_fromUtf8("spinBox_frame"))
        self.horizontalLayout_3.addWidget(self.spinBox_frame)
        self.label_frame = QtGui.QLabel(self.centralWidget)
        self.label_frame.setObjectName(_fromUtf8("label_frame"))
        self.horizontalLayout_3.addWidget(self.label_frame)
        self.spinBox_step = QtGui.QSpinBox(self.centralWidget)
        self.spinBox_step.setMaximum(999999999)
        self.spinBox_step.setProperty("value", 1)
        self.spinBox_step.setObjectName(_fromUtf8("spinBox_step"))
        self.horizontalLayout_3.addWidget(self.spinBox_step)
        self.label_step = QtGui.QLabel(self.centralWidget)
        self.label_step.setObjectName(_fromUtf8("label_step"))
        self.horizontalLayout_3.addWidget(self.label_step)
        self.doubleSpinBox_fps = QtGui.QDoubleSpinBox(self.centralWidget)
        self.doubleSpinBox_fps.setMaximum(100.0)
        self.doubleSpinBox_fps.setProperty("value", 25.0)
        self.doubleSpinBox_fps.setObjectName(_fromUtf8("doubleSpinBox_fps"))
        self.horizontalLayout_3.addWidget(self.doubleSpinBox_fps)
        self.label_fps = QtGui.QLabel(self.centralWidget)
        self.label_fps.setObjectName(_fromUtf8("label_fps"))
        self.horizontalLayout_3.addWidget(self.label_fps)
        self.checkBox_showLabel = QtGui.QCheckBox(self.centralWidget)
        self.checkBox_showLabel.setChecked(True)
        self.checkBox_showLabel.setObjectName(_fromUtf8("checkBox_showLabel"))
        self.horizontalLayout_3.addWidget(self.checkBox_showLabel)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setMargin(11)
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.imageSlider = QtGui.QSlider(self.centralWidget)
        self.imageSlider.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.imageSlider.setMouseTracking(True)
        self.imageSlider.setAutoFillBackground(False)
        self.imageSlider.setMaximum(100)
        self.imageSlider.setOrientation(QtCore.Qt.Horizontal)
        self.imageSlider.setInvertedAppearance(False)
        self.imageSlider.setInvertedControls(False)
        self.imageSlider.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.imageSlider.setObjectName(_fromUtf8("imageSlider"))
        self.horizontalLayout_2.addWidget(self.imageSlider)
        self.playButton = QtGui.QPushButton(self.centralWidget)
        self.playButton.setCheckable(False)
        self.playButton.setObjectName(_fromUtf8("playButton"))
        self.horizontalLayout_2.addWidget(self.playButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setMargin(11)
        self.horizontalLayout_4.setSpacing(6)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.comboBox_h5path = QtGui.QComboBox(self.centralWidget)
        self.comboBox_h5path.setEditable(True)
        self.comboBox_h5path.setObjectName(_fromUtf8("comboBox_h5path"))
        self.comboBox_h5path.addItem(_fromUtf8(""))
        self.comboBox_h5path.addItem(_fromUtf8(""))
        self.horizontalLayout_4.addWidget(self.comboBox_h5path)
        self.pushButton_h5groups = QtGui.QPushButton(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Maximum,
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_h5groups.sizePolicy().hasHeightForWidth())
        self.pushButton_h5groups.setSizePolicy(sizePolicy)
        self.pushButton_h5groups.setObjectName(
            _fromUtf8("pushButton_h5groups"))
        self.horizontalLayout_4.addWidget(self.pushButton_h5groups)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setMargin(11)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.lineEdit_video = QtGui.QLineEdit(self.centralWidget)
        self.lineEdit_video.setText(_fromUtf8(""))
        self.lineEdit_video.setObjectName(_fromUtf8("lineEdit_video"))
        self.gridLayout.addWidget(self.lineEdit_video, 2, 0, 1, 1)
        self.pushButton_video = QtGui.QPushButton(self.centralWidget)
        self.pushButton_video.setObjectName(_fromUtf8("pushButton_video"))
        self.gridLayout.addWidget(self.pushButton_video, 2, 1, 1, 1)
        self.pushButton_skel = QtGui.QPushButton(self.centralWidget)
        self.pushButton_skel.setObjectName(_fromUtf8("pushButton_skel"))
        self.gridLayout.addWidget(self.pushButton_skel, 3, 1, 1, 1)
        self.lineEdit_skel = QtGui.QLineEdit(self.centralWidget)
        self.lineEdit_skel.setObjectName(_fromUtf8("lineEdit_skel"))
        self.gridLayout.addWidget(self.lineEdit_skel, 3, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        ImageViewer.setCentralWidget(self.centralWidget)
        self.menuBar = QtGui.QMenuBar(ImageViewer)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 642, 22))
        self.menuBar.setObjectName(_fromUtf8("menuBar"))
        ImageViewer.setMenuBar(self.menuBar)
        self.mainToolBar = QtGui.QToolBar(ImageViewer)
        self.mainToolBar.setObjectName(_fromUtf8("mainToolBar"))
        ImageViewer.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.toolBar = QtGui.QToolBar(ImageViewer)
        self.toolBar.setObjectName(_fromUtf8("toolBar"))
        ImageViewer.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.retranslateUi(ImageViewer)
        QtCore.QMetaObject.connectSlotsByName(ImageViewer)

    def retranslateUi(self, ImageViewer):
        ImageViewer.setWindowTitle(
            _translate(
                "ImageViewer",
                "ImageViewer",
                None))
        self.label_frame.setText(_translate("ImageViewer", "Frame", None))
        self.label_step.setText(_translate("ImageViewer", "Step Size", None))
        self.label_fps.setText(_translate("ImageViewer", "FPS display", None))
        self.checkBox_showLabel.setText(
            _translate("ImageViewer", "Show Skeleton", None))
        self.playButton.setText(_translate("ImageViewer", "Play", None))
        self.comboBox_h5path.setItemText(
            0, _translate("ImageViewer", "/mask", None))
        self.comboBox_h5path.setItemText(
            1, _translate("ImageViewer", "/full_data", None))
        self.pushButton_h5groups.setText(
            _translate("ImageViewer", "Update Groups", None))
        self.pushButton_video.setText(
            _translate(
                "ImageViewer",
                "Select Video File",
                None))
        self.pushButton_skel.setText(
            _translate(
                "ImageViewer",
                "Select Skeletons File",
                None))
        self.toolBar.setWindowTitle(_translate("ImageViewer", "toolBar", None))
