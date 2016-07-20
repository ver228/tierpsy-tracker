# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'HDF5VideoPlayer.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_HDF5VideoPlayer(object):

    def setupUi(self, HDF5VideoPlayer):
        HDF5VideoPlayer.setObjectName("HDF5VideoPlayer")
        HDF5VideoPlayer.resize(600, 760)
        self.centralWidget = QtWidgets.QWidget(HDF5VideoPlayer)
        self.centralWidget.setEnabled(True)
        self.centralWidget.setObjectName("centralWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralWidget)
        self.verticalLayout_2.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.scrollArea = QtWidgets.QScrollArea(self.centralWidget)
        self.scrollArea.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 572, 525))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(
            self.scrollAreaWidgetContents)
        self.verticalLayout_3.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_3.setSpacing(6)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.mainGraphicsView = QtWidgets.QGraphicsView(
            self.scrollAreaWidgetContents)
        self.mainGraphicsView.setAutoFillBackground(False)
        self.mainGraphicsView.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.mainGraphicsView.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff)
        self.mainGraphicsView.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff)
        self.mainGraphicsView.setDragMode(
            QtWidgets.QGraphicsView.ScrollHandDrag)
        self.mainGraphicsView.setTransformationAnchor(
            QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.mainGraphicsView.setResizeAnchor(
            QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.mainGraphicsView.setObjectName("mainGraphicsView")
        self.verticalLayout_3.addWidget(self.mainGraphicsView)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout.addWidget(self.scrollArea)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.spinBox_frame = QtWidgets.QSpinBox(self.centralWidget)
        self.spinBox_frame.setMaximum(999999999)
        self.spinBox_frame.setObjectName("spinBox_frame")
        self.horizontalLayout_3.addWidget(self.spinBox_frame)
        self.label_frame = QtWidgets.QLabel(self.centralWidget)
        self.label_frame.setObjectName("label_frame")
        self.horizontalLayout_3.addWidget(self.label_frame)
        self.spinBox_step = QtWidgets.QSpinBox(self.centralWidget)
        self.spinBox_step.setMaximum(999999999)
        self.spinBox_step.setProperty("value", 1)
        self.spinBox_step.setObjectName("spinBox_step")
        self.horizontalLayout_3.addWidget(self.spinBox_step)
        self.label_step = QtWidgets.QLabel(self.centralWidget)
        self.label_step.setObjectName("label_step")
        self.horizontalLayout_3.addWidget(self.label_step)
        self.doubleSpinBox_fps = QtWidgets.QDoubleSpinBox(self.centralWidget)
        self.doubleSpinBox_fps.setMaximum(100.0)
        self.doubleSpinBox_fps.setProperty("value", 25.0)
        self.doubleSpinBox_fps.setObjectName("doubleSpinBox_fps")
        self.horizontalLayout_3.addWidget(self.doubleSpinBox_fps)
        self.label_fps = QtWidgets.QLabel(self.centralWidget)
        self.label_fps.setObjectName("label_fps")
        self.horizontalLayout_3.addWidget(self.label_fps)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_6.setSpacing(6)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.comboBox_h5path = QtWidgets.QComboBox(self.centralWidget)
        self.comboBox_h5path.setEditable(True)
        self.comboBox_h5path.setObjectName("comboBox_h5path")
        self.comboBox_h5path.addItem("")
        self.comboBox_h5path.addItem("")
        self.horizontalLayout_6.addWidget(self.comboBox_h5path)
        self.pushButton_h5groups = QtWidgets.QPushButton(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Maximum,
            QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_h5groups.sizePolicy().hasHeightForWidth())
        self.pushButton_h5groups.setSizePolicy(sizePolicy)
        self.pushButton_h5groups.setObjectName("pushButton_h5groups")
        self.horizontalLayout_6.addWidget(self.pushButton_h5groups)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.imageSlider = QtWidgets.QSlider(self.centralWidget)
        self.imageSlider.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.imageSlider.setMouseTracking(True)
        self.imageSlider.setAutoFillBackground(False)
        self.imageSlider.setMaximum(100)
        self.imageSlider.setOrientation(QtCore.Qt.Horizontal)
        self.imageSlider.setInvertedAppearance(False)
        self.imageSlider.setInvertedControls(False)
        self.imageSlider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.imageSlider.setObjectName("imageSlider")
        self.horizontalLayout_2.addWidget(self.imageSlider)
        self.playButton = QtWidgets.QPushButton(self.centralWidget)
        self.playButton.setCheckable(False)
        self.playButton.setObjectName("playButton")
        self.horizontalLayout_2.addWidget(self.playButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lineEdit_video = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_video.setText("")
        self.lineEdit_video.setObjectName("lineEdit_video")
        self.horizontalLayout.addWidget(self.lineEdit_video)
        self.pushButton_video = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_video.setObjectName("pushButton_video")
        self.horizontalLayout.addWidget(self.pushButton_video)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        HDF5VideoPlayer.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(HDF5VideoPlayer)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 600, 22))
        self.menuBar.setObjectName("menuBar")
        HDF5VideoPlayer.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(HDF5VideoPlayer)
        self.mainToolBar.setObjectName("mainToolBar")
        HDF5VideoPlayer.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.toolBar = QtWidgets.QToolBar(HDF5VideoPlayer)
        self.toolBar.setObjectName("toolBar")
        HDF5VideoPlayer.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.retranslateUi(HDF5VideoPlayer)
        QtCore.QMetaObject.connectSlotsByName(HDF5VideoPlayer)

    def retranslateUi(self, HDF5VideoPlayer):
        _translate = QtCore.QCoreApplication.translate
        HDF5VideoPlayer.setWindowTitle(
            _translate(
                "HDF5VideoPlayer",
                "HDF5VideoPlayer"))
        self.label_frame.setText(_translate("HDF5VideoPlayer", "Frame"))
        self.label_step.setText(_translate("HDF5VideoPlayer", "Step Size"))
        self.label_fps.setText(_translate("HDF5VideoPlayer", "FPS display"))
        self.comboBox_h5path.setItemText(
            0, _translate("HDF5VideoPlayer", "/mask"))
        self.comboBox_h5path.setItemText(
            1, _translate("HDF5VideoPlayer", "/full_data"))
        self.pushButton_h5groups.setText(
            _translate("HDF5VideoPlayer", "Update Groups"))
        self.playButton.setText(_translate("HDF5VideoPlayer", "Play"))
        self.pushButton_video.setText(
            _translate(
                "HDF5VideoPlayer",
                "Select File"))
        self.toolBar.setWindowTitle(_translate("HDF5VideoPlayer", "toolBar"))
