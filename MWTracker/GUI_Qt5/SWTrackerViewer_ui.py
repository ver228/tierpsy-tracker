# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SWTrackerViewer.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SWTrackerViewer(object):

    def setupUi(self, SWTrackerViewer):
        SWTrackerViewer.setObjectName("SWTrackerViewer")
        SWTrackerViewer.resize(522, 760)
        self.centralWidget = QtWidgets.QWidget(SWTrackerViewer)
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
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 494, 449))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(
            self.scrollAreaWidgetContents)
        self.verticalLayout_3.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_3.setSpacing(6)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.mainGraphicsView = QtWidgets.QGraphicsView(
            self.scrollAreaWidgetContents)
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
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_6.setSpacing(6)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.checkBox_showLabel = QtWidgets.QCheckBox(self.centralWidget)
        self.checkBox_showLabel.setChecked(True)
        self.checkBox_showLabel.setObjectName("checkBox_showLabel")
        self.horizontalLayout_6.addWidget(self.checkBox_showLabel)
        self.label_skelBlock = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Maximum,
            QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.label_skelBlock.sizePolicy().hasHeightForWidth())
        self.label_skelBlock.setSizePolicy(sizePolicy)
        self.label_skelBlock.setObjectName("label_skelBlock")
        self.horizontalLayout_6.addWidget(self.label_skelBlock)
        self.spinBox_skelBlock = QtWidgets.QSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Maximum,
            QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.spinBox_skelBlock.sizePolicy().hasHeightForWidth())
        self.spinBox_skelBlock.setSizePolicy(sizePolicy)
        self.spinBox_skelBlock.setObjectName("spinBox_skelBlock")
        self.horizontalLayout_6.addWidget(self.spinBox_skelBlock)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
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
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_4.setSpacing(6)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.comboBox_h5path = QtWidgets.QComboBox(self.centralWidget)
        self.comboBox_h5path.setEditable(True)
        self.comboBox_h5path.setObjectName("comboBox_h5path")
        self.comboBox_h5path.addItem("")
        self.comboBox_h5path.addItem("")
        self.horizontalLayout_4.addWidget(self.comboBox_h5path)
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
        self.horizontalLayout_4.addWidget(self.pushButton_h5groups)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setContentsMargins(11, 11, 11, 11)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit_video = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_video.setText("")
        self.lineEdit_video.setObjectName("lineEdit_video")
        self.gridLayout.addWidget(self.lineEdit_video, 2, 0, 1, 1)
        self.pushButton_video = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_video.setObjectName("pushButton_video")
        self.gridLayout.addWidget(self.pushButton_video, 2, 1, 1, 1)
        self.pushButton_skel = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_skel.setObjectName("pushButton_skel")
        self.gridLayout.addWidget(self.pushButton_skel, 3, 1, 1, 1)
        self.lineEdit_skel = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_skel.setObjectName("lineEdit_skel")
        self.gridLayout.addWidget(self.lineEdit_skel, 3, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        SWTrackerViewer.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(SWTrackerViewer)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 522, 22))
        self.menuBar.setObjectName("menuBar")
        SWTrackerViewer.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(SWTrackerViewer)
        self.mainToolBar.setObjectName("mainToolBar")
        SWTrackerViewer.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.toolBar = QtWidgets.QToolBar(SWTrackerViewer)
        self.toolBar.setObjectName("toolBar")
        SWTrackerViewer.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.retranslateUi(SWTrackerViewer)
        QtCore.QMetaObject.connectSlotsByName(SWTrackerViewer)

    def retranslateUi(self, SWTrackerViewer):
        _translate = QtCore.QCoreApplication.translate
        SWTrackerViewer.setWindowTitle(
            _translate(
                "SWTrackerViewer",
                "SWTrackerViewer"))
        self.checkBox_showLabel.setText(
            _translate("SWTrackerViewer", "Show Skeleton"))
        self.label_skelBlock.setText(_translate("SWTrackerViewer", "Block"))
        self.label_frame.setText(_translate("SWTrackerViewer", "Frame"))
        self.label_step.setText(_translate("SWTrackerViewer", "Step Size"))
        self.label_fps.setText(_translate("SWTrackerViewer", "FPS display"))
        self.playButton.setText(_translate("SWTrackerViewer", "Play"))
        self.comboBox_h5path.setItemText(
            0, _translate("SWTrackerViewer", "/mask"))
        self.comboBox_h5path.setItemText(
            1, _translate("SWTrackerViewer", "/full_data"))
        self.pushButton_h5groups.setText(
            _translate("SWTrackerViewer", "Update Groups"))
        self.pushButton_video.setText(
            _translate(
                "SWTrackerViewer",
                "Select Video File"))
        self.pushButton_skel.setText(
            _translate(
                "SWTrackerViewer",
                "Select Skeletons File"))
        self.toolBar.setWindowTitle(_translate("SWTrackerViewer", "toolBar"))
