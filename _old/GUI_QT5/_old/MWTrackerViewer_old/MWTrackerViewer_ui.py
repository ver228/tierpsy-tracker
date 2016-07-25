# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'imageviewer.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ImageViewer(object):

    def setupUi(self, ImageViewer):
        ImageViewer.setObjectName("ImageViewer")
        ImageViewer.resize(1645, 802)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            ImageViewer.sizePolicy().hasHeightForWidth())
        ImageViewer.setSizePolicy(sizePolicy)
        ImageViewer.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.centralWidget = QtWidgets.QWidget(ImageViewer)
        self.centralWidget.setEnabled(True)
        self.centralWidget.setObjectName("centralWidget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.centralWidget)
        self.gridLayout_4.setContentsMargins(11, 11, 11, 11)
        self.gridLayout_4.setSpacing(6)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.gridLayout_3.setContentsMargins(11, 11, 11, 11)
        self.gridLayout_3.setSpacing(6)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.checkBox_ROI1 = QtWidgets.QCheckBox(self.centralWidget)
        self.checkBox_ROI1.setChecked(True)
        self.checkBox_ROI1.setObjectName("checkBox_ROI1")
        self.gridLayout_3.addWidget(self.checkBox_ROI1, 1, 7, 1, 1)
        self.comboBox_ROI1 = QtWidgets.QComboBox(self.centralWidget)
        self.comboBox_ROI1.setEditable(True)
        self.comboBox_ROI1.setObjectName("comboBox_ROI1")
        self.gridLayout_3.addWidget(self.comboBox_ROI1, 1, 6, 1, 1)
        self.wormCanvas2 = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.wormCanvas2.sizePolicy().hasHeightForWidth())
        self.wormCanvas2.setSizePolicy(sizePolicy)
        self.wormCanvas2.setFrameShape(QtWidgets.QFrame.Box)
        self.wormCanvas2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.wormCanvas2.setText("")
        self.wormCanvas2.setObjectName("wormCanvas2")
        self.gridLayout_3.addWidget(self.wormCanvas2, 2, 5, 1, 6)
        self.radioButton_ROI2 = QtWidgets.QRadioButton(self.centralWidget)
        self.radioButton_ROI2.setText("")
        self.radioButton_ROI2.setObjectName("radioButton_ROI2")
        self.gridLayout_3.addWidget(self.radioButton_ROI2, 4, 5, 1, 1)
        self.checkBox_ROI2 = QtWidgets.QCheckBox(self.centralWidget)
        self.checkBox_ROI2.setChecked(True)
        self.checkBox_ROI2.setObjectName("checkBox_ROI2")
        self.gridLayout_3.addWidget(self.checkBox_ROI2, 4, 7, 1, 1)
        self.checkBox_showLabel = QtWidgets.QCheckBox(self.centralWidget)
        self.checkBox_showLabel.setChecked(True)
        self.checkBox_showLabel.setObjectName("checkBox_showLabel")
        self.gridLayout_3.addWidget(self.checkBox_showLabel, 4, 0, 1, 1)
        self.radioButton_ROI1 = QtWidgets.QRadioButton(self.centralWidget)
        self.radioButton_ROI1.setText("")
        self.radioButton_ROI1.setChecked(True)
        self.radioButton_ROI1.setObjectName("radioButton_ROI1")
        self.gridLayout_3.addWidget(self.radioButton_ROI1, 1, 5, 1, 1)
        self.comboBox_ROI2 = QtWidgets.QComboBox(self.centralWidget)
        self.comboBox_ROI2.setEditable(True)
        self.comboBox_ROI2.setObjectName("comboBox_ROI2")
        self.gridLayout_3.addWidget(self.comboBox_ROI2, 4, 6, 1, 1)
        self.wormCanvas1 = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.wormCanvas1.sizePolicy().hasHeightForWidth())
        self.wormCanvas1.setSizePolicy(sizePolicy)
        self.wormCanvas1.setFrameShape(QtWidgets.QFrame.Box)
        self.wormCanvas1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.wormCanvas1.setText("")
        self.wormCanvas1.setObjectName("wormCanvas1")
        self.gridLayout_3.addWidget(self.wormCanvas1, 0, 5, 1, 6)
        self.pushButton_ROI1_RW = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_ROI1_RW.setObjectName("pushButton_ROI1_RW")
        self.gridLayout_3.addWidget(self.pushButton_ROI1_RW, 1, 8, 1, 1)
        self.comboBox_labelType = QtWidgets.QComboBox(self.centralWidget)
        self.comboBox_labelType.setObjectName("comboBox_labelType")
        self.comboBox_labelType.addItem("")
        self.comboBox_labelType.addItem("")
        self.gridLayout_3.addWidget(self.comboBox_labelType, 4, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(
            600, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem, 4, 3, 1, 1)
        self.imageCanvas = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.imageCanvas.sizePolicy().hasHeightForWidth())
        self.imageCanvas.setSizePolicy(sizePolicy)
        self.imageCanvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.imageCanvas.setFrameShape(QtWidgets.QFrame.Box)
        self.imageCanvas.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.imageCanvas.setText("")
        self.imageCanvas.setObjectName("imageCanvas")
        self.gridLayout_3.addWidget(self.imageCanvas, 0, 0, 3, 5)
        self.pushButton_feats = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_feats.setObjectName("pushButton_feats")
        self.gridLayout_3.addWidget(self.pushButton_feats, 4, 4, 1, 1)
        self.pushButton_ROI2_FF = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_ROI2_FF.setObjectName("pushButton_ROI2_FF")
        self.gridLayout_3.addWidget(self.pushButton_ROI2_FF, 4, 9, 1, 1)
        self.pushButton_ROI2_RW = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_ROI2_RW.setObjectName("pushButton_ROI2_RW")
        self.gridLayout_3.addWidget(self.pushButton_ROI2_RW, 4, 8, 1, 1)
        self.pushButton_ROI1_FF = QtWidgets.QPushButton(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Maximum,
            QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_ROI1_FF.sizePolicy().hasHeightForWidth())
        self.pushButton_ROI1_FF.setSizePolicy(sizePolicy)
        self.pushButton_ROI1_FF.setObjectName("pushButton_ROI1_FF")
        self.gridLayout_3.addWidget(self.pushButton_ROI1_FF, 1, 9, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem1, 0, 2, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem2, 0, 0, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setContentsMargins(11, 11, 11, 11)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.spinBox_frame = QtWidgets.QSpinBox(self.centralWidget)
        self.spinBox_frame.setObjectName("spinBox_frame")
        self.gridLayout.addWidget(self.spinBox_frame, 0, 1, 1, 1)
        self.lineEdit_video = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_video.setText("")
        self.lineEdit_video.setObjectName("lineEdit_video")
        self.gridLayout.addWidget(self.lineEdit_video, 3, 3, 1, 5)
        self.pushButton_video = QtWidgets.QPushButton(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_video.sizePolicy().hasHeightForWidth())
        self.pushButton_video.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton_video.setFont(font)
        self.pushButton_video.setObjectName("pushButton_video")
        self.gridLayout.addWidget(self.pushButton_video, 3, 0, 1, 2)
        self.label_frame = QtWidgets.QLabel(self.centralWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_frame.setFont(font)
        self.label_frame.setAlignment(QtCore.Qt.AlignCenter)
        self.label_frame.setObjectName("label_frame")
        self.gridLayout.addWidget(self.label_frame, 0, 0, 1, 1)
        self.pushButton_skel = QtWidgets.QPushButton(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_skel.sizePolicy().hasHeightForWidth())
        self.pushButton_skel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton_skel.setFont(font)
        self.pushButton_skel.setObjectName("pushButton_skel")
        self.gridLayout.addWidget(self.pushButton_skel, 4, 0, 1, 2)
        self.lineEdit_skel = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_skel.setObjectName("lineEdit_skel")
        self.gridLayout.addWidget(self.lineEdit_skel, 4, 3, 1, 5)
        self.playButton = QtWidgets.QPushButton(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.playButton.sizePolicy().hasHeightForWidth())
        self.playButton.setSizePolicy(sizePolicy)
        self.playButton.setCheckable(False)
        self.playButton.setObjectName("playButton")
        self.gridLayout.addWidget(self.playButton, 0, 7, 1, 1)
        self.spinBox_step = QtWidgets.QSpinBox(self.centralWidget)
        self.spinBox_step.setMaximum(100000)
        self.spinBox_step.setProperty("value", 1)
        self.spinBox_step.setObjectName("spinBox_step")
        self.gridLayout.addWidget(self.spinBox_step, 1, 1, 1, 1)
        self.label_fps = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Maximum,
            QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.label_fps.sizePolicy().hasHeightForWidth())
        self.label_fps.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_fps.setFont(font)
        self.label_fps.setAlignment(QtCore.Qt.AlignCenter)
        self.label_fps.setObjectName("label_fps")
        self.gridLayout.addWidget(self.label_fps, 2, 0, 1, 1)
        self.doubleSpinBox_fps = QtWidgets.QDoubleSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Maximum,
            QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.doubleSpinBox_fps.sizePolicy().hasHeightForWidth())
        self.doubleSpinBox_fps.setSizePolicy(sizePolicy)
        self.doubleSpinBox_fps.setMaximum(100.0)
        self.doubleSpinBox_fps.setProperty("value", 25.0)
        self.doubleSpinBox_fps.setObjectName("doubleSpinBox_fps")
        self.gridLayout.addWidget(self.doubleSpinBox_fps, 2, 1, 1, 1)
        self.comboBox_h5path = QtWidgets.QComboBox(self.centralWidget)
        self.comboBox_h5path.setEditable(True)
        self.comboBox_h5path.setObjectName("comboBox_h5path")
        self.comboBox_h5path.addItem("")
        self.comboBox_h5path.addItem("")
        self.gridLayout.addWidget(self.comboBox_h5path, 5, 3, 1, 5)
        self.label_step = QtWidgets.QLabel(self.centralWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_step.setFont(font)
        self.label_step.setAlignment(QtCore.Qt.AlignCenter)
        self.label_step.setObjectName("label_step")
        self.gridLayout.addWidget(self.label_step, 1, 0, 1, 1)
        self.spinBox_join1 = QtWidgets.QSpinBox(self.centralWidget)
        self.spinBox_join1.setProperty("showGroupSeparator", False)
        self.spinBox_join1.setMaximum(999999999)
        self.spinBox_join1.setObjectName("spinBox_join1")
        self.gridLayout.addWidget(self.spinBox_join1, 2, 5, 1, 1)
        self.spinBox_join2 = QtWidgets.QSpinBox(self.centralWidget)
        self.spinBox_join2.setMaximum(999999999)
        self.spinBox_join2.setObjectName("spinBox_join2")
        self.gridLayout.addWidget(self.spinBox_join2, 2, 6, 1, 1)
        self.pushButton_join = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_join.setObjectName("pushButton_join")
        self.gridLayout.addWidget(self.pushButton_join, 2, 3, 1, 1)
        self.pushButton_split = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_split.setObjectName("pushButton_split")
        self.gridLayout.addWidget(self.pushButton_split, 2, 4, 1, 1)
        self.pushButton_save = QtWidgets.QPushButton(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Maximum,
            QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_save.sizePolicy().hasHeightForWidth())
        self.pushButton_save.setSizePolicy(sizePolicy)
        self.pushButton_save.setObjectName("pushButton_save")
        self.gridLayout.addWidget(self.pushButton_save, 1, 7, 2, 1)
        self.pushButton_U = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_U.setObjectName("pushButton_U")
        self.gridLayout.addWidget(self.pushButton_U, 1, 3, 1, 1)
        self.pushButton_W = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_W.setObjectName("pushButton_W")
        self.gridLayout.addWidget(self.pushButton_W, 1, 4, 1, 1)
        self.pushButton_WS = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_WS.setObjectName("pushButton_WS")
        self.gridLayout.addWidget(self.pushButton_WS, 1, 5, 1, 1)
        self.pushButton_B = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_B.setObjectName("pushButton_B")
        self.gridLayout.addWidget(self.pushButton_B, 1, 6, 1, 1)
        self.pushButton_h5groups = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_h5groups.setObjectName("pushButton_h5groups")
        self.gridLayout.addWidget(self.pushButton_h5groups, 5, 0, 1, 2)
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
        self.gridLayout.addWidget(self.imageSlider, 0, 3, 1, 4)
        self.gridLayout_4.addLayout(self.gridLayout, 1, 1, 1, 1)
        ImageViewer.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(ImageViewer)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1645, 22))
        self.menuBar.setObjectName("menuBar")
        ImageViewer.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(ImageViewer)
        self.mainToolBar.setObjectName("mainToolBar")
        ImageViewer.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.toolBar = QtWidgets.QToolBar(ImageViewer)
        self.toolBar.setObjectName("toolBar")
        ImageViewer.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.retranslateUi(ImageViewer)
        QtCore.QMetaObject.connectSlotsByName(ImageViewer)

    def retranslateUi(self, ImageViewer):
        _translate = QtCore.QCoreApplication.translate
        ImageViewer.setWindowTitle(_translate("ImageViewer", "ImageViewer"))
        self.checkBox_ROI1.setText(_translate("ImageViewer", "Skeleton"))
        self.checkBox_ROI2.setText(_translate("ImageViewer", "Skeleton"))
        self.checkBox_showLabel.setText(
            _translate("ImageViewer", "Show Labels"))
        self.pushButton_ROI1_RW.setText(_translate("ImageViewer", "<<"))
        self.comboBox_labelType.setItemText(
            0, _translate("ImageViewer", "Manual"))
        self.comboBox_labelType.setItemText(
            1, _translate("ImageViewer", "Auto"))
        self.pushButton_feats.setText(
            _translate(
                "ImageViewer",
                "Calc Individual Feat"))
        self.pushButton_ROI2_FF.setText(_translate("ImageViewer", ">>"))
        self.pushButton_ROI2_RW.setText(_translate("ImageViewer", "<<"))
        self.pushButton_ROI1_FF.setText(_translate("ImageViewer", ">>"))
        self.pushButton_video.setText(
            _translate(
                "ImageViewer",
                "Select Video File"))
        self.label_frame.setText(_translate("ImageViewer", "Frame"))
        self.pushButton_skel.setText(
            _translate(
                "ImageViewer",
                "Select Skeletons File"))
        self.playButton.setText(_translate("ImageViewer", "Play"))
        self.label_fps.setText(_translate("ImageViewer", "FPS display"))
        self.comboBox_h5path.setItemText(0, _translate("ImageViewer", "/mask"))
        self.comboBox_h5path.setItemText(
            1, _translate("ImageViewer", "/full_data"))
        self.label_step.setText(_translate("ImageViewer", "Step Size"))
        self.pushButton_join.setText(
            _translate(
                "ImageViewer",
                "Join Trajectory"))
        self.pushButton_split.setText(
            _translate(
                "ImageViewer",
                "Split Trajectory"))
        self.pushButton_save.setText(_translate("ImageViewer", "SAVE"))
        self.pushButton_U.setText(_translate("ImageViewer", "Undefined"))
        self.pushButton_W.setText(_translate("ImageViewer", "Single Worms"))
        self.pushButton_WS.setText(_translate("ImageViewer", "Worm Cluster"))
        self.pushButton_B.setText(_translate("ImageViewer", "Bad"))
        self.pushButton_h5groups.setText(
            _translate("ImageViewer", "Update Groups"))
        self.toolBar.setWindowTitle(_translate("ImageViewer", "toolBar"))
