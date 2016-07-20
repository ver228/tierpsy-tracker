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
        ImageViewer.resize(837, 792)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Preferred,
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            ImageViewer.sizePolicy().hasHeightForWidth())
        ImageViewer.setSizePolicy(sizePolicy)
        ImageViewer.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.centralWidget = QtGui.QWidget(ImageViewer)
        self.centralWidget.setEnabled(True)
        self.centralWidget.setObjectName(_fromUtf8("centralWidget"))
        self.gridLayout_4 = QtGui.QGridLayout(self.centralWidget)
        self.gridLayout_4.setMargin(11)
        self.gridLayout_4.setSpacing(6)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setMargin(11)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.spinBox_frame = QtGui.QSpinBox(self.centralWidget)
        self.spinBox_frame.setObjectName(_fromUtf8("spinBox_frame"))
        self.gridLayout.addWidget(self.spinBox_frame, 0, 1, 1, 1)
        self.lineEdit_video = QtGui.QLineEdit(self.centralWidget)
        self.lineEdit_video.setText(_fromUtf8(""))
        self.lineEdit_video.setObjectName(_fromUtf8("lineEdit_video"))
        self.gridLayout.addWidget(self.lineEdit_video, 3, 3, 1, 5)
        self.pushButton_video = QtGui.QPushButton(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Expanding,
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_video.sizePolicy().hasHeightForWidth())
        self.pushButton_video.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton_video.setFont(font)
        self.pushButton_video.setObjectName(_fromUtf8("pushButton_video"))
        self.gridLayout.addWidget(self.pushButton_video, 3, 0, 1, 2)
        self.label_frame = QtGui.QLabel(self.centralWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_frame.setFont(font)
        self.label_frame.setAlignment(QtCore.Qt.AlignCenter)
        self.label_frame.setObjectName(_fromUtf8("label_frame"))
        self.gridLayout.addWidget(self.label_frame, 0, 0, 1, 1)
        self.pushButton_skel = QtGui.QPushButton(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Expanding,
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_skel.sizePolicy().hasHeightForWidth())
        self.pushButton_skel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton_skel.setFont(font)
        self.pushButton_skel.setObjectName(_fromUtf8("pushButton_skel"))
        self.gridLayout.addWidget(self.pushButton_skel, 4, 0, 1, 2)
        self.lineEdit_skel = QtGui.QLineEdit(self.centralWidget)
        self.lineEdit_skel.setObjectName(_fromUtf8("lineEdit_skel"))
        self.gridLayout.addWidget(self.lineEdit_skel, 4, 3, 1, 5)
        self.playButton = QtGui.QPushButton(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Minimum,
            QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.playButton.sizePolicy().hasHeightForWidth())
        self.playButton.setSizePolicy(sizePolicy)
        self.playButton.setCheckable(False)
        self.playButton.setObjectName(_fromUtf8("playButton"))
        self.gridLayout.addWidget(self.playButton, 0, 7, 1, 1)
        self.spinBox_step = QtGui.QSpinBox(self.centralWidget)
        self.spinBox_step.setMaximum(100000)
        self.spinBox_step.setProperty("value", 1)
        self.spinBox_step.setObjectName(_fromUtf8("spinBox_step"))
        self.gridLayout.addWidget(self.spinBox_step, 1, 1, 1, 1)
        self.label_fps = QtGui.QLabel(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Maximum,
            QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.label_fps.sizePolicy().hasHeightForWidth())
        self.label_fps.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_fps.setFont(font)
        self.label_fps.setAlignment(QtCore.Qt.AlignCenter)
        self.label_fps.setObjectName(_fromUtf8("label_fps"))
        self.gridLayout.addWidget(self.label_fps, 2, 0, 1, 1)
        self.doubleSpinBox_fps = QtGui.QDoubleSpinBox(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Maximum,
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.doubleSpinBox_fps.sizePolicy().hasHeightForWidth())
        self.doubleSpinBox_fps.setSizePolicy(sizePolicy)
        self.doubleSpinBox_fps.setMaximum(100.0)
        self.doubleSpinBox_fps.setProperty("value", 25.0)
        self.doubleSpinBox_fps.setObjectName(_fromUtf8("doubleSpinBox_fps"))
        self.gridLayout.addWidget(self.doubleSpinBox_fps, 2, 1, 1, 1)
        self.comboBox_h5path = QtGui.QComboBox(self.centralWidget)
        self.comboBox_h5path.setEditable(True)
        self.comboBox_h5path.setObjectName(_fromUtf8("comboBox_h5path"))
        self.comboBox_h5path.addItem(_fromUtf8(""))
        self.comboBox_h5path.addItem(_fromUtf8(""))
        self.gridLayout.addWidget(self.comboBox_h5path, 5, 3, 1, 5)
        self.label_step = QtGui.QLabel(self.centralWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_step.setFont(font)
        self.label_step.setAlignment(QtCore.Qt.AlignCenter)
        self.label_step.setObjectName(_fromUtf8("label_step"))
        self.gridLayout.addWidget(self.label_step, 1, 0, 1, 1)
        self.spinBox_join1 = QtGui.QSpinBox(self.centralWidget)
        self.spinBox_join1.setProperty("showGroupSeparator", False)
        self.spinBox_join1.setMaximum(999999999)
        self.spinBox_join1.setObjectName(_fromUtf8("spinBox_join1"))
        self.gridLayout.addWidget(self.spinBox_join1, 2, 5, 1, 1)
        self.spinBox_join2 = QtGui.QSpinBox(self.centralWidget)
        self.spinBox_join2.setMaximum(999999999)
        self.spinBox_join2.setObjectName(_fromUtf8("spinBox_join2"))
        self.gridLayout.addWidget(self.spinBox_join2, 2, 6, 1, 1)
        self.pushButton_join = QtGui.QPushButton(self.centralWidget)
        self.pushButton_join.setObjectName(_fromUtf8("pushButton_join"))
        self.gridLayout.addWidget(self.pushButton_join, 2, 3, 1, 1)
        self.pushButton_split = QtGui.QPushButton(self.centralWidget)
        self.pushButton_split.setObjectName(_fromUtf8("pushButton_split"))
        self.gridLayout.addWidget(self.pushButton_split, 2, 4, 1, 1)
        self.pushButton_save = QtGui.QPushButton(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Maximum,
            QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_save.sizePolicy().hasHeightForWidth())
        self.pushButton_save.setSizePolicy(sizePolicy)
        self.pushButton_save.setObjectName(_fromUtf8("pushButton_save"))
        self.gridLayout.addWidget(self.pushButton_save, 1, 7, 2, 1)
        self.pushButton_U = QtGui.QPushButton(self.centralWidget)
        self.pushButton_U.setObjectName(_fromUtf8("pushButton_U"))
        self.gridLayout.addWidget(self.pushButton_U, 1, 3, 1, 1)
        self.pushButton_W = QtGui.QPushButton(self.centralWidget)
        self.pushButton_W.setObjectName(_fromUtf8("pushButton_W"))
        self.gridLayout.addWidget(self.pushButton_W, 1, 4, 1, 1)
        self.pushButton_WS = QtGui.QPushButton(self.centralWidget)
        self.pushButton_WS.setObjectName(_fromUtf8("pushButton_WS"))
        self.gridLayout.addWidget(self.pushButton_WS, 1, 5, 1, 1)
        self.pushButton_B = QtGui.QPushButton(self.centralWidget)
        self.pushButton_B.setObjectName(_fromUtf8("pushButton_B"))
        self.gridLayout.addWidget(self.pushButton_B, 1, 6, 1, 1)
        self.pushButton_h5groups = QtGui.QPushButton(self.centralWidget)
        self.pushButton_h5groups.setObjectName(
            _fromUtf8("pushButton_h5groups"))
        self.gridLayout.addWidget(self.pushButton_h5groups, 5, 0, 1, 2)
        self.imageSlider = QtGui.QSlider(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Expanding,
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.imageSlider.sizePolicy().hasHeightForWidth())
        self.imageSlider.setSizePolicy(sizePolicy)
        self.imageSlider.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.imageSlider.setMouseTracking(True)
        self.imageSlider.setAutoFillBackground(False)
        self.imageSlider.setMaximum(100)
        self.imageSlider.setOrientation(QtCore.Qt.Horizontal)
        self.imageSlider.setInvertedAppearance(False)
        self.imageSlider.setInvertedControls(False)
        self.imageSlider.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.imageSlider.setObjectName(_fromUtf8("imageSlider"))
        self.gridLayout.addWidget(self.imageSlider, 0, 3, 1, 4)
        self.gridLayout_4.addLayout(self.gridLayout, 1, 0, 1, 1)
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setSizeConstraint(QtGui.QLayout.SetMaximumSize)
        self.gridLayout_3.setMargin(11)
        self.gridLayout_3.setSpacing(6)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.checkBox_ROI1 = QtGui.QCheckBox(self.centralWidget)
        self.checkBox_ROI1.setChecked(True)
        self.checkBox_ROI1.setObjectName(_fromUtf8("checkBox_ROI1"))
        self.gridLayout_3.addWidget(self.checkBox_ROI1, 1, 7, 1, 1)
        self.comboBox_ROI1 = QtGui.QComboBox(self.centralWidget)
        self.comboBox_ROI1.setEditable(True)
        self.comboBox_ROI1.setObjectName(_fromUtf8("comboBox_ROI1"))
        self.gridLayout_3.addWidget(self.comboBox_ROI1, 1, 6, 1, 1)
        self.wormCanvas2 = QtGui.QLabel(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Ignored,
            QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.wormCanvas2.sizePolicy().hasHeightForWidth())
        self.wormCanvas2.setSizePolicy(sizePolicy)
        self.wormCanvas2.setFrameShape(QtGui.QFrame.Box)
        self.wormCanvas2.setFrameShadow(QtGui.QFrame.Sunken)
        self.wormCanvas2.setText(_fromUtf8(""))
        self.wormCanvas2.setObjectName(_fromUtf8("wormCanvas2"))
        self.gridLayout_3.addWidget(self.wormCanvas2, 2, 5, 1, 6)
        self.radioButton_ROI2 = QtGui.QRadioButton(self.centralWidget)
        self.radioButton_ROI2.setText(_fromUtf8(""))
        self.radioButton_ROI2.setObjectName(_fromUtf8("radioButton_ROI2"))
        self.gridLayout_3.addWidget(self.radioButton_ROI2, 4, 5, 1, 1)
        self.checkBox_ROI2 = QtGui.QCheckBox(self.centralWidget)
        self.checkBox_ROI2.setChecked(True)
        self.checkBox_ROI2.setObjectName(_fromUtf8("checkBox_ROI2"))
        self.gridLayout_3.addWidget(self.checkBox_ROI2, 4, 7, 1, 1)
        self.checkBox_showLabel = QtGui.QCheckBox(self.centralWidget)
        self.checkBox_showLabel.setChecked(True)
        self.checkBox_showLabel.setObjectName(_fromUtf8("checkBox_showLabel"))
        self.gridLayout_3.addWidget(self.checkBox_showLabel, 4, 0, 1, 1)
        self.radioButton_ROI1 = QtGui.QRadioButton(self.centralWidget)
        self.radioButton_ROI1.setText(_fromUtf8(""))
        self.radioButton_ROI1.setChecked(True)
        self.radioButton_ROI1.setObjectName(_fromUtf8("radioButton_ROI1"))
        self.gridLayout_3.addWidget(self.radioButton_ROI1, 1, 5, 1, 1)
        self.comboBox_ROI2 = QtGui.QComboBox(self.centralWidget)
        self.comboBox_ROI2.setEditable(True)
        self.comboBox_ROI2.setObjectName(_fromUtf8("comboBox_ROI2"))
        self.gridLayout_3.addWidget(self.comboBox_ROI2, 4, 6, 1, 1)
        self.wormCanvas1 = QtGui.QLabel(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Ignored,
            QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.wormCanvas1.sizePolicy().hasHeightForWidth())
        self.wormCanvas1.setSizePolicy(sizePolicy)
        self.wormCanvas1.setFrameShape(QtGui.QFrame.Box)
        self.wormCanvas1.setFrameShadow(QtGui.QFrame.Sunken)
        self.wormCanvas1.setText(_fromUtf8(""))
        self.wormCanvas1.setObjectName(_fromUtf8("wormCanvas1"))
        self.gridLayout_3.addWidget(self.wormCanvas1, 0, 5, 1, 6)
        self.pushButton_ROI1_RW = QtGui.QPushButton(self.centralWidget)
        self.pushButton_ROI1_RW.setObjectName(_fromUtf8("pushButton_ROI1_RW"))
        self.gridLayout_3.addWidget(self.pushButton_ROI1_RW, 1, 8, 1, 1)
        self.comboBox_labelType = QtGui.QComboBox(self.centralWidget)
        self.comboBox_labelType.setObjectName(_fromUtf8("comboBox_labelType"))
        self.comboBox_labelType.addItem(_fromUtf8(""))
        self.comboBox_labelType.addItem(_fromUtf8(""))
        self.gridLayout_3.addWidget(self.comboBox_labelType, 4, 1, 1, 1)
        spacerItem = QtGui.QSpacerItem(
            600, 20, QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem, 4, 3, 1, 1)
        self.imageCanvas = QtGui.QLabel(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Ignored,
            QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.imageCanvas.sizePolicy().hasHeightForWidth())
        self.imageCanvas.setSizePolicy(sizePolicy)
        self.imageCanvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.imageCanvas.setFrameShape(QtGui.QFrame.Box)
        self.imageCanvas.setFrameShadow(QtGui.QFrame.Sunken)
        self.imageCanvas.setText(_fromUtf8(""))
        self.imageCanvas.setObjectName(_fromUtf8("imageCanvas"))
        self.gridLayout_3.addWidget(self.imageCanvas, 0, 0, 3, 5)
        self.pushButton_feats = QtGui.QPushButton(self.centralWidget)
        self.pushButton_feats.setObjectName(_fromUtf8("pushButton_feats"))
        self.gridLayout_3.addWidget(self.pushButton_feats, 4, 4, 1, 1)
        self.pushButton_ROI2_FF = QtGui.QPushButton(self.centralWidget)
        self.pushButton_ROI2_FF.setObjectName(_fromUtf8("pushButton_ROI2_FF"))
        self.gridLayout_3.addWidget(self.pushButton_ROI2_FF, 4, 9, 1, 1)
        self.pushButton_ROI2_RW = QtGui.QPushButton(self.centralWidget)
        self.pushButton_ROI2_RW.setObjectName(_fromUtf8("pushButton_ROI2_RW"))
        self.gridLayout_3.addWidget(self.pushButton_ROI2_RW, 4, 8, 1, 1)
        self.pushButton_ROI1_FF = QtGui.QPushButton(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Maximum,
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_ROI1_FF.sizePolicy().hasHeightForWidth())
        self.pushButton_ROI1_FF.setSizePolicy(sizePolicy)
        self.pushButton_ROI1_FF.setObjectName(_fromUtf8("pushButton_ROI1_FF"))
        self.gridLayout_3.addWidget(self.pushButton_ROI1_FF, 1, 9, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(
            10, 1000, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        self.gridLayout_4.addItem(spacerItem1, 0, 1, 1, 1)
        ImageViewer.setCentralWidget(self.centralWidget)
        self.menuBar = QtGui.QMenuBar(ImageViewer)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 837, 22))
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
        self.pushButton_video.setText(
            _translate(
                "ImageViewer",
                "Select Video File",
                None))
        self.label_frame.setText(_translate("ImageViewer", "Frame", None))
        self.pushButton_skel.setText(
            _translate(
                "ImageViewer",
                "Select Skeletons File",
                None))
        self.playButton.setText(_translate("ImageViewer", "Play", None))
        self.label_fps.setText(_translate("ImageViewer", "FPS display", None))
        self.comboBox_h5path.setItemText(
            0, _translate("ImageViewer", "/mask", None))
        self.comboBox_h5path.setItemText(
            1, _translate("ImageViewer", "/full_data", None))
        self.label_step.setText(_translate("ImageViewer", "Step Size", None))
        self.pushButton_join.setText(
            _translate(
                "ImageViewer",
                "Join Trajectory",
                None))
        self.pushButton_split.setText(
            _translate(
                "ImageViewer",
                "Split Trajectory",
                None))
        self.pushButton_save.setText(_translate("ImageViewer", "SAVE", None))
        self.pushButton_U.setText(_translate("ImageViewer", "Undefined", None))
        self.pushButton_W.setText(
            _translate(
                "ImageViewer",
                "Single Worms",
                None))
        self.pushButton_WS.setText(
            _translate(
                "ImageViewer",
                "Worm Cluster",
                None))
        self.pushButton_B.setText(_translate("ImageViewer", "Bad", None))
        self.pushButton_h5groups.setText(
            _translate("ImageViewer", "Update Groups", None))
        self.checkBox_ROI1.setText(_translate("ImageViewer", "Skeleton", None))
        self.checkBox_ROI2.setText(_translate("ImageViewer", "Skeleton", None))
        self.checkBox_showLabel.setText(
            _translate("ImageViewer", "Show Labels", None))
        self.pushButton_ROI1_RW.setText(_translate("ImageViewer", "<<", None))
        self.comboBox_labelType.setItemText(
            0, _translate("ImageViewer", "Manual", None))
        self.comboBox_labelType.setItemText(
            1, _translate("ImageViewer", "Auto", None))
        self.pushButton_feats.setText(
            _translate(
                "ImageViewer",
                "Calc Individual Feat",
                None))
        self.pushButton_ROI2_FF.setText(_translate("ImageViewer", ">>", None))
        self.pushButton_ROI2_RW.setText(_translate("ImageViewer", "<<", None))
        self.pushButton_ROI1_FF.setText(_translate("ImageViewer", ">>", None))
        self.toolBar.setWindowTitle(_translate("ImageViewer", "toolBar", None))
