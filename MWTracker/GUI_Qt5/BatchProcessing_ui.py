# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'BatchProcessing.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_BatchProcessing(object):

    def setupUi(self, BatchProcessing):
        BatchProcessing.setObjectName("BatchProcessing")
        BatchProcessing.resize(712, 425)
        self.centralwidget = QtWidgets.QWidget(BatchProcessing)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.lineEdit_videosDir = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_videosDir.setObjectName("lineEdit_videosDir")
        self.gridLayout_2.addWidget(self.lineEdit_videosDir, 0, 2, 1, 1)
        self.pushButton_videosDir = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_videosDir.setObjectName("pushButton_videosDir")
        self.gridLayout_2.addWidget(self.pushButton_videosDir, 0, 1, 1, 1)
        self.lineEdit_tmpDir = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_tmpDir.setObjectName("lineEdit_tmpDir")
        self.gridLayout_2.addWidget(self.lineEdit_tmpDir, 5, 2, 1, 1)
        self.lineEdit_txtFileList = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_txtFileList.setEnabled(True)
        self.lineEdit_txtFileList.setObjectName("lineEdit_txtFileList")
        self.gridLayout_2.addWidget(self.lineEdit_txtFileList, 1, 2, 1, 1)
        self.lineEdit_masksDir = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_masksDir.setObjectName("lineEdit_masksDir")
        self.gridLayout_2.addWidget(self.lineEdit_masksDir, 2, 2, 1, 1)
        self.pushButton_tmpDir = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_tmpDir.setObjectName("pushButton_tmpDir")
        self.gridLayout_2.addWidget(self.pushButton_tmpDir, 5, 1, 1, 1)
        self.pushButton_txtFileList = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_txtFileList.setEnabled(True)
        self.pushButton_txtFileList.setObjectName("pushButton_txtFileList")
        self.gridLayout_2.addWidget(self.pushButton_txtFileList, 1, 1, 1, 1)
        self.pushButton_masksDir = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_masksDir.setObjectName("pushButton_masksDir")
        self.gridLayout_2.addWidget(self.pushButton_masksDir, 2, 1, 1, 1)
        self.lineEdit_resultsDir = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_resultsDir.setObjectName("lineEdit_resultsDir")
        self.gridLayout_2.addWidget(self.lineEdit_resultsDir, 3, 2, 1, 1)
        self.pushButton_paramFile = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_paramFile.setObjectName("pushButton_paramFile")
        self.gridLayout_2.addWidget(self.pushButton_paramFile, 4, 1, 1, 1)
        self.pushButton_resultsDir = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_resultsDir.setObjectName("pushButton_resultsDir")
        self.gridLayout_2.addWidget(self.pushButton_resultsDir, 3, 1, 1, 1)
        self.checkBox_txtFileList = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Maximum,
            QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.checkBox_txtFileList.sizePolicy().hasHeightForWidth())
        self.checkBox_txtFileList.setSizePolicy(sizePolicy)
        self.checkBox_txtFileList.setText("")
        self.checkBox_txtFileList.setObjectName("checkBox_txtFileList")
        self.gridLayout_2.addWidget(self.checkBox_txtFileList, 1, 0, 1, 1)
        self.checkBox_tmpDir = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Maximum,
            QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.checkBox_tmpDir.sizePolicy().hasHeightForWidth())
        self.checkBox_tmpDir.setSizePolicy(sizePolicy)
        self.checkBox_tmpDir.setText("")
        self.checkBox_tmpDir.setObjectName("checkBox_tmpDir")
        self.gridLayout_2.addWidget(self.checkBox_tmpDir, 5, 0, 1, 1)
        self.lineEdit_paramFile = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_paramFile.setObjectName("lineEdit_paramFile")
        self.gridLayout_2.addWidget(self.lineEdit_paramFile, 4, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_2)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.pushButton_start = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_start.setObjectName("pushButton_start")
        self.gridLayout_3.addWidget(self.pushButton_start, 0, 4, 2, 1)
        self.checkBox_isCompress = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_isCompress.setObjectName("checkBox_isCompress")
        self.gridLayout_3.addWidget(self.checkBox_isCompress, 0, 3, 1, 1)
        self.label_numMaxProc = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.label_numMaxProc.sizePolicy().hasHeightForWidth())
        self.label_numMaxProc.setSizePolicy(sizePolicy)
        self.label_numMaxProc.setObjectName("label_numMaxProc")
        self.gridLayout_3.addWidget(self.label_numMaxProc, 2, 3, 1, 1)
        self.checkBox_isTrack = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_isTrack.setObjectName("checkBox_isTrack")
        self.gridLayout_3.addWidget(self.checkBox_isTrack, 1, 3, 1, 1)
        self.checkBox_isSingleWorm = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_isSingleWorm.setObjectName("checkBox_isSingleWorm")
        self.gridLayout_3.addWidget(self.checkBox_isSingleWorm, 0, 0, 1, 2)
        self.spinBox_numMaxProc = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_numMaxProc.setObjectName("spinBox_numMaxProc")
        self.gridLayout_3.addWidget(self.spinBox_numMaxProc, 3, 3, 1, 1)
        self.lineEdit_patternInTrack = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_patternInTrack.setObjectName("lineEdit_patternInTrack")
        self.gridLayout_3.addWidget(self.lineEdit_patternInTrack, 4, 1, 1, 1)
        self.lineEdit_patternExcTrack = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_patternExcTrack.setObjectName("lineEdit_patternExcTrack")
        self.gridLayout_3.addWidget(self.lineEdit_patternExcTrack, 4, 2, 1, 1)
        self.lineEdit_patternExcComp = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_patternExcComp.setObjectName("lineEdit_patternExcComp")
        self.gridLayout_3.addWidget(self.lineEdit_patternExcComp, 3, 2, 1, 1)
        self.lineEdit_patternInComp = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_patternInComp.setObjectName("lineEdit_patternInComp")
        self.gridLayout_3.addWidget(self.lineEdit_patternInComp, 3, 1, 1, 1)
        self.label_track = QtWidgets.QLabel(self.centralwidget)
        self.label_track.setObjectName("label_track")
        self.gridLayout_3.addWidget(self.label_track, 4, 0, 1, 1)
        self.label_comp = QtWidgets.QLabel(self.centralwidget)
        self.label_comp.setObjectName("label_comp")
        self.gridLayout_3.addWidget(self.label_comp, 3, 0, 1, 1)
        self.label_patternIn = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.label_patternIn.sizePolicy().hasHeightForWidth())
        self.label_patternIn.setSizePolicy(sizePolicy)
        self.label_patternIn.setObjectName("label_patternIn")
        self.gridLayout_3.addWidget(self.label_patternIn, 2, 1, 1, 1)
        self.label_patternExc = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.label_patternExc.sizePolicy().hasHeightForWidth())
        self.label_patternExc.setSizePolicy(sizePolicy)
        self.label_patternExc.setObjectName("label_patternExc")
        self.gridLayout_3.addWidget(self.label_patternExc, 2, 2, 1, 1)
        self.checkBox_isCopyVideo = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_isCopyVideo.setObjectName("checkBox_isCopyVideo")
        self.gridLayout_3.addWidget(self.checkBox_isCopyVideo, 1, 0, 1, 2)
        self.horizontalLayout_2.addLayout(self.gridLayout_3)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        BatchProcessing.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(BatchProcessing)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 712, 22))
        self.menubar.setObjectName("menubar")
        BatchProcessing.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(BatchProcessing)
        self.statusbar.setObjectName("statusbar")
        BatchProcessing.setStatusBar(self.statusbar)

        self.retranslateUi(BatchProcessing)
        QtCore.QMetaObject.connectSlotsByName(BatchProcessing)

    def retranslateUi(self, BatchProcessing):
        _translate = QtCore.QCoreApplication.translate
        BatchProcessing.setWindowTitle(
            _translate(
                "BatchProcessing",
                "Batch Processing"))
        self.pushButton_videosDir.setText(_translate(
            "BatchProcessing", "Original Videos Dir"))
        self.pushButton_tmpDir.setText(
            _translate(
                "BatchProcessing",
                "Temporary Dir"))
        self.pushButton_txtFileList.setText(_translate(
            "BatchProcessing", "Individual Files List"))
        self.pushButton_masksDir.setText(_translate(
            "BatchProcessing", "Masked Videos Dir"))
        self.pushButton_paramFile.setText(
            _translate("BatchProcessing", "Parameters File"))
        self.pushButton_resultsDir.setText(_translate(
            "BatchProcessing", "Tracking Results Dir"))
        self.pushButton_start.setText(_translate("BatchProcessing", "START"))
        self.checkBox_isCompress.setText(
            _translate(
                "BatchProcessing",
                "Execute compression"))
        self.label_numMaxProc.setText(
            _translate(
                "BatchProcessing",
                "Maximum number of processes:"))
        self.checkBox_isTrack.setText(
            _translate(
                "BatchProcessing",
                "Execute tracking"))
        self.checkBox_isSingleWorm.setText(
            _translate(
                "BatchProcessing",
                "Is single worm (Shafer Lab)?"))
        self.label_track.setText(_translate("BatchProcessing", "Tracking"))
        self.label_comp.setText(_translate("BatchProcessing", "Compression"))
        self.label_patternIn.setText(
            _translate(
                "BatchProcessing",
                "File pattern to include:"))
        self.label_patternExc.setText(
            _translate(
                "BatchProcessing",
                "File pattern to exclude:"))
        self.checkBox_isCopyVideo.setText(
            _translate(
                "BatchProcessing",
                "Copy original videos to tmp dir"))
