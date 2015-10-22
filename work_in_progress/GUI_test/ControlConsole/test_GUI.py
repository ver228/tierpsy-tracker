#!/usr/bin/env python

from PyQt5.QtCore import (QDir, QIODevice, QFile, QFileInfo, Qt, QTextStream,
        QUrl)
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QComboBox,
        QDialog, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
        QProgressDialog, QPushButton, QSizePolicy, QTableWidget,
        QTableWidgetItem, QMessageBox)

import os
from functools import partial

class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        default_file = "*.avi"
        default_videoDir = '/Users/ajaver/Desktop/Gecko_compressed/Alex_Anderson/Worm_Videos/Locomotion_videos_for_analysis_2015/'
        default_finalDir = '/Users/ajaver/Desktop/Gecko_compressed/Alex_Anderson'
        default_tmpDir = os.path.join(os.path.expanduser("~"), 'Tmp')
        default_paramFile = ''


        findButton = self.createButton("&Add", self.add)
        startButton = self.createButton("&Start", self.start)
        stopButton = self.createButton("&Stop", self.stop)
        delButton = self.createButton("&Delete", self.delete)

        self.fileNameComboBox = self.createComboBox(default_file) 
        fileNameLabel = QLabel("Named:")

        self.videoDirComboBox = self.createComboBox(default_videoDir) #QDir.currentPath())
        videoDirLabel = QLabel("Video Directory:")
        videoDirButton = self.createButton("&Browse...", 
            partial(self.browse, self.videoDirComboBox, 'Find results root directory...'))

        self.finalDirComboBox = self.createComboBox(default_finalDir)
        finalDirLabel = QLabel("Results Root Directory:")
        finalDirButton = self.createButton("&Browse...", 
            partial(self.browse, self.finalDirComboBox, 'Find final results root directory...'))

        self.tmpDirComboBox = self.createComboBox(default_tmpDir)
        tmpDirLabel = QLabel("Temp Files Directory:")
        tmpDirButton = self.createButton("&Browse...", 
            partial(self.browse, self.tmpDirComboBox, 'Find temporary results root directory...'))
        
        self.paramComboBox = self.createComboBox(default_paramFile)
        paramLabel = QLabel("Parameters File:")
        paramButton = self.createButton("&Browse...", 
            partial(self.browse_file, self.paramComboBox, 'Find json file with the parameters data...'))
        

        self.createFilesTable()

        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch()
        
        buttonsLayout.addWidget(stopButton)
        buttonsLayout.addWidget(startButton)
        buttonsLayout.addWidget(delButton)
        buttonsLayout.addWidget(findButton)
        
        mainLayout = QGridLayout()
        mainLayout.addWidget(fileNameLabel, 0, 0)
        mainLayout.addWidget(self.fileNameComboBox, 0, 1, 1, 2)
        
        mainLayout.addWidget(videoDirLabel, 1, 0)
        mainLayout.addWidget(self.videoDirComboBox, 1, 1)
        mainLayout.addWidget(videoDirButton, 1, 2)

        mainLayout.addWidget(finalDirLabel, 2, 0)
        mainLayout.addWidget(self.finalDirComboBox, 2, 1)
        mainLayout.addWidget(finalDirButton, 2, 2)

        mainLayout.addWidget(tmpDirLabel, 3, 0)
        mainLayout.addWidget(self.tmpDirComboBox, 3, 1)
        mainLayout.addWidget(tmpDirButton, 3, 2)

        mainLayout.addWidget(paramLabel, 4, 0)
        mainLayout.addWidget(self.paramComboBox, 4, 1)
        mainLayout.addWidget(paramButton, 4, 2)

        mainLayout.addWidget(self.filesTable, 6, 0, 1, 3)
        #mainLayout.addWidget(self.filesFoundLabel, 5, 0)
        mainLayout.addLayout(buttonsLayout, 7, 0, 1, 3)
        
        self.setLayout(mainLayout)

        self.setWindowTitle("Find Files")
        self.resize(700, 300)

    def browse(self, comboBox, text):
        currentDir = comboBox.currentText() 
        if not currentDir: QDir.currentPath()

        directory = QFileDialog.getExistingDirectory(self, text, currentDir)

        if directory:
            if comboBox.findText(directory) == -1:
                comboBox.addItem(directory)

            comboBox.setCurrentIndex(comboBox.findText(directory))

    def browse_file(self, comboBox, text):
        currentFile = comboBox.currentText() 
        if not currentFile:
            currentDir = QDir.currentPath()
        else:
            currentDir = currentFile.rpartition(os.sep)

        newFile, _ = QFileDialog.getOpenFileName(self, "Find video file", 
            currentDir, "Parameters File (*.json);;All files (*)")

        if newFile:
            if comboBox.findText(newFile) == -1:
                comboBox.addItem(newFile)

            comboBox.setCurrentIndex(comboBox.findText(newFile))

    @staticmethod
    def updateComboBox(comboBox):
        if comboBox.findText(comboBox.currentText()) == -1:
            comboBox.addItem(comboBox.currentText())

    def add(self):
        #self.filesTable.setRowCount(0)
        fileName = self.fileNameComboBox.currentText()
        videoDir = self.videoDirComboBox.currentText()
        finalDir = self.finalDirComboBox.currentText()
        tmpDir = self.tmpDirComboBox.currentText()

        self.updateComboBox(self.fileNameComboBox)
        self.updateComboBox(self.videoDirComboBox)
        self.updateComboBox(self.finalDirComboBox)
        self.updateComboBox(self.tmpDirComboBox)

        

        self.currentDir = QDir(videoDir)
        if not fileName:
            fileName = "*"
        
        files = self.currentDir.entryList([fileName],
                QDir.Files | QDir.NoSymLinks)



        if not files:
            QMessageBox.critical(self, '', 'No valid files in the current directory',
                    QMessageBox.Ok)
            return
        else:
            subdir_base = os.path.split(videoDir)[-1]

            masks_dir = os.path.join(finalDir, 'MaskedVideos', subdir_base) + os.sep
            if not os.path.exists(masks_dir): os.makedirs(masks_dir)
            results_dir = os.path.join(finalDir, 'Results', subdir_base) + os.sep
            if not os.path.exists(results_dir): os.makedirs(results_dir)      
            
            tmp_masks_dir = os.path.join(tmpDir, 'MaskedVideos', subdir_base) + os.sep
            if not os.path.exists(tmp_masks_dir): os.makedirs(tmp_masks_dir)
            tmp_results_dir = os.path.join(tmpDir, 'Results', subdir_base) + os.sep
            if not os.path.exists(tmp_results_dir): os.makedirs(tmp_results_dir)

            

            print(files)
            self.addFiles(files)


    def addFiles(self, files):
        for fn in files:
            file = QFile(self.currentDir.absoluteFilePath(fn))
            size = QFileInfo(file).size()
            

            fileNameItem = QTableWidgetItem(fn)
            fileNameItem.setFlags(fileNameItem.flags() ^ Qt.ItemIsEditable)
            
            sizeItem = QTableWidgetItem("%d KB" % (int((size + 1023) / 1024)))
            sizeItem.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
            sizeItem.setFlags(sizeItem.flags() ^ Qt.ItemIsEditable)

            row = self.filesTable.rowCount()
            self.filesTable.insertRow(row)
            self.filesTable.setItem(row, 1, fileNameItem)
            self.filesTable.setItem(row, 0, sizeItem)
            

        #self.filesFoundLabel.setText("%d file(s) found (Double click on a file to open it)" % len(files))
 
    def start(self):
        pass
    def stop(self):
        pass
    def delete(self):
        pass
 
    def createButton(self, text, member):
        button = QPushButton(text)
        button.clicked.connect(member)
        return button

    def createComboBox(self, text=""):
        comboBox = QComboBox()
        comboBox.setEditable(True)
        comboBox.addItem(text)
        comboBox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        return comboBox

    def createFilesTable(self):
        headerLabels = ("Status", "File Name", "Subdirectory", "Prefix")

        self.filesTable = QTableWidget(0, len(headerLabels))
        self.filesTable.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.filesTable.setHorizontalHeaderLabels(headerLabels)
        self.filesTable.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.filesTable.verticalHeader().hide()
        self.filesTable.setShowGrid(True)

        self.filesTable.cellActivated.connect(self.openFileOfItem)

    def openFileOfItem(self, row, column):
        item = self.filesTable.item(row, 0)
        print(item.text())
        #QDesktopServices.openUrl(QUrl(self.currentDir.absoluteFilePath(item.text())))

if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
