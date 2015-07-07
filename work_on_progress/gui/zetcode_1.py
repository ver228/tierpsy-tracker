#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ZetCode PyQt4 tutorial 

In this example, we create a simple
window in PyQt4.

author: Jan Bodnar
website: zetcode.com 
last edited: October 2011
"""

import sys
from PyQt4 import QtGui, QtCore


class MainProgram(QtGui.QWidget):
    
    def __init__(self):
        super(MainProgram, self).__init__()
        
        self.initUI()
        
        
    def initUI(self):
        QtGui.QToolTip.setFont(QtGui.QFont('SansSerif', 10))
        
        
        btn = QtGui.QPushButton('Button', self)
        btn.setToolTip('This is a <b>QPushButton</b> widget')
        btn.resize(btn.sizeHint())
        btn.move(50, 50)       
        
        
        qbtn = QtGui.QPushButton('Quit', self)
        qbtn.clicked.connect(QtGui.qApp.quit)#QtCore.QCoreApplication.instance().quit)
        qbtn.resize(qbtn.sizeHint())
        qbtn.move(10, 10)  
        
        self.setToolTip('This is a <b>QWidget</b> widget')
        self.resize(300, 300)#, 250, 150)
        self.center()
        
        self.setWindowTitle('Main Program')
        #self.setWindowIcon(QtGui.QIcon('web.png'))        
    
        self.show()


    def closeEvent(self, event):
        
        reply = QtGui.QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QtGui.QMessageBox.Yes | 
            QtGui.QMessageBox.No, QtGui.QMessageBox.No)

        if reply == QtGui.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore() 
            
    def center(self):
        
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

def main():
    
    app = QtGui.QApplication(sys.argv)

    ex = MainProgram()
    #ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()