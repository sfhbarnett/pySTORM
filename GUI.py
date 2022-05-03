#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:41:26 2022

@author: sbarnett
"""

import sys
import matplotlib

matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtGui, QtWidgets
from superqt import QRangeSlider

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import tifffile


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        path = '/Users/sbarnett/Documents/PIVData/withDoxy_200_1DC.tif'
        self.imstack = tiffstack(path)
        self.plothandle = self.sc.axes.imshow(self.imstack.getimage(0))
        self.plothandle.set_cmap('gray')

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.

        vlayout2 = QtWidgets.QHBoxLayout()
        canvaslayout = QtWidgets.QVBoxLayout()
        canvaslayout.addWidget(self.sc)

        stacksliderholder = QtWidgets.QHBoxLayout()
        self.stackslider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.stackslider.setRange(0, self.imstack.nfiles - 1)
        self.stackslider.valueChanged.connect(self.move_through_stack)
        stacksliderholder.addWidget(self.stackslider)
        self.label = QtWidgets.QLabel('0', self)
        self.label.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.label.setMinimumWidth(20)
        stacksliderholder.addSpacing(1)
        stacksliderholder.addWidget(self.label)
        canvaslayout.addLayout(stacksliderholder)

        vlayout2.addLayout(canvaslayout)

        self.qtr = QRangeSlider(QtCore.Qt.Orientation.Vertical)
        self.qtr.setValue((11, 33))
        vlayout2.addWidget(self.qtr)


        vlayout1 = QtWidgets.QVBoxLayout()
        self.btn = QtWidgets.QPushButton('Open File')
        self.btn.clicked.connect(self.getFile)
        vlayout1.addWidget(self.btn)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addLayout(vlayout1)
        hbox.addLayout(vlayout2)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(hbox)
        self.setCentralWidget(widget)
        self.setGeometry(100,100,1200,600)

        self.close()

    def getFile(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            return fileName

    def move_through_stack(self,value):
        self.plothandle.set_data(self.imstack.getimage(value))
        #self.sc.fig.canvas.flush_events()
        self.sc.fig.canvas.draw()
        #self.sc.fig.canvas.draw_idle()
        self.label.setText(str(value))


class tiffstack():

    def __init__(self,pathname):
        self.ims = tifffile.TiffFile(pathname)
        self.nfiles = len(self.ims.pages)

    def getimage(self,index):
        return self.ims.pages[index].asarray()


def main():
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    app.setQuitOnLastWindowClosed(True)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    m = main()


