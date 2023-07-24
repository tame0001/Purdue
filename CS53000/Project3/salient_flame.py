#!/usr/bin/env python

from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QSlider, QGridLayout, QLabel, QPushButton, QTextEdit
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import Qt
import sys
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import os
import argparse


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName('The Main Window')
        MainWindow.setWindowTitle('')

        self.centralWidget = QWidget(MainWindow)

        self.gridlayout = QGridLayout(self.centralWidget)
        self.vtkWidget = QVTKRenderWindowInteractor(self.centralWidget)

        self.gridlayout.addWidget(self.vtkWidget, 0, 0, 4, 4)
        MainWindow.setCentralWidget(self.centralWidget)


class PyQtDemo(QMainWindow):

    def __init__(self, args, parent=None):
        QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.params = [
            {
                'isovalue': 700,
                'rgb': (0.1, 0.3, 0.9),
                'opacity': 0.03
            },
            {
                'isovalue': 13000,
                'rgb': (1.0, 1.0, 1.0),
                'opacity': 0.1
            },
            {
                'isovalue': 42000,
                'rgb': (1.0, 1.0, 1.0),
                'opacity': 0.15
            },
            {
                'isovalue': 53000,
                'rgb': (0.9, 0.2, 0.1),
                'opacity': 0.8
            },
        ]

        self.reader = vtk.vtkXMLImageDataReader()
        self.reader.SetFileName(args.data)
        self.reader.Update()

        self.renderer = vtk.vtkRenderer()

        for param in self.params:

            iso = vtk.vtkContourFilter()
            iso.SetInputConnection(self.reader.GetOutputPort())
            iso.SetValue(0, param['isovalue'])

            normal = vtk.vtkPolyDataNormals()
            normal.SetInputConnection(iso.GetOutputPort())
            normal.SetFeatureAngle(60.0)

            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputConnection(normal.GetOutputPort())
            mapper.ScalarVisibilityOff()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetDiffuseColor(param['rgb'])
            actor.GetProperty().SetOpacity(param['opacity'])

            self.renderer.AddActor(actor)

        self.renderer.GetActiveCamera().SetPosition(454.45, 882.71, 1046.44)
        self.renderer.GetActiveCamera().SetFocalPoint(239.0, 359.88, 59.0)
        self.renderer.GetActiveCamera().SetViewUp(-0.88, -0.32, 0.36)
        self.renderer.GetActiveCamera().SetClippingRange(786.76, 1582.64)

        # self.renderer.GetActiveCamera().SetPosition(163.22, 329.13, -465.50)
        # self.renderer.GetActiveCamera().SetFocalPoint(239.0, 359.88, 59.0)
        # self.renderer.GetActiveCamera().SetViewUp(-0.92, -0.35, 0.15)
        # self.renderer.GetActiveCamera().SetClippingRange(324.29, 791.99)

        self.renderer.SetBackground(0.5, 0.5, 0.5)

        self.ui.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.ui.vtkWidget.GetRenderWindow().GetInteractor()


def main():
    global window
    global args
    global renderer

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('data', metavar='data',
                        type=str, help='Input data file')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = PyQtDemo(args)
    window.ui.vtkWidget.GetRenderWindow().SetSize(800, 800)
    window.show()
    window.setWindowState(Qt.WindowMaximized)
    window.interactor.Initialize()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
