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
                'rgb': (1.0, 0.8, 0.7),
                'opacity': 0.3
            },
            {
                'isovalue': 1020,
                'rgb': (1.0, 0.2, 0.2),
                'opacity': 0.1
            },
            {
                'isovalue': 1300,
                'rgb': (1.0, 1.0, 1.0),
                'opacity': 1.0
            },
            {
                'isovalue': 2900,
                'rgb': (0.0, 0.0, 1.0),
                'opacity': 1.0
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

        self.renderer.ResetCamera()

        # self.renderer.GetActiveCamera().Azimuth(180)
        # self.renderer.GetActiveCamera().Elevation(-90)

        # self.renderer.GetActiveCamera().SetPosition(-381.54, -594.56, -52.30)
        # self.renderer.GetActiveCamera().SetFocalPoint(134.47, 138.27, 146.10)
        # self.renderer.GetActiveCamera().SetViewUp(0.20, 0.13, -0.97)
        # self.renderer.GetActiveCamera().SetClippingRange(489.684, 1460.58)

        self.renderer.GetActiveCamera().SetPosition(538.51, -339.47, 168.34)
        self.renderer.GetActiveCamera().SetFocalPoint(134.47, 138.48, 146.36)
        self.renderer.GetActiveCamera().SetViewUp(-0.02, -0.06, -0.99)
        self.renderer.GetActiveCamera().SetClippingRange(258.61, 1100.50)

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
