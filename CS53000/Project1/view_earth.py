#!/usr/bin/env python

from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QSlider, QGridLayout, QLabel, QPushButton, QTextEdit
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import Qt
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import argparse
import sys


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName('The Main Window')
        MainWindow.setWindowTitle('Earth')

        self.centralWidget = QWidget(MainWindow)

        self.gridlayout = QGridLayout(self.centralWidget)
        self.vtkWidget = QVTKRenderWindowInteractor(self.centralWidget)

        self.slider_factor = QSlider()
        self.slider_radius = QSlider()

        self.gridlayout.addWidget(self.vtkWidget, 0, 0, 4, 4)
        self.gridlayout.addWidget(QLabel("Radius"), 4, 0, 1, 1)
        self.gridlayout.addWidget(self.slider_radius, 4, 1, 1, 1)
        self.gridlayout.addWidget(QLabel("Scale Factor"), 5, 0, 1, 1)
        self.gridlayout.addWidget(self.slider_factor, 5, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralWidget)


class PyQtDemo(QMainWindow):

    def __init__(self, elevation_file, satellite_file, parent=None):
        QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.scalar_factor = 50
        self.radius = 20

        self.reader = vtk.vtkXMLPolyDataReader()
        self.reader.SetFileName(elevation_file)

        self.warp = vtk.vtkWarpScalar()
        self.warp.SetInputConnection(self.reader.GetOutputPort())
        self.warp.SetScaleFactor(self.scalar_factor)

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.warp.GetOutputPort())
        self.mapper.ScalarVisibilityOff()

        self.jpeg_reader = vtk.vtkJPEGReader()
        self.jpeg_reader.SetFileName(satellite_file)
        self.jpeg_reader.Update()

        self.atext = vtk.vtkTexture()
        self.atext.SetInputConnection(self.jpeg_reader.GetOutputPort())
        self.atext.InterpolateOn()

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.SetTexture(self.atext)

        self.iso = vtk.vtkContourFilter()
        self.iso.SetInputConnection(self.warp.GetOutputPort())
        self.iso.GenerateValues(19, -8000, 10000)

        self.tubes = vtk.vtkTubeFilter()
        self.tubes.SetInputConnection(self.iso.GetOutputPort())
        self.tubes.SetRadius(100*self.radius)

        self.color_tf = vtk.vtkColorTransferFunction()
        self.color_tf.AddRGBPoint(-10000, 0, 0.5, 1) # Blue
        self.color_tf.AddRGBPoint(-1000, 1, 1, 1) # White
        self.color_tf.AddRGBPoint(0, 1, 0, 0) # Red
        self.color_tf.AddRGBPoint(1000, 1, 1, 1) # White
        self.color_tf.AddRGBPoint(8000, 1, 1, 0) # Yellow

        self.iso_mapper = vtk.vtkDataSetMapper()
        self.iso_mapper.SetInputConnection(self.tubes.GetOutputPort())
        self.iso_mapper.ScalarVisibilityOn()
        self.iso_mapper.SetLookupTable(self.color_tf)

        self.iso_actor = vtk.vtkActor()
        self.iso_actor.SetMapper(self.iso_mapper)

        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.actor)
        self.renderer.AddActor(self.iso_actor)
        self.renderer.ResetCamera()

        self.ui.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.ui.vtkWidget.GetRenderWindow().GetInteractor()

        def slider_setup(slider, val, bounds, interv):
            slider.setOrientation(QtCore.Qt.Horizontal)
            slider.setValue(float(val))
            slider.setTracking(False)
            slider.setTickInterval(interv)
            slider.setTickPosition(QSlider.TicksAbove)
            slider.setRange(bounds[0], bounds[1])

        slider_setup(self.ui.slider_radius, self.radius, [10, 50], 2)
        slider_setup(self.ui.slider_factor, self.scalar_factor, [0, 100], 2)

    def factor_callback(self, val):
        self.scalar_factor = val
        self.warp.SetScaleFactor(self.scalar_factor)
        self.ui.vtkWidget.GetRenderWindow().Render()

    def radius_callback(self, val):
        self.radius = val
        self.tubes.SetRadius(100*self.radius)
        self.ui.vtkWidget.GetRenderWindow().Render()


if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = PyQtDemo(elevation_file=sys.argv[1], satellite_file=sys.argv[2])
    window.ui.vtkWidget.GetRenderWindow().SetSize(800, 800)
    window.show()
    window.setWindowState(Qt.WindowMaximized)
    window.interactor.Initialize()

    window.ui.slider_radius.valueChanged.connect(window.radius_callback)
    window.ui.slider_factor.valueChanged.connect(window.factor_callback)
    sys.exit(app.exec_())
