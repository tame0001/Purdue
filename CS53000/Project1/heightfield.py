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
        MainWindow.setWindowTitle('Height Field')

        self.centralWidget = QWidget(MainWindow)

        self.gridlayout = QGridLayout(self.centralWidget)
        self.vtkWidget = QVTKRenderWindowInteractor(self.centralWidget)

        self.slider_factor = QSlider()

        self.gridlayout.addWidget(self.vtkWidget, 0, 0, 4, 4)
        self.gridlayout.addWidget(QLabel("Scale Factor"), 4, 0, 1, 1)
        self.gridlayout.addWidget(self.slider_factor, 4, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralWidget)

class PyQtDemo(QMainWindow):

    def __init__(self, elevation_file, satellite_file, parent = None):
        QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.scalar_factor = 50
        
        # Pipe for hieght data
        self.reader = vtk.vtkXMLImageDataReader()
        self.reader.SetFileName(elevation_file)
        
        self.geometry = vtk.vtkImageDataGeometryFilter()
        self.geometry.SetInputConnection(self.reader.GetOutputPort())
        
        self.warp = vtk.vtkWarpScalar()
        self.warp.SetInputConnection(self.geometry.GetOutputPort())
        self.warp.SetScaleFactor(-1*self.scalar_factor)

        self.mapper = vtk.vtkDataSetMapper()
        self.mapper.SetInputConnection(self.warp.GetOutputPort())
        self.mapper.ScalarVisibilityOff()
        
        # Pipe for texture data
        self.jpeg_reader = vtk.vtkJPEGReader()
        self.jpeg_reader.SetFileName(satellite_file)
        self.jpeg_reader.Update()
        
        self.atext = vtk.vtkTexture()
        self.atext.SetInputConnection(self.jpeg_reader.GetOutputPort())
        self.atext.InterpolateOn()

        # Feed into actor   
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.SetTexture(self.atext)
        
        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.actor)
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().Azimuth(180)
        self.renderer.GetActiveCamera().Roll(180)
        
        self.ui.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.ui.vtkWidget.GetRenderWindow().GetInteractor()

        def slider_setup(slider, val, bounds, interv):
            slider.setOrientation(QtCore.Qt.Horizontal)
            slider.setValue(float(val))
            slider.setTracking(False)
            slider.setTickInterval(interv)
            slider.setTickPosition(QSlider.TicksAbove)
            slider.setRange(bounds[0], bounds[1])

        slider_setup(self.ui.slider_factor, self.scalar_factor, [0, 100], 2)
 

    def factor_callback(self, val):
        self.scalar_factor = val
        self.warp.SetScaleFactor(-1*self.scalar_factor)
        self.ui.vtkWidget.GetRenderWindow().Render()


if __name__=="__main__":
    
    
    app = QApplication(sys.argv)
    window = PyQtDemo(elevation_file=sys.argv[1], satellite_file=sys.argv[2])
    window.ui.vtkWidget.GetRenderWindow().SetSize(800, 800)
    window.show()
    window.setWindowState(Qt.WindowMaximized)
    window.interactor.Initialize() 

    window.ui.slider_factor.valueChanged.connect(window.factor_callback)
    sys.exit(app.exec_())
