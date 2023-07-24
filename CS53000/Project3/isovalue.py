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

        self.slider_isovalue = QSlider()

        self.gridlayout.addWidget(self.vtkWidget, 0, 0, 4, 4)
        self.gridlayout.addWidget(QLabel("Iso Value"), 4, 0, 1, 1)
        self.gridlayout.addWidget(self.slider_isovalue, 4, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralWidget)


class PyQtDemo(QMainWindow):

    def __init__(self, args, parent=None):
        QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.starting_isovalue = 13000
        self.scale = 250

        self.reader = vtk.vtkXMLImageDataReader()
        self.reader.SetFileName(args.data)
        self.reader.Update()
        self.scalar_range = self.reader.GetOutput().GetPointData().GetArray(0).GetRange()

        self.isovalue_factor = (
            self.scalar_range[1] - self.scalar_range[0]) / self.scale
        self.isovalue = self.starting_isovalue / self.isovalue_factor

        self.iso = vtk.vtkContourFilter()
        self.iso.SetInputConnection(self.reader.GetOutputPort())
        self.iso.SetValue(0, self.isovalue * self.isovalue_factor)

        self.normal = vtk.vtkPolyDataNormals()
        self.normal.SetInputConnection(self.iso.GetOutputPort())
        self.normal.SetFeatureAngle(60.0)

        self.iso_mapper = vtk.vtkDataSetMapper()
        self.iso_mapper.SetInputConnection(self.normal.GetOutputPort())
        self.iso_mapper.ScalarVisibilityOn()
        self.iso_mapper.SetScalarRange(self.scalar_range)

        self.iso_actor = vtk.vtkActor()
        self.iso_actor.SetMapper(self.iso_mapper)

        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.iso_actor)
        self.renderer.SetBackground(0.5, 0.5, 0.5)
        self.renderer.ResetCamera()
        # self.renderer.GetActiveCamera().Azimuth(180)
        # self.renderer.GetActiveCamera().Elevation(-90)

        # # Head 1
        # self.renderer.GetActiveCamera().SetPosition(-381.54, -594.56, -52.30)
        # self.renderer.GetActiveCamera().SetFocalPoint(134.47, 138.27, 146.10)
        # self.renderer.GetActiveCamera().SetViewUp(0.20, 0.13, -0.97)
        # self.renderer.GetActiveCamera().SetClippingRange(489.684, 1460.58)

        # # Head 2
        # self.renderer.GetActiveCamera().SetPosition(538.51, -339.47, 168.34)
        # self.renderer.GetActiveCamera().SetFocalPoint(134.47, 138.48, 146.36)
        # self.renderer.GetActiveCamera().SetViewUp(-0.02, -0.06, -0.99)
        # self.renderer.GetActiveCamera().SetClippingRange(258.61, 1100.50)

        # # Flame 1
        self.renderer.GetActiveCamera().SetPosition(454.45, 882.71, 1046.44)
        self.renderer.GetActiveCamera().SetFocalPoint(239.0, 359.88, 59.0)
        self.renderer.GetActiveCamera().SetViewUp(-0.88, -0.32, 0.36)
        self.renderer.GetActiveCamera().SetClippingRange(786.76, 1582.64)
        
        # # Flame 2
        # self.renderer.GetActiveCamera().SetPosition(163.22, 329.13, -465.50)
        # self.renderer.GetActiveCamera().SetFocalPoint(239.0, 359.88, 59.0)
        # self.renderer.GetActiveCamera().SetViewUp(-0.92, -0.35, 0.15)
        # self.renderer.GetActiveCamera().SetClippingRange(324.29, 791.99)

        # self.renderer.ResetCameraClippingRange()

        self.ui.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.ui.vtkWidget.GetRenderWindow().GetInteractor()
        self.interactor.AddObserver("KeyPressEvent", self.key_pressed_callback)

        def slider_setup(slider, val, bounds, interv):
            slider.setOrientation(QtCore.Qt.Horizontal)
            slider.setValue(float(val))
            slider.setTracking(False)
            slider.setTickInterval(interv)
            slider.setTickPosition(QSlider.TicksAbove)
            slider.setRange(bounds[0], bounds[1])

        slider_setup(self.ui.slider_isovalue, self.isovalue, [0, self.scale], 1)

    def isovalue_callback(self, val):
        self.isovalue = val
        self.iso.SetValue(0, self.isovalue * self.isovalue_factor)
        self.ui.vtkWidget.GetRenderWindow().Render()

    def print_camera_settings(self):
        camera = self.renderer.GetActiveCamera()
        print("Camera settings:")
        print("  * position:        %s" % (camera.GetPosition(),))
        print("  * focal point:     %s" % (camera.GetFocalPoint(),))
        print("  * up vector:       %s" % (camera.GetViewUp(),))
        print("  * clipping range:  %s" % (camera.GetClippingRange(),))

    def print_isovalue(self):
        print("  * isovalue:  %s" % (self.isovalue * self.isovalue_factor))

    def key_pressed_callback(self, obj, event):

        key = obj.GetKeySym()
        if key == "c":
            self.print_camera_settings()
        elif key == "v":
            self.print_isovalue()
        elif key == "q":
            sys.exit()


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

    window.ui.slider_isovalue.valueChanged.connect(window.isovalue_callback)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
