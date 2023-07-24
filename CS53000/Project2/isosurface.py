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

        self.slider_isovalue = QSlider()
        self.slider_xclipper = QSlider()
        self.slider_yclipper = QSlider()
        self.slider_zclipper = QSlider()

        self.gridlayout.addWidget(self.vtkWidget, 0, 0, 4, 6)
        self.gridlayout.addWidget(QLabel("X-Clipper"), 4, 0, 1, 1)
        self.gridlayout.addWidget(self.slider_xclipper, 4, 1, 1, 1)
        self.gridlayout.addWidget(QLabel("Y-Clipper"), 4, 2, 1, 1)
        self.gridlayout.addWidget(self.slider_yclipper, 4, 3, 1, 1)
        self.gridlayout.addWidget(QLabel("Z-Clipper"), 4, 4, 1, 1)
        self.gridlayout.addWidget(self.slider_zclipper, 4, 5, 1, 1)
        self.gridlayout.addWidget(QLabel("Iso Value"), 5, 0, 1, 1)
        self.gridlayout.addWidget(self.slider_isovalue, 5, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralWidget)


class PyQtDemo(QMainWindow):

    def __init__(self, args, parent=None):
        QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.xclipper_value_multi = 2
        self.yclipper_value_multi = 2.5
        self.zclipper_value_multi = 2.8

        self.lut = vtk.vtkLookupTable()
        self.lut.Build()

        self.reader = vtk.vtkXMLImageDataReader()
        self.reader.SetFileName(args.data)
        self.reader.Update()
        self.scalar_range = self.reader.GetOutput().GetPointData().GetArray(0).GetRange()

        self.isovalue_factor = (self.scalar_range[1] - self.scalar_range[0]) / 100
        self.isovalue = args.val / self.isovalue_factor
        self.xclipper_value = args.clip[0] / self.xclipper_value_multi
        self.yclipper_value = args.clip[1] / self.yclipper_value_multi
        self.zclipper_value = args.clip[2] / self.zclipper_value_multi

        self.iso = vtk.vtkContourFilter()
        self.iso.SetInputConnection(self.reader.GetOutputPort())
        self.iso.SetValue(0, self.isovalue * self.isovalue_factor)
        
        self.normal = vtk.vtkPolyDataNormals()
        self.normal.SetInputConnection(self.iso.GetOutputPort())
        self.normal.SetFeatureAngle(60.0)
        
        self.xplane = vtk.vtkPlane()
        self.xplane.SetOrigin(self.xclipper_value * self.xclipper_value_multi, 0, 0)
        self.xplane.SetNormal(1, 0, 0)      
        self.xclipper = vtk.vtkClipPolyData()
        self.xclipper.SetInputConnection(self.normal.GetOutputPort())
        self.xclipper.SetClipFunction(self.xplane)
        self.xclipper.GenerateClipScalarsOff()
        self.xclipper.GenerateClippedOutputOn()
        
        self.yplane = vtk.vtkPlane()
        self.yplane.SetOrigin(0, self.yclipper_value * self.yclipper_value_multi, 0)
        self.yplane.SetNormal(0, 1, 0)      
        self.yclipper = vtk.vtkClipPolyData()
        self.yclipper.SetInputConnection(self.xclipper.GetOutputPort())
        self.yclipper.SetClipFunction(self.yplane)
        self.yclipper.GenerateClipScalarsOff()
        self.yclipper.GenerateClippedOutputOn()

        self.zplane = vtk.vtkPlane()
        self.zplane.SetOrigin(0, 0, self.zclipper_value * self.zclipper_value_multi)
        self.zplane.SetNormal(0, 0, 1)      
        self.zclipper = vtk.vtkClipPolyData()
        self.zclipper.SetInputConnection(self.yclipper.GetOutputPort())
        self.zclipper.SetClipFunction(self.zplane)
        self.zclipper.GenerateClipScalarsOff()
        self.zclipper.GenerateClippedOutputOn()
   
        self.iso_mapper = vtk.vtkDataSetMapper()
        self.iso_mapper.SetInputConnection(self.zclipper.GetOutputPort())
        self.iso_mapper.ScalarVisibilityOn()
        self.iso_mapper.SetScalarRange(self.scalar_range)
        self.iso_mapper.SetLookupTable(self.lut)

        self.iso_actor = vtk.vtkActor()
        self.iso_actor.SetMapper(self.iso_mapper)

        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.iso_actor)
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().Azimuth(180)
        self.renderer.GetActiveCamera().Elevation(-90)
        self.renderer.ResetCameraClippingRange()

        self.ui.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.ui.vtkWidget.GetRenderWindow().GetInteractor()

        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetOrientationToHorizontal()
        self.scalar_bar.SetLookupTable(self.lut)

        self.scalar_bar_widget = vtk.vtkScalarBarWidget()
        self.scalar_bar_widget.SetInteractor(self.interactor)
        self.scalar_bar_widget.SetScalarBarActor(self.scalar_bar)
        self.scalar_bar_widget.On()

        def slider_setup(slider, val, bounds, interv):
            slider.setOrientation(QtCore.Qt.Horizontal)
            slider.setValue(float(val))
            slider.setTracking(False)
            slider.setTickInterval(interv)
            slider.setTickPosition(QSlider.TicksAbove)
            slider.setRange(bounds[0], bounds[1])

        slider_setup(self.ui.slider_xclipper, self.xclipper_value, [0, 100], 10)
        slider_setup(self.ui.slider_yclipper, self.yclipper_value, [0, 100], 10)
        slider_setup(self.ui.slider_zclipper, self.zclipper_value, [0, 100], 10)
        slider_setup(self.ui.slider_isovalue, self.isovalue, [0, 100], 10)

    def isovalue_callback(self, val):
        self.isovalue = val
        self.iso.SetValue(0, self.isovalue * self.isovalue_factor)
        self.ui.vtkWidget.GetRenderWindow().Render()

    def xclipper_callback(self, val):
        self.xclipper_value = val
        self.xplane.SetOrigin(self.xclipper_value * self.xclipper_value_multi, 0, 0)
        self.ui.vtkWidget.GetRenderWindow().Render()

    def yclipper_callback(self, val):
        self.yclipper_value = val
        self.yplane.SetOrigin(0, self.yclipper_value * self.yclipper_value_multi, 0)
        self.ui.vtkWidget.GetRenderWindow().Render()

    def zclipper_callback(self, val):
        self.zclipper_value = val
        self.zplane.SetOrigin(0, 0, self.zclipper_value * self.zclipper_value_multi)
        self.ui.vtkWidget.GetRenderWindow().Render()


if __name__ == "__main__":

    global args

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('data', metavar='data', type=str, help='Input data file')
    parser.add_argument('--val', type=int, metavar='isovalue', help='Initial isovalue', default=1700)
    parser.add_argument('--clip', type=int, metavar=('x', 'y', 'z'), nargs=3, help='Initail clipper x y z', default=[0, 0, 0])

    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = PyQtDemo(args)
    window.ui.vtkWidget.GetRenderWindow().SetSize(800, 800)
    window.show()
    window.setWindowState(Qt.WindowMaximized)
    window.interactor.Initialize()

    window.ui.slider_xclipper.valueChanged.connect(window.xclipper_callback)
    window.ui.slider_yclipper.valueChanged.connect(window.yclipper_callback)
    window.ui.slider_zclipper.valueChanged.connect(window.zclipper_callback)
    window.ui.slider_isovalue.valueChanged.connect(window.isovalue_callback)
    sys.exit(app.exec_())
