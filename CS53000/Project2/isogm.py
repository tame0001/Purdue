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
        MainWindow.setCentralWidget(self.centralWidget)


class PyQtDemo(QMainWindow):

    def __init__(self, args, parent=None):
        QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.xclipper_value_multi = 2
        self.yclipper_value_multi = 2.5
        self.zclipper_value_multi = 2.8

        self.isovals = []

        with open(args.isoval) as f:
            lines = f.readlines()
            for line in lines:
                isoval = line.rstrip().split(' ')
                try:
                    self.isovals.append(float(isoval[0]))
                except ValueError:
                    pass

        self.color_tf = vtk.vtkColorTransferFunction()
        if args.cmap:
            with open(args.cmap) as f:
                lines = f.readlines()
            
            for line in lines:
                if line[0] == '#':
                    continue
                colors = line.rstrip().split(' ')
                rgb_point = []
                for color in colors:
                    try: 
                        rgb_point.append(float(color))
                    except ValueError:
                        pass
                
                try: 
                    self.color_tf.AddRGBPoint(rgb_point[0],
                                              rgb_point[1],
                                              rgb_point[2],
                                              rgb_point[3])
                except IndexError:
                    pass

        else:
            self.color_tf.AddRGBPoint(25000, 1, 0.8, 0.7) # Skin
            self.color_tf.AddRGBPoint(55000, 1, 0.8, 0.7) # Skin
            self.color_tf.AddRGBPoint(0, 1, 0.2, 0.2) # Muscle
            self.color_tf.AddRGBPoint(20000, 1, 0.2, 0.2) # Muscle
            self.color_tf.AddRGBPoint(60000, 1, 1, 1) # Bone

        self.reader_gradient = vtk.vtkXMLImageDataReader()
        self.reader_gradient.SetFileName(args.gradmag)
        self.reader_gradient.Update()

        self.reader = vtk.vtkXMLImageDataReader()
        self.reader.SetFileName(args.data)
        self.reader.Update()
        self.scalar_range = self.reader_gradient.GetOutput().GetPointData().GetArray(0).GetRange()

        self.xclipper_value = args.clip[0] / self.xclipper_value_multi
        self.yclipper_value = args.clip[1] / self.yclipper_value_multi
        self.zclipper_value = args.clip[2] / self.zclipper_value_multi

        self.xplane = vtk.vtkPlane()
        self.xplane.SetOrigin(self.xclipper_value * self.xclipper_value_multi, 0, 0)
        self.xplane.SetNormal(1, 0, 0)  

        self.yplane = vtk.vtkPlane()
        self.yplane.SetOrigin(0, self.yclipper_value * self.yclipper_value_multi, 0)
        self.yplane.SetNormal(0, 1, 0) 

        self.zplane = vtk.vtkPlane()
        self.zplane.SetOrigin(0, 0, self.zclipper_value * self.zclipper_value_multi)
        self.zplane.SetNormal(0, 0, 1)

        self.renderer = vtk.vtkRenderer()

        for isoval in self.isovals:
            iso = vtk.vtkContourFilter()
            iso.SetInputConnection(self.reader.GetOutputPort())
            iso.SetValue(0, isoval)

            probe = vtk.vtkProbeFilter()
            probe.SetInputConnection(iso.GetOutputPort())
            probe.SetSourceConnection(self.reader_gradient.GetOutputPort())
        
            normal = vtk.vtkPolyDataNormals()
            normal.SetInputConnection(probe.GetOutputPort())
            normal.SetFeatureAngle(60.0)
            
            xclipper = vtk.vtkClipPolyData()
            xclipper.SetInputConnection(normal.GetOutputPort())
            xclipper.SetClipFunction(self.xplane)
            xclipper.GenerateClipScalarsOff()
            xclipper.GenerateClippedOutputOn()
             
            yclipper = vtk.vtkClipPolyData()
            yclipper.SetInputConnection(xclipper.GetOutputPort())
            yclipper.SetClipFunction(self.yplane)
            yclipper.GenerateClipScalarsOff()
            yclipper.GenerateClippedOutputOn()     

            zclipper = vtk.vtkClipPolyData()
            zclipper.SetInputConnection(yclipper.GetOutputPort())
            zclipper.SetClipFunction(self.zplane)
            zclipper.GenerateClipScalarsOff()
            zclipper.GenerateClippedOutputOn()
   
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputConnection(zclipper.GetOutputPort())
            mapper.ScalarVisibilityOn()
            mapper.SetScalarRange(self.scalar_range)
            mapper.SetLookupTable(self.color_tf)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            self.renderer.AddActor(actor)

        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().Azimuth(180)
        self.renderer.GetActiveCamera().Elevation(-90)
        self.renderer.ResetCameraClippingRange()

        self.ui.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.ui.vtkWidget.GetRenderWindow().GetInteractor()

        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetOrientationToHorizontal()
        self.scalar_bar.SetLookupTable(self.color_tf)

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
    parser.add_argument('data', metavar='data', type=str, help='CT data file')
    parser.add_argument('gradmag', type=str, metavar='gradmag', help='Gradient  data file')
    parser.add_argument('isoval', type=str, metavar='isoval', help='Iso Value')
    parser.add_argument('--cmap', type=str, metavar='colors',help='Color function')
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
    sys.exit(app.exec_())
