#!/usr/bin/env python

import vtk
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('data', metavar='data', type=str, help='Input data file')
args = parser.parse_args()

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(args.data)
reader.Update()

opacityTransferFunction = vtk.vtkPiecewiseFunction()
opacityTransferFunction.AddPoint(200, 0.0)
opacityTransferFunction.AddPoint(10000, 0.03)
opacityTransferFunction.AddPoint(42000, 0.15)
opacityTransferFunction.AddPoint(60000, 0.8)
opacityTransferFunction.AddPoint(65000, 0.0)


colorTransferFunction = vtk.vtkColorTransferFunction()
colorTransferFunction.AddRGBPoint(25000, 0.1, 0.3, 0.9)
colorTransferFunction.AddRGBPoint(42000.0, 1.0, 1.0, 1.0)
colorTransferFunction.AddRGBPoint(53000.0, 0.9, 0.2, 0.1)

volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(colorTransferFunction)
volumeProperty.SetScalarOpacity(opacityTransferFunction)
volumeProperty.ShadeOn()
volumeProperty.SetInterpolationTypeToLinear()

volumeMapper = vtk.vtkSmartVolumeMapper()
volumeMapper.SetBlendModeToComposite()
volumeMapper.SetInputConnection(reader.GetOutputPort())

volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

renderer = vtk.vtkRenderer()
renderer.ResetCamera()

renderer.GetActiveCamera().SetPosition(454.45, 882.71, 1046.44)
renderer.GetActiveCamera().SetFocalPoint(239.0, 359.88, 59.0)
renderer.GetActiveCamera().SetViewUp(-0.88, -0.32, 0.36)
renderer.GetActiveCamera().SetClippingRange(786.76, 1582.64)

# renderer.GetActiveCamera().SetPosition(163.22, 329.13, -465.50)
# renderer.GetActiveCamera().SetFocalPoint(239.0, 359.88, 59.0)
# renderer.GetActiveCamera().SetViewUp(-0.92, -0.35, 0.15)
# renderer.GetActiveCamera().SetClippingRange(324.29, 791.99)

renderer.SetBackground(0.5, 0.5, 0.5)


renderer_window = vtk.vtkRenderWindow()
renderer_window.AddRenderer(renderer)
renderer_interactive = vtk.vtkRenderWindowInteractor()
renderer_interactive.SetRenderWindow(renderer_window)

renderer.AddVolume(volume)
renderer_window.SetSize(800, 800)
renderer_window.Render()

renderer_interactive.Initialize()
renderer_window.Render()
renderer_interactive.Start()
