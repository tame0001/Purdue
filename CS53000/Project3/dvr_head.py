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
# Skin
opacityTransferFunction.AddPoint(550, 0.0)
opacityTransferFunction.AddPoint(700, 0.3)
opacityTransferFunction.AddPoint(800, 0.0)
# Muscle
opacityTransferFunction.AddPoint(980, 0.0)
opacityTransferFunction.AddPoint(1020, 0.1)
opacityTransferFunction.AddPoint(1070, 0.0)
# Bone
opacityTransferFunction.AddPoint(1125, 0.0)
opacityTransferFunction.AddPoint(1300, 1.0)
opacityTransferFunction.AddPoint(1600, 0.0)
# Teeth
opacityTransferFunction.AddPoint(2700, 0.0)
opacityTransferFunction.AddPoint(2900, 1.0)
opacityTransferFunction.AddPoint(4000, 0.0)

colorTransferFunction = vtk.vtkColorTransferFunction()
colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
colorTransferFunction.AddRGBPoint(550.0, 1.0, 0.8, 0.7)
colorTransferFunction.AddRGBPoint(800.0, 1.0, 0.8, 0.7)
colorTransferFunction.AddRGBPoint(980.0, 1.0, 0.2, 0.2)
colorTransferFunction.AddRGBPoint(1070.0, 1.0, 0.2, 0.2)
colorTransferFunction.AddRGBPoint(1125.0, 1.0, 1.0, 1.0)

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

# renderer.GetActiveCamera().Azimuth(180)
# renderer.GetActiveCamera().Elevation(-90)

# renderer.GetActiveCamera().SetPosition(-381.54, -594.56, -52.30)
# renderer.GetActiveCamera().SetFocalPoint(134.47, 138.27, 146.10)
# renderer.GetActiveCamera().SetViewUp(0.20, 0.13, -0.97)
# renderer.GetActiveCamera().SetClippingRange(489.684, 1460.58)

renderer.GetActiveCamera().SetPosition(538.51, -339.47, 168.34)
renderer.GetActiveCamera().SetFocalPoint(134.47, 138.48, 146.36)
renderer.GetActiveCamera().SetViewUp(-0.02, -0.06, -0.99)
renderer.GetActiveCamera().SetClippingRange(258.61, 1100.50)

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
