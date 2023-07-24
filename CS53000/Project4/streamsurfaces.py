import vtk
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('tdelta', metavar='tdelta', type=str, help='Vector data file')
parser.add_argument('wing', metavar='wing', type=str, help='Geometry data file')
args = parser.parse_args()

renderer = vtk.vtkRenderer()

reader_tdelta = vtk.vtkStructuredPointsReader()
reader_tdelta.SetFileName(args.tdelta)
reader_tdelta.Update()

arrayCalc = vtk.vtkArrayCalculator()
arrayCalc.SetInputConnection(reader_tdelta.GetOutputPort())
arrayCalc.AddVectorArrayName('velocity')
arrayCalc.SetFunction('mag(velocity)')
arrayCalc.SetResultArrayName('velocityMag')
arrayCalc.Update()

vector_range = arrayCalc.GetOutput().GetPointData().GetArray(1).GetRange()

color_tf = vtk.vtkColorTransferFunction()
color_tf.AddRGBPoint(0, 1, 0, 0)
color_tf.AddRGBPoint(80, 1, 1, 1)
color_tf.AddRGBPoint(160, 0, 0, 1)

# -----------------------------------------------

rake = vtk.vtkLineSource()
rake.SetPoint1(50, -80, 7)
rake.SetPoint2(50, 80, 7)
rake.SetResolution(100)

integ = vtk.vtkRungeKutta4()
streamer = vtk.vtkStreamTracer()
streamer.SetInputConnection(arrayCalc.GetOutputPort())
streamer.SetSourceConnection(rake.GetOutputPort())
streamer.SetMaximumPropagation(600)
streamer.SetInitialIntegrationStep(0.5)
streamer.SetIntegrationDirectionToBoth()
streamer.SetIntegrator(integ)

surface = vtk.vtkRuledSurfaceFilter()
surface.SetInputConnection(streamer.GetOutputPort())
surface.SetOffset(0)
surface.SetOnRatio(2)
surface.PassLinesOn()
surface.SetRuledModeToPointWalk()
surface.SetDistanceFactor(30)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(surface.GetOutputPort())
mapper.SetScalarRange(vector_range)
mapper.SetLookupTable(color_tf)

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer.AddActor(actor)

# -----------------------------------------------

reader_wing = vtk.vtkUnstructuredGridReader()
reader_wing.SetFileName(args.wing)
reader_wing.Update()

mapper_wing = vtk.vtkDataSetMapper()
mapper_wing.SetInputConnection(reader_wing.GetOutputPort())

actor_wing = vtk.vtkActor()
actor_wing.SetMapper(mapper_wing)
actor_wing.GetProperty().SetDiffuseColor(0, 0, 0)

# -----------------------------------------------

renderer.AddActor(actor_wing)
renderer.ResetCamera()
renderer.SetBackground(0.3, 0.3, 0.3)

window = vtk.vtkRenderWindow()
window.AddRenderer(renderer)

interactive = vtk.vtkRenderWindowInteractor()
interactive.SetRenderWindow(window)

scalar_bar = vtk.vtkScalarBarActor()
scalar_bar.SetOrientationToHorizontal()
scalar_bar.SetLookupTable(color_tf)

scalar_bar_widget = vtk.vtkScalarBarWidget()
scalar_bar_widget.SetInteractor(interactive)
scalar_bar_widget.SetScalarBarActor(scalar_bar)
scalar_bar_widget.On()

window.SetSize(800, 800)

interactive.Initialize()
window.Render()
interactive.Start()
