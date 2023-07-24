import vtk
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('tdelta', metavar='tdelta', type=str, help='Vector data file')
parser.add_argument('wing', metavar='wing', type=str, help='Geometry data file')
args = parser.parse_args()

renderer = vtk.vtkRenderer()

lut = vtk.vtkLookupTable()
lut.Build()

reader_tdelta = vtk.vtkStructuredPointsReader()
reader_tdelta.SetFileName(args.tdelta)
reader_tdelta.Update()

array_calc = vtk.vtkArrayCalculator()
array_calc.SetInputConnection(reader_tdelta.GetOutputPort())
array_calc.AddVectorArrayName('velocity')
array_calc.SetFunction('mag(velocity)')
array_calc.SetResultArrayName('velocityMag')
array_calc.Update()

vector_range = array_calc.GetOutput().GetPointData().GetArray(1).GetRange()

# -----------------------------------------------

plane_position = [300]
# plane_position = [50, 150, 300]

for position in plane_position:

    plane = vtk.vtkPlane()
    plane.SetOrigin(position, 0, 0)
    plane.SetNormal(1, 0.0, 0.0)

    cutter = vtk.vtkCutter()
    cutter.SetInputConnection(reader_tdelta.GetOutputPort())
    cutter.SetCutFunction(plane)

    probe = vtk.vtkProbeFilter()
    probe.SetInputConnection(cutter.GetOutputPort())
    probe.SetSourceData(reader_tdelta.GetOutput())

    arrow_source = vtk.vtkArrowSource()
    glyph = vtk.vtkGlyph3D()
    glyph.SetInputConnection(probe.GetOutputPort())
    glyph.SetSourceConnection(arrow_source.GetOutputPort())
    glyph.SetVectorModeToUseVector()
    glyph.SetColorModeToColorByVector()
    glyph.SetScaleFactor(2)
    glyph.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())
    mapper.SetScalarRange(vector_range)
    mapper.SetLookupTable(lut)

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
scalar_bar.SetLookupTable(lut)

scalar_bar_widget = vtk.vtkScalarBarWidget()
scalar_bar_widget.SetInteractor(interactive)
scalar_bar_widget.SetScalarBarActor(scalar_bar)
scalar_bar_widget.On()

window.SetSize(800, 800)

interactive.Initialize()
window.Render()
interactive.Start()
