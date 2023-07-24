import vtk

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName('temperature.vti')

# Pipe for contour line
iso = vtk.vtkContourFilter()
iso.SetInputConnection(reader.GetOutputPort())
iso.GenerateValues(11, 32, 42)

color_tf = vtk.vtkColorTransferFunction()
color_tf.AddRGBPoint(32, 0, 0.5, 1)  # Blue
color_tf.AddRGBPoint(42, 1, 0, 0)  # Red

iso_mapper = vtk.vtkDataSetMapper()
iso_mapper.SetInputConnection(iso.GetOutputPort())
iso_mapper.ScalarVisibilityOn()
iso_mapper.SetLookupTable(color_tf)

iso_actor = vtk.vtkActor()
iso_actor.SetMapper(iso_mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(iso_actor)
renderer.ResetCamera()
renderer.GetActiveCamera().Azimuth(180)
renderer.GetActiveCamera().Roll(180)

window = vtk.vtkRenderWindow()
window.AddRenderer(renderer)

interactive = vtk.vtkRenderWindowInteractor()
interactive.SetRenderWindow(window)

window.SetSize(800, 800)

interactive.Initialize()
window.Render()
interactive.Start()
