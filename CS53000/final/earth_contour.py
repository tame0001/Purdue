import vtk

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName('elevation_small.vti')

# print(reader.GetOutput().GetPointData())

radius = 50

mapper = vtk.vtkDataSetMapper()
mapper.SetInputConnection(reader.GetOutputPort())
mapper.ScalarVisibilityOff()

# jpeg_reader = vtk.vtkJPEGReader()
# jpeg_reader.SetFileName(satellite_file)
# jpeg_reader.Update()

# atext = vtk.vtkTexture()
# atext.SetInputConnection(jpeg_reader.GetOutputPort())
# atext.InterpolateOn()

actor = vtk.vtkActor()
actor.SetMapper(mapper)
# actor.SetTexture(atext)

# Pipe for contour line
iso = vtk.vtkContourFilter()
iso.SetInputConnection(reader.GetOutputPort())
iso.GenerateValues(19, -8000, 10000)

tubes = vtk.vtkTubeFilter()
tubes.SetInputConnection(iso.GetOutputPort())
tubes.SetRadius(100*radius)

color_tf = vtk.vtkColorTransferFunction()
color_tf.AddRGBPoint(-10000, 0, 0.5, 1)  # Blue
color_tf.AddRGBPoint(-1000, 1, 1, 1)  # White
color_tf.AddRGBPoint(0, 1, 0, 0)  # Red
color_tf.AddRGBPoint(1000, 1, 1, 1)  # White
color_tf.AddRGBPoint(8000, 1, 1, 0)  # Yellow



iso_mapper = vtk.vtkDataSetMapper()
iso_mapper.SetInputConnection(tubes.GetOutputPort())
iso_mapper.ScalarVisibilityOn()
iso_mapper.SetLookupTable(color_tf)

iso_actor = vtk.vtkActor()
iso_actor.SetMapper(iso_mapper)

# Feed both pipes to renderer
renderer = vtk.vtkRenderer()
# renderer.AddActor(actor)
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
