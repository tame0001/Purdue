import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

imr = vtk.vtkXMLImageDataReader()
imr.SetFileName('elevation_small.vti')
# imr.SetFileName('image.vti')
imr.Update()

im = imr.GetOutput()
rows, cols, _ = im.GetDimensions()
sc = im.GetPointData().GetScalars()
a = vtk_to_numpy(sc)
a = a.reshape(rows, cols, -1)

print(a.shape)

assert a.shape==im.GetDimensions()