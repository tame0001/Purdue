import vtk
import pandas as pd

raw_data_file = '041602_temp.csv'
output_name = 'temperature.vti'

raw_data = pd.read_csv(raw_data_file, header=None, sep=',')
print(raw_data)

for i in range(9):
    print(raw_data[i][0])

imageData = vtk.vtkImageData()
imageData.SetDimensions(9, 5, 1)
if vtk.VTK_MAJOR_VERSION <= 5:
    imageData.SetNumberOfScalarComponents(1)
    imageData.SetScalarTypeToDouble()
else:
    imageData.AllocateScalars(vtk.VTK_DOUBLE, 1)

dims = imageData.GetDimensions()

# Fill every entry of the image data with "2.0"
for z in range(dims[2]):
    for y in range(dims[1]):
        for x in range(dims[0]):
            imageData.SetScalarComponentFromDouble(x, y, z, 0, raw_data[x][y])


writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName(output_name)
if vtk.VTK_MAJOR_VERSION <= 5:
    writer.SetInputConnection(imageData.GetProducerPort())
else:
    writer.SetInputData(imageData)
writer.Write()