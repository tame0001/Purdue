from pyevtk.hl import imageToVTK
import numpy as np

# Dimensions
nx, ny, nz = 100, 20, 0
ncells = nx * ny * nz
npoints = (nx + 1) * (ny + 1) * (nz + 1)

# Variables
# pressure = np.random.rand(ncells).reshape( (nx, ny, nz), order = 'C')
temp = np.random.rand(npoints).reshape( (nx + 1, ny + 1, nz + 1))
temp *= 10000

print(temp.shape)


imageToVTK("./image", pointData = {"temp" : temp} )

# fluxx = np.random.rand(ncells).reshape( (nx, ny, nz), order='F')
# fluxy = np.random.rand(ncells).reshape( (nx, ny, nz), order='F')
# fluxz = np.random.rand(ncells).reshape( (nx, ny, nz), order='F')
# flux = (fluxx, fluxy, fluxz)

# Efieldx = np.random.rand(npoints).reshape( (nx + 1, ny + 1, nz + 1), order='F')
# Efieldy = np.random.rand(npoints).reshape( (nx + 1, ny + 1, nz + 1), order='F')
# Efieldz = np.random.rand(npoints).reshape( (nx + 1, ny + 1, nz + 1), order='F')
# Efield = (Efieldx,Efieldy)

# # # imageToVTK("./image", cellData={"flux" : flux}, pointData = {"Efield" : Efieldx} )
# imageToVTK("./image", pointData = {"Efield" : Efieldx} )