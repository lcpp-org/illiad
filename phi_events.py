import numpy as np

from coordtrans import XYZ_to_RTP
from mesh import *

#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi9(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh.R0)
	return p_RTP[2] - 1. * (np.pi/20.) 
isphi9.direction = 1.0

#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi18(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh.R0)
	return p_RTP[2] - 2. * (np.pi/20.)
isphi18.direction = 1.0

#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi27(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh.R0)
	return p_RTP[2] - 3. * (np.pi/20.)
isphi27.direction = 1.0

#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi36(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh.R0)
	return p_RTP[2] - 4. * (np.pi/20.)
isphi36.direction = 1.0

#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi45(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh.R0)
	return p_RTP[2] - 5. * (np.pi/20.)
isphi45.direction = 1.0

#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi54(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh.R0)
	return p_RTP[2] - 6. * (np.pi/20.)
isphi54.direction = 1.0

#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi63(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh.R0)
	return p_RTP[2] - 7. * (np.pi/20.) 
isphi63.direction = 1.0

#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi72(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh.R0)
	return p_RTP[2] - 8. * (np.pi/20.)
isphi72.direction = 1.0