import numpy as np

from coordtrans import XYZ_to_RTP
from mesh import *

#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi0(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh.R0)
	return 0. * (np.pi/20.) - p_RTP[2]
isphi0.direction = 1.0

#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi9(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh.R0)
	return 1. * (np.pi/20.) - p_RTP[2]
isphi9.direction = 1.0

#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi18(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh.R0)
	return 2. * (np.pi/20.) - p_RTP[2]
isphi18.direction = 1.0

#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi27(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh.R0)
	return 3. * (np.pi/20.) - p_RTP[2]
isphi27.direction = 1.0

#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi36(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh.R0)
	return 4. * (np.pi/20.) - p_RTP[2]
isphi36.direction = 1.0

#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi45(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh.R0)
	return 5. * (np.pi/20.) - p_RTP[2]
isphi45.direction = 1.0

#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi54(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh.R0)
	return 6. * (np.pi/20.) - p_RTP[2]
isphi54.direction = 1.0

#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi63(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh.R0)
	return 7. * (np.pi/20.) - p_RTP[2]
isphi63.direction = 1.0

#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi72(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh.R0)
	return 8. * (np.pi/20.) - p_RTP[2]
isphi72.direction = 1.0