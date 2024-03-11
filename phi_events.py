import numpy as np
from coordtrans import XYZ_to_RTP
from mesh import *

#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi0(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh)
	return 0. * (np.pi/20.) - p_RTP[2]


#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi9(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh)
	return 1. * (np.pi/20.) - p_RTP[2]


#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi18(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh)
	return 2. * (np.pi/20.) - p_RTP[2]


#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi27(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh)
	return 3. * (np.pi/20.) - p_RTP[2]


#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi36(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh)
	return 4. * (np.pi/20.) - p_RTP[2]


#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi45(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh)
	return 5. * (np.pi/20.) - p_RTP[2]


#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi54(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh)
	return 6. * (np.pi/20.) - p_RTP[2]


#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi63(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh)
	return 7. * (np.pi/20.) - p_RTP[2]


#@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi72(t, p_XYZ, mesh):
	p_RTP = XYZ_to_RTP(p_XYZ[:3], mesh)
	return 8. * (np.pi/20.) - p_RTP[2]