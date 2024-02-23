import numpy as np
from numpy import ndarray as a
from coordtrans import *
import numba as nb
from numba import jit


## ====== ##
## EVENTS ##
## ====== ##

## DETERMINE WHETHER POINT IS WITHIN VACUUM VESSEL
## >0, WITHIN VESSEL, <=0, OUTSIDE VESSEL
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def inVV(t, point):
	return Rmin - toR(point[0],point[1], point[2],  Rmaj) - 0.0001
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi0(t, point):
	return 0.*(np.pi/20.) - toPHI(point[0],point[1], point[2],  Rmaj)
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi9(t, point):
	return 1. * (np.pi/20.) - toPHI(point[0],point[1], point[2],  Rmaj)
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi18(t, point):
	return 2. * (np.pi/20.) - toPHI(point[0],point[1], point[2],  Rmaj)
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi27(t, point):
	return 3. * (np.pi/20.) - toPHI(point[0],point[1], point[2],  Rmaj)
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi36(t, point):
	return 4. * (np.pi/20.) - toPHI(point[0],point[1], point[2],  Rmaj)
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi45(t, point):
	return 5. * (np.pi/20.) - toPHI(point[0],point[1], point[2],  Rmaj)
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi54(t, point):
	return 6. * (np.pi/20.) - toPHI(point[0],point[1], point[2],  Rmaj)
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi63(t, point):
	return 7. * (np.pi/20.) - toPHI(point[0],point[1], point[2],  Rmaj)
@jit(nb.float64(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def isphi72(t, point):
	return 8. * (np.pi/20.) - toPHI(point[0],point[1], point[2],  Rmaj)
