import numpy as np
from numpy import ndarray as a
from coordtrans import *
import numba as nb
from numba import jit

## ==================================== ##-
## FIELD LINE SOLVER (FROM "Bfield.py") ##
## ==================================== ##
@jit(nb.types.Array(nb.float64, 1, "C")(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def blines(t,y):
	B = np.zeros((3,1))
	X=y[0]
	Y=y[1]
	Z=y[2]
	direction=y[3]
	point = np.array([ X, Y, Z ])

	B = interpField(point, R, THETA, PHI, Bx, By, Bz)
	Bnorm = np.sqrt(B[0]**2 + B[1]**2 + B[2]**2)
	dY    = np.zeros(4)
	dY[0] = direction * B[0]/Bnorm
	dY[1] = direction * B[1]/Bnorm
	dY[2] = direction * B[2]/Bnorm
	dY[3] = 0.0
	return dY
