import numpy as np
from numba import jit
import numba as nb

# COORDINATE TRANSFORMATION: 
#       FROM -- CYLINDRICAL COORDINATES ON THE POLOIDAL PLANE (r,theta,phi)
#         TO -- CARTESIAN COORDINATES (x,y,z)

#@jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def toX(r, theta, phi, Rmajor):
    return (Rmajor + r*np.cos(theta))*np.cos(phi)

#@jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def toY(r, theta, phi, Rmajor):
    return (Rmajor + r*np.cos(theta))*np.sin(phi)

#@jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def toZ(r, theta, phi, Rmajor):
    return r*np.sin(theta)

## TRANSFORM TO TOROIDAL COORDINATES
#############
@jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def toR(x, y, z, Rmajor):
    return np.sqrt( x**2 + y**2 + z**2 + Rmajor**2 - 2*Rmajor*np.sqrt(x**2 + y**2) )
@jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def toTHETA(x, y, z, Rmajor):
    den = np.sqrt(x**2 + y**2) - Rmajor
    temp = np.arctan2(z,den)
    temp2 = np.where(temp<0, 2*np.pi+temp, temp)
    return temp2.item()
@jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def toPHI(x, y, z, Rmajor):
    temp = np.arctan2(y,x)
    temp2 = np.where(temp<0, 2*np.pi+temp, temp)
    return temp2.item()