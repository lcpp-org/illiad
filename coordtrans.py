import numpy as np
from numba import jit
import numba as nb

from mesh import *

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
def toZ(r, theta, phi, R0):
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

import numpy as np

##
## TRANSFORM TO CARTESIAN COORDINATES
##
def RTP_to_XYZ(p_RTP, mesh):
    r, theta, phi = p_RTP[:3]
    
    x = (mesh.R0 + r*np.cos(theta))*np.cos(phi)
    y = (mesh.R0 + r*np.cos(theta))*np.sin(phi)
    z = r*np.sin(theta)
    p_XYZ = np.array([x, y, z])
    
    return p_XYZ


##
## TRANSFORM TO TOROIDAL COORDINATES
##
def XYZ_to_RTP(p_XYZ, mesh):
    x, y, z = p_XYZ

    r = np.sqrt( x**2 + y**2 + z**2 + mesh.R0**2 - 2*mesh.R0*np.sqrt(x**2 + y**2) )

    den = np.sqrt(x**2 + y**2) - mesh.R0
    #theta = np.arctan2(z,den)
    temp1 = np.arctan2(z,den)
    temp2 = np.where(temp1<0, 2*np.pi+temp1, temp1)
    theta = temp2.item()

    #phi = np.arctan2(y,x)
    temp3 = np.arctan2(y,x)
    temp4 = np.where(temp3<0, 2*np.pi+temp3, temp3)
    phi = temp4.item()

    p_RTP = np.array([r, theta, phi])

    return p_RTP