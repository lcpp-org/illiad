import numpy as np
from numpy import ndarray as a

from coordtrans import *
from mesh import *
#import mesh

import numba as nb
from numba import jit 

# "Point in mesh" function

## FIND THE 8 CORNER POINTS OF THE CELL CONTAINING THE PARTICLE
## RETURN AS A LIST OF TUPLES
#############
###@jit(nb.types.Array(nb.int32, 2, "C")
###(nb.types.Array(nb.float64, 1, "C"), nb.types.Array(nb.float64, 1, "C", readonly=True), nb.types.Array(nb.float64, 1, "C", readonly=True), nb.types.Array(nb.float64, 1, "C", readonly=True)), nopython=True)
#  Inputs:
#     point: 3D array with the coordinates of the particle
#     r: 1D array with the radial grid
#     theta: 1D array with the poloidal grid
#     phi: 1D array with the toroidal grid
#    
#     Output:
#   
#     iNodes: 2D array with the indices of the 8 corner points of the cell
#
def findNodes(point, mesh):
  
    #th_loc = np.float64
    #ph_loc = np.float64
    
    # Get the location of the particle	
    r_loc  = point[0]
    th_loc = np.fmod(point[1], (2*np.pi)) # keep theta within 0 and 2pi
    ph_loc = np.fmod(point[2], (2*np.pi)) # keep phi within 0 and 2pi
    
    # Get array size
    #nr     = len(r)
    #ntheta = len(theta)
    #nphi   = len(phi)
    # inherit values from mesh class

    if r_loc > mesh.a:
        print('POINT OUTSIDE OF MESH!')
        
        # Cast the indices to the last element of the array
        # This is to make sure the interpolation function does not fail
        rlb  = mesh.nr-2
        thlb = mesh.ntheta-2 
        phlb = mesh.nphi-2

    else:
        
        # Point is within the mesh, and we can find the indices
        
        # (here "lb" stands for "lower bound")
        rlb = np.floor(r_loc/mesh.dr)
        thlb = np.floor(th_loc/mesh.dtheta)
        phlb = np.floor(ph_loc/mesh.dphi)

        # Keeping the theta and phi indices within the range of the angular grid
        #thlb = np.fmod(thlb, mesh.ntheta)
        #phlb = np.fmod(phlb, mesh.nphi)
        # This is already accomplished when definig th_loc and ph_loc

    # Return the indices of the 8 corner points of the cell
    # Validation of the indices is not done here
    iNodes = np.array(
        [(rlb,  thlb,   phlb   ), (rlb+1, thlb,    phlb   ),
         (rlb,  thlb+1, phlb   ), (rlb+1, thlb+1, phlb    ),
         (rlb,  thlb,   phlb+1 ), (rlb+1, thlb,    phlb+1 ),
         (rlb,  thlb+1, phlb+1 ), (rlb+1, thlb+1, phlb+1  )],
        dtype=np.int32)

    return iNodes

## CALCULATE THE VOLUME OF THE CELL DEFINED BY TWO OPPOSITE CORNERS
#############
###@jit(nb.float64(nb.types.Array(nb.float64, 1, "C"), nb.types.Array(nb.float64, 1, "C")), nopython=True)

def volume(point1, point2, mesh):
    # Calculate the volume of the cell defined by two opposite corners
    # point1 and point2 are 3D arrays with the (r,th,phi) coordinates of two opposite corners
    # Each corner is defined by its (r,th,phi) coordinates
    # The volume is calculated as the absolute value of the determinant of the Jacobian matrix
    # The Jacobian matrix is defined by the differences between the coordinates of the two corners
    # The volume is returned as a float

    r1, r2 = np.sort([point1[0], point2[0]])
    theta1, theta2 = np.sort([point1[0], point2[0]])
    phi1, phi2 = np.sort([point1[0], point2[0]])
    
    return ( (mesh.R0/2.)*(r2**2 - r1**2)*(theta2 - theta1) + (1./3.)*(r2**3 - r1**3)*np.sin(theta2 - theta1) ) * (phi2 - phi1)
    
    #return abs(  (Rmaj/2)*(point2[0]**2 - point1[0]**2)*(point2[1] - point1[1])*(point2[2] - point1[2]) + (1/3)*(point2[0]**3 - point1[0]**3)*np.sin(point2[1] - point1[1])*(point2[2] - point1[2]) )


## INTERPOLATE THE VALUES OF THE MAGNETIC FIELDS AT THE PARTICLE LOCATION
#############
###@jit(nb.types.Array(nb.float64, 1, "C")
###(nb.types.Array(nb.float64, 1, "C"),
###nb.types.Array(nb.float64, 1, "C", readonly=True), nb.types.Array(nb.float64, 1, "C", readonly=True), nb.types.Array(nb.float64, 1, "C", readonly=True),
###nb.types.Array(nb.float64, 3, "C", readonly=True),nb.types.Array(nb.float64, 3, "C", readonly=True),nb.types.Array(nb.float64, 3, "C", readonly=True)),
###nopython=True)

#def interpField(point, mesh, field):
def interpField(point, field):
    vols = np.zeros(8)
    #t_bx = 0.
    #t_by = 0.
    #t_bz = 0.
    
    point_rtp = XYZ_to_RTP(point, field)
    cellpts = findNodes(point_rtp, field)
    
    for j in range(8):
        rj = cellpts[j][0]
        thetaj = np.fmod(cellpts[j][1], field.ntheta)
        phij = np.fmod(cellpts[j][2], field.nphi)

        cpoint = np.array([field.r[rj],  field.theta[thetaj], field.phi[phij]])

        vols[j] = volume(point_rtp, cpoint, field)

        t_bx += vols[j] * field.Bx[phij, thetaj, rj]
        t_by += vols[j] * field.By[phij, thetaj, rj]
        t_bz += vols[j] * field.Bz[phij, thetaj, rj]
    
    t_bx = t_bx/sum(vols)
    t_by = t_by/sum(vols)
    t_bz = t_bz/sum(vols)
    t_b = np.array([t_bx,t_by,t_bz])

    return t_b
