import numpy as np
from numpy import ndarray as a
from coordtrans import *
import numba as nb
from numba import jit 

# "Point in mesh" function

## FIND THE 8 CORNER POINTS OF THE CELL CONTAINING THE PARTICLE
## RETURN AS A LIST OF TUPLES
#############
@jit(nb.types.Array(nb.int32, 2, "C")
(nb.types.Array(nb.float64, 1, "C"), nb.types.Array(nb.float64, 1, "C", readonly=True), nb.types.Array(nb.float64, 1, "C", readonly=True), nb.types.Array(nb.float64, 1, "C", readonly=True)), nopython=True)
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
def findNodes(point, r, theta, phi):
  
    th_loc = np.float64
    ph_loc = np.float64
    
    # Get the location of the particle	
    r_loc  = float(point[0])
    th_loc = np.fmod(point[1], (2*np.pi)) # keep theta within 0 and 2pi
    ph_loc = np.fmod(point[2], (2*np.pi)) # keep phi within 0 and 2pi
    
    # Get array size
    nr     = len(r)
    ntheta = len(theta)
    nphi   = len(phi)

    if r_loc > Rmin:
		
        print('POINT OUTSIDE OF MESH!')
			
        # Cast the indices to the last element of the array
		# This is to make sure the interpolation function does not fail
        rlb = nr-1
		thlb = ntheta-1 
		phlb = nphi-1
		
	else:
		
        # Point is within the mesh, and we can find the indices
		
        # (here "lb" stands for "lower bound")

		#temp1 = np.where(r_loc < (r + dr)) 
		#rlb = temp1[0][0]
		
		#rlb = r_loc // dr
		rlb = np.floor(r_loc/dr)

		#temp2 = np.where(th_loc < (theta + dtheta))
		#thlb = temp2[0][0]
		
		#thlb = th_loc // dtheta
		thlb = np.floor(th_loc/dtheta)
		#thlb = np.fmod(thlb, ntheta)
		
		#temp3 = np.where(ph_loc < (phi + dphi))
		#phlb = temp3[0][0]
		
		#phlb = ph_loc // dphi
		phlb = np.floor(ph_loc/dphi)
		#phlb = np.fmod(phlb, nphi)
		
        # Keeping the theta and phi indices within the range of the angular grid
		thlb = np.fmod(thlb, ntheta)
		phlb = np.fmod(phlb, nphi)

    # Return the indices of the 8 corner points of the cell
    # Validation of the indices is not done here
	iNodes = np.array(
			[(rlb, thlb, phlb      ), (rlb+1, thlb,    phlb   ),
			 (rlb,  thlb+1, phlb   ), (rlb+1, thlb+1, phlb    ),
			 (rlb,  thlb,   phlb+1 ), (rlb+1, thlb,    phlb+1 ),
			 (rlb,  thlb+1, phlb+1 ), (rlb+1, thlb+1, phlb+1  )],
			dtype=np.int32)
			
	return iNodes

## CALCULATE THE VOLUME OF THE CELL DEFINED BY TWO OPPOSITE CORNERS
#############
@jit(nb.float64(nb.types.Array(nb.float64, 1, "C"), nb.types.Array(nb.float64, 1, "C")), nopython=True)

def volume(point1, point2):
	#return abs(((Rmaj/2)*(point2[0]**2 - point1[0]**2) + (1/3)*(point2[0]**3 - point1[0]**3)) * (point2[1] - point1[1])*(point2[2] - point1[2])) # using small angle theorem: sin(x)~x
	return abs(  (Rmaj/2)*(point2[0]**2 - point1[0]**2)*(point2[1] - point1[1])*(point2[2] - point1[2]) + (1/3)*(point2[0]**3 - point1[0]**3)*np.sin(point2[1] - point1[1])*(point2[2] - point1[2]) )

## INTERPOLATE THE VALUES OF THE MAGNETIC FIELDS AT THE PARTICLE LOCATION
#############
@jit(nb.types.Array(nb.float64, 1, "C")
(nb.types.Array(nb.float64, 1, "C"),
nb.types.Array(nb.float64, 1, "C", readonly=True), nb.types.Array(nb.float64, 1, "C", readonly=True), nb.types.Array(nb.float64, 1, "C", readonly=True),
nb.types.Array(nb.float64, 3, "C", readonly=True),nb.types.Array(nb.float64, 3, "C", readonly=True),nb.types.Array(nb.float64, 3, "C", readonly=True)),
nopython=True)
def interpField(point, r, theta, phi, bx, by, bz):
	vols = np.zeros(8)
	t_bx = 0.
	t_by = 0.
	t_bz = 0.
	
	point_tor = np.array([toR(point[0], point[1], point[2], Rmaj), toTHETA(point[0], point[1], point[2], Rmaj), toPHI(point[0], point[1], point[2], Rmaj)])
	cellpts = findNodes(point_tor, r, theta, phi)
	for j in range(8):
		thetaj = np.fmod(cellpts[j][1],len(THETA))
		phij = np.fmod(cellpts[j][2],len(PHI))
		# fmod already done in 'findNodes'
		#cpoint = np.array([R[cellpts[j][0]], 
		#					THETA[cellpts[j][1]],
		#					PHI[cellpts[j][2]]])
		cpoint = np.array([R[cellpts[j][0]], 
							THETA[thetaj],
							PHI[phij]]) #modulus on periodic boundaries
		vols[j] = volume(point_tor, cpoint)
		t_bx += vols[j] * Bx[cellpts[j][2], cellpts[j][1], cellpts[j][0]]
		t_by += vols[j] * By[cellpts[j][2], cellpts[j][1], cellpts[j][0]]
		t_bz += vols[j] * Bz[cellpts[j][2], cellpts[j][1], cellpts[j][0]]
	t_bx = t_bx/sum(vols)
	t_by = t_by/sum(vols)
	t_bz = t_bz/sum(vols)
	t_b = np.array([t_bx,t_by,t_bz])

	return t_b
