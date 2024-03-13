import numpy as np
import os as os
from coordtrans import XYZ_to_RTP

# define a class with mesh information
class Mesh:
    # Class to store the mesh information
    #
    # r, theta, and phi are the 1D arrays with the grid points
    #
    #  r: minor radius coordinate, measured from the geometrical center of the poloidal cross section
    #  theta: poloidal angle [0,2pi)
    #  phi: toroidal angle [0,2pi)
    #
    # Rmaj: major radius of the tokamak
    # Rmin: minor radius of the tokamak
    #
    # dr: grid spacing in the r direction
    # dtheta: grid spacing in the theta direction
    # dphi: grid spacing in the phi direction

    def __init__(self):
        self.R0 = 0.0
        self.a = 0.0

        self.dr = 0.0
        self.dtheta = 0.0
        self.dphi = 0.0

        self.nr = 0
        self.ntheta = 0
        self.nphi = 0

    def setToroidalGeometry(self, R0, a):
        self.R0 = R0
        self.a = a 

    def subElementVolume(self, point1, point2):
        # this method takes in 2 points defined (r, theta, phi) coordinates
        # returns a scalar volume of the subelement defined these 2 diagonal points
        r1, theta1, phi1 = point1
        r2, theta2, phi2 = point2

        term1 = self.R0 * (r2**2 - r1**2)/2. * (theta2 - theta1) * (phi2 - phi1) 
        term2 =           (r2**3 - r1**3)/3. * np.sin(theta2 - theta1) * (phi2 - phi1) 
        return abs(term1 + term2)
        #return abs(( (self.R0/2.)*(r2**2 - r1**2)*(theta2 - theta1) + (1./3.)*(r2**3 - r1**3)*np.sin(theta2 - theta1) ) * (phi2 - phi1))


class CartesianField(Mesh):

    def loadCartesianField(self, Bx, By, Bz):
        # This function loads a vector field as a 3-dimensional scalar array for each cartesian vector
        # The grid properties are assumed from the dimensions of the input arrays
        #!! A better input data structure is needed to define dimension order, dimension size, and periodicity
        if Bx.shape == By.shape and Bx.shape == Bz.shape:
            self.nphi, self.ntheta, self.nr = Bx.shape

            self.Bx = Bx
            self.By = By
            self.Bz = Bz

            self.dr     = self.a / (self.nr-1)
            self.dtheta = 2*np.pi / (self.ntheta-1)
            self.dphi   = 2*np.pi / (self.nphi-1)
        else: print("ERROR: Input Array dimensions do not match!")


    def loadCartesianField_fromFile(self, name):
        # This function loads a vector field as a 3-dimensional scalar array for each cartesian vector
        # The grid properties are assumed from the dimensions of the input arrays
        #!! A better input data structure is needed to define dimension order, dimension size, and periodicity

        #self.module_path = os.path.realpath(os.path.dirname(__file__))
        try:
            self.Bx, self.By, self.Bz = np.load('input_files/'+name)
        except OSError as error:
            print(error)

        if self.Bx.shape == self.By.shape and self.Bx.shape == self.Bz.shape:
            self.nphi, self.ntheta, self.nr = self.Bx.shape

            self.dr     = self.a / (self.nr-1)
            self.dtheta = 2*np.pi / (self.ntheta-1)
            self.dphi   = 2*np.pi / (self.nphi-1)
        else: print("ERROR: Input Array dimensions do not match!")


    def interpField(self, point_XYZ):
        # Method to return the interpolated field values at a point defined in Cartesian coordinates
        # Interpolation  done via a weighted sum of field values at each node of the enclosing cell
        # The weight function is calculated as the volume of the octant opposed to the node

        # Get the location of the point in RTP coordinates
        point_RTP = XYZ_to_RTP(point_XYZ, self.R0)

        r_loc  = point_RTP[0]
        th_loc = point_RTP[1]
        ph_loc = point_RTP[2]

        #th_loc = np.fmod(point_RTP[1], (2*np.pi)) # keep theta within 0 and theta_max!
        #ph_loc = np.fmod(point_RTP[2], (2*np.pi)) # keep phi within 0 and phi_max!

        if r_loc > self.a:
            # determine whether point is within mesh domain
            # Cast the indices to the last element of the array
            # This is to make sure the interpolation function does not fail
            #print('POINT OUTSIDE OF MESH!')
            rlb  = self.nr - 2

        else:
            # Point is within the mesh, and we can find the indices
            # (here "lb" stands for "lower bound")
            rlb = np.floor(r_loc/self.dr)

        thlb = np.floor(th_loc/self.dtheta)
        phlb = np.floor(ph_loc/self.dphi)

        # Return the indices of the 8 corner points of the cell
        # Validation of the indices is not done here
        nodeIndexArray = np.array(
            [[rlb,  thlb,   phlb   ], [rlb+1, thlb,   phlb   ],
             [rlb,  thlb+1, phlb   ], [rlb+1, thlb+1, phlb   ],
             [rlb,  thlb,   phlb+1 ], [rlb+1, thlb,   phlb+1 ],
             [rlb,  thlb+1, phlb+1 ], [rlb+1, thlb+1, phlb+1 ]],
            dtype=np.int32)

        # antiNodeArray will return the node indices diagonally "opposite" the node in nodeArray
        antiNodeArray = np.flip(nodeIndexArray, 0)

        t_bx = 0.
        t_by = 0.
        t_bz = 0.
        totalVolume = 0.
        #for node, antiNode in enumerate(zip(nodeArray, antiNodeArray)):
        for j, node in enumerate(nodeIndexArray):
            r_index = node[0]
            theta_index = node[1]
            phi_index = node[2]

            antiNode = antiNodeArray[j]*np.array([self.dr, self.dtheta, self.dphi])
            subVolume = self.subElementVolume(point_RTP, antiNode)

            totalVolume += subVolume
            t_bx += subVolume * self.Bx[phi_index, theta_index, r_index]
            t_by += subVolume * self.By[phi_index, theta_index, r_index]
            t_bz += subVolume * self.Bz[phi_index, theta_index, r_index]
        #  return the sum of weighted values divided by the total
        t_b = np.array([t_bx,t_by,t_bz])/totalVolume

        return t_b