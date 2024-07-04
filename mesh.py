import numpy as np
from math import degrees, sin, cos, floor
import os as os
from coordtrans import XYZ_to_RTP #, rot_vecXYZ_byPHI
import logging

import numba as nb
#from numba.experimental import jitclass
#spec = [
#        ('R0', nb.float32),
#        ('a', nb.float32),
#        ('dr', nb.float32),
#        ('dtheta', nb.float32),
#        ('dphi', nb.float32),
#        ('nr', nb.int32),
#        ('ntheta', nb.int32),
#        ('nphi', nb.int32),
#        ('r_max', nb.float32),
#        ('theta_max', nb.float32),
#        ('phi_max', nb.float32),
#        ('r_min', nb.float32),
#        ('theta_min', nb.float32),
#        ('phi_min', nb.float32),
#        ('Bx', nb.types.Array(nb.float64, 3, "C")),
#        ('By', nb.types.Array(nb.float64, 3, "C")),
#        ('Bz', nb.types.Array(nb.float64, 3, "C")),
#        ('periodicity', nb.int32[:]),
#    ]


# define a class with mesh information
#@jitclass(spec)
class Mesh:
    """
    Class to store the mesh data, properties, and interpolation methods
    
    r, theta, and phi are the 1D arrays with the grid points
    
    r: minor radius coordinate, measured from the geometrical center of the poloidal cross section
    theta: poloidal angle [0,2pi)
    phi: toroidal angle [0,2pi)
    
    Rmaj: major radius of the tokamak
    Rmin: minor radius of the tokamak
    
    dr: grid spacing in the r direction
    dtheta: grid spacing in the theta direction
    dphi: grid spacing in the phi direction
    """

    def __init__(self, R0=0.0, a=0.0):
        #self.log = logging.getLogger()
        self.R0 = R0
        self.a = a
        self.dr = 0.0
        self.dtheta = 0.0
        self.dphi = 0.0
        self.nr = 0
        self.ntheta = 0
        self.nphi = 0
        self.r_max = 0
        self.theta_max = 0
        self.phi_max = 0
        self.r_min = 0
        self.theta_min = 0
        self.phi_min = 0
        self.Bx: np.float64[:][:][:]
        self.By: np.float64[:][:][:]
        self.Bz: np.float64[:][:][:]
        self.periodicity: np.int32[:]
        self.errField: np.bool


    def loadCartesianField(self, Bx_: np.ndarray, By_: np.ndarray, Bz_: np.ndarray, period_ = np.array([0, 1, 5]), errField=False):
        """ 
        This function loads a vector field as a 3-dimensional scalar array for each cartesian vector.
        The grid properties are assumed from the dimensions of the input arrays
        """
        if Bx_.shape != By_.shape or Bx_.shape != Bz_.shape:
            print("INPUT ARRAY DIMENSIONS DO NOT MATCH!!")
        else:
            self.nr, self.ntheta, self.nphi = Bx_.shape
            self.periodicity = period_
            self.Bx = Bx_
            self.By = By_
            self.Bz = Bz_
            self.errField = errField

            if self.errField:
                self.Bx += 0. #0.0002
                self.By += 0. #-0.0002

            if self.periodicity[0]:
                self.r_max = self.a / self.periodicity[0]
                self.dr = self.r_max / self.nr
                self.r_min = self.dr
            else:
                self.r_max = self.a
                self.dr = self.r_max / (self.nr-1)
                self.r_min = 0.

            if self.periodicity[1]:
                self.theta_max = (2*np.pi) / self.periodicity[1]
                self.dtheta = self.theta_max / self.ntheta
                self.theta_min = self.dtheta
            else:
                self.theta_max = (2*np.pi)
                self.dtheta = self.theta_max / (self.ntheta-1)
                self.theta_min = 0.

            if self.periodicity[2]:
                self.phi_max = (2*np.pi) / self.periodicity[2]
                self.dphi = self.phi_max / self.nphi
                self.phi_min = self.dphi
            else:
                self.phi_max = (2*np.pi)
                self.dphi = self.phi_max / (self.nphi-1)
                self.phi_min = 0.
            #self.log.info('Cartesian Vector field loaded:\n'
            #               +'# -----------------------------------\n'
            #              +'# Shape: {}\n.format(Bx.shape)'
            #              +'# dr = {} m.\n'.format(self.dr)
            #              +'# dtheta = {} deg.\n'.format(degrees(self.dtheta))
            #              +'# dphi = {} deg.\n'.format(degrees(self.dphi))
            #               +'# -----------------------------------\n')

    def subElementVolume(self, point1, point2):
        """
        This method takes in 2 points defined (r, theta, phi) coordinates
        returns a scalar volume of the subelement defined these 2 diagonal points
        """
        r1, theta1, phi1 = point1
        r2, theta2, phi2 = point2
        term1 = self.R0 * (r2**2 - r1**2)/2. *       (theta2 - theta1) * (phi2 - phi1) 
        term2 =           (r2**3 - r1**3)/3. * np.sin(theta2 - theta1) * (phi2 - phi1) 
        return np.abs(term1 + term2)

    def rot_vecXYZ_byPHI(self, vec_XYZ, delta_phi):
        """
        Function takes in a cartesian vector and a phi angle
        Returns the cartesian values of the vector rotated by phi degrees
        convention: When looking at across-section to the right of the +z axis, theta is counterclockwise
        convention: +phi is clockwise when viewed from above
        """
        rotated_XYZ = np.zeros(3)
        xFormMatrix = np.array([[ np.cos(delta_phi), -np.sin(delta_phi), 0.0],
                                [ np.sin(delta_phi),  np.cos(delta_phi), 0.0],
                                [               0.0,                0.0, 1.0]])

        rotated_XYZ = np.dot(vec_XYZ, xFormMatrix)

        return rotated_XYZ

    def interpField(self, point_XYZ):
        """
        Method to return the interpolated field values at a point defined in Cartesian coordinates
        Interpolation done via a weighted sum of field values at each node of the enclosing cell
        The weight function is calculated as the volume of the octant opposed to the node
        Get the location of the point in RTP coordinates,
        keep in domains:
        r:        0.0 -> r_max     (minor Radius)
        theta: dtheta -> theta_max (2pi/Nperiods)
        phi:     dphi -> phi_max   (2pi/Nperiods)
        """
        point_RTP = XYZ_to_RTP(point_XYZ, self.R0)
        r_local  = point_RTP[0]
        th_localN, th_local = np.divmod(point_RTP[1], self.theta_max) # keep theta within 0 and theta_max!
        ph_localN, ph_local = np.divmod(point_RTP[2], self.phi_max) # keep phi within 0 and phi_max!
        point_RTP_local = np.array([r_local, th_local, ph_local])

        if r_local > self.r_max:
            # determine whether point is within mesh domain
            # Cast the indices to the last element of the array
            # This is to make sure the interpolation function does not fail
            rindex_lo  = self.nr - 2
        else:
            # Point is within the mesh, and we can find the indices
            # (here "lo" stands for lower bound
            rindex_lo = np.floor(r_local/self.dr)

        rindex_hi = rindex_lo + 1
        thindex_hi = np.floor(th_local/self.dtheta)
        thindex_lo = thindex_hi - 1
        phindex_hi = np.floor(ph_local/self.dphi)
        phindex_lo = phindex_hi - 1

        # Return the indices of the 8 corner points of the cell
        # Validation of the indices is not done here
        nodeIndexArray = np.array(
            [[rindex_lo,  thindex_lo, phindex_lo ], [rindex_hi, thindex_lo, phindex_lo ],
             [rindex_lo,  thindex_hi, phindex_lo ], [rindex_hi, thindex_hi, phindex_lo ],
             [rindex_lo,  thindex_lo, phindex_hi ], [rindex_hi, thindex_lo, phindex_hi ],
             [rindex_lo,  thindex_hi, phindex_hi ], [rindex_hi, thindex_hi, phindex_hi ]],
            dtype=np.int32)

        # antiNodeArray will return the node indices diagonally "opposite" the node in nodeArray
        # note that proper functioning depends on ordering of nodeIndexArray
        antiNodeArray = np.flip(nodeIndexArray, 0)
        totalVolume = 0.
        node_vecXYZ = np.zeros(3)
        local_vecXYZ = np.zeros(3)
        global_vecXYZ = np.zeros(3)

        # cycle through nodes, solving for the field and the weight function
        for n, node in enumerate(nodeIndexArray):
            # get node and antiNode indices
            node_i, node_j, node_k = node
            antiNode_i, antiNode_j, antiNode_k = antiNodeArray[n]
            node_vecXYZ[0] = self.Bx[node_i, node_j, node_k]
            node_vecXYZ[1] = self.By[node_i, node_j, node_k]
            node_vecXYZ[2] = self.Bz[node_i, node_j, node_k]
            # transform the field if the node is < dphi
            if node_k < 0.:
                node_vecXYZ = self.rot_vecXYZ_byPHI(node_vecXYZ, -self.phi_max)
            # calculate antiNode rtp values from indices for input in to 'subElementVolume'
            antiNode_r = antiNode_i * self.dr
            antiNode_theta = (antiNode_j + 1) * self.dtheta
            antiNode_phi = (antiNode_k + 1) * self.dphi
            antiNode_rtp = np.array([antiNode_r, antiNode_theta, antiNode_phi])
            # calculate the wieght function as the volume of the point-antiNode subelement
            antiNode_subVolume = self.subElementVolume(point_RTP_local, antiNode_rtp)
            totalVolume += antiNode_subVolume
            local_vecXYZ += node_vecXYZ * antiNode_subVolume
        #  return the sum of weighted values divided by the total
        local_vecXYZ = local_vecXYZ / totalVolume

        # if the mesh is defined with periodic symmetry, we must 
        # perform a rotational transform based on which 'period' of the mesh
        # the point is located
        # -defined for phi, not sure if necessary for theta, (and almost surely not for r)
        phi_rotation = int(ph_localN) * self.phi_max  # angle of transform
        global_vecXYZ = self.rot_vecXYZ_byPHI(local_vecXYZ, phi_rotation)
        
        return global_vecXYZ, ph_localN