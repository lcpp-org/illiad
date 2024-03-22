import numpy as np
from math import degrees, sin, cos, floor
import os as os
from coordtrans import XYZ_to_RTP, rot_vecXYZ_byPHI
import logging

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
        self.log = logging.getLogger()
        self.R0 = 0.0
        self.a = 0.0

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

            self.log.info(f'Cartesian Vector field loaded:\n'
                           +'# -----------------------------------\n'
                          +f'# Shape: {Bx.shape}\n'
                          +f'# dr = {self.dr} m.\n'
                          +f'# dtheta = {degrees(self.dtheta)} deg.\n'
                          +f'# dphi = {degrees(self.dphi)} deg.\n'
                           +'# -----------------------------------\n')
            
        else: self.log.critical("INPUT ARRAY DIMENSIONS DO NOT MATCH!!")


    def loadCartesianField_fromFile(self, name, r_period, theta_period, phi_period):
        # This function loads a vector field as a 3-dimensional scalar array for each cartesian vector
        # The grid properties are assumed from the dimensions of the input arrays
        #!! A better input data structure is needed to define dimension order, dimension size, and periodicity

        try:
            self.Bx, self.By, self.Bz = np.load('input_files/'+name)
        except OSError as error:
            self.log.critical(error)
            self.log.critical("INPUT FILE NOT FOUND!!")

        if self.Bx.shape == self.By.shape and self.Bx.shape == self.Bz.shape:

            self.nr, self.ntheta, self.nphi = self.Bx.shape
            self.periodicity = np.array([r_period, theta_period, phi_period])

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


            self.log.info(f'Cartesian Vector field loaded from file {name}:\n'
                           +'# --------- FIELD MESH DATA --------- #\n'
                          +f'# Size: {self.Bx.shape}\n'
                          +f'# Periodicity: {self.periodicity}\n'
                          +f'# r min/max [meters]= {self.r_min}/{self.r_max}\n'
                          +f'# theta min/max [deg.]= {degrees(self.theta_min)}/{degrees(self.theta_max)}\n'
                          +f'# phi min/max [deg.]= {degrees(self.phi_min)}/{degrees(self.phi_max)}\n'
                          +f'# dr = {self.dr} m.\n'
                          +f'# dtheta = {degrees(self.dtheta)} deg.\n'
                          +f'# dphi = {degrees(self.dphi)} deg.\n'
                           +'# ------------------------------------ #\n')

        else: self.log.critical("INPUT ARRAY DIMENSIONS DO NOT MATCH!!")


    def interpField(self, point_XYZ):
        # Method to return the interpolated field values at a point defined in Cartesian coordinates
        # Interpolation done via a weighted sum of field values at each node of the enclosing cell
        # The weight function is calculated as the volume of the octant opposed to the node

        # Get the location of the point in RTP coordinates,
        # keep in domains:
        # r:        0.0 -> r_max     (minor Radius)
        # theta: dtheta -> theta_max (2pi/Nperiods)
        # phi:     dphi -> phi_max   (2pi/Nperiods)
        point_RTP = XYZ_to_RTP(point_XYZ, self.R0)
        
        r_local  = point_RTP[0]
        th_localN, th_local = divmod(point_RTP[1], self.theta_max) # keep theta within 0 and theta_max!
        ph_localN, ph_local = divmod(point_RTP[2], self.phi_max) # keep phi within 0 and phi_max!

        point_RTP_local = np.array([r_local, th_local, ph_local])

        if r_local > self.r_max:
            # determine whether point is within mesh domain
            # Cast the indices to the last element of the array
            # This is to make sure the interpolation function does not fail
            #print('POINT OUTSIDE OF MESH!')
            rindex_lo  = self.nr - 2

        else:
            # Point is within the mesh, and we can find the indices
            # (here "lo" stands for lower bound
            rindex_lo = floor(r_local/self.dr)

        rindex_hi = rindex_lo + 1

        thindex_hi = floor(th_local/self.dtheta)
        thindex_lo = thindex_hi - 1        

        phindex_hi = floor(ph_local/self.dphi)
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
                node_vecXYZ = rot_vecXYZ_byPHI(node_vecXYZ, -self.phi_max)

            # calculate anitNode rtp values from indices for input in to 'subElementVolume'
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
        global_vecXYZ = rot_vecXYZ_byPHI(local_vecXYZ, phi_rotation)

        return global_vecXYZ