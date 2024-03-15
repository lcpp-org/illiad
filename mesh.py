import numpy as np
from math import degrees, sin, cos, floor
import os as os
from coordtrans import XYZ_to_RTP
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

        #self.module_path = os.path.realpath(os.path.dirname(__file__))
        try:
            self.Bx, self.By, self.Bz = np.load('input_files/'+name)
        except OSError as error:
            self.log.critical(error)
            self.log.critical("INPUT FILE NOT FOUND!!")

        if self.Bx.shape == self.By.shape and self.Bx.shape == self.Bz.shape:

            self.nr, self.ntheta, self.nphi = self.Bx.shape

            #self.periodicity = np.array([0, 1, 1]) #hack! set outside
            self.periodicity = np.array([r_period, theta_period, phi_period]) #hack! set outside


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
                           +'# ----------------------------------- #\n'
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
        # Interpolation  done via a weighted sum of field values at each node of the enclosing cell
        # The weight function is calculated as the volume of the octant opposed to the node

        # Get the location of the point in RTP coordinates
        # domains:
        # r: 0 -> minor Radius
        # theta: 0 -> 2pi
        # phi: 0 -> 2pi
        point_RTP = XYZ_to_RTP(point_XYZ, self.R0)
        r_loc  = point_RTP[0]

        th_loc_old = point_RTP[1]
        ph_loc_old = point_RTP[2]

        th_locN, th_loc_new = divmod(point_RTP[1], self.theta_max) # keep theta within 0 a   nd theta_max!
        ph_locN, ph_loc_new = divmod(point_RTP[2], self.phi_max) # keep phi within 0 and phi_max!

        #elf.log.debug(f'modTheta = old:{degrees(th_loc_old)}, new:{degrees(th_loc_new)}')
        #elf.log.debug(f'modPhi = old:{degrees(ph_loc_old)}, new:{degrees(ph_loc_new)}')

        #elf.log.debug(f'Theta current N = {int(th_locN)}, modTheta = old:{degrees(th_loc_old)}, new:{degrees(th_loc_new)}')
        #elf.log.debug(f'Phi current N = {int(ph_locN)}, modPhi = old:{degrees(ph_loc_old)}, new:{degrees(ph_loc_new)}')

        if r_loc > self.a:
            # determine whether point is within mesh domain
            # Cast the indices to the last element of the array
            # This is to make sure the interpolation function does not fail
            #print('POINT OUTSIDE OF MESH!')
            rindex_lo  = self.nr - 2

        else:
            # Point is within the mesh, and we can find the indices
            # (here "lb" stands for "lower bound")
            rindex_lo = floor(r_loc/self.dr)

        rindex_hi = rindex_lo + 1

        thindex_hi = floor(th_loc_new/self.dtheta)
        phindex_hi = floor(ph_loc_new/self.dphi)

        thindex_lo = thindex_hi - 1
        phindex_lo = phindex_hi - 1

        # Return the indices of the 8 corner points of the cell
        # Validation of the indices is not done here
        nodeIndexArray = np.array(
             [[rindex_lo,  thindex_lo, phindex_lo ], [rindex_hi, thindex_lo, phindex_lo],
             [rindex_lo,  thindex_hi, phindex_lo ], [rindex_hi, thindex_hi, phindex_lo ],
             [rindex_lo,  thindex_lo, phindex_hi ], [rindex_hi, thindex_lo, phindex_hi ],
             [rindex_lo,  thindex_hi, phindex_hi ], [rindex_hi, thindex_hi, phindex_hi ]],
            dtype=np.int32)

        # antiNodeArray will return the node indices diagonally "opposite" the node in nodeArray
        antiNodeArray = np.flip(nodeIndexArray, 0)

        t_bx = 0.
        t_by = 0.
        t_bz = 0.
        totalVolume = 0.

        bx_rotatedPhi = 0.
        by_rotatedPhi = 0.
        bz_rotatedPhi = 0.

        #for node, antiNode in enumerate(zip(nodeArray, antiNodeArray)):
        for n, node in enumerate(nodeIndexArray):
            node_i = node[0]
            node_j  = node[1]
            node_k = node[2]

            antiNode_i = antiNodeArray[n][0]
            antiNode_j  = antiNodeArray[n][1]
            antiNode_k = antiNodeArray[n][2]

            antiNode_r = antiNode_i * self.dr
            antiNode_theta = (antiNode_j + 1) * self.dtheta
            antiNode_phi = (antiNode_k + 1) * self.dphi

            antiNode_rtp = np.array([antiNode_r, antiNode_theta, antiNode_phi])

            antiNode_subVolume = self.subElementVolume(point_RTP, antiNode_rtp)

            totalVolume += antiNode_subVolume
            t_bx += antiNode_subVolume * self.Bx[node_i, node_j, node_k]
            t_by += antiNode_subVolume * self.By[node_i, node_j, node_k]
            t_bz += antiNode_subVolume * self.Bz[node_i, node_j, node_k]
        #  return the sum of weighted values divided by the total
        #t_b = np.array([t_bx, t_by, t_bz])/totalVolume
        t_bx = t_bx/totalVolume
        t_by = t_by/totalVolume
        t_bz = t_bz/totalVolume

        # if the mesh is defined with periodic symmetry, we must 
        # perform a rotational transform based on which 'period' of the mesh
        # the point is located
        # -defined for phi, not sure if necessary for theta, (and almost surely not for r)

        #if self.periodicity[2]>1: # phi periodicity
        #if ph_locN > 0:
        phi_rotation = int(ph_locN) * self.phi_max #(2*np.pi)/self.periodicity[2] # angle of transform
        bx_rotatedPhi = t_bx*cos(phi_rotation) + t_by*sin(phi_rotation)
        by_rotatedPhi = t_bx*sin(phi_rotation) - t_by*cos(phi_rotation)
        bz_rotatedPhi = t_bz

        t_b = np.array([bx_rotatedPhi, by_rotatedPhi, bz_rotatedPhi])#/totalVolume

        return t_b