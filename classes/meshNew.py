import numpy as np
from math import degrees, sin, cos, floor
import os as os
from utility.coordtrans import XYZ_to_RTP2 #, rot_vecXYZ_byPHI
import logging

#import numba as nb
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# define a class with mesh information
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
        ## Align with new mesh.py 
        self.err_mag = 2.828427E-4 * 1.2
        self.err_dir = 270.* np.pi/180
        self.cos_err_dir = np.cos(self.err_dir)
        self.sin_err_dir = np.sin(self.err_dir)   

        self.att_mult = 1.0 

    def loadCartesianField(self, file_path, period_ = np.array([0, 1, 5], dtype=np.int32), errField=False, att_mult=1.0):
    #def loadCartesianField(self, Bx_: np.ndarray, By_: np.ndarray, Bz_: np.ndarray, period_ = np.array([0, 1, 5]), errField=False):
        """ 
        This function loads a vector field as a 3-dimensional scalar array for each cartesian vector.
        The grid properties are assumed from the dimensions of the input arrays
        """

        #self.log.info('Loading Cartesian Vector field from file: {}'.format(file_path))
        Bx_, By_, Bz_ = np.load(file_path)

        if Bx_.shape != By_.shape or Bx_.shape != Bz_.shape:
            print("INPUT ARRAY DIMENSIONS DO NOT MATCH!!")
        else:
            self.nr, self.ntheta, self.nphi = Bx_.shape
            self.periodicity = period_
            self.Bx = torch.tensor(Bx_ * att_mult, dtype=torch.float64).to(device)
            self.By = torch.tensor(By_ * att_mult, dtype=torch.float64).to(device)
            self.Bz = torch.tensor(Bz_ * att_mult, dtype=torch.float64).to(device)

            self.errField = errField

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

            self.phi_max = torch.tensor(self.phi_max, dtype=torch.float64).to(device)
            self.dr = torch.tensor(self.dr, dtype=torch.float64).to(device)
            self.dtheta = torch.tensor(self.dtheta, dtype=torch.float64).to(device)
            self.dphi = torch.tensor(self.dphi, dtype=torch.float64).to(device)
            # align with new mesh.py
            self.err_mag = torch.tensor(self.err_mag, dtype=torch.float64).to(device)
            self.cos_err_dir = torch.tensor(self.cos_err_dir, dtype=torch.float64).to(device)
            self.sin_err_dir = torch.tensor(self.sin_err_dir, dtype=torch.float64).to(device)


    # Align with new mesh.py
    def addFieldPerturbation(self, file_path, att_mult=1.0):
        """ 
        This function adds a vector field from a file to an existing vector field.
        The array sizes must match the existing mesh dimensions and periodicity is assumed the same
        """
        Bx_, By_, Bz_ = np.load(file_path)

        if Bx_.shape != self.Bx.shape or By_.shape != self.By.shape or Bz_.shape != self.Bz.shape:
            print("INPUT ARRAY DIMENSIONS DO NOT MATCH!!")
        else:
            self.Bx += torch.tensor((Bx_ * att_mult), dtype=torch.float64).to(device)
            self.By += torch.tensor((By_ * att_mult), dtype=torch.float64).to(device)
            self.Bz += torch.tensor((Bz_ * att_mult), dtype=torch.float64).to(device)

    def set_nonPer_errField(self, err_mag, err_dir):
        """This function sets the magnitude and direction (measured from the phi_c=0, i.e. 18degrees CW from the South Split)
          of the non-periodic error field. It also calculates the cosine and sine of the error direction for efficiency in the interpolation function"""
        self.err_mag = err_mag
        self.err_dir = err_dir
        self.cos_err_dir = np.cos(err_dir)
        self.sin_err_dir = np.sin(err_dir)

    def rot_vecXYZ_byPHI(self, vec_XYZ, delta_phi):
        #print(f'{vec_XYZ=}')
        # Function takes in a cartesian vector and a phi angle
        # Returns the cartesian values of the vector rotated by phi degrees
        # convention: When looking at across-section to the right of the +z axis, theta is counterclockwise
        # convention: +phi is clockwise when viewed from above
        rotated_XYZ = torch.zeros(vec_XYZ.shape, dtype=torch.float64).to(device)
        #xFormMat = np.array([[ np.cos(delta_phi), (-1)*np.sin(delta_phi), zeros_],
        #                     [ np.sin(delta_phi),      np.cos(delta_phi), zeros_],
        #                     [           zeros_,             zeros_,       ones_]],
        #                    )#dtype=torch.float64)
        #xFormMatrix = torch.tensor(xFormMat)

        #rotated_XYZ = torch.tensordot( vec_XYZ.T, xFormMatrix, dims=([0],[1]) )
        rotated_XYZ[0] = torch.cos(delta_phi)*vec_XYZ[0] - torch.sin(delta_phi)*vec_XYZ[1]
        rotated_XYZ[1] = torch.sin(delta_phi)*vec_XYZ[0] + torch.cos(delta_phi)*vec_XYZ[1]
        rotated_XYZ[2] = vec_XYZ[2]

        return rotated_XYZ

    def interpField(self, point_XYZ, Cart=True):
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

        ## Sanitize input (or make sure we are passing torch tensors!)
        #print(f'{len(point_XYZ.shape)=}')
        #if len(point_XYZ.shape) < 2: # if we passed a single vector
        #point_XYZ = torch.tensor([point_XYZ], dtype=torch.float64).to(device)
        #print(f'{point_XYZ.shape=}')
        
        Npts = torch.int64
        if len(point_XYZ.shape) < 2:
            Npts = 1
        else:
            Npts = point_XYZ.shape[0]

        if Cart:
            point_RTP = XYZ_to_RTP2(point_XYZ, self.R0).to(device)
        else:
            point_RTP = point_XYZ.to(device)

        point_RTP = point_RTP.permute(*torch.arange(point_RTP.ndim - 1, -1, -1)) #= point_RTP.T throws a warning

        r_local  = point_RTP[0]

        th_local = torch.remainder(point_RTP[1], self.theta_max) # keep theta within 0 and theta_max!

        ph_localN = torch.div(point_RTP[2], self.phi_max, rounding_mode='floor') # keep phi within 0 and phi_max!  #floor?
        ph_local = torch.remainder(point_RTP[2], self.phi_max) # keep phi within 0 and phi_max!

        vecXYZ = torch.zeros([3,Npts], dtype=torch.float64).to(device)

        rindex = torch.zeros([3,Npts], dtype=torch.int).to(device)
        thindex = torch.zeros([3,Npts], dtype=torch.int).to(device)
        phindex = torch.zeros([3,Npts], dtype=torch.int).to(device)
        rindex = torch.where( r_local >= self.r_max, self.nr - 2, torch.div(r_local, self.dr, rounding_mode='floor'))
        r_el = torch.remainder(r_local, self.dr)

        thindex = torch.div(th_local, self.dtheta, rounding_mode='floor')
        th_el = torch.remainder(th_local, self.dtheta)

        phindex = torch.div(ph_local, self.dphi, rounding_mode='floor')
        ph_el = torch.remainder(ph_local, self.dphi)

        r_low = rindex * self.dr
        th_low = (thindex+1) * self.dtheta

        invr_el = self.dr - r_el
        invth_el = self.dtheta - th_el
        invph_el = self.dphi - ph_el

        r_lowr_el = r_low * r_el
        r_localinvr_el = r_local * invr_el
        # sub-element volumes
        A1 = (self.R0 + r_low*torch.cos(th_low))     * r_lowr_el      * th_el    * ph_el
        A2 = (self.R0 + r_local*torch.cos(th_low))   * r_localinvr_el * th_el    * ph_el
        A3 = (self.R0 + r_low*torch.cos(th_local))   * r_lowr_el      * invth_el * ph_el
        A4 = (self.R0 + r_local*torch.cos(th_local)) * r_localinvr_el * invth_el * ph_el
        A5 = (self.R0 + r_low*torch.cos(th_low))     * r_lowr_el      * th_el    * invph_el
        A6 = (self.R0 + r_local*torch.cos(th_low))   * r_localinvr_el * th_el    * invph_el
        A7 = (self.R0 + r_low*torch.cos(th_local))   * r_lowr_el      * invth_el * invph_el
        A8 = (self.R0 + r_local*torch.cos(th_local)) * r_localinvr_el * invth_el * invph_el

        ir_hi = (rindex + 1).type(torch.int)
        ir_lo = rindex.type(torch.int)
        ith_hi = thindex.type(torch.int)
        ith_lo = (thindex - 1).type(torch.int)
        iph_hi = phindex.type(torch.int)
        iph_lo = (phindex - 1).type(torch.int)

        # node vectors
        Bvec1 = torch.stack([ self.Bx[ir_hi, ith_hi, iph_hi], self.By[ir_hi, ith_hi, iph_hi], self.Bz[ir_hi, ith_hi, iph_hi] ], dim = 0)
        Bvec2 = torch.stack([ self.Bx[ir_lo, ith_hi, iph_hi], self.By[ir_lo, ith_hi, iph_hi], self.Bz[ir_lo, ith_hi, iph_hi] ], dim = 0)
        Bvec3 = torch.stack([ self.Bx[ir_hi, ith_lo, iph_hi], self.By[ir_hi, ith_lo, iph_hi], self.Bz[ir_hi, ith_lo, iph_hi] ], dim = 0)
        Bvec4 = torch.stack([ self.Bx[ir_lo, ith_lo, iph_hi], self.By[ir_lo, ith_lo, iph_hi], self.Bz[ir_lo, ith_lo, iph_hi] ], dim = 0)
        Bvec5 = torch.stack([ self.Bx[ir_hi, ith_hi, iph_lo], self.By[ir_hi, ith_hi, iph_lo], self.Bz[ir_hi, ith_hi, iph_lo] ], dim = 0)
        Bvec6 = torch.stack([ self.Bx[ir_lo, ith_hi, iph_lo], self.By[ir_lo, ith_hi, iph_lo], self.Bz[ir_lo, ith_hi, iph_lo] ], dim = 0)
        Bvec7 = torch.stack([ self.Bx[ir_hi, ith_lo, iph_lo], self.By[ir_hi, ith_lo, iph_lo], self.Bz[ir_hi, ith_lo, iph_lo] ], dim = 0)
        Bvec8 = torch.stack([ self.Bx[ir_lo, ith_lo, iph_lo], self.By[ir_lo, ith_lo, iph_lo], self.Bz[ir_lo, ith_lo, iph_lo] ], dim = 0)
        # have to perform vector rotation if wrapping around in phi direction

        if Npts > 1:
            toRotate = torch.where( iph_hi >= self.nphi)[0]
            if len(toRotate) > 0:
                Bvec5[:,toRotate] = self.rot_vecXYZ_byPHI( Bvec5[:,toRotate], self.phi_max )
                Bvec6[:,toRotate] = self.rot_vecXYZ_byPHI( Bvec6[:,toRotate], self.phi_max )
                Bvec7[:,toRotate] = self.rot_vecXYZ_byPHI( Bvec7[:,toRotate], self.phi_max )
                Bvec8[:,toRotate] = self.rot_vecXYZ_byPHI( Bvec8[:,toRotate], self.phi_max )
        else:
            if iph_hi >= self.nphi:
                Bvec5 = self.rot_vecXYZ_byPHI( Bvec5, self.phi_max )
                Bvec6 = self.rot_vecXYZ_byPHI( Bvec6, self.phi_max )
                Bvec7 = self.rot_vecXYZ_byPHI( Bvec7, self.phi_max )
                Bvec8 = self.rot_vecXYZ_byPHI( Bvec8, self.phi_max )

        # sum of vectors, weighted by 'anti-node' volume
        vecXYZ = (Bvec1*A1 + Bvec2*A2 + Bvec3*A3 + Bvec4*A4 + Bvec5*A5 + Bvec6*A6 + Bvec7*A7 + Bvec8*A8) / (A1+A2+A3+A4+A5+A6+A7+A8)

        # if the mesh is defined with periodic symmetry, we must 
        # perform a rotational transform based on which 'period' of the mesh
        # the point is located
        # -defined for phi, not sure if necessary for theta, (and almost surely not for r)
        phi_rotation = ph_localN * self.phi_max  # angle of transform
        vecXYZ = self.rot_vecXYZ_byPHI(vecXYZ, -phi_rotation)

        if self.errField:
            vecXYZ[0] += self.err_mag * self.cos_err_dir
            vecXYZ[1] -= self.err_mag * self.sin_err_dir

        return vecXYZ

###
def XYZ_to_RTP2(p_XYZ, Rmajor):
    # Function to take in a point defined in Cartesian coordinates
    # And return a point in r-theta-phi coordinates
    # convention: When looking at a cross-section to the right of the +z axis, +theta is counterclockwise
    # convention: +phi is clockwise when viewed from above
    #p_XYZ = torch.tensor(p_XYZ).to(device)
    p_XYZ = p_XYZ.clone().detach().to(device)
    p_RTP = torch.zeros(p_XYZ.shape, dtype=torch.float64).to(device)
    x, y, z = p_XYZ.permute(*torch.arange(p_XYZ.ndim - 1, -1, -1)) #= p_XYZ.T throws a warning
    x2 = x*x
    y2 = y*y
    z2 = z*z
    R = torch.sqrt(x2 + y2)

    p_RTP[..., 0] =  torch.sqrt(x2 + y2 + z2 + Rmajor * Rmajor - 2 * Rmajor * R)

    den = R - Rmajor
    theta = torch.arctan2(z,den) # arctan2 returns radians from (-pi to +pi)
    # here we shift the domain to (0 to 2*pi)
    p_RTP[..., 1] = torch.where(theta<0, theta + 2*torch.pi, theta)
    #p_RTP.T[1] = theta

    phi = (-1) * torch.arctan2(y,x) # arctan2 returns radians from (-pi to +pi)
    # here we shift the domain to (0 to 2*pi)
    p_RTP[..., 2] = torch.where(phi<0, phi + 2*torch.pi, phi)
    #p_RTP.T[2] = phi

    return p_RTP
