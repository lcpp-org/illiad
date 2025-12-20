import numpy as np
from math import degrees, sin, cos, floor
import os as os
from utility.coordtrans import XYZ_to_RTP2
import logging

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

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
        """Initializes the Mesh object.

        Args:
            R0 (float, optional): Major radius of the tokamak. Defaults to 0.0.
            a (float, optional): Minor radius of the tokamak. Defaults to 0.0.
        """
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
        self.err_mag = 0.0 #1.5654e-4 # [Tesla]
        self.err_dir = 0.0 #271.5 * np.pi/180 # [radians]
        self.cos_err_dir = 1.0 #np.cos(self.err_dir)
        self.sin_err_dir = 0.0 #np.sin(self.err_dir)
        self.xerr_adder = 0.0
        self.yerr_adder = 0.0
        self.err_adder = np.array([self.xerr_adder, self.yerr_adder, 0.0], dtype=np.float64).T

        self.att_mult = 1.0 

    def loadCartesianField(self, file_path='input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy', period_ = np.array([0, 1, 5], dtype=np.int32), coilCurrent=1.0, errField=False, att_mult='default_toroidal'):
        """Loads a vector field from a file and sets mesh properties.

        The function loads a 3D vector field from a .npy file and initializes the mesh grid properties
        based on the input array dimensions. The field arrays are assumed to be in Cartesian coordinates.

        Args:
            file_path (str, optional): Path to the .npy file containing the field arrays. Defaults to 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy'.
            period_ (np.ndarray, optional): Array specifying periodicity in [r, theta, phi] directions. Defaults to np.array([0, 1, 5], dtype=np.int32).
            coilCurrent (float, optional): Scaling factor for the coil current. Defaults to 1.0.
            errField (bool, optional): If True, enables error field addition. Defaults to False.
            att_mult (str or float, optional): Attenuation multiplier. Can be a float or one of ['default_toroidal', 'default_poloidal', 'default_poloidal_rev']. Defaults to 'default_toroidal'.

        Raises:
            ValueError: If the input array dimensions do not match.
        """
        Bx_, By_, Bz_ = np.load(file_path)

        if Bx_.shape != By_.shape or Bx_.shape != Bz_.shape:
            print("INPUT ARRAY DIMENSIONS DO NOT MATCH!!")
        else:
            # check the attenuation multiplier
            if att_mult =='default_toroidal':
                att_mult = 0.9448
            elif att_mult =='default_poloidal':
                att_mult = -0.955 * 0.9448
            elif att_mult =='default_poloidal_rev':
                att_mult = 0.955 * 0.9448
            elif isinstance(att_mult, float):
                att_mult = att_mult
            else:
                print(f"{self}: ATT_MULT IS NOT A FLOAT OR DEFAULT VALUE!!")
                att_mult = 1.0

            total_mult = att_mult * coilCurrent
            self.nr, self.ntheta, self.nphi = Bx_.shape
            self.periodicity = period_
            self.Bx = torch.tensor(Bx_ * total_mult, dtype=torch.float64).to(device)
            self.By = torch.tensor(By_ * total_mult, dtype=torch.float64).to(device)
            self.Bz = torch.tensor(Bz_ * total_mult, dtype=torch.float64).to(device)

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
            self.errField = torch.tensor(errField, dtype=torch.bool).to(device)

    def addFieldPerturbation(self, file_path='input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy', coilCurrent=1.0, att_mult='default_helical'):
        """Adds a vector field perturbation from a file to the existing mesh field.

        The perturbation field is loaded from a file and added to the current mesh field.
        Array sizes must match the existing mesh dimensions, and periodicity is assumed to be the same.

        Args:
            file_path (str, optional): Path to the .npy file containing the perturbation field arrays. 
                Defaults to 'input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy'.
            coilCurrent (float, optional): Scaling factor for the coil current. Defaults to 1.0.
            att_mult (str or float, optional): Attenuation multiplier. Can be a float or one of 
                ['default_toroidal', 'default_helical', 'default_helical_rev']. Defaults to 'default_helical'.

        Raises:
            ValueError: If the input array dimensions do not match the mesh field dimensions.
        """
        Bx_, By_, Bz_ = np.load(file_path)

        if Bx_.shape != self.Bx.shape or By_.shape != self.By.shape or Bz_.shape != self.Bz.shape:
            print("INPUT ARRAY DIMENSIONS DO NOT MATCH!!")
        else:
            # check the attenuation multiplier
            if att_mult =='default_toroidal':
                att_mult = 0.9448
            elif att_mult =='default_helical':
                att_mult = -0.955 * 0.9448
            elif att_mult =='default_helical_rev':
                att_mult = 0.955 * 0.9448
            elif isinstance(att_mult, float):
                att_mult = att_mult
            else:
                print(f"{self}: ATT_MULT IS NOT A FLOAT OR DEFAULT VALUE!!")
                att_mult = 1.0

            total_mult = att_mult * coilCurrent

            self.Bx += torch.tensor((Bx_ * total_mult), dtype=torch.float64).to(device)
            self.By += torch.tensor((By_ * total_mult), dtype=torch.float64).to(device)
            self.Bz += torch.tensor((Bz_ * total_mult), dtype=torch.float64).to(device)

    def setErrorField(self, err_mag=1.5654e-4, err_dir=271.5*np.pi/180):
        """Sets the magnitude and direction of the non-periodic error field.

        The direction is measured from phi_c=0 (i.e., 18 degrees clockwise from the South Split).
        Also computes and stores the cosine and sine of the error direction for efficient interpolation.

        Args:
            err_mag (float, optional): Magnitude of the error field. Defaults to 1.5654e-4.
            err_dir (float, optional): Direction of the error field in radians. Defaults to 271.5*np.pi/180.
        """
        self.errField = True
        self.err_mag = err_mag
        self.err_dir = err_dir
        self.cos_err_dir = np.cos(err_dir)
        self.sin_err_dir = np.sin(err_dir)
        self.xerr_adder = self.err_mag * self.cos_err_dir
        self.yerr_adder = -1 * self.err_mag * self.sin_err_dir
        self.err_adder = np.array([self.xerr_adder, self.yerr_adder, 0.0], dtype=np.float64).T
        self.err_adder = torch.tensor(self.err_adder, dtype=torch.float64).to(device)

    def rot_vecXYZ_byPHI(self, vec_XYZ, delta_phi):
        """
        Rotates a 3D Cartesian vector around the z-axis by a given angle phi.

        Args:
            vec_XYZ (torch.Tensor): A 3-element tensor representing the Cartesian vector [x, y, z].
            delta_phi (torch.Tensor or float): The rotation angle in radians. Positive values rotate clockwise when viewed from above.

        Returns:
            torch.Tensor: The rotated 3D Cartesian vector as a tensor of shape (3,).

        Notes:
            - Rotation is performed around the z-axis.
            - The z-component remains unchanged.
            - When viewed from above, positive phi rotates the vector clockwise.
        Function takes in a cartesian vector and a phi angle
        Returns the cartesian values of the vector rotated by phi degrees
        convention: When looking at across-section to the right of the +z axis, theta is counterclockwise
        convention: +phi is clockwise when viewed from above
        """
        rotated_XYZ = torch.zeros(vec_XYZ.shape, dtype=torch.float64).to(device)
        #xFormMat = np.array([[ np.cos(delta_phi), (-1)*np.sin(delta_phi), zeros_],
        #                     [ np.sin(delta_phi),      np.cos(delta_phi), zeros_],
        #                     [           zeros_,             zeros_,       ones_]],
        #                    )#dtype=torch.float64)
        #xFormMatrix = torch.tensor(xFormMat)
        #rotated_XYZ = torch.tensordot( vec_XYZ.T, xFormMatrix, dims=([0],[1]) )

        cos_phi = torch.cos(delta_phi)
        sin_phi = torch.sin(delta_phi)
        rotated_XYZ[0] = cos_phi*vec_XYZ[0] - sin_phi*vec_XYZ[1]
        rotated_XYZ[1] = sin_phi*vec_XYZ[0] + cos_phi*vec_XYZ[1]
        rotated_XYZ[2] = vec_XYZ[2]

        # mismatch in sign between 'Mesh' and 'eshNew' inmplementations?
        return rotated_XYZ

    def interpField(self, point_INPUT, Cart=True, basis='physical'):
        """
        Returns the interpolated field values at a point defined in Cartesian coordinates.

        Interpolation is performed using a weighted sum of field values at each node of the enclosing cell.
        The weights are calculated as the volume of the octant opposed to each node.

        The input point is converted to RTP coordinates and kept within the following domains:
            r: 0.0 to r_max (minor radius)
            theta: dtheta to theta_max (2pi / Nperiods)
            phi: dphi to phi_max (2pi / Nperiods)

        Args:
            point_INPUT (torch.Tensor): Input point(s) in Cartesian coordinates. (N,3) if multiple points.
            Cart (bool): If True, input is in Cartesian coordinates; otherwise, RTP coordinates.

        Returns:
            torch.Tensor: Tensor of shape (3, Npts), where Npts is the number of input points.
            Each column corresponds to the interpolated field components [Bx, By, Bz] at the respective point.
        """
        ## Sanitize input (or make sure we are passing torch tensors!)
        #print(f'{len(point_XYZ.shape)=}')
        #if len(point_XYZ.shape) < 2: # if we passed a single vector
        #point_XYZ = torch.tensor([point_XYZ], dtype=torch.float64).to(device)
        #print(f'{point_XYZ.shape=}')
        Npts = torch.int64
        if len(point_INPUT.shape) < 2:
            Npts = 1
        else:
            Npts = point_INPUT.shape[0]

        if Cart:
            point_RTP = XYZ_to_RTP2(point_INPUT, self.R0).to(device)
        else:
            point_RTP = point_INPUT.to(device)

        point_RTP = point_RTP.permute(*torch.arange(point_RTP.ndim - 1, -1, -1)) #= point_RTP.T throws a warning

        # Is there proper handling of rho< 0.0 cases? does there need to be? (not for point_XYZ cases)
        r_local  = point_RTP[0]
        th_local = torch.remainder(point_RTP[1], self.theta_max) # keep theta within 0 and theta_max!
        # pytorch has no built-in divmod function?
        ph_local = torch.remainder(point_RTP[2], self.phi_max) # keep phi within 0 and phi_max!
        ph_localN = torch.div(point_RTP[2], self.phi_max, rounding_mode='floor') # keep phi within 0 and phi_max!  #floor?

        vecOUT = torch.zeros([3,Npts], dtype=torch.float64).to(device)

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

        # r_lowr_el = r_low * r_el
        # r_localinvr_el = r_local * invr_el
        # above replaced with below in mesh.py to fix failing interp at r=0
        r_lowr_el = (r_low + r_el/2) * r_el
        r_localinvr_el = (r_local + invr_el/2) * invr_el

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

        # perform vector rotation if wrapping around in phi direction
        if Npts > 1:
            toRotate = torch.where(iph_hi == 0)[0]
            #toRotate = torch.where(iph_hi >= self.phi_max)[0]
            if toRotate.numel() > 0:
                Bvec5[:,toRotate] = self.rot_vecXYZ_byPHI( Bvec5[:,toRotate], self.phi_max )
                Bvec6[:,toRotate] = self.rot_vecXYZ_byPHI( Bvec6[:,toRotate], self.phi_max )
                Bvec7[:,toRotate] = self.rot_vecXYZ_byPHI( Bvec7[:,toRotate], self.phi_max )
                Bvec8[:,toRotate] = self.rot_vecXYZ_byPHI( Bvec8[:,toRotate], self.phi_max )
        else:
            if iph_hi == 0:
            #if iph_hi >= self.phi_max:
                Bvec5 = self.rot_vecXYZ_byPHI( Bvec5, self.phi_max )
                Bvec6 = self.rot_vecXYZ_byPHI( Bvec6, self.phi_max )
                Bvec7 = self.rot_vecXYZ_byPHI( Bvec7, self.phi_max )
                Bvec8 = self.rot_vecXYZ_byPHI( Bvec8, self.phi_max )

        # sum of vectors, weighted by 'anti-node' volume
        total_vol = A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8
        vecOUT = (Bvec1*A1 + Bvec2*A2 + Bvec3*A3 + Bvec4*A4 + Bvec5*A5 + Bvec6*A6 + Bvec7*A7 + Bvec8*A8) / total_vol

        # if the mesh is defined with periodic symmetry, we must 
        # perform a rotational transform based on which 'period' of the mesh the point is located
        # -defined for phi, not sure if necessary for theta, (and almost surely not for r)
        phi_rotation = ph_localN * self.phi_max  # angle of transform
        vecOUT = self.rot_vecXYZ_byPHI(vecOUT, -phi_rotation)

        if self.errField:
            if basis == 'physical':
                vecOUT += self.err_adder.unsqueeze(-1) if Npts > 1 else self.err_adder
            elif basis == 'contravariant':
                # convert error field to contravariant basis before adding
                bxerr = self.err_adder[0]
                byerr = self.err_adder[1]

                rho = point_RTP[0]
                sin_theta = torch.sin(point_RTP[1])
                cos_theta = torch.cos(point_RTP[1])
                sin_phi = torch.sin(point_RTP[2])
                cos_phi = torch.cos(point_RTP[2])

                R_cyl = self.R0 + rho * cos_theta
                term = (bxerr*cos_phi - byerr*sin_phi)

                err_adder_rho = term * cos_theta
                err_adder_theta = term * (-sin_theta) / torch.clamp(rho, min=self.dr/2)
                err_adder_phi = -(bxerr*sin_phi + byerr*cos_phi) / R_cyl

                err_adder_contra = torch.stack([err_adder_rho, err_adder_theta, err_adder_phi], dim=0)
                vecOUT += err_adder_contra
                # if Npts > 1:
                #     vecOUT += err_adder_contra.unsqueeze(-1) 
                # else:
                #     vecOUT += err_adder_contra
            else:   
                print(f"{self}: BASIS TYPE NOT RECOGNIZED FOR ERROR FIELD ADDITION!!")

        return vecOUT
