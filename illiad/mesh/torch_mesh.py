"""PyTorch-backed field mesh used by particle transport."""

import numpy as np
from math import degrees, sin, cos, floor
import os as os
from illiad.utilities.coordtrans import XYZ_to_RTP2
import logging

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

class TorchMesh:
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
        """Initialize a PyTorch-backed mesh.

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
        self.value = None
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
            self.B = torch.as_tensor(np.stack([Bx_, By_, Bz_], axis=-1) * total_mult,
                                        dtype=torch.float64, device=device)
            self.Bx = self.B[..., 0]
            self.By = self.B[..., 1]
            self.Bz = self.B[..., 2]

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

            self.B += torch.as_tensor(np.stack([Bx_, By_, Bz_], axis=-1) * total_mult,
                                        dtype=torch.float64, device=device)

    def loadScalarField(self, file_path, period_=np.array([0, 1, 1], dtype=np.int32), att_mult=1.0):
        """Loads a scalar field from a file and sets mesh properties.

        The input array is expected in the saved Flux_Interpolator ordering of
        (phi, theta, r) and is transposed into the internal (r, theta, phi)
        layout used by the interpolation routines.
        """
        val_ = np.load(file_path)
        val_ = np.transpose(val_, (2, 1, 0))

        self.nr, self.ntheta, self.nphi = val_.shape
        self.periodicity = period_
        self.att_mult = float(att_mult)

        value = torch.as_tensor(val_ * self.att_mult, dtype=torch.float64, device=device)
        self.value = torch.clamp(value, min=0.0)
        self.errField = torch.tensor(False, dtype=torch.bool).to(device)

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
        #rotated_XYZ = torch.zeros(vec_XYZ.shape, dtype=torch.float64).to(device)
        rotated_XYZ = torch.empty_like(vec_XYZ, dtype=torch.float64, device=vec_XYZ.device)
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

    def get_weights(self, point_INPUT, Cart=True):
        """
        Returns the corner indices and interpolation weights for a point defined in XYZ or RTP coordinates.

        The input point is converted to RTP coordinates and kept within the following domains:
            r: 0.0 to r_max (minor radius)
            theta: dtheta to theta_max (2pi / Nperiods)
            phi: dphi to phi_max (2pi / Nperiods)

        Args:
            point_INPUT (torch.Tensor): Input point(s) in Cartesian coordinates. (N,3) if multiple points.
            Cart (bool): If True, input is in Cartesian coordinates; otherwise, RTP coordinates.

        Returns:
            weights (torch.Tensor): Tensor of shape (8, Npts) containing the interpolation weights for each of the 8 cell corners.
            corner_vecs (torch.Tensor): Tensor of shape (8, 3, Npts) containing the field vectors at each of the 8 cell corners.
            ph_localN (torch.Tensor): Tensor of shape (Npts,) containing the local phi period index for each point.
        """
        ## Sanitize input (or make sure we are passing torch tensors!)
        #if len(point_INPUT.shape) < 2: # if we passed a single vector
        #point_XYZ = torch.tensor([point_XYZ], dtype=torch.float64).to(device)

        Npts = torch.int64
        if len(point_INPUT.shape) < 2:
            Npts = 1
        else:
            Npts = point_INPUT.shape[0]

        if Cart:
            point_RTP = XYZ_to_RTP2(point_INPUT, self.R0) #.to(device)
        else:
            point_RTP = point_INPUT #.to(device)

        point_RTP = point_RTP.permute(*torch.arange(point_RTP.ndim - 1, -1, -1)) #= point_RTP.T throws a warning

        # Is there proper handling of rho< 0.0 cases? does there need to be? (not for point_XYZ cases)
        r_local  = point_RTP[0]
        th_local = torch.remainder(point_RTP[1], self.theta_max) # keep theta within 0 and theta_max!
        # pytorch has no built-in divmod function?
        ph_local = torch.remainder(point_RTP[2], self.phi_max) # keep phi within 0 and phi_max!
        ph_localN = torch.div(point_RTP[2], self.phi_max, rounding_mode='floor') # keep phi within 0 and phi_max!  #floor?

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

        # fix failing interp at r=0
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
        weights = torch.stack([A1, A2, A3, A4, A5, A6, A7, A8], dim=0)

        ir_hi = (rindex + 1).type(torch.int)
        ir_lo = rindex.type(torch.int)
        ith_hi = thindex.type(torch.int)
        ith_lo = (thindex - 1).type(torch.int)
        iph_hi = phindex.type(torch.int)
        iph_lo = (phindex - 1).type(torch.int)

        # Gather the eight cell corners in one batched tensor shaped (8, 3, Npts).
        corner_r = torch.stack([ir_hi, ir_lo, ir_hi, ir_lo, ir_hi, ir_lo, ir_hi, ir_lo], dim=0)
        corner_theta = torch.stack([ith_hi, ith_hi, ith_lo, ith_lo, ith_hi, ith_hi, ith_lo, ith_lo], dim=0)
        corner_phi = torch.stack([iph_hi, iph_hi, iph_hi, iph_hi, iph_lo, iph_lo, iph_lo, iph_lo], dim=0)
        corner_indices = torch.stack([corner_r, corner_theta, corner_phi], dim=0)

        return weights, corner_indices, ph_localN

    def return_vecs(self, weights, corner_idx, ph_localN):
        """
        Returns the interpolated field values at a point defined in Cartesian coordinates.

        Interpolation is performed using a weighted sum of field values at each node of the enclosing cell.
        The weights are calculated as the volume of the octant opposed to each node.

        Args:
            weights (torch.Tensor): Tensor of shape (8, Npts) containing the interpolation weights for each of the 8 cell corners.
            corner_vecs (torch.Tensor): Tensor of shape (8, 3, Npts) containing the field vectors at each of the 8 cell corners.
            ph_localN (torch.Tensor): Tensor of shape (Npts,) containing the local phi period index for each point.

        Returns:
            vecOUT (torch.Tensor): Tensor of shape (3, Npts) containing the interpolated field vectors at the input points.
        """

        Npts = 1 if corner_idx.ndim == 2 else corner_idx.shape[-1]
        iph_hi = corner_idx[2][0]  # index of the 'high' phi corner (the one that may need to be rotated if wrapping around in phi direction)
        corner_vecs = torch.movedim(self.B[corner_idx[0], corner_idx[1], corner_idx[2]], -1, 1)
        # sum of vectors, weighted by 'anti-node' volume
        total_vol = weights.sum(dim=0)

        # perform vector rotation if wrapping around in phi direction
        if Npts > 1:
            toRotate = torch.where(iph_hi == 0)[0]
            if toRotate.numel() > 0:
                corner_vecs[4, :, toRotate] = self.rot_vecXYZ_byPHI(corner_vecs[4, :, toRotate], self.phi_max)
                corner_vecs[5, :, toRotate] = self.rot_vecXYZ_byPHI(corner_vecs[5, :, toRotate], self.phi_max)
                corner_vecs[6, :, toRotate] = self.rot_vecXYZ_byPHI(corner_vecs[6, :, toRotate], self.phi_max)
                corner_vecs[7, :, toRotate] = self.rot_vecXYZ_byPHI(corner_vecs[7, :, toRotate], self.phi_max)
            vecOUT = (corner_vecs * weights.unsqueeze(1)).sum(dim=0) / total_vol.unsqueeze(0)
        else:
            if iph_hi == 0:
                corner_vecs[4] = self.rot_vecXYZ_byPHI(corner_vecs[4], self.phi_max)
                corner_vecs[5] = self.rot_vecXYZ_byPHI(corner_vecs[5], self.phi_max)
                corner_vecs[6] = self.rot_vecXYZ_byPHI(corner_vecs[6], self.phi_max)
                corner_vecs[7] = self.rot_vecXYZ_byPHI(corner_vecs[7], self.phi_max)
            vecOUT = (corner_vecs * weights[:, None]).sum(dim=0) / total_vol

        # if the mesh is defined with periodic symmetry, we must
        # perform a rotational transform based on which 'period' of the mesh the point is located
        # -defined for phi, not sure if necessary for theta, (and almost surely not for r)
        if self.periodicity[2] > 1:
            phi_rotation = ph_localN * self.phi_max  # angle of transform
            vecOUT = self.rot_vecXYZ_byPHI(vecOUT, -phi_rotation)

        if self.errField:
            vecOUT += self.err_adder.unsqueeze(-1) if vecOUT.ndim == 2 else self.err_adder
        return vecOUT

    def return_scalars(self, weights, corner_idx):
        """Returns interpolated scalar values using precomputed corner weights."""
        node_vals = self.value[corner_idx[0], corner_idx[1], corner_idx[2]]
        total_vol = weights.sum(dim=0)
        scalar_out = (node_vals * weights).sum(dim=0) / total_vol
        return torch.atleast_1d(scalar_out)

    def interpField(self, point_INPUT, Cart=True, basis='physical'):
        weights, corner_indices, ph_localN = self.get_weights(point_INPUT, Cart)
        vecOUT = self.return_vecs(weights, corner_indices, ph_localN)
        return vecOUT

    def interpScalarField(self, point_INPUT, Cart=True):
        weights, corner_indices, _ = self.get_weights(point_INPUT, Cart)
        return self.return_scalars(weights, corner_indices)

