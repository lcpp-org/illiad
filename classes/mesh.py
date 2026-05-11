import numpy as np
import os as os
from utility.coordtrans import XYZ_to_RTP


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
    def __init__(self, R0=0.72, a=0.19):
        """Initializes the Mesh object.

        Args:
            R0 (float, optional): Major radius of the tokamak. Defaults to 0.0.
            a (float, optional): Minor radius of the tokamak. Defaults to 0.0.
        """
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
        self.value: np.float64[:][:][:]
        self.periodicity: np.int32[:]

        self.errField: np.bool
        # self.err_mag = 2.828427E-4 * 1.2
        # self.err_dir = 270.* np.pi/180
        # self.cos_err_dir = np.cos(self.err_dir)
        # self.sin_err_dir = np.sin(self.err_dir)   
        self.err_mag = 0.0 #1.5654e-4 # [Tesla]
        self.err_dir = 0.0 #271.5 * np.pi/180 # [radians]
        self.cos_err_dir = 1.0 #np.cos(self.err_dir)
        self.sin_err_dir = 0.0 #np.sin(self.err_dir)   
        self.xerr_adder = 0.0
        self.yerr_adder = 0.0
        self.err_adder = np.array([self.xerr_adder, self.yerr_adder, 0.0], dtype=np.float64)

        self.att_mult = 1.0 

    def loadCartesianField(self, file_path='input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy', period = np.array([0, 1, 5], dtype=np.int32), coilCurrent=1.0, errField=False, att_mult='default_toroidal'):
        """Loads a vector field from a file and sets mesh properties.

        The function loads a 3D vector field from a .npy file and initializes the mesh grid properties
        based on the input array dimensions. The field arrays are assumed to be in Cartesian coordinates.

        Args:
            file_path (str, optional): Path to the .npy file containing the field arrays. Defaults to 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy'.
            period (np.ndarray, optional): Array specifying periodicity in [r, theta, phi] directions. Defaults to np.array([0, 1, 5], dtype=np.int32).
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
            elif att_mult =='ideal_toroidal':
                att_mult = 1.0
            elif att_mult =='ideal_poloidal':
                att_mult = -1.0
            elif isinstance(att_mult, float):
                att_mult = att_mult
            else:
                print(f"{self}: ATT_MULT IS NOT A FLOAT OR DEFAULT VALUE!!")
                att_mult = 1.0

            total_mult = att_mult * coilCurrent
            self.nr, self.ntheta, self.nphi = Bx_.shape
            self.periodicity = period
            #self.att_mult = att_mult
            self.Bx = Bx_ * total_mult
            self.By = By_ * total_mult
            self.Bz = Bz_ * total_mult
            self.errField = errField

            # r periodicity
            if self.periodicity[0]:
                self.r_max = self.a / self.periodicity[0]
                self.dr = self.r_max / self.nr
                self.r_min = self.dr
            else:
                self.r_max = self.a
                self.dr = self.r_max / (self.nr-1)
                self.r_min = 0.
            
            # theta periodicity
            if self.periodicity[1]:
                self.theta_max = (2*np.pi) / self.periodicity[1]
                self.dtheta = self.theta_max / self.ntheta
                self.theta_min = self.dtheta
            else:
                self.theta_max = (2*np.pi)
                self.dtheta = self.theta_max / (self.ntheta-1)
                self.theta_min = 0.
            
            # phi periodicity
            if self.periodicity[2]:
                self.phi_max = (2*np.pi) / self.periodicity[2]
                self.dphi = self.phi_max / self.nphi
                self.phi_min = self.dphi
            else:
                self.phi_max = (2*np.pi)
                self.dphi = self.phi_max / (self.nphi-1)
                self.phi_min = 0.

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
            elif att_mult =='ideal_toroidal':
                att_mult = 1.0
            elif att_mult =='ideal_poloidal':
                att_mult = -1.0
            elif isinstance(att_mult, float):
                att_mult = att_mult
            else:
                print(f"{self}: ATT_MULT IS NOT A FLOAT OR DEFAULT VALUE!!")
                att_mult = 1.0

            total_mult = att_mult * coilCurrent

            self.Bx += (Bx_ * total_mult)
            self.By += (By_ * total_mult)
            self.Bz += (Bz_ * total_mult)

    def setErrorField(self, mag=1.5654e-4, dir_deg=271.5*np.pi/180):
        """Sets the magnitude and direction of the non-periodic error field.

        The direction is measured from phi_c=0 (i.e., 18 degrees clockwise from the South Split).
        Also computes and stores the cosine and sine of the error direction for efficient interpolation.

        Args:
            err_mag (float, optional): Magnitude of the error field. Defaults to 1.5654e-4.
            err_dir (float, optional): Direction of the error field in radians. Defaults to 271.5*np.pi/180.
        """
        self.err_mag = mag
        self.err_dir = dir_deg
        self.cos_err_dir = np.cos(dir_deg)
        self.sin_err_dir = np.sin(dir_deg)
        self.xerr_adder = self.err_mag * self.cos_err_dir
        self.yerr_adder = -1 * self.err_mag * self.sin_err_dir
        self.err_adder = np.array([self.xerr_adder, self.yerr_adder, 0.0], dtype=np.float64).T
        #self.err_adder = torch.tensor(self.err_adder, dtype=torch.float64).to(device)


    def set_nonPer_errField(self, err_mag=1.5654e-4, err_dir=271.5*np.pi/180):
        """Sets the magnitude and direction of the non-periodic error field.

        The direction is measured from phi_c=0 (i.e., 18 degrees clockwise from the South Split).
        Also computes and stores the cosine and sine of the error direction for efficient interpolation.

        Args:
            err_mag (float, optional): Magnitude of the error field. Defaults to 1.5654e-4.
            err_dir (float, optional): Direction of the error field in radians. Defaults to 271.5*np.pi/180.
        """
        self.err_mag = err_mag
        self.err_dir = err_dir
        self.cos_err_dir = np.cos(err_dir)
        self.sin_err_dir = np.sin(err_dir)
        self.xerr_adder = self.err_mag * self.cos_err_dir
        self.yerr_adder = -1 * self.err_mag * self.sin_err_dir
        self.err_adder = np.array([self.xerr_adder, self.yerr_adder, 0.0], dtype=np.float64)

    def rot_vecXYZ_byPHI(self, vec_XYZ, delta_phi):
        """
        Rotates a 3D Cartesian vector around the z-axis by a given angle phi.

        Args:
            vec_XYZ (np.ndarray): A 3-element array representing the Cartesian vector [x, y, z].
            delta_phi (np.ndarray or float): The rotation angle in radians. Positive values rotate clockwise when viewed from above.

        Returns:
            np.ndarray: The rotated 3D Cartesian vector as a array of shape (3,).

        Notes:
            - Rotation is performed around the z-axis.
            - The z-component remains unchanged.
            - When viewed from above, positive phi rotates the vector clockwise.
        Function takes in a cartesian vector and a phi angle
        Returns the cartesian values of the vector rotated by phi degrees
        convention: When looking at across-section to the right of the +z axis, theta is counterclockwise
        convention: +phi is clockwise when viewed from above
        """
        rotated_XYZ = np.zeros(vec_XYZ.shape)
        xFormMatrix = np.array([[ np.cos(delta_phi), -np.sin(delta_phi), 0.0],
                                [ np.sin(delta_phi),  np.cos(delta_phi), 0.0],
                                [               0.0,                0.0, 1.0]])

        rotated_XYZ = np.dot(vec_XYZ, xFormMatrix)

        # mismatch in sign between 'Mesh' and 'eshNew' inmplementations?
        return rotated_XYZ

    def interpField(self, point_XYZ, Cart=True):
        """
        Returns the interpolated field values at a point defined in Cartesian coordinates.

        Interpolation is performed using a weighted sum of field values at each node of the enclosing cell.
        The weights are calculated as the volume of the octant opposed to each node.

        The input point is converted to RTP coordinates and kept within the following domains:
            r: 0.0 to r_max (minor radius)
            theta: dtheta to theta_max (2pi / Nperiods)
            phi: dphi to phi_max (2pi / Nperiods)

        Args:
            point_XYZ (np.ndarray): Input point(s) in Cartesian coordinates.
            Cart (bool): If True, input is in Cartesian coordinates; otherwise, RTP coordinates.

        Returns:
            np.ndarray: Array of shape (3, Npts), where Npts is the number of input points.
            Each column corresponds to the interpolated field components [Bx, By, Bz] at the respective point.
        """
        ## Sanitize input (or make sure we are passing torch tensors!)
        Npts = 1 #Npts = point_XYZ.shape[0]

        if Cart: point_RTP = XYZ_to_RTP(point_XYZ, self.R0) #.to(device)
        else: point_RTP = point_XYZ #.to(device)
        
        # periodic boundarys
        r_local  = point_RTP[0]
        if r_local < 0.0:
            #print(f"WARNING: r_local < 0.0 : {r_local} !! Setting r_local = 0.0")
            #exit()
            r_local *= -1.0
            point_RTP[1] += np.pi

        th_local = np.remainder(point_RTP[1], self.theta_max) # keep theta within 0 and theta_max!
        ph_localN, ph_local = np.divmod(point_RTP[2], self.phi_max) # keep phi within 0 and phi_max!



        vecXYZ = np.zeros([3,Npts], dtype=np.float64) 
        rindex : np.int16
        thindex : np.int16
        phindex : np.int16

        # indices and local (within cell) values
        thindex, th_el = np.divmod(th_local, self.dtheta)
        phindex, ph_el = np.divmod(ph_local, self.dphi)
        rindex, r_el = np.divmod(r_local, self.dr)
        if r_local >= self.r_max:
            rindex = self.nr - 2

        r_low = rindex * self.dr
        th_low = (thindex+1) * self.dtheta

        invr_el = self.dr - r_el
        invth_el = self.dtheta - th_el
        invph_el = self.dphi - ph_el

        r_lowr_el = (r_low + r_el/2) * r_el
        r_localinvr_el = (r_local + invr_el/2) * invr_el
        
        cos_th_low = np.cos(th_low)
        cos_th_local = np.cos(th_local)

        cos_th_low_r_low = r_low * cos_th_low
        cos_th_low_r_local = r_local * cos_th_low
        cos_th_local_r_low = r_low * cos_th_local
        cos_th_local_r_local = r_local * cos_th_local

        # sub-element volumes
        A1 = (self.R0 + cos_th_low_r_low)     * r_lowr_el      * th_el    #* ph_el
        A2 = (self.R0 + cos_th_low_r_local)   * r_localinvr_el * th_el    #* ph_el
        A3 = (self.R0 + cos_th_local_r_low)   * r_lowr_el      * invth_el #* ph_el
        A4 = (self.R0 + cos_th_local_r_local) * r_localinvr_el * invth_el #* ph_el
        A5 = (self.R0 + cos_th_low_r_low)     * r_lowr_el      * th_el    #* invph_el
        A6 = (self.R0 + cos_th_low_r_local)   * r_localinvr_el * th_el    #* invph_el
        A7 = (self.R0 + cos_th_local_r_low)   * r_lowr_el      * invth_el #* invph_el
        A8 = (self.R0 + cos_th_local_r_local) * r_localinvr_el * invth_el #* invph_el
        
        Areas = np.array([A1, A2, A3, A4, A5, A6, A7, A8])
        Areas[:4] *= ph_el
        Areas[4:] *= invph_el

        # indices of the 8 corner nodes of the cell
        ir_hi = rindex + 1
        ir_lo = rindex
        ith_hi = thindex
        ith_lo = thindex - 1
        iph_hi = phindex
        iph_lo = phindex - 1

        # indices of the 8 corner nodes of the cell
        index_array = np.array([[ir_hi, ith_hi, iph_hi], [ir_lo, ith_hi, iph_hi],
                    [ir_hi, ith_lo, iph_hi], [ir_lo, ith_lo, iph_hi],
                    [ir_hi, ith_hi, iph_lo], [ir_lo, ith_hi, iph_lo],
                    [ir_hi, ith_lo, iph_lo], [ir_lo, ith_lo, iph_lo]], dtype=np.int16)
        
        # B field vectors at the 8 corner nodes
        Bvecs = np.zeros((8, 3))
        Bvecs[:, 0] = self.Bx[index_array[:, 0], index_array[:, 1], index_array[:, 2]]
        Bvecs[:, 1] = self.By[index_array[:, 0], index_array[:, 1], index_array[:, 2]]
        Bvecs[:, 2] = self.Bz[index_array[:, 0], index_array[:, 1], index_array[:, 2]]

        # have to perform vector rotation if wrapping around in phi direction
        #if iph_lo < 0: Bvecs[4:] = self.rot_vecXYZ_byPHI( Bvecs[4:], -self.phi_max )
        if iph_hi == 0: Bvecs[4:] = self.rot_vecXYZ_byPHI( Bvecs[4:], -self.phi_max )

        # result is sum of B field vectors weighted with sub-element volumes
        vecXYZ = np.dot(Areas, Bvecs) / np.sum(Areas)

        # if mesh is defined with periodic symmetry, we perform a rotational transform based
        # on which 'period' of the mesh the point is located
        phi_rotation = ph_localN * self.phi_max  # angle of transform
        vecXYZ = self.rot_vecXYZ_byPHI(vecXYZ, phi_rotation)
    
        if self.errField: # non-periodic perturbative error field applied
            #vecXYZ *= self.att_mult
            # vecXYZ[0] += self.err_mag * self.cos_err_dir
            # vecXYZ[1] -= self.err_mag * self.sin_err_dir
            vecXYZ += self.err_adder

        return vecXYZ, ph_localN

    def loadScalarField(self, file_path, period = np.array([0, 1, 5], dtype=np.int32), errField=False, att_mult=1.0):
        """ 
        This function loads a vector field as a 3-dimensional scalar array for each cartesian vector.
        The grid properties are assumed from the dimensions of the input arrays
        """
        # self.value: np.float64[:][:][:]

        val_= np.load(file_path)
        # transpose val_
        val_ = np.transpose(val_, (2, 1, 0))


        self.nr, self.ntheta, self.nphi = val_.shape
        self.periodicity = period
        self.att_mult = att_mult
        self.value = val_ * att_mult
        self.errField = errField

        print(f"self.value.shape: {self.value.shape}")
        #print(f"self.value: {self.value}")
        # r periodicity
        if self.periodicity[0]:
            self.r_max = self.a / self.periodicity[0]
            self.dr = self.r_max / self.nr
            self.r_min = self.dr
        else:
            self.r_max = self.a
            self.dr = self.r_max / (self.nr-1)
            self.r_min = 0.
        
        # theta periodicity
        if self.periodicity[1]:
            self.theta_max = (2*np.pi) / self.periodicity[1]
            self.dtheta = self.theta_max / self.ntheta
            self.theta_min = self.dtheta
        else:
            self.theta_max = (2*np.pi)
            self.dtheta = self.theta_max / (self.ntheta-1)
            self.theta_min = 0.
        
        # phi periodicity
        if self.periodicity[2]:
            self.phi_max = (2*np.pi) / self.periodicity[2]
            self.dphi = self.phi_max / self.nphi
            self.phi_min = self.dphi
        else:
            self.phi_max = (2*np.pi)
            self.dphi = self.phi_max / (self.nphi-1)
            self.phi_min = 0.

    def interpScalarField(self, point_XYZ, Cart=True):
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
        Npts = 1 #Npts = point_XYZ.shape[0]

        if Cart: point_RTP = XYZ_to_RTP(point_XYZ, self.R0) #.to(device)
        else: point_RTP = point_XYZ #.to(device)
        
        # periodic boundarys
        r_local  = point_RTP[0]
        th_local = np.remainder(point_RTP[1], self.theta_max) # keep theta within 0 and theta_max!
        ph_localN, ph_local = np.divmod(point_RTP[2], self.phi_max) # keep phi within 0 and phi_max!

        scalarVal = np.zeros([1,Npts], dtype=np.float64) 
        rindex : np.int16
        thindex : np.int16
        phindex : np.int16

        # indices and local (within cell) values
        thindex, th_el = np.divmod(th_local, self.dtheta)
        phindex, ph_el = np.divmod(ph_local, self.dphi)
        rindex, r_el = np.divmod(r_local, self.dr)
        if r_local >= self.r_max:
            rindex = self.nr - 2

        r_low = rindex * self.dr
        th_low = (thindex+1) * self.dtheta

        invr_el = self.dr - r_el
        invth_el = self.dtheta - th_el
        invph_el = self.dphi - ph_el

        r_lowr_el = (r_low + r_el/2) * r_el
        r_localinvr_el = (r_local + invr_el/2) * invr_el
        
        cos_th_low = np.cos(th_low)
        cos_th_local = np.cos(th_local)

        cos_th_low_r_low = r_low * cos_th_low
        cos_th_low_r_local = r_local * cos_th_low
        cos_th_local_r_low = r_low * cos_th_local
        cos_th_local_r_local = r_local * cos_th_local

        # sub-element volumes
        A1 = (self.R0 + cos_th_low_r_low)     * r_lowr_el      * th_el    #* ph_el
        A2 = (self.R0 + cos_th_low_r_local)   * r_localinvr_el * th_el    #* ph_el
        A3 = (self.R0 + cos_th_local_r_low)   * r_lowr_el      * invth_el #* ph_el
        A4 = (self.R0 + cos_th_local_r_local) * r_localinvr_el * invth_el #* ph_el
        A5 = (self.R0 + cos_th_low_r_low)     * r_lowr_el      * th_el    #* invph_el
        A6 = (self.R0 + cos_th_low_r_local)   * r_localinvr_el * th_el    #* invph_el
        A7 = (self.R0 + cos_th_local_r_low)   * r_lowr_el      * invth_el #* invph_el
        A8 = (self.R0 + cos_th_local_r_local) * r_localinvr_el * invth_el #* invph_el
        
        Areas = np.array([A1, A2, A3, A4, A5, A6, A7, A8])
        Areas[:4] *= ph_el
        Areas[4:] *= invph_el

        # indices of the 8 corner nodes of the cell
        ir_hi = rindex + 1
        ir_lo = rindex
        ith_hi = thindex
        ith_lo = thindex - 1
        iph_hi = phindex
        iph_lo = phindex - 1

        # indices of the 8 corner nodes of the cell
        index_array = np.array([[ir_hi, ith_hi, iph_hi], [ir_lo, ith_hi, iph_hi],
                    [ir_hi, ith_lo, iph_hi], [ir_lo, ith_lo, iph_hi],
                    [ir_hi, ith_hi, iph_lo], [ir_lo, ith_hi, iph_lo],
                    [ir_hi, ith_lo, iph_lo], [ir_lo, ith_lo, iph_lo]], dtype=np.int16)
        
        # B field vectors at the 8 corner nodes
        nodeVals = np.zeros((8, 1))
        nodeVals[:, 0] = self.value[index_array[:, 0], index_array[:, 1], index_array[:, 2]]

        # result is sum of B field vectors weighted with sub-element volumes
        scalarVal = np.dot(Areas, nodeVals) / np.sum(Areas)

        return scalarVal, ph_localN