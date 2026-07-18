import gc
import numpy as np
from scipy.interpolate import splev, splrep
from scipy.integrate import dblquad
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from illiad.mesh import Mesh
from illiad.utilities.coordtrans import axisShift, RTP_to_XYZ, XYZ_to_RTP

np.set_printoptions(threshold=np.inf)
mpl.rcParams.update({
    # --- fonts & text (simIOP-friendly, ~8–12 pt at final size) ---
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

class FluxCalculator:
    """Class to handle toroidal flux calculation."""
    def __init__(self, IO_handler, b_hidra, input_params):
        """
        Initialize the flux calculator.

        Args:
            IO_handler: An object responsible for handling output operations, such as logging and directory creation.
            b_hidra: Magnetic field mesh/object.  
            input_params (dict): Dictionary of flux calculation input parameters. 
        """

        self.simIO = IO_handler
        self.field = b_hidra

        # Input parameters
        self.anlys_subdir = input_params['ANLYS_SUBDIR']
        self.lcfs_index = input_params['LCFS_INDEX']
        self.ntheta = input_params['NTHETA']
        self.phi_gens = input_params['PHI_GENs']
        self.max_subsets = input_params['MAX_SUBSETS']
        self.smooth_fctr = input_params['SMOOTH_FCTR']
        self.integrate_epsabs = input_params['INTEGRATE_EPSABS']
        self.integrate_epsrel = input_params['INTEGRATE_EPSREL']
        self.island_algorithm = input_params['ISLAND_ALGORITHM']
        self.hist_bins = input_params['HIST_BINS']
        self.plot_all = input_params['PLOT_ALL']
        self.big_mesh = input_params['BIG_MESH']

        # Calculation state
        self.nsurfaces = None
        self.smallest_island_index = None

        # Main output arrays
        self.tot_flux_array = None
        self.total_flux_norm = None
        self.centers_array = None
        self.valid_surfs = None

        # Optional mesh output arrays
        self.lcfs_flat_point = None
        self.flat_point_meshes = None
        

    def process_all_phi_angles(self):
        """Process every toroidal angle requested in 'self.phi_gens'.

        For each toroidal angle, this method:
        - Loads the corresponding Poincare data.
        - Initializes output arrays on the first angle.
        - Initializes optional spline plots.
        - Finds the magnetic axis.
        - Runs a first pass over surfaces to identify magnetic islands (subsets).
        - Runs a second pass to spline surfaces, calculate fluxes, validate surfaces, and store point/center data.

        Notes:
            This method fills the main output attributes, 'tot_flux_array',
            'centers_array', 'valid_surfs', and the point mesh arrays.
        """

        for phi_index, phi_deg in enumerate(self.phi_gens):
            flux_surfaces = self.load_poincare_data(phi_deg)
            self.nsurfaces = len(flux_surfaces)
            if phi_index == 0:
                self.init_output_arrays()
            if self.plot_all:
                fig, ax_rect, ax_hist, ax_polar = self.init_plotting()
            else:
                ax_rect = None
                ax_hist = None
                ax_polar = None
            mag_axis, mag_axis_rev = self.get_magnetic_axis(flux_surfaces)
            # process each flux surface to identify magnetic islands
            first_loop_output = self.identify_surface_subsets(flux_surfaces, mag_axis)
            smallest_island_index, num_subsets, subset_data, subset_centers, hist_output = first_loop_output
            self.smallest_island_index = smallest_island_index
            # spline surfaces, calculate surface fluxes, validate fits, and store output data.
            surface_fluxes = self.process_surfaces_for_phi(phi_index, phi_deg, flux_surfaces, mag_axis, 
                                        mag_axis_rev, smallest_island_index, num_subsets, 
                                        subset_data, subset_centers, hist_output, 
                                        ax_rect=ax_rect, ax_hist=ax_hist, ax_polar=ax_polar)

            for flux_index, flux in enumerate(surface_fluxes, start=self.lcfs_index):
                self.tot_flux_array[flux_index][phi_index] = np.sum(flux, axis=-1)

            if self.plot_all: 
                self.finalize_plotting(fig, ax_rect, ax_hist, ax_polar, phi_deg)

            del flux_surfaces
            if phi_index % 5 == 0:
                gc.collect()
        

    def normalize_fluxes(self):
        """Normalize toroidal flux values relative to the LCFS flux.

        Notes:
            Fluxes outside the LCFS are set equal to the LCFS flux before normalization.
            The normalized profile is clipped to the range [0, 1], and surfaces outside
            the LCFS are set to zero.
        """
         
        self.tot_flux_array[:self.lcfs_index][:] = self.tot_flux_array[self.lcfs_index][:]
        self.tot_flux_array = np.clip(self.tot_flux_array, 0.0, self.tot_flux_array[self.lcfs_index])
        # Set range of data to be between 0 and 1, and set data outside of LCFS to be equal to the 0
        self.total_flux_norm = 1 - self.tot_flux_array / self.tot_flux_array[self.lcfs_index]
        self.total_flux_norm = np.clip(self.total_flux_norm, 0.0, 1.0)
        self.total_flux_norm[:self.lcfs_index][:] = 0.0


    def save_output(self):
        # save the core flux arrays
        filename_fluxes = self.anlys_subdir + '/CalculatedFLuxes.npy'
        filename_fluxNorms = self.anlys_subdir + '/CalculatedFLuxes-normalized.npy'
        filename_validSurfaces = self.anlys_subdir + '/ValidSurfaces.npy'
        self.simIO.saveNumpyData(self.tot_flux_array, filename_fluxes)
        self.simIO.saveNumpyData(self.total_flux_norm, filename_fluxNorms)
        self.simIO.saveNumpyData(self.valid_surfs, filename_validSurfaces)
    
        # plot a big array of fluxes
        self.simIO.log.info('Plotting Flux v Surface Index...')
        fig_post = plt.figure()
        ax_psi = fig_post.add_subplot(211, ylabel='Toroidal Flux $\psi_{\phi}$\n$[g*m^2]$')
        ax_norm = fig_post.add_subplot(212, xlabel='Surface index', ylabel='Surface Parameter $\hat{\psi}_n$')

        for i in range(len(self.phi_gens)):
            ax_psi.plot(self.tot_flux_array[:,i]*10_000, label='{:d}'.format(int(self.phi_gens[i])), linewidth=1)
            ax_norm.plot(self.total_flux_norm.T[i], label='{:d}'.format(int(self.phi_gens[i])), linewidth=1)
        ax_psi.set_title('Toroidal Angles $\phi[\degree]$', loc='right', fontsize=8)
        ax_psi.set_ylim(0, 1.1*10_000*np.max(self.tot_flux_array)) 
        ax_psi.grid(which='both', linestyle=':', linewidth=0.5)
        ax_psi.legend(loc='upper right', fontsize=5,ncols=3)
        ax_norm.set_ylim(0, 1.1)
        ax_norm.grid(which='both', linestyle=':', linewidth=0.5)

        self.simIO.saveFig(self.anlys_subdir+'/Flux_v_Surface.png', dpi=250)
        self.simIO.log.info('Finished, LCFS=#{}, ISLAND AXIS=#{}'.format(self.lcfs_index, self.smallest_island_index))

        # save the numpy arrays to individual files using simIO method
        for surf_index in range(self.lcfs_index, self.nsurfaces):
            filename_center = self.anlys_subdir + '/fSurf_{:03d}_center.npy'.format(surf_index)
            self.simIO.saveNumpyData(self.centers_array[surf_index], filename_center)

            if self.big_mesh:
                filename_pt_mesh = self.anlys_subdir + '/fSurf_{:03d}_POINTmesh.npy'.format(surf_index)
                self.simIO.saveNumpyData(self.flat_point_meshes[surf_index], filename_pt_mesh)
            elif surf_index == self.lcfs_index:
                filename_pt_mesh = self.anlys_subdir + '/fSurf_{:03d}_POINTmesh.npy'.format(surf_index)
                self.simIO.saveNumpyData(self.lcfs_flat_point, filename_pt_mesh)
    
    
    def load_poincare_data(self, phi_deg):
        """Loads Poincare data for a specific toroidal angle."""
        filename = 'Poincare_{:03d}.npy'.format(int(phi_deg))
        flux_surfaces = self.simIO.loadNumpyData(filename, subdir="Poincare")
        return flux_surfaces
    

    def init_output_arrays(self):
        """Initializes the output arrays."""
        self.tot_flux_array = np.zeros([self.nsurfaces, len(self.phi_gens)])
        self.total_flux_norm = np.zeros([self.nsurfaces, len(self.phi_gens)])
        self.centers_array = np.zeros([self.nsurfaces, len(self.phi_gens), self.max_subsets, 2])
        if self.big_mesh:
            self.flat_point_meshes = np.full([self.nsurfaces, len(self.phi_gens), self.ntheta*self.max_subsets, 2], np.nan)
        else:
            self.lcfs_flat_point = np.full([len(self.phi_gens), self.ntheta*self.max_subsets, 2], np.nan)

        self.valid_surfs = np.zeros([self.nsurfaces, len(self.phi_gens)], dtype=bool) 
        self.valid_surfs[self.lcfs_index:][:] = True


    def get_magnetic_axis(self, flux_surfaces):
        """Finds the magnetic axis from the innermost (last) flux surface."""
        mag_axis = self.find_axis(*flux_surfaces[-1], self.field)
        mag_axis_rev = np.copy(mag_axis)
        mag_axis_rev[0] += np.pi

        return mag_axis, mag_axis_rev
    
    
    def process_surfaces_for_phi(self, phi_index, phi_deg, flux_surfaces, mag_axis, 
                                        mag_axis_rev, smallest_island_index, num_subsets, 
                                        subset_data, subset_centers, hist_output, 
                                        ax_rect=None, ax_hist=None, ax_polar=None):        
        """Process all flux surfaces for one toroidal angle.

        For each surface from the LCFS inward, this method fits splines, integrates the
        toroidal flux, stores flux/center/validity data, filters wild point meshes
        against the LCFS, and optionally updates the plots.

        Args:
            phi_index (int): Index of the current toroidal angle in 'self.phi_gens'.
            phi_deg (float): Toroidal angle in degrees.
            flux_surfaces (list of tuple): List where each element is a tuple (theta_array, r_array) representing a flux surface.
            mag_axis (np.ndarray): Magnetic axis in (r, theta) coordinates.
            mag_axis_rev (np.ndarray): Reversed magnetic axis, used to transform data to the geometric axis frame.
            smallest_island_index (int): Surface index used as the island alignment reference.
            num_subsets (np.ndarray): Number of detected subsets for each surface.
            subset_data (list): Surface/subset point data relative to the magnetic axis.
            subset_centers (list): Center of each subset.
            hist_output (tuple): Histogram data from the subset-identification pass.
            ax_rect (optional): Rectangular plot axis.
            ax_hist (optional): Histogram plot axis.
            ax_polar (optional): Polar plot axis.

        Returns:
            surface_fluxes (list): Flux values for each processed surface, starting at 'self.lcfs_index'.
        """
        surface_fluxes = []
        hist, bin_edges, wrap_flag = hist_output
        for surf_index in range(self.lcfs_index, self.nsurfaces):
            n_subsets = num_subsets[surf_index]
            # loop to spline, calculate flux, and create regularly-spaced points
            output = self.fit_and_integrate_surface_subsets(subset_data, subset_centers, surf_index, 
                                                            smallest_island_index, n_subsets, wrap_flag, 
                                                            mag_axis_rev, phi_deg)
            points_tr_geoAxis, spline_tr_magAxis, subCenters_geo, subset_flux_list, valid_surface = output
            # storing flux, center data, and validity
            surface_fluxes += [subset_flux_list]
            self.centers_array[surf_index][phi_index] = subCenters_geo
            self.valid_surfs[surf_index][phi_index] = valid_surface
            # use lcfs as reference for surface validation
            if surf_index == self.lcfs_index:
                spline_lcfs_magaxis = spline_tr_magAxis
            else:
                self.surface_validation(surf_index, phi_index,
                                        spline_tr_magAxis,
                                        n_subsets,
                                        spline_lcfs_magaxis)
                
            # flatten spline points for output storage and polar plotting
            total_npts = n_subsets * self.ntheta
            surface_tr_points = points_tr_geoAxis.reshape((total_npts, 2), order='C', copy=True)
            surface_thetas = surface_tr_points.T[0]
            surface_radii = surface_tr_points.T[1]
            # store lcfs data 
            if surf_index == self.lcfs_index:
                lcfs_points = surface_tr_points
                if not self.big_mesh:
                    self.lcfs_flat_point[phi_index][:total_npts] = lcfs_points
            # filter out wild fits
            lcfs_radii = axisShift(*lcfs_points.T, *mag_axis_rev)[1]
            surface_inside_lcfs = np.max(lcfs_radii) > np.max(surface_radii)
            if surface_inside_lcfs and self.big_mesh:
                self.flat_point_meshes[surf_index][phi_index][:total_npts] = surface_tr_points

            if self.plot_all:
                self.plot_flux_surfaces_for_phi(surf_index, phi_index, flux_surfaces,
                                                mag_axis, spline_tr_magAxis, num_subsets,
                                                ax_rect, ax_hist, ax_polar, bin_edges,
                                                hist, surface_thetas, surface_radii, surface_inside_lcfs)
                        
        return surface_fluxes
    

    def plot_flux_surfaces_for_phi(self, surf_index, phi_index, flux_surfaces, 
                                  mag_axis, spline_tr_magAxis, num_subsets, 
                                  ax_rect, ax_hist, ax_polar,bin_edges, 
                                  hist, surface_thetas, surface_radii, surface_inside_lcfs):
        """Plot rectangular, histogram, and polar diagnostics for one flux surface."""
        th_in, r_in = flux_surfaces[surf_index]
        r_in = r_in[~np.isnan(r_in)]
        th_in = th_in[~np.isnan(th_in)]

        points_tr_magAxis = axisShift(th_in, r_in, *mag_axis).T

        unique_indices = np.unique(points_tr_magAxis[:, 0], return_index=True)[1]
        points_tr_magAxis = points_tr_magAxis[unique_indices]
        points_tr_magAxis = points_tr_magAxis[np.argsort(points_tr_magAxis[:, 0])]
        # Plot thera-r Poincare points
        ax_rect.scatter(points_tr_magAxis.T[0]*180.0/np.pi, points_tr_magAxis.T[1], color='k', s=0.25, linewidths=0.0)
        
        # Overlay the spline fit used for integration.        
        if self.valid_surfs[surf_index][phi_index]:
            for i in range(0, num_subsets[surf_index]):
                ax_rect.scatter(spline_tr_magAxis[i].T[0]*180.0/np.pi, spline_tr_magAxis[i].T[1], s=0.5, linewidths=0.05)  

              
            ax_hist.bar(bin_edges[surf_index][:-1]*180.0/np.pi, hist[surf_index], 
                        width=np.diff(bin_edges[surf_index])*180.0/np.pi, align='edge', 
                        edgecolor='k', linewidth=0.1)
        
        if not surface_inside_lcfs:
            return
    
        if num_subsets[surf_index] == self.max_subsets:
            surface_centers_geoAxis = self.centers_array[surf_index][phi_index].T
        else:
            surface_centers_geoAxis = self.centers_array[surf_index][phi_index][0].T

        ax_polar.scatter(surface_thetas, surface_radii, s=0.3, linewidths=0.0)
        if surface_centers_geoAxis.shape[-1] > 1:
            ax_polar.scatter(surface_centers_geoAxis[0], surface_centers_geoAxis[1], s=15, color='k', marker='x',  linewidths=0.5)
    
        
    def finalize_plotting(self, fig, ax_rect, ax_hist, ax_polar, phi_deg):
        """Format, save, and close the plot for one toroidal angle."""
        fig.suptitle('Toroidal Angle $\phi={}\degree$'.format(int(phi_deg)), fontsize=12, x=0.18, y=0.98)
        # rectangular plot
        ax_rect.set_ylim(0, 0.19)
        ax_rect.set_xlim(0, 360)
        ax_rect.tick_params(axis='both', which='major', labelsize=6)
        ax_rect.set_xticks(np.arange(0, 361, 90))
        ax_rect.set_xticklabels([])
        ax_rect.set_yticks(np.arange(0.0, 0.19, 0.025))
        ax_rect.set_yticklabels(['', '2.5', '5', '7.5', '10', '12.5', '15', '17.5'])
        ax_rect.grid(linewidth=0.5, linestyle=':', c='grey')
        ax_rect.set_ylabel('[cm]', fontsize=8)
        # histogram plot
        ax_hist.set_xlim(0, 360)
        ax_hist.set_xticks(np.arange(0, 361, 90))
        ax_hist.tick_params(axis='both', which='major', labelsize=6)
        ax_hist.grid(linewidth=0.5, linestyle=':', c='grey')
        ax_hist.set_xlabel('Poloidal Angle $\\theta[\degree]$', fontsize=8)
        ax_hist.set_ylabel('[#]', fontsize=8)
        # polar plot
        ax_polar.set_rlim(0, 0.19)
        ax_polar.tick_params(axis='both', which='major', labelsize=8) #,pad=10)
        ax_polar.grid(linewidth=0.5, linestyle=':', c='grey')
        ax_polar.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                            labels=['Low\nField', '', '', '', 'High\nField', '', '', ''], fontsize=8)
        ax_polar.set_rgrids([0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175],
                        labels=['', '5cm', '', '10cm', '', '15cm', ''], angle=0, fontsize=4)

        self.simIO.log.info('Plotting Flux Surfaces @ phi={}'.format(phi_deg))
        self.simIO.saveFig(self.anlys_subdir +'/Splines_{:03d}deg.png'.format(int(phi_deg)), dpi=300)
        plt.close()
    

    def find_axis(self, theta_vals: np.ndarray, r_vals: np.ndarray, field: Mesh) -> np.ndarray:
        """Computes the geometric center (axis) of a set of points in (r, theta) coordinates.

        The function filters NaNs from the input arrays, converts the remaining(r, theta)
        points to Cartesian coordinates, and calculates the averages to determine the axis.
        The axis is then converted back to (theta, r) coordinates.

        Args:
            theta_vals (np.ndarray): Array of theta values (angles in radians).
            r_vals (np.ndarray): Array of r values (radial distances).
            field (Mesh): Mesh object with attribute R0, used for coordinate transformations.

        Returns:
            axis_thetar (np.ndarray): Array containing the axis position in (theta, r) coordinates.
        """

        r_vals = r_vals[~np.isnan(r_vals)]
        theta_vals = theta_vals[~np.isnan(theta_vals)]
        theta_vals[theta_vals < 0.0] += 2*np.pi # ensure all theta values are positive for averaging

        # Vectorized coordinate transformation instead of loop
        coords_3d = np.array([RTP_to_XYZ(np.array([r, theta, 0.0]), field.R0) for r, theta in zip(r_vals, theta_vals)])

        axis_xyz = np.mean(coords_3d, axis=0)
        axis_thetar = XYZ_to_RTP(axis_xyz, field.R0)[1::-1]

        # find the indices of all theta_vals within 20 degrees of axis_thetar[0]
        theta_bins = np.deg2rad(20.0)  # 20 degree in radians
        dtheta_to_axis = np.minimum(np.abs(theta_vals - axis_thetar[0]), 2 * np.pi - np.abs(theta_vals - axis_thetar[0]))
        theta_indices = np.where(dtheta_to_axis < theta_bins)[0]

        # Compute the average radius for the selected indices
        axis_thetar[1] = np.mean(r_vals[theta_indices])

        return axis_thetar
    
    
    def find_subsets(self, theta_r_pts, mag_axis):
        """Function to find contiguous subsets of points in theta-r space
        Args:
            theta_r_pts (np.ndarray): Array of points in (theta, r) coordinates, wrt to the magnetic Axis.
            mag_axis (np.ndarray): Magnetic axis position in (theta, r) coordinates, wrt to the geometric Axis.

        Returns:
            split_data (list): List of arrays containing the split data points wrt to magnetic axis.
            found_centers (np.ndarray): Array of found centers wrt to magnetic axis for each subset.
            hist (np.ndarray): Histogram of point density vs theta.
            bin_edges (np.ndarray): Edges of the bins used for the histogram.
            wrapped_flag (bool): Flag indicating if the data wraps around.
        """
        # build theta histogram to identify separated angular regions.
        hist, bin_edges = np.histogram(theta_r_pts.T[0], bins=self.hist_bins, range=(0.0, 2*np.pi))
        dtheta_bin = bin_edges[1] - bin_edges[0]

        # find contiguous sets of non-zero bins
        non_empty_bins = np.where(hist > 0)[0]
        contiguous_sets = np.split(non_empty_bins, np.where(np.diff(non_empty_bins) != 1)[0]+1)
        # if first and last bins are non-empty, then the first and last sets of bins are contiguous
        if hist[0] > 0 and hist[-1] > 0 and len(contiguous_sets) > 1:
            contiguous_sets[0] = np.concatenate((contiguous_sets[-1], contiguous_sets[0]))
            contiguous_sets.pop()
            wrapped_flag = True
        else:
            wrapped_flag = False
        num_sets = len(contiguous_sets)

        # split the data points into subsets
        split_data_magAxis = []
        found_centers = np.zeros([num_sets, 2])
        for i, contiguous_set in enumerate(contiguous_sets):
            subset_points = []
            lower_bound = contiguous_set*dtheta_bin
            upper_bound = lower_bound + dtheta_bin
            # append data within each bin belonging to the subset
            for lo, hi in zip(lower_bound, upper_bound):
                mask = (theta_r_pts[:, 0] >= lo) & (theta_r_pts[:, 0] < hi)
                subset_points += list(theta_r_pts[mask])
            subset_points = np.array(subset_points)
            subset_points = subset_points[np.argsort(subset_points[:, 0])] # sort by theta

            # split if the # of sets is equal to input 'MAX_SUBSETS': 
            if num_sets == self.max_subsets:
                found_centers[i][:] = self.find_axis(subset_points.T[0], subset_points.T[1], self.field)
                split_data_magAxis += [subset_points]
            else: # keep the original magnetic axis
                found_centers[i][:] = mag_axis[:2]
                split_data_magAxis = [theta_r_pts]

        return split_data_magAxis, found_centers, hist, bin_edges, wrapped_flag
        

    def identify_surface_subsets(self, flux_surfaces, mag_axis):
        """
        Processes a range of flux surfaces to identify and analyze magnetic islands.

        For each flux surface in the specified range, this method:
        - Shifts the origin of (r, theta) coordinates to the magnetic axis.
        - Sorts the data and removes duplicate theta values.
        - Finds subsets (potential magnetic islands) and their local centers.
        - Computes mean radii for each subset.
        - Identifies the island (subset) with the smallest mean radius among surfaces with multiple subsets.

        Args:
            flux_surfaces (list of tuple): List where each element is a tuple (theta_array, r_array) representing a flux surface.
            mag_axis (array-like): Coordinates of the magnetic axis (e.g., [x, y]).

        Returns:
            smallest_island_index (int): Index of the flux surface containing the smallest-radius island among those with multiple islands.
            num_subsets (np.ndarray): Array of the number of subsets found for each surface.
            split_data_magAxis (list): List of lists containing subset data wrt to magnetic axis for each surface.
            surface_axes_magAxis (list): List of local centers wrt to magnetic axis for each subset in each surface.
            hist_data (tuple): Tuple containing (hist, bin_edges, wrap_flag) for each surface, useful for diagnostics or plotting.
        """
        num_subsets = np.zeros(self.nsurfaces, dtype=int)
        set_mean_rads = np.zeros(self.nsurfaces)
        split_data_magAxis = [0]*self.nsurfaces
        surface_axes_magAxis = [0]*self.nsurfaces
        hist = [0]*self.nsurfaces
        bin_edges = [0]*self.nsurfaces
        wrap_flag = [0]*self.nsurfaces
        for surf_index in range(self.lcfs_index, self.nsurfaces):
            th_in, r_in = flux_surfaces[surf_index]
            r_in = r_in[~np.isnan(r_in)]
            th_in = th_in[~np.isnan(th_in)]
            # shift origin of r, theta coords from geo center to mag axis
            pts_tr_magAxis = axisShift(th_in, r_in, *mag_axis).T
            # remove duplicate theta values and sort data in increasing theta
            unique_indices = np.unique(pts_tr_magAxis[:, 0], return_index=True, return_counts=False)[1:]
            pts_tr_magAxis = pts_tr_magAxis[unique_indices]
            pts_tr_magAxis = pts_tr_magAxis[np.argsort(pts_tr_magAxis[:, 0])]

            # Find subsets and local centers. Returned points are in the magnetic-axis frame.
            output_histogram_method = self.find_subsets(pts_tr_magAxis, mag_axis)
            hist[surf_index], bin_edges[surf_index] = output_histogram_method[2:4]
            wrap_flag[surf_index] = output_histogram_method[-1]
            
            if self.island_algorithm == 'histogram':  output = output_histogram_method
            else: raise ValueError(f"Unknown ISLAND_ALGORITHM: {self.island_algorithm}")

            # initialize split_data with data w.r.t. magnetic axis
            split_data_magAxis[surf_index], surface_axes_magAxis[surf_index] = output[:2]
            num_subsets[surf_index] = len(split_data_magAxis[surf_index])

            # loop through subsets
            subset_mean_rads = np.zeros(num_subsets[surf_index])
            for subset_index in range(num_subsets[surf_index]):
                subset_points = split_data_magAxis[surf_index][subset_index]
                if num_subsets[surf_index] > 1:
                    rad_toSpline = axisShift(
                        subset_points[:, 0],
                        subset_points[:, 1],
                        *surface_axes_magAxis[surf_index][subset_index],
                    )[1]
                else:
                    rad_toSpline = subset_points[:, 1]
                subset_mean_rads[subset_index] = np.mean(rad_toSpline)
            set_mean_rads[surf_index] = np.mean(subset_mean_rads)

        # find the index of the islands of smallest radius
        island_indices = np.where(num_subsets > 1)[0]
        if island_indices.size > 0: smallest_island_index = island_indices[np.argmin(set_mean_rads[island_indices])]
        else: smallest_island_index = self.nsurfaces - 1  #If no islands found, return the last surface index

        if np.any(np.isnan(surface_axes_magAxis[smallest_island_index])):
            print('NaN detected in smallest island centers!!!!')
            print(f'{surface_axes_magAxis[smallest_island_index]=}' )
        hist_data = (hist, bin_edges, wrap_flag)
        return smallest_island_index, num_subsets, split_data_magAxis, surface_axes_magAxis, hist_data
    

    def init_plotting(self):
        """Initialize the plotting for the flux calculations."""
        width_in = 15 / 2.54 # 15 cm in inches
        height_in = width_in * (9/16)  # Maintain 16:9 aspect ratio
        fig = plt.figure(figsize=(width_in, height_in))

        gs = gridspec.GridSpec(2, 2, width_ratios=[1.3, 1], wspace=0.35, hspace=0.0)
        ax_polar = fig.add_subplot(gs[:,0], polar=True)
        ax_rect = fig.add_subplot(gs[0,1], polar=False)
        ax_hist = fig.add_subplot(gs[1,1], polar=False)

        # Reduce margins around the entire figure
        plt.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.12)

        return fig, ax_rect, ax_hist, ax_polar


    def integrate_flux(self, spline_parms, spline_axis, phi, field, err_abs=1e-5, err_rel=1e-3):
        """
        Integrates the total toroidal flux contained within the given spline, defined relative to the specified center.

        Args:
            spline_parms (tuple): Spline parameters as returned by `scipy.interpolate.splrep`. w.r.t. spline_axis
            spline_axis (array-like): The (theta, r) coordinates of the spline's center. (w.r.t. magnetic axis)
            phi (float): Toroidal angle (in radians) at which to evaluate the flux.
            field (Mesh): Mesh object providing the `interpField` method for magnetic field interpolation.
            err_abs (float, optional): Absolute error tolerance for integration. Defaults to 1e-5.
            err_rel (float, optional): Relative error tolerance for integration. Defaults to 1e-3.

        Returns:
            float: Scalar value of the toroidal flux for the (sub)set.
        """
        # integration helper functions
        def flux_integrand(r, theta, phi, field, axis):
            """Function to calculate the toroidal field times radius at a given point in space"""
            geo_point = np.array([*axisShift(r, theta, *axis), phi])
            bxy = field.interpField(geo_point, Cart=False)[0][:2]

            # calculate the toroidal flux integrand: r*B_toroidal = r*( -Bx*sin(phi) - By*cos(phi) )
            return -r*( bxy[0]*np.sin(phi) + bxy[1]*np.cos(phi) )

        # define the upper radial bound of the integration
        def hfun(theta): return splev(theta, spline_parms)

        # integrate toroidal flux
        spline_axis_rev = np.array(spline_axis, copy=True)
        spline_axis_rev[0] += np.pi
        PSI, abserr = dblquad(flux_integrand, 0., 2*np.pi,
                            0.0, hfun, args=(phi, field, spline_axis_rev),
                            epsabs=err_abs, epsrel=err_rel)

        return float(PSI)


    def shift_the_subcenters(self, surf_index, smallest_island_index, subset_centers, num_subsets, wrap_flag):
        """Function performs tests to see if there is a misalignment of subset centers 
        between the smallest island set and the current island set. If so, it returns the 
        appropriate r and theta values to shift the data set to be relative to the smallest island subcenters """
    
        smallest_centers = subset_centers[smallest_island_index]
        these_centers = subset_centers[surf_index]

        shifted_data = np.zeros([num_subsets, 2])
        shifted_data = axisShift(*smallest_centers.T, *these_centers.T).T
        dtheta_this_to_smallest = smallest_centers[0][0] - these_centers[0][0]
        misalign_cond = abs(dtheta_this_to_smallest) > np.pi / 2.  # subset centers don't line up between current and smallest island subset

        first_set_theta = smallest_centers[0][0]  # first set's dist. to 0
        last_set_theta = 2*np.pi - smallest_centers[-1][0]  # last set's dist. to 0
        proximity_cond = first_set_theta > last_set_theta

        shift_flag = wrap_flag and proximity_cond and misalign_cond
        if shift_flag:
            for subset_index in range(num_subsets):
                    shifted_data[subset_index] = axisShift(*smallest_centers[subset_index-1], *these_centers[subset_index])

        return shifted_data, shift_flag
    

    def spline_data(self, theta_pts: np.ndarray, rad_pts: np.ndarray, smoothing=1e-5):
        """Create a smoothing spline fit of the data points.

        This function constructs a cubic smoothing spline for the given (theta, radius) points.
        To improve endpoint behavior and approximate periodicity, the data is extended at both
        ends before fitting. The function returns the spline parameters as produced by `scipy.interpolate.splrep`.

        Args:
            theta_pts (np.ndarray): Array of theta values (angles in radians).
            rad_pts (np.ndarray): Array of corresponding radius values.
            smoothing (float, optional): Smoothing factor for the spline. Defaults to 1e-5.

        Returns:
            tuple: Spline parameters as returned by `scipy.interpolate.splrep`, including the
            tuple (t, c, k), the sum of squared residuals, a flag indicating failure, and a message.
        """

        valid_mask = ~(np.isnan(theta_pts) | np.isnan(rad_pts))
        theta_valid = theta_pts[valid_mask]
        rad_valid = rad_pts[valid_mask]

        if len(theta_valid) <= 3:
            return None, None, True, "Not enough points for cubic spline (need > 3)"
        # copy data to both ends for pseudo-periodicity (smooth spline endpoints) 
        append_length = int(len(theta_valid)/2)

        th_A = theta_valid[append_length:-1] - 2*np.pi
        th_B = theta_valid[1:append_length] + 2*np.pi
        theta_spl = np.concatenate((th_A, theta_valid, th_B))

        rad_A = rad_valid[append_length:-1]
        rad_B = rad_valid[1:append_length]
        rad_spl = np.concatenate((rad_A, rad_valid, rad_B))

        spline_rep = splrep(theta_spl, rad_spl, k=3, s=smoothing, per=False, full_output=1, quiet=1)

        return spline_rep


    def fit_and_integrate_surface_subsets(self, subset_data, subset_centers, surf_index, 
                      smallest_island_index, num_subsets, wrap_flag, 
                      mag_axis_rev, phi_deg):
        """
    Loop through subsets of a flux surface, fit spline, calculate flux, and generate
        regularly-spaced points evaluated on the spline fit.

        Args:
            subset_data (list): List of data arrays for each subset of each surface.
            subset_centers (list): List of center coordinates for each subset of each surface.
            surf_index (int): Index of the current flux surface.
            smallest_island_index (int): Index of the smallest island subset.
            num_subsets (int): Number of subsets for the current flux surface.
            wrap_flag (list): List of flags indicating if wrapping is needed for each surface.
            mag_axis_rev (tuple): Magnetic axis coordinates for reverse axis shift.
            phi_deg (float): Phi angle in degrees.

        Returns:
            splined_tr_GeoAxis (np.ndarray): Regularly-spaced points in geometric coordinates.
            splined_tr_MagAxis (np.ndarray): Regularly-spaced points in magnetic axis coordinates.
            subset_flux_list (list): List of flux values for each subset.
            subCenters_geo (np.ndarray): Centers of subsets in geometric coordinates.
            valid_surface (bool): Flag indicating if the surface is valid for interpolationn.
        """

        radius_evals_locAxis = np.zeros([num_subsets, self.ntheta])
        splined_tr_MagAxis   = np.zeros([num_subsets, self.ntheta, 2])
        splined_tr_GeoAxis   = np.zeros([num_subsets, self.ntheta, 2])
        subCenters_geo       = np.zeros([num_subsets, 2])
        theta_gens = np.linspace(2*np.pi/self.ntheta, 2*np.pi, self.ntheta)

        # align current island subsets with the smallest island reference set        
        if num_subsets > 1: 
            shifted_subcenters, shiftint = self.shift_the_subcenters(surf_index, smallest_island_index, 
                                                                     subset_centers, num_subsets, wrap_flag[surf_index] )
        else: shiftint = 0
        # loop through subsets to spline, calculate flux, create regularly-spaced points
        subset_flux_list = []
        for subset_idx in range(num_subsets):
            current_center = subset_centers[surf_index][subset_idx]
            current_data = np.array(subset_data[surf_index][subset_idx], copy=True)

            # shift data points from [rel. to magaxis] to [rel. to the centers of the smallest islands]
            if num_subsets == self.max_subsets:
                current_data = axisShift(current_data[:, 0], current_data[:, 1], *current_center).T
                current_data = axisShift(current_data[:, 0], current_data[:, 1], *shifted_subcenters[subset_idx]).T

                current_data = current_data[np.argsort(current_data[:, 0])]
                current_center = subset_centers[smallest_island_index][subset_idx-shiftint]
                subCenters_geo[subset_idx][:] = axisShift(*current_center, *mag_axis_rev)
            else:
                subCenters_geo[subset_idx] = current_center

            # fit spline to the current subset.
            surface_spline_params, res, fail, msg = self.spline_data(*current_data.T, smoothing=self.smooth_fctr)

            if fail:
                valid_surface = False # not a valid surface for interpolation
                self.simIO.log.info('\t(B)Surface #{} NOT A VALID SURFACE!!! Residual: {}'.format(surf_index, res) )
                self.simIO.log.info('\tSurface #{}, fail: {}\tmsg: {}'.format(surf_index, bool(fail), msg) )
                continue
            else:
                self.simIO.log.info( '\t(B)Surface #{} valid. Residual: {:.3e}'.format(surf_index, res) )
                valid_surface = True # valid surface for interpolation
            # integrate an amount of toroidal field bounded by the flux surface [Tesla*m^2]
            this_flux = self.integrate_flux(surface_spline_params, np.array(current_center, copy=True),
                                            phi_deg*np.pi/180, self.field, err_abs=self.integrate_epsabs, 
                                            err_rel=self.integrate_epsrel)
            subset_flux_list += [this_flux]

            # create a set of regularly-spaced points evaluated on the spline fit
            current_theta_gens = (theta_gens + current_center[0]) % (2*np.pi)
            radius_evals_locAxis[subset_idx] = splev(current_theta_gens, surface_spline_params)
            # shift island subsets back into the magnetic axis frame
            if num_subsets > 1:
                shift_r = current_center[1]
                shift_theta = current_center[0] + np.pi
                splined_tr_MagAxis[subset_idx] = axisShift(current_theta_gens,
                                                           radius_evals_locAxis[subset_idx],
                                                           shift_theta, shift_r,).T
            else:
                splined_tr_MagAxis[subset_idx, :, 0] = current_theta_gens
                splined_tr_MagAxis[subset_idx, :, 1] = radius_evals_locAxis[subset_idx]

            # shift r, theta back relative to geometric axis
            splined_tr_GeoAxis[subset_idx] = axisShift(
                splined_tr_MagAxis[subset_idx, :, 0],
                splined_tr_MagAxis[subset_idx, :, 1],
                *mag_axis_rev,
            ).T

        return splined_tr_GeoAxis, splined_tr_MagAxis, subCenters_geo, subset_flux_list, valid_surface
    

    def surface_validation(self, surf_index, phi_index, spline_tr_magAxis, n_subsets, spline_lcfs_magaxis):
        """Mark a surface invalid if its spline geometry is unphysical (negative radii or extension beyond LCFS).""" 

        if n_subsets == 1:
            has_negative_radius = np.any(spline_tr_magAxis[:,:,1] < 0.0)
            delta_radius_to_lcfs = spline_lcfs_magaxis[0,:,1] - spline_tr_magAxis[0,:,1] 
            extends_beyond_lcfs = np.any(delta_radius_to_lcfs <= 0.0)
            if has_negative_radius or extends_beyond_lcfs:
                self.valid_surfs[surf_index][phi_index] = False # not a valid surface for interpolation
                self.simIO.log.info( '\t(A)Surface #{} NOT A VALID SURFACE!!!'.format(surf_index) )

        elif np.any(spline_tr_magAxis[:,:,1] < 0.0):
            self.valid_surfs[surf_index][phi_index] = False # not a valid surface for interpolation
            self.simIO.log.info( '\t(A)Surface #{} NOT A VALID SURFACE!!!'.format(surf_index) )


    def run(self):
        """
        Run the full flux calculation workflow.

        This method processes all requested toroidal angles to calculate fluxes, normalizes the calculated
        fluxes relative to the LCFS, and saves output files. Results are stored in
        'tot_flux_array', 'total_flux_norm', and 'valid_surfs' attributes.
        """
        self.process_all_phi_angles()
        self.normalize_fluxes()
        self.save_output()    
