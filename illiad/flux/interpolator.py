"""Normalized-flux interpolation onto the field mesh."""

import gc
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchrbf import RBFInterpolator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _polar_interp_points_to_xy(thetas, rads):
    """Convert polar interpolation points from (theta, r) to Cartesian-like (x, y)."""
    return np.array([rads * np.cos(thetas), rads * np.sin(thetas)]).T


def _surface_indices(indices):
    if indices is None:
        return np.array([], dtype=int)
    return np.atleast_1d(np.asarray(indices, dtype=int))


class FluxInterpolator:
    """Classs to handle interpolation of the normalized flux profiles onto the field mesh."""
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
        self.input_params = input_params

        # Input parameters
        self.anlys_subdir = input_params["ANLYS_SUBDIR"]
        self.plot_subdir = os.path.join(self.anlys_subdir, "nfield")
        self.lcfs_index = input_params["LCFS_INDEX"]
        self.phi_gens = input_params["PHI_GENs"]
        self.alpha = input_params.get("ALPHA", 1.0)
        self.guess_phi_index = input_params.get("GUESS_PHI_INDEX", None)
        self.inv_surf_indices = _surface_indices(input_params.get("INV_SURF_INDICES", []))
        self.output_file_name = input_params.get("OUTPUT_FILE_NAME", "default")
        self.debug = input_params.get("DEBUG", False)

        self.rbf_kernel = input_params.get("RBF_KERNEL")
        self.rbf_neighbors = input_params.get("RBF_NEIGHBORS")
        self.rbf_smoothing = input_params.get("RBF_SMOOTHING")
        self.rbf_epsilon = input_params.get("RBF_EPSILON")
        
        # Future parameters for shifted flux
        self.set_lcfs_value = 0.01



        # Loaded data
        self.tot_flux_array = None
        self.n_surfaces = None
        self.valid_surface = None
        self.axis_array = None

        # Processed flux data
        self.total_flux_norm = None
        self.shifted_flux_norm = None

        # Selected profile
        self.best_phi_index = None
        self.linear_flux_array = None
        self.profile_select_str = None

        # Interpolation output
        self.grid_theta = None
        self.grid_rad = None
        self.interpolated_surface_parm = None


    def load_flux_data(self):
        """Load normalized flux values, surface validity, and magnetic axis data."""

        filepath = self.anlys_subdir + '/'
        tot_flux_array_name = filepath + 'CalculatedFLuxes.npy'
        self.tot_flux_array = self.simIO.loadNumpyData(tot_flux_array_name)
        self.n_surfaces = self.tot_flux_array.shape[0]

        valid_surf_name = filepath + 'ValidSurfaces.npy'
        self.valid_surface = self.simIO.loadNumpyData(valid_surf_name)

        filename_center = filepath + 'fSurf_{:03d}_center.npy'.format(self.n_surfaces-1)
        self.axis_array = self.simIO.loadNumpyData(filename_center)


    def select_best_flux_profile(self):
        """
        Method chooses one toroidal angle profile from self.total_flux_norm.
        The selected profile is adjusted using the ALPHA parameter,
        filtered using valid surface flags, and stored as
        self.linear_flux_array.

        Notes:
            Surfaces outside the LCFS and surfaces listed in
            INV_SURF_INDICES are manually marked invalid. The LCFS itself is
            manually marked valid.
        """

        sum_flux = np.nansum(self.total_flux_norm, axis=0)
        if self.guess_phi_index is not None:
            self.best_phi_index = np.argsort(sum_flux)[self.guess_phi_index]
        else:
            self.best_phi_index = np.argsort(sum_flux)[-5]
        linear_flux_array = self.total_flux_norm[:, self.best_phi_index]
        # adjust profile with ALPHA parameter
        linear_flux_array = 1 - (1 - linear_flux_array)**self.alpha
        # shift the profile to prescribe nonzero value at LCFS (Not currently used)
        self.shifted_flux_norm = self.set_lcfs_value + self.total_flux_norm * (1.0 - self.set_lcfs_value)

        # if valid_surface has shape (surface, phi), select the chosen phi profile
        # if it already has shape (surface,), leave it alone
        if self.valid_surface.ndim == 2:
            self.valid_surface = self.valid_surface[:, self.best_phi_index]
        self.valid_surface[self.lcfs_index] = True # manually set LCFS surface to valid
        self.valid_surface[:self.lcfs_index] = False # manually set surfaces outside LCFS to invalid
        self.valid_surface[self.inv_surf_indices] = False # manually set picked surfaces outside LCFS to invalid

        self.linear_flux_array = linear_flux_array
        self.profile_select_str = '"Best" flux profile, at phi={:03d} deg'.format(int(self.phi_gens[self.best_phi_index]))
        self.simIO.log.info(self.profile_select_str)

        self.simIO.log.info(f'Valid Surfaces: {self.valid_surface}')


    def save_best_flux_profile(self):
        """Saves plot of the "best" flux profile.

        If DEBUG is enabled , the plot is also displayedad.
        """

        fig, ax = plt.subplots()
        ax.bar(range(len(self.linear_flux_array)), self.linear_flux_array)
        ax.set_xlabel('Surface Index')
        ax.set_ylabel('Flux')
        ax.grid(True)
        ax.set_title(self.profile_select_str)
        self.simIO.saveFig(self.plot_subdir + '/Best_Flux_Profile.png', dpi=300)
        if self.debug:
            plt.show()
        else:
            plt.close(fig)


    def save_interpolated_data(self):
        """
        Converts the interpolated torch tensor to a NumPy array, saves it as an
        .npy file, and writes one polar plot for each toroidal angle.
        """

        interpolated_surface_parm_np = self.interpolated_surface_parm.detach().to("cpu").numpy()
        self.simIO.saveNumpyData(interpolated_surface_parm_np,
                                 self.anlys_subdir + '/nField_' + self.output_file_name + '.npy')

        for phi_index, phi_deg in enumerate(self.phi_gens):
            self.output_phi_plots(phi_deg, interpolated_surface_parm_np[phi_index],
                                  name='psi_hat',
                                  subdir=self.plot_subdir, output_handler=self.simIO,
                                  colormap='Blues', plotmin=0.0, plotmax=1.0)


    def perform_interpolation(self):
        """Interpolate the selected flux profile onto the mesh for every phi.

        Method loops through all toroidal angles in self.phi_gens, interpolates
        each angle independently, and stores the results in self.interpolated_surface_parm.
        """

        grid_shape, self.grid_theta, self.grid_rad, interpol_pts = self.create_meshgrid()
        for phi_index, phi_deg in enumerate(self.phi_gens):
            interpolated_angle = self.interpolate_one_phi(phi_deg, grid_shape, interpol_pts)

            # Skip theta=0 row because it was only added for interpolation periodicity.
            self.interpolated_surface_parm[phi_index] = interpolated_angle[1:]

            if phi_index % 10 == 0:
                gc.collect()


    def interpolate_one_phi(self, phi_deg, grid_shape, interpol_pts):
        """Interpolate the flux profile for a single toroidal angle."""

        points, flux_norm = self.obtain_poincare_data(phi_deg)
        source_points_xy = _polar_interp_points_to_xy(points[:, 0], points[:, 1])

        points_torch = torch.as_tensor(source_points_xy, device=device, dtype=torch.float32)
        flux_norm_torch = torch.as_tensor(flux_norm, device=device, dtype=torch.float32)

        interpolation = RBFInterpolator(points_torch, flux_norm_torch, kernel=self.rbf_kernel,
                                            neighbors=self.rbf_neighbors, smoothing=self.rbf_smoothing, epsilon=self.rbf_epsilon)

        # Work around torchrbf device placement: ensure internal tensors/buffers are on the same device.
        interpolation = interpolation.to(device)
        interpolation.smoothing = interpolation.smoothing.to(device)
        interpolated_angle = interpolation(interpol_pts).reshape(grid_shape)

        ## HACKY SOLUTIONS HERE!!!
        # copying values out for r=0.0
        fred3 = interpolated_angle.T[1]
        fred4 = interpolated_angle.T[2]
        fred3[fred3==0] = fred4[fred3==0]
        interpolated_angle.T[1] = fred3
        interpolated_angle.T[0] = interpolated_angle.T[1]

        return interpolated_angle


    def create_meshgrid(self):
        """Create the target mesh used by the RBF interpolator.

        Method builds a (theta, r) mesh from the field mesh limits, converts the
        mesh points to Cartesian (x, y) points, and initializes
        self.interpolated_surface_parm to store the final interpolated output.
        """

        # Create a meshgrid for the interpolation
        rads = np.linspace(self.field.r_min, self.field.r_max, self.field.nr)
        thetas = np.linspace(0, self.field.theta_max, self.field.ntheta+1) #add theta=0 for proper interpolation
        grid_theta, grid_rad = np.meshgrid(thetas, rads, indexing='ij')
        grid_shape = grid_theta.shape
        interpol_pts = _polar_interp_points_to_xy(grid_theta.ravel(), grid_rad.ravel())
        interpol_pts = torch.as_tensor(interpol_pts, device=device, dtype=torch.float32)

        self.interpolated_surface_parm = torch.zeros([len(self.phi_gens), len(thetas)-1, len(rads)], device=device, dtype=torch.float32)

        return grid_shape, grid_theta, grid_rad, interpol_pts


    def obtain_poincare_data(self, phi_deg):
        """Load Poincare points and matching flux values for one phi angle."""
        filename = 'Poincare_{:03d}.npy'.format(int(phi_deg))
        flux_surfaces = self.simIO.loadNumpyData(filename, subdir="Poincare")
        points = np.zeros([1,2])
        flux_norm = np.ones(1)
        points[0] = self.axis_array[0][0]
        for surface_index in range(self.lcfs_index, self.n_surfaces):
            if self.valid_surface[surface_index] == False:
                self.simIO.log.info(f'Skipping surface {surface_index} (not valid)')
            else:
                thetas = flux_surfaces[surface_index][0]
                rads = flux_surfaces[surface_index][1]
                # filter NaNs
                thetas = thetas[~np.isnan(thetas)]
                rads = rads[~np.isnan(rads)]
                N_pts = len(thetas)
                # concatenate to big array of points
                these_points = np.array([thetas, rads]).T
                points = np.concatenate((points, these_points))

                these_flux_norms = np.full(N_pts, self.linear_flux_array[surface_index])
                flux_norm = np.concatenate((flux_norm, these_flux_norms))

        return points, flux_norm


    def output_phi_plots(self, phi_deg, data, name, subdir, output_handler, colormap='inferno', plotmin=None, plotmax=None):
        """Save a polar plot of interpolated data for one toroidal angle."""

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_title(name + '\n$\phi_{{phy}}$={:02.0f}$\degree$ CW from North Split\n$\phi_c$={:02.0f}$\degree$'.format((phi_deg+198.)%360., phi_deg), loc='left')
        plot_data = np.vstack((data[-1], data))
        c = ax.pcolormesh(self.grid_theta, self.grid_rad, plot_data, shading='gouraud', cmap=colormap, vmin=plotmin, vmax=plotmax)

        ax.set_rmax(0.19)
        ax.set_rticks([])
        plt.grid(False)
        fig.colorbar(c, ax=ax, label='Flux')
        fig_path = subdir + '/' + name + '_{:03d}.png'.format(int(phi_deg))
        output_handler.saveFig(fig_path, dpi=300)
        output_handler.log.info('Saved figure: ' + fig_path)
        plt.close()


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
        

    def run(self):
        """
        Run the full flux interpolation workflow."""
        self.load_flux_data()
        self.normalize_fluxes()
        self.select_best_flux_profile()
        self.save_best_flux_profile()
        self.perform_interpolation()
        self.save_interpolated_data()
        self.simIO.log.info("## Flux interpolation complete. ##")
