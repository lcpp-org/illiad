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


def _wrap_to_pi(angle):
    """Wrap an angle in radians to [-pi, pi)."""
    return (angle + np.pi) % (2*np.pi) - np.pi


def _surface_points(thetas, rads, max_points=None):
    """Return finite, unique, poloidally ordered points from one surface."""
    thetas = np.asarray(thetas)
    rads = np.asarray(rads)
    finite = np.isfinite(thetas) & np.isfinite(rads)
    pairs = np.column_stack((thetas[finite], rads[finite]))
    if not pairs.size:
        return pairs

    _, first_indices = np.unique(pairs, axis=0, return_index=True)
    pairs = pairs[np.sort(first_indices)]
    pairs = pairs[np.argsort(np.mod(pairs[:, 0], 2*np.pi), kind="stable")]
    if max_points is not None and len(pairs) > max_points:
        keep = np.linspace(0, len(pairs) - 1, max_points, dtype=int)
        pairs = pairs[keep]
    return pairs


def _axis_point_at_phi(axis_array, phi_deg, axis_phi_gens=None):
    """Periodically interpolate the magnetic axis to one toroidal plane."""
    axis_points = []
    for phi_index, plane in enumerate(np.asarray(axis_array)):
        candidates = plane.reshape(-1, 2)
        finite = np.all(np.isfinite(candidates), axis=1)
        if not np.any(finite):
            raise ValueError(
                f"No finite magnetic-axis point for upstream phi index {phi_index}"
            )
        axis_points.append(candidates[finite][0])

    axis_points = np.asarray(axis_points)
    if axis_phi_gens is None:
        axis_phi_gens = np.linspace(
            360.0 / len(axis_points), 360.0, len(axis_points)
        )
    source_phi = np.radians(axis_phi_gens)
    axis_u = axis_points[:, 1] * np.cos(axis_points[:, 0])
    axis_v = axis_points[:, 1] * np.sin(axis_points[:, 0])
    target_phi = np.radians(phi_deg)
    u = np.interp(target_phi, source_phi, axis_u, period=2*np.pi)
    v = np.interp(target_phi, source_phi, axis_v, period=2*np.pi)
    return np.array([np.mod(np.arctan2(v, u), 2*np.pi), np.hypot(u, v)])


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

        self.interpolation_mode = str(
            input_params.get("FLUX_INTERPOLATION_MODE", "3d")
        )
        if self.interpolation_mode not in {"2d", "3d"}:
            raise ValueError("FLUX_INTERPOLATION_MODE must be '2d' or '3d'")

        self.rbf_phi_half_window = int(input_params.get("RBF_PHI_HALF_WINDOW", 2))
        self.rbf_phi_scale = float(input_params.get("RBF_PHI_SCALE", self.field.R0))
        self.rbf_points_per_surface_per_phi = int(
            input_params.get("RBF_POINTS_PER_SURFACE_PER_PHI", 72)
        )
        if self.rbf_phi_half_window < 0:
            raise ValueError("RBF_PHI_HALF_WINDOW must be nonnegative")
        if self.interpolation_mode == "3d" and self.rbf_phi_half_window < 1:
            raise ValueError("3d interpolation requires RBF_PHI_HALF_WINDOW >= 1")
        if self.rbf_phi_scale <= 0:
            raise ValueError("RBF_PHI_SCALE must be positive")
        if self.rbf_points_per_surface_per_phi < 1:
            raise ValueError("RBF_POINTS_PER_SURFACE_PER_PHI must be positive")
        if self.rbf_neighbors is not None:
            self.rbf_neighbors = int(self.rbf_neighbors)
            if self.rbf_neighbors < 1:
                raise ValueError("RBF_NEIGHBORS must be positive or null")

        self.plane_sample_cache = {}

        # Loaded data
        self.flux_norm_array = None
        self.n_surfaces = None
        self.valid_surface = None
        self.axis_array = None
        self.axis_phi_gens = None
        self.profile_phi_gens = None

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
        flux_norm_name = filepath + 'CalculatedFLuxes-normalized.npy'
        self.flux_norm_array = self.simIO.loadNumpyData(flux_norm_name)
        self.n_surfaces = self.flux_norm_array.shape[0]

        valid_surf_name = filepath + 'ValidSurfaces.npy'
        self.valid_surface = self.simIO.loadNumpyData(valid_surf_name)

        filename_center = filepath + 'fSurf_{:03d}_center.npy'.format(self.n_surfaces-1)
        self.axis_array = self.simIO.loadNumpyData(filename_center)
        profile_nphi = (
            self.flux_norm_array.shape[1]
            if self.flux_norm_array.ndim == 2 else 1
        )
        if profile_nphi == len(self.phi_gens):
            self.profile_phi_gens = np.asarray(self.phi_gens, dtype=np.float64)
        else:
            self.profile_phi_gens = np.linspace(
                360.0 / profile_nphi, 360.0, profile_nphi
            )
            self.simIO.log.info(
                "Mapping upstream flux/axis data from "
                f"{profile_nphi} planes onto {len(self.phi_gens)} output planes"
            )
        if len(self.axis_array) == len(self.phi_gens):
            self.axis_phi_gens = np.asarray(self.phi_gens, dtype=np.float64)
        elif len(self.axis_array) == len(self.profile_phi_gens):
            self.axis_phi_gens = self.profile_phi_gens


    def select_best_flux_profile(self):
        """
        Method chooses one toroidal angle profile from self.flux_norm_array.
        The selected profile is adjusted using the ALPHA parameter,
        filtered using valid surface flags, and stored as
        self.linear_flux_array.

        Notes:
            Surfaces outside the LCFS and surfaces listed in
            INV_SURF_INDICES are manually marked invalid. The LCFS itself is
            manually marked valid.
        """

        sum_flux = np.nansum(self.flux_norm_array, axis=0)
        if self.guess_phi_index is not None:
            self.best_phi_index = np.argsort(sum_flux)[self.guess_phi_index]
        else:
            self.best_phi_index = np.argsort(sum_flux)[-5]
        linear_flux_array = self.flux_norm_array[:, self.best_phi_index]
        # Adjust profile with ALPHA while retaining 1 at the axis and 0 at the LCFS.
        linear_flux_array = 1 - (1 - linear_flux_array)**self.alpha
        if self.valid_surface.ndim == 2:
            self.valid_surface = self.valid_surface[:, self.best_phi_index]
        self.valid_surface = np.asarray(self.valid_surface, dtype=bool).copy()
        self.valid_surface[self.lcfs_index] = True
        self.valid_surface[:self.lcfs_index] = False
        self.valid_surface[self.inv_surf_indices] = False

        self.linear_flux_array = linear_flux_array
        self.profile_select_str = '"Best" flux profile, at phi={:03d} deg'.format(
            int(self.profile_phi_gens[self.best_phi_index])
        )
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
        """Interpolate the selected flux profile onto the mesh for every phi."""

        grid_shape, self.grid_theta, self.grid_rad, query_points_2d = (
            self.create_meshgrid()
        )
        query_points_3d = torch.column_stack((
            query_points_2d,
            torch.zeros(
                query_points_2d.shape[0],
                device=query_points_2d.device,
                dtype=query_points_2d.dtype,
            ),
        ))
        for phi_index, phi_deg in enumerate(self.phi_gens):
            interpolated_angle = self.interpolate_one_phi(
                phi_index, grid_shape, query_points_2d, query_points_3d
            )

            # Skip theta=0 row because it was only added for interpolation periodicity.
            self.interpolated_surface_parm[phi_index] = interpolated_angle[1:]

            if phi_index % 10 == 0:
                gc.collect()


    def _fit_rbf(self, source_points, source_values, query_points, grid_shape):
        """Fit and evaluate one local RBF using the shared RBF settings."""
        points_torch = torch.as_tensor(source_points, device=device, dtype=torch.float64)
        values_torch = torch.as_tensor(source_values, device=device, dtype=torch.float64)
        query_torch = torch.as_tensor(query_points, device=device, dtype=torch.float64)
        neighbors = self.rbf_neighbors
        if neighbors is not None:
            neighbors = min(neighbors, len(source_points))

        interpolation = RBFInterpolator(
            points_torch,
            values_torch,
            kernel=self.rbf_kernel,
            neighbors=neighbors,
            smoothing=self.rbf_smoothing,
            epsilon=self.rbf_epsilon,
        )
        # Work around torchrbf device placement: ensure internal tensors/buffers are on the same device.
        interpolation = interpolation.to(device)
        interpolation.smoothing = interpolation.smoothing.to(device)
        return interpolation(query_torch).reshape(grid_shape)

    def _regularize_axis(self, interpolated_angle, phi_deg):
        """Apply the existing near-axis repair and poloidal axis average."""
        # Repair zero values on the innermost finite-radius shell from the next
        # radial shell, then enforce a single poloidally averaged value at
        # rho=0. This removes unphysical angular variation at the coordinate
        # singularity while retaining the nearby interpolated radial profile.
        inner_shell = interpolated_angle[:, 1].clone()
        next_shell = interpolated_angle[:, 2]
        invalid_inner = (~torch.isfinite(inner_shell)) | (inner_shell == 0)
        inner_shell[invalid_inner] = next_shell[invalid_inner]
        if not torch.all(torch.isfinite(inner_shell)):
            raise ValueError(
                f"Flux interpolation produced nonfinite values near rho=0 at phi={phi_deg}"
            )
        interpolated_angle[:, 1] = inner_shell
        interpolated_angle[:, 0] = torch.mean(inner_shell)
        return interpolated_angle

    def _interpolation_sources_2d(self, phi_index):
        """Build source coordinates and labels for one 2-D interpolation."""
        source_points, source_values, _ = self.load_plane_samples(phi_index)
        source_points = _polar_interp_points_to_xy(
            source_points[:, 0], source_points[:, 1]
        )
        return source_points, source_values

    def _interpolation_sources_3d(self, phi_index):
        """Build periodic local source coordinates and labels for 3-D mode."""
        nphi = len(self.phi_gens)
        window_size = 2*self.rbf_phi_half_window + 1
        if window_size > nphi:
            raise ValueError(
                "The 3d interpolation phi window cannot contain more planes "
                "than PHI_GENs"
            )

        target_phi = np.radians(self.phi_gens[phi_index])
        offsets = np.arange(
            -self.rbf_phi_half_window, self.rbf_phi_half_window + 1
        )
        source_indices = (phi_index + offsets) % nphi
        source_points = []
        source_values = []

        for source_index in source_indices:
            points, values, source_phi = self.load_plane_samples(
                source_index,
                max_points=self.rbf_points_per_surface_per_phi,
            )
            delta_phi = _wrap_to_pi(source_phi - target_phi)
            w = self.rbf_phi_scale * delta_phi
            theta = points[:, 0]
            rho = points[:, 1]
            source_points.append(np.column_stack((
                rho*np.cos(theta),
                rho*np.sin(theta),
                np.full(len(points), w),
            )))
            source_values.append(values)

        return np.vstack(source_points), np.concatenate(source_values)

    def interpolate_one_phi(
        self, phi_index, grid_shape, query_points_2d, query_points_3d
    ):
        """Interpolate one target plane using the configured dimensionality."""
        if self.interpolation_mode == "2d":
            source_points, source_values = self._interpolation_sources_2d(
                phi_index
            )
            query_points = query_points_2d
        else:
            source_points, source_values = self._interpolation_sources_3d(
                phi_index
            )
            query_points = query_points_3d

        interpolated_angle = self._fit_rbf(
            source_points, source_values, query_points, grid_shape
        )
        return self._regularize_axis(
            interpolated_angle, self.phi_gens[phi_index]
        )

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
        interpol_pts = torch.as_tensor(interpol_pts, device=device, dtype=torch.float64)

        self.interpolated_surface_parm = torch.zeros(
            [len(self.phi_gens), len(thetas)-1, len(rads)],
            device=device,
            dtype=torch.float64,
        )

        return grid_shape, grid_theta, grid_rad, interpol_pts


    def load_plane_samples(self, phi_index, max_points=None):
        """Load and cache labeled Poincare samples for one toroidal plane."""
        phi_index = int(phi_index) % len(self.phi_gens)
        cache_key = (phi_index, max_points)
        if cache_key in self.plane_sample_cache:
            return self.plane_sample_cache[cache_key]

        phi_deg = self.phi_gens[phi_index]
        filename = 'Poincare_{:03.0f}.npy'.format(phi_deg)
        flux_surfaces = self.simIO.loadNumpyData(filename, subdir="Poincare")
        axis_point = _axis_point_at_phi(
            self.axis_array, phi_deg, self.axis_phi_gens
        )
        point_blocks = [axis_point[None, :]]
        value_blocks = [np.array([1.0])]
        for surface_index in range(self.lcfs_index, self.n_surfaces):
            if not self.valid_surface[surface_index]:
                self.simIO.log.info(f'Skipping surface {surface_index} (not valid)')
                continue

            points = _surface_points(
                flux_surfaces[surface_index][0],
                flux_surfaces[surface_index][1],
                max_points=max_points,
            )
            if not points.size:
                continue
            point_blocks.append(points)
            value_blocks.append(np.full(
                len(points), self.linear_flux_array[surface_index]
            ))

        if len(point_blocks) == 1:
            raise ValueError(f"No valid Poincare samples found in {filename}")

        samples = (
            np.vstack(point_blocks),
            np.concatenate(value_blocks),
            np.radians(phi_deg),
        )
        self.plane_sample_cache[cache_key] = samples
        return samples


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
        fig_path = subdir + '/' + name + '_{:03.0f}.png'.format(phi_deg)
        output_handler.saveFig(fig_path, dpi=300)
        output_handler.log.info('Saved figure: ' + fig_path)
        plt.close()


    def run(self):
        """
        Run the full flux interpolation workflow."""
        self.load_flux_data()
        self.select_best_flux_profile()
        self.save_best_flux_profile()
        self.perform_interpolation()
        self.save_interpolated_data()
        self.simIO.log.info("## Flux interpolation complete. ##")
