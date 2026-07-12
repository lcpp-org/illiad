## IN THIS FILE, WE WILL TAKE THE CALCULATED FLUX FOR EACH SURFACE,
## APPLY IT TO THE POINTS ON THE THEIR RESPECTIVE SURFACE, 
## AND THEN INTERPOLATE IT ONTO A MESH THE SAME SHAPE/RESOLUTION AS THE BFIELD MESH
from re import DEBUG
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#from scipy.interpolate import griddata, RBFInterpolator, CloughTocher2DInterpolator
from torchrbf import RBFInterpolator
import matplotlib.pyplot as plt
import gc
from classes.mesh import *
from classes.iohandler import IOHandler


def _polar_interp_points(thetas, rads):
    return np.array([rads * np.cos(thetas), rads * np.sin(thetas)]).T


def _wrap_to_pi(angle):
    """Wrap angular differences to [-pi, pi)."""
    return (angle + np.pi) % (2*np.pi) - np.pi


def _surface_indices(indices):
    if indices is None:
        return np.array([], dtype=int)
    return np.atleast_1d(np.asarray(indices, dtype=int))


def _axis_point_at_phi(axis_array, phi_deg):
    """Periodically interpolate the stored magnetic-axis anchors to phi_deg."""
    axis_points = np.asarray(axis_array)[:, 0]
    axis_xy = _polar_interp_points(axis_points[:, 0], axis_points[:, 1])
    axis_phi = np.linspace(360.0/len(axis_points), 360.0, len(axis_points))
    periodic_phi = np.concatenate(([0.0], axis_phi, [axis_phi[0] + 360.0]))
    periodic_xy = np.vstack((axis_xy[-1], axis_xy, axis_xy[0]))
    target_phi = float(phi_deg) % 360.0
    axis_u = np.interp(target_phi, periodic_phi, periodic_xy[:, 0])
    axis_v = np.interp(target_phi, periodic_phi, periodic_xy[:, 1])
    axis_r = np.hypot(axis_u, axis_v)
    axis_theta = np.arctan2(axis_v, axis_u) % (2*np.pi)
    return np.array([axis_theta, axis_r])


def _thin_surface_points(thetas, rads, max_points):
    """Select at most max_points approximately angle-balanced contour samples."""
    finite = np.isfinite(thetas) & np.isfinite(rads)
    thetas = np.asarray(thetas)[finite]
    rads = np.asarray(rads)[finite]
    if max_points is None or len(thetas) <= max_points:
        return thetas, rads
    order = np.argsort(thetas, kind='stable')
    select = np.linspace(0, len(order) - 1, max_points, dtype=int)
    selected = order[select]
    return thetas[selected], rads[selected]


def _load_plane_samples(simIO, phi_deg, axis_array, linear_flux_array,
                        valid_surface, lcfs_index, n_surfaces, max_points=None):
    """Load labeled Poincare samples and the LCFS contour for one phi plane."""
    filename = 'Poincare_{:03d}.npy'.format(int(phi_deg))
    flux_surfaces = simIO.loadNumpyData(filename)
    point_blocks = [_axis_point_at_phi(axis_array, phi_deg)[None, :]]
    value_blocks = [np.ones(1)]

    for surface_index in range(lcfs_index, n_surfaces):
        if not valid_surface[surface_index]:
            continue
        thetas, rads = _thin_surface_points(
            flux_surfaces[surface_index][0], flux_surfaces[surface_index][1], max_points
        )
        if len(thetas) == 0:
            continue
        point_blocks.append(np.column_stack((thetas, rads)))
        value_blocks.append(np.full(len(thetas), linear_flux_array[surface_index]))

    lcfs_thetas, lcfs_rads = flux_surfaces[lcfs_index]
    lcfs_valid = np.isfinite(lcfs_thetas) & np.isfinite(lcfs_rads)
    return (
        np.concatenate(point_blocks),
        np.concatenate(value_blocks),
        (lcfs_thetas[lcfs_valid], lcfs_rads[lcfs_valid]),
    )


def fluxInterpolator(input_params=None):
    ## LOAD INPUT PARAMETERS
    if input_params is not None:
        print(f'{input_params.keys()=}')
        for key, value in input_params.items():
            print(f'{key}: {value}')
            globals()[str(key)] = value

    rbf_kernel = globals().get("RBF_KERNEL", "multiquadric")
    rbf_neighbors = globals().get("RBF_NEIGHBORS", 45)
    rbf_smoothing = globals().get("RBF_SMOOTHING", 1e-0)
    rbf_epsilon = globals().get("RBF_EPSILON", 1000)
    interpolation_mode = str(globals().get("FLUX_INTERPOLATION_MODE", "per_plane_2d")).lower()
    rbf_phi_half_window = int(globals().get("RBF_PHI_HALF_WINDOW", 2))
    rbf_phi_scale = float(globals().get("RBF_PHI_SCALE", 0.72))
    rbf_points_per_surface = int(globals().get("RBF_POINTS_PER_SURFACE_PER_PHI", 72))
    rbf_neighbors_3d = int(globals().get("RBF_NEIGHBORS_3D", 256))
    if interpolation_mode not in {"per_plane_2d", "periodic_3d"}:
        raise ValueError("FLUX_INTERPOLATION_MODE must be 'per_plane_2d' or 'periodic_3d'")
    if rbf_phi_half_window < 1:
        raise ValueError("RBF_PHI_HALF_WINDOW must be at least 1")
    if rbf_points_per_surface < 1 or rbf_neighbors_3d < 1:
        raise ValueError("3-D RBF point and neighbor counts must be positive")
    inv_surf_indices = _surface_indices(globals().get("INV_SURF_INDICES", []))
    input_log_params = globals().copy()
    input_log_params.update({
        "RBF_KERNEL": rbf_kernel,
        "RBF_NEIGHBORS": rbf_neighbors,
        "RBF_SMOOTHING": rbf_smoothing,
        "RBF_EPSILON": rbf_epsilon,
        "FLUX_INTERPOLATION_MODE": interpolation_mode,
        "RBF_PHI_HALF_WINDOW": rbf_phi_half_window,
        "RBF_PHI_SCALE": rbf_phi_scale,
        "RBF_POINTS_PER_SURFACE_PER_PHI": rbf_points_per_surface,
        "RBF_NEIGHBORS_3D": rbf_neighbors_3d,
        "INV_SURF_INDICES": inv_surf_indices.tolist(),
    })

    ## DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
    simIO = IOHandler(ANLYS_DIR)
    simIO.setActiveSubDir(ANLYS_SUBDIR)
    simIO.startLog(log_name="fluxInterpolator.log", subdir=ANLYS_SUBDIR, logger_name="FluxInterpolator")
    simIO.inputsBoilerplate(
        "FLUX INTERPOLATOR INPUTS",
        input_log_params,
        [
            "ANLYS_DIR",
            "ANLYS_SUBDIR",
            "CURRENT_TOR",
            "CURRENT_HEL",
            "CONFIG_TOR",
            "CONFIG_HEL",
            "ENABLE_ERRFIELD",
            "LCFS_INDEX",
            "SMALLEST_ISLAND_INDEX",
            "PHI_GENs",
            "MAX_SUBSETS",
            "ALPHA",
            "INV_SURF_INDICES",
            "GUESS_PHI_INDEX",
            "OUTPUT_FILE_NAME",
            "RBF_KERNEL",
            "RBF_NEIGHBORS",
            "RBF_SMOOTHING",
            "RBF_EPSILON",
            "FLUX_INTERPOLATION_MODE",
            "RBF_PHI_HALF_WINDOW",
            "RBF_PHI_SCALE",
            "RBF_POINTS_PER_SURFACE_PER_PHI",
            "RBF_NEIGHBORS_3D",
        ],
    )
    ## DEFINE MESH AND LOAD MAGNETIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField(coilCurrent=CURRENT_TOR, errField=ENABLE_ERRFIELD, att_mult=CONFIG_TOR)
    b_hidra.addFieldPerturbation(coilCurrent=CURRENT_HEL, att_mult=CONFIG_HEL)
    b_hidra.set_nonPer_errField()

    ## LOAD FLUX DATA
    filepath = ANLYS_SUBDIR  + '/'
    flux_norm_name = filepath + 'CalculatedFLuxes-normalized.npy'
    flux_norm_array = simIO.loadNumpyData(flux_norm_name)
    N_surfaces = flux_norm_array.shape[0]
    # LOAD VALID SURFACE DATA
    validSurf_name = filepath + 'ValidSurfaces.npy'
    valid_surface = simIO.loadNumpyData(validSurf_name)
    # Load Magnetic Axis point:
    filename_center = filepath + 'fSurf_{:03d}_center.npy'.format(N_surfaces-1)
    axis_array = simIO.loadNumpyData(filename_center)

    # CAN GET AWAY WITH FUXING NPHI<360!
    # if SMALLEST_ISLAND_INDEX:
    #     filename_center_island = filepath + 'fSurf_{:03d}_center.npy'.format(SMALLEST_ISLAND_INDEX)
    #     island_axis_array = simIO.loadNumpyData(filename_center_island)

    # CHOOSING ONE 'WELL-BEHAVED' ANGLE FOR THE CALCULATION (NO FAILED CALCULATIONS)
    # sum flux_norm_array along first axis and find index of max value
    sum_flux = np.nansum(flux_norm_array, axis=0)
    #best_phi_index = np.argsort(sum_flux)[-17]
    if GUESS_PHI_INDEX:
        best_phi_index = np.argsort(sum_flux)[GUESS_PHI_INDEX]
    else:
        best_phi_index = np.argsort(sum_flux)[-5]
    linear_flux_array = flux_norm_array[:, best_phi_index]
    # Adjust profile with ALPHA parameter, keeping 0.1 at the LCFS and 1.0 at the axis
    linear_flux_array = 0.1 + 0.9 * (1 - (1 - linear_flux_array)**ALPHA)

    if valid_surface.ndim == 2: # if valid_surface has multiple phi angles
        valid_surface = valid_surface[:, best_phi_index]
    valid_surface[LCFS_INDEX] = True # manually set LCFS surface to valid
    valid_surface[:LCFS_INDEX] = False # manually set surfaces outside LCFS to invalid
    valid_surface[inv_surf_indices] = False # manually set surfaces outside LCFS to invalid

    profile_select_str = '"Best" flux profile, at phi={:03d} deg'.format(int(PHI_GENs[best_phi_index]))
    print(profile_select_str)
    print(f'{valid_surface=}')
    #valid_indices = np.where(valid_surface)[0] # find the indices where valid_surface is True
    # flux parameter vs surface index plot
    fig, ax = plt.subplots()
    #ax.plot(valid_indices, linear_flux_array[valid_indices])
    ax.bar(range(len(linear_flux_array)), linear_flux_array)
    ax.set_xlabel('Surface Index')
    ax.set_ylabel('Flux')
    ax.grid(True)
    ax.set_title(profile_select_str)
    simIO.saveFig(ANLYS_SUBDIR + '/' + 'Best_Flux_Profile.png', dpi=300)
    if DEBUG:
        plt.show()

    # Create a meshgrid for the interpolation
    RADS = np.linspace(b_hidra.r_min, b_hidra.r_max, b_hidra.nr)
    THETAS = np.linspace(0, b_hidra.theta_max, b_hidra.ntheta+1) #add theta=0 for proper interpolation
    grid_theta, grid_rad = np.meshgrid(THETAS, RADS, indexing='ij')
    grid_shape = grid_theta.shape
    interpol_pts_2d = _polar_interp_points(grid_theta.ravel(), grid_rad.ravel())
    interpol_pts_2d = torch.as_tensor(interpol_pts_2d, device=device, dtype=torch.float32)
    interpol_pts_3d = torch.column_stack((
        interpol_pts_2d,
        torch.zeros(len(interpol_pts_2d), device=device, dtype=torch.float32),
    ))

    big_grid_linear = torch.zeros([len(PHI_GENs), len(THETAS)-1, len(RADS)], device=device, dtype=torch.float32)
    plane_sample_cache = {}

    ## LOOP THROUGH PHI ANGLES
    for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):
        if interpolation_mode == "per_plane_2d":
            points, flux_norm, target_lcfs = _load_plane_samples(
                simIO, PHI_GEN_DEG, axis_array, linear_flux_array, valid_surface,
                LCFS_INDEX, N_surfaces,
            )
            interp_points = _polar_interp_points(points[:, 0], points[:, 1])
            query_points = interpol_pts_2d
            neighbor_count = min(rbf_neighbors, len(interp_points))
        else:
            source_blocks = []
            value_blocks = []
            target_lcfs = None
            for phi_offset in range(-rbf_phi_half_window, rbf_phi_half_window + 1):
                source_index = (phi_index + phi_offset) % len(PHI_GENs)
                if source_index not in plane_sample_cache:
                    plane_sample_cache[source_index] = _load_plane_samples(
                        simIO, PHI_GENs[source_index], axis_array, linear_flux_array,
                        valid_surface, LCFS_INDEX, N_surfaces,
                        max_points=rbf_points_per_surface,
                    )
                source_points, source_values, source_lcfs = plane_sample_cache[source_index]
                if source_index == phi_index:
                    target_lcfs = source_lcfs
                source_uv = _polar_interp_points(source_points[:, 0], source_points[:, 1])
                delta_phi = _wrap_to_pi(
                    np.radians(float(PHI_GENs[source_index]) - float(PHI_GEN_DEG))
                )
                source_w = np.full((len(source_uv), 1), rbf_phi_scale*delta_phi)
                source_blocks.append(np.column_stack((source_uv, source_w)))
                value_blocks.append(source_values)

            interp_points = np.concatenate(source_blocks)
            flux_norm = np.concatenate(value_blocks)
            query_points = interpol_pts_3d
            neighbor_count = min(rbf_neighbors_3d, len(interp_points))

        points_torch = torch.as_tensor(interp_points, device=device, dtype=torch.float32)
        flux_norm_torch = torch.as_tensor(flux_norm, device=device, dtype=torch.float32)
        interpolation = RBFInterpolator(
            points_torch, flux_norm_torch, kernel=rbf_kernel,
            neighbors=neighbor_count, smoothing=rbf_smoothing, epsilon=rbf_epsilon,
        )
        
        #interpolation = RBFInterpolator(points_torch, flux_norm_torch, kernel='linear', neighbors=15, smoothing=1e-5, degree=1) #, epsilon=1e4)
        #interpolation = RBFInterpolator(points_torch, flux_norm_torch, kernel='linear', neighbors=55, smoothing=1e-5, degree=1) #, epsilon=1e4)

        # Work around torchrbf device placement: ensure internal tensors/buffers are on the same device.
        interpolation = interpolation.to(device)
        interpolation.smoothing = interpolation.smoothing.to(device)

        grid_linear = interpolation(query_points).reshape(grid_shape)

        ## HACKY SOLUTIONS HERE!!!
        # copying values out for r=0.0
        fred3 = grid_linear.T[1]
        fred4 = grid_linear.T[2]
        fred3[fred3==0] = fred4[fred3==0]
        grid_linear.T[1] = fred3
        grid_linear.T[0] = grid_linear.T[1]
        if interpolation_mode == "periodic_3d":
            grid_linear = torch.clamp(grid_linear, min=0.1, max=1.0)

        # Exponentially decay from 0.1 at the actual LCFS contour to 0.01 at the wall
        lcfs_thetas, lcfs_rads = target_lcfs
        theta_distances = np.abs((lcfs_thetas[:, None] - THETAS[None, :] + np.pi) % (2*np.pi) - np.pi)
        lcfs_rads_on_grid = lcfs_rads[np.argmin(theta_distances, axis=0)]
        distance_fraction = np.clip(
            (grid_rad - lcfs_rads_on_grid[:, None]) / (b_hidra.r_max - lcfs_rads_on_grid[:, None]),
            0.0,
            1.0,
        )
        outside_lcfs = grid_rad >= lcfs_rads_on_grid[:, None]
        exterior_profile = 0.1 * np.exp(np.log(0.01 / 0.1) * distance_fraction)
        grid_linear = torch.where(
            torch.as_tensor(outside_lcfs, device=device),
            torch.as_tensor(exterior_profile, device=device, dtype=grid_linear.dtype),
            grid_linear,
        )

        """       # set all points outside the LCFS to zero
        ## GET LCFS POINTS
        lcfs_points = lcfs_points_full[phi_index][:NTHETA].T # LCFS IS ONLY 1 SUBSET OF POINTS
        # if phi_index==0 and DEBUG:
        #     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        #     ax.set_title('LCFS at phi={:03d} deg'.format(int(PHI_GEN_DEG)))
        #     ax.plot(lcfs_points[0], lcfs_points[1], 'b.-', label='LCFS Points')
        #     ax.set_rmax(0.19)
        #     ax.set_xlabel('Theta [rad]')
        #     ax.set_ylabel('Radius [m]')
        #     ax.grid(True)
        #     plt.show()

        for theta_index, this_theta in enumerate(THETAS):

            # find the index of the value in lcfs_points[0] closest to this_theta
            mintheta1 = np.abs(lcfs_points[0] - this_theta)
            mintheta2 = np.abs(lcfs_points[0] - this_theta + 2*np.pi)
            # calculate the minimum of the two
            mintheta = np.fmin(mintheta1, mintheta2)

            # Use boolean indexing to set all radii greater than (lcfs_rad - 0.01) to zero for this theta
            lcfs_theta_index = np.argmin(mintheta)
            lcfs_rad = lcfs_points[1][lcfs_theta_index]
            mask = RADS > (lcfs_rad + 0.0005) # add buffer to avoid numerical issues
            grid_linear[theta_index][mask] = 0.0
        """

        # Add to big mesh array (3D)
        big_grid_linear[phi_index] = grid_linear[1:]  # skip the first row (theta=0) to match the shape of the b_hidra mesh

        if phi_index % 10 == 0: gc.collect()
    #### END OF LOOP THROUGH PHI ANGLES ####
    # save numpy data using simIO method
    big_grid_linear_np = big_grid_linear.detach().to("cpu").numpy()
    simIO.saveNumpyData(big_grid_linear_np, ANLYS_SUBDIR + '/' + 'nField_' + OUTPUT_FILE_NAME + '.npy')

    ## LOOP THROUGH PHI ANGLES for plotting
    for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_grid_linear_np[phi_index], 'LinearFluxNorm', ANLYS_SUBDIR, simIO, 'Blues', 0.0, 1.0)

    simIO.log.info("## Flux interpolation complete. ##")


def output_phi_plots(phi_deg, mesh_theta, mesh_rad, data, name, subdir, output_handler, colormap='inferno', plotmin=None, plotmax=None):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title(name + '\n$\phi_{{phy}}$={:02.0f}$\degree$ CW from North Split\n$\phi_c$={:02.0f}$\degree$'.format((phi_deg+198.)%360., phi_deg), loc='left')
    plot_data = np.vstack((data[-1], data))
    c = ax.pcolormesh(mesh_theta, mesh_rad, plot_data, shading='gouraud', cmap=colormap, vmin=plotmin, vmax=plotmax)

    ax.set_rmax(0.19)
    ax.set_rticks([])
    plt.grid(False)
    fig.colorbar(c, ax=ax, label='Flux')
    fig_path = subdir + '/' + name +'_{:03d}deg.png'.format(int(phi_deg))
    output_handler.saveFig(fig_path, dpi=300)
    output_handler.log.info('Saved figure: ' + fig_path)
    plt.close()

if __name__ == '__main__':
    #### DEFINE ANALYSIS PARAMETERS ####
    ## RUN DIRECTORY AND SUBDIRECTORY
    # ANLYS_DIR = "It-0486_Ih-0790_1500sp_LSODA2p49e8"
    #ANLYS_SUBDIR = "LCFS6_360x180_newFilter2"
    #ANLYS_SUBDIR = "LCFS26_360x180"
    #ANLYS_SUBDIR = "LCFS6_360x180_Calc-GeoR-test1"
    #ANLYS_SUBDIR = "LCFS6_360x180_Calc-locR-test"
    #ANLYS_SUBDIR = "LCFS6_360x180_atol-12_rtol-2"
    # ANLYS_SUBDIR = "LCFS6_360x180_atol-12_rtol-2_nthet"

    ## IDEAL iota 1/3
    # ANLYS_DIR = "It-0486_Ih-0900_noErr_1500sp_LSODA1e8"
    # #ANLYS_SUBDIR = "LCFS29_360x180_smooth7p5e6"
    # ANLYS_SUBDIR = "LCFS30_360x180_smooth1e-4"

    ANLYS_DIR = "It-0486_Ih-0790_PHI324_1500sp_LSODA2p49e8"
    ANLYS_SUBDIR = "LCFS15_360x180_smooth1e-4"

    ## DEFINE FIELDS
    FIELD_FILE_TOR = 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy'
    FIELD_FILE_HEL = 'input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy'
    CURRENT_TOR = 0.486 #[kA]
    CURRENT_HEL = 0.790 #[kA]
    CONFIG_TOR = 'default_toroidal'
    CONFIG_HEL = 'default_helical'
    ENABLE_ERRFIELD = True

    ## DEFINE LCFS AND ANGLES TO EVALUATE
    LCFS_INDEX = 15 #30 #29 #100  #1f00 #40 #22 #29?
    NPHI = 360
    NTHETA = 180
    PHI_GENs = np.linspace(360//NPHI, 360, NPHI)

    ## FLUX INTEGRATION PARAMETERS
    MAX_SUBSETS = 4
    SMALLEST_ISLAND_INDEX = None #57 #104 #39
    ALPHA = 1.0 #0.85  # flux profile adjustment parameter
    GUESS_PHI_INDEX = -20 #-71
    OUTPUT_FILE_NAME = "default"
    # Stop for flux profile selection
    DEBUG = True
    fluxInterpolator()
