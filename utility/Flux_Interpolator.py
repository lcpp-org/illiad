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


<<<<<<< HEAD
=======
def _surface_indices(indices):
    if indices is None:
        return np.array([], dtype=int)
    return np.atleast_1d(np.asarray(indices, dtype=int))


>>>>>>> origin/main
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
<<<<<<< HEAD
=======
    inv_surf_indices = _surface_indices(globals().get("INV_SURF_INDICES", []))
>>>>>>> origin/main
    input_log_params = globals().copy()
    input_log_params.update({
        "RBF_KERNEL": rbf_kernel,
        "RBF_NEIGHBORS": rbf_neighbors,
        "RBF_SMOOTHING": rbf_smoothing,
        "RBF_EPSILON": rbf_epsilon,
<<<<<<< HEAD
=======
        "INV_SURF_INDICES": inv_surf_indices.tolist(),
>>>>>>> origin/main
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
<<<<<<< HEAD
=======
            "OUTPUT_FILE_NAME",
>>>>>>> origin/main
            "RBF_KERNEL",
            "RBF_NEIGHBORS",
            "RBF_SMOOTHING",
            "RBF_EPSILON",
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

    # Load LCFS file:
    lcfs_filename = ANLYS_SUBDIR + '/fSurf_{:03d}_POINTmesh.npy'.format(int(LCFS_INDEX))
    lcfs_points_full = simIO.loadNumpyData(lcfs_filename)

    # CHOOSING ONE 'WELL-BEHAVED' ANGLE FOR THE CALCULATION (NO FAILED CALCULATIONS)
    # sum flux_norm_array along first axis and find index of max value
    sum_flux = np.nansum(flux_norm_array, axis=0)
    #best_phi_index = np.argsort(sum_flux)[-17]
    if GUESS_PHI_INDEX:
        best_phi_index = np.argsort(sum_flux)[GUESS_PHI_INDEX]
    else:
        best_phi_index = np.argsort(sum_flux)[-5]
    linear_flux_array = flux_norm_array[:, best_phi_index]
    # Adjust profile with ALPHA parameter
    linear_flux_array = 1 - (1 - linear_flux_array)**ALPHA

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
    interpol_pts = _polar_interp_points(grid_theta.ravel(), grid_rad.ravel())
    interpol_pts = torch.as_tensor(interpol_pts, device=device, dtype=torch.float32)

    big_grid_linear = torch.zeros([len(PHI_GENs), len(THETAS)-1, len(RADS)], device=device, dtype=torch.float32)

    ## LOOP THROUGH PHI ANGLES
    for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):

        # CAN GET AWAY WITH FUXING NPHI<360!
        # # LOAD AXES POINTS
        # if SMALLEST_ISLAND_INDEX:
        #     points = np.zeros([MAX_SUBSETS+1,2])
        #     flux_norm = np.ones([MAX_SUBSETS+1]) # peak values for the axes points
        #     points[1:] = island_axis_array[phi_index]
        # else:
        points = np.zeros([1,2])
        flux_norm = np.ones(1) # peak values for the axes points
        # CAN GET AWAY WITH FUXING NPHI<360!
        #points[0] = axis_array[phi_index][0]
        points[0] = axis_array[0][0]

        ## LOAD SCATTER POINTS (POINCARE DATA)
        filename = 'Poincare_{:03d}.npy'.format(int(PHI_GEN_DEG))
        flux_surfaces = simIO.loadNumpyData(filename)

        ## LOOP THROUGH SURFACES
        for surface_index in range(LCFS_INDEX, N_surfaces):
            if valid_surface[surface_index] == False:
                simIO.log.info(f'Skipping surface {surface_index} (not valid)')
            else:
                ### GET VALUES
                thetas = flux_surfaces[surface_index][0]
                rads = flux_surfaces[surface_index][1]
                # filter NaNs
                thetas = thetas[~np.isnan(thetas)]
                rads = rads[~np.isnan(rads)]
                N_pts = len(thetas)

                # concatenate to big array of points
                these_points = np.array([thetas, rads]).T
                points = np.concatenate((points, these_points))

                these_flux_norms = np.full(N_pts, linear_flux_array[surface_index])
                flux_norm = np.concatenate((flux_norm, these_flux_norms))

        #grid_linear = griddata(points, flux_norm, (grid_theta, grid_rad), method='linear', fill_value=0.0, rescale=True)
        interp_points = _polar_interp_points(points[:, 0], points[:, 1])
        points_torch = torch.as_tensor(interp_points, device=device, dtype=torch.float32)
        flux_norm_torch = torch.as_tensor(flux_norm, device=device, dtype=torch.float32)
        #interpolation = RBFInterpolator(points_torch, flux_norm_torch, kernel='multiquadric', neighbors=25, smoothing=1e-0, epsilon=1000)
        interpolation = RBFInterpolator(points_torch, flux_norm_torch, kernel=rbf_kernel, neighbors=rbf_neighbors, smoothing=rbf_smoothing, epsilon=rbf_epsilon)
        
        #interpolation = RBFInterpolator(points_torch, flux_norm_torch, kernel='linear', neighbors=15, smoothing=1e-5, degree=1) #, epsilon=1e4)
        #interpolation = RBFInterpolator(points_torch, flux_norm_torch, kernel='linear', neighbors=55, smoothing=1e-5, degree=1) #, epsilon=1e4)

        # Work around torchrbf device placement: ensure internal tensors/buffers are on the same device.
        interpolation = interpolation.to(device)
        interpolation.smoothing = interpolation.smoothing.to(device)

        grid_linear = interpolation(interpol_pts).reshape(grid_shape)

        ## HACKY SOLUTIONS HERE!!!
        # copying values out for r=0.0
        fred3 = grid_linear.T[1]
        fred4 = grid_linear.T[2]
        fred3[fred3==0] = fred4[fred3==0]
        grid_linear.T[1] = fred3
        grid_linear.T[0] = grid_linear.T[1]

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

        del flux_surfaces
        if phi_index % 10 == 0: gc.collect()
    #### END OF LOOP THROUGH PHI ANGLES ####
    # save numpy data using simIO method
    big_grid_linear_np = big_grid_linear.detach().to("cpu").numpy()
<<<<<<< HEAD
    simIO.saveNumpyData(big_grid_linear_np, ANLYS_SUBDIR + '/density_field.npy')
=======
    simIO.saveNumpyData(big_grid_linear_np, ANLYS_SUBDIR + '/' + 'nField_' + OUTPUT_FILE_NAME + '.npy')
>>>>>>> origin/main

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
