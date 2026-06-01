#import logging
import gc
import numpy as np
from scipy.interpolate import splev, splrep, make_splrep
from scipy.integrate import dblquad
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
mpl.rcParams.update({
    # --- fonts & text (IOP-friendly, ~8–12 pt at final size) ---
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

from classes.mesh import *
from classes.iohandler import IOHandler
from utility.coordtrans import axisShift, RTP_to_XYZ, XYZ_to_RTP
np.set_printoptions(threshold=np.inf)

def fluxCalculator(input_params=None):
    ## LOAD INPUT PARAMETERS
    if input_params is not None:
        print(f'{input_params.keys()=}')
        for key, value in input_params.items():
            print(f'{key}: {value}')
            globals()[str(key)] = value
    ## DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
    simIO = IOHandler(ANLYS_DIR)
    simIO.setActiveSubDir(ANLYS_SUBDIR)
    simIO.startLog(log_name="fluxCalc.log", subdir=ANLYS_SUBDIR, logger_name="FluxCalculator")
    simIO.inputsBoilerplate(
        "FLUX CALCULATOR INPUTS",
        globals(),
        [
            "ANLYS_DIR",
            "ANLYS_SUBDIR",
            "FIELD_FILE_TOR",
            "FIELD_FILE_HEL",
            "CURRENT_TOR",
            "CURRENT_HEL",
            "CONFIG_TOR",
            "CONFIG_HEL",
            "ENABLE_ERRFIELD",
            "LCFS_INDEX",
            "NPHI",
            "NTHETA",
            "PHI_GENs",
            "MAX_SUBSETS",
            "SMOOTH_FCTR",
            "INTEGRATE_EPSABS",
            "INTEGRATE_EPSREL",
            "ISLAND_ALGORITHM",
            "HIST_BINS",
            "PLOT_ALL",
            "BIG_MESH",
        ],
    )

    ## DEFINE MESH AND LOAD MAGNETIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField(coilCurrent=CURRENT_TOR, errField=ENABLE_ERRFIELD, att_mult=CONFIG_TOR)
    b_hidra.set_nonPer_errField()
    b_hidra.addFieldPerturbation(coilCurrent=CURRENT_HEL, att_mult=CONFIG_HEL)

    for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):
        ## LOAD POINCARE DATA
        filename = 'Poincare_{:03d}.npy'.format(int(PHI_GEN_DEG))
        flux_surfaces = simIO.loadNumpyData(filename)
        if PLOT_ALL: fig, ax_rect, ax_hist, ax_polar = init_plotting()

        # DECLARE BIG ARRAYS TO STORE DATA OVER SURFACES AND PHI ANGLES
        NSURFACE = len(flux_surfaces)
        if phi_index == 0:
            tot_flux_array = np.zeros([NSURFACE, len(PHI_GENs)])
            total_flux_norm = np.zeros([NSURFACE, len(PHI_GENs)])
            centers_array = np.zeros([NSURFACE, len(PHI_GENs), MAX_SUBSETS, 2])
            if BIG_MESH:
                flat_point_meshes = np.full([NSURFACE, len(PHI_GENs), NTHETA*MAX_SUBSETS, 2], np.nan)
            else:
                lcfs_flat_point = np.full([len(PHI_GENs), NTHETA*MAX_SUBSETS, 2], np.nan)
            valid_surfs = np.zeros([NSURFACE, len(PHI_GENs)], dtype=bool) # keep track of valid surfaces
            valid_surfs[LCFS_INDEX:][:] = True  # set all surfaces after LCFS to valid

        ## FIND THE MAGNETIC AXIS FROM SMALLEST FLUX SURFACE (ASSUMED LAST FLUX SURFACE IN SET)
        mag_axis = find_Axis(*flux_surfaces[-1], b_hidra)
        mag_axis_rev = np.copy(mag_axis)
        mag_axis_rev[0] += np.pi

        ## LOOP THROUGH FLUX SURFACES TO FIND SUBSETS (ISLAND) AND THE SET OF SMALLEST ISLANDS
        first_loop_output = first_surface_loop(flux_surfaces, mag_axis, b_hidra, LCFS_INDEX, NSURFACE, MAX_SUBSETS)
        smallest_island_index, num_subsets, subsetData, subsetCenters, hist_output = first_loop_output
        hist, bin_edges, wrap_flag = hist_output

        ## (2ND) LOOP THROUGH FLUX SURFACES TO SHIFT DATA AND SPLINE FIT
        Fluxes = []
        for surf_index in range(LCFS_INDEX, NSURFACE):
            N_subsets = num_subsets[surf_index]
            ## SUBSET LOOP: LOOP TO SPLINE, CALCULATE FLUX, AND CREATE REGULARLY-SPACED POINTS
            output = subset_looper(subsetData, subsetCenters, surf_index,
                                    smallest_island_index, N_subsets, wrap_flag,
                                    b_hidra, mag_axis_rev, NTHETA, PHI_GEN_DEG,
                                    INTEGRATE_EPSABS, INTEGRATE_EPSREL, SMOOTH_FCTR, simIO)
            points_tr_geoAxis, spline_tr_magAxis, subCenters_geo, subset_flux_list, valid_surfs[surf_index] = output

            ## APPENDING LISTS AND RESHAPING ARRAYS
            total_npts = num_subsets[surf_index] * NTHETA
            Fluxes += [subset_flux_list]
            centers_array[surf_index][phi_index] = subCenters_geo

            if surf_index==LCFS_INDEX:
                spline_lcfs_magaxis = spline_tr_magAxis
            elif N_subsets == 1:
                filter_cond1 = np.any(spline_tr_magAxis[:,:,1] < 0.0)
                fred = spline_lcfs_magaxis[0,:,1] - spline_tr_magAxis[0,:,1] 
                filter_cond2 = np.any(fred <= 0.0)
                if filter_cond1 or filter_cond2:
                    valid_surfs[surf_index][phi_index] = False # not a valid surface for interpolation
                    simIO.log.info( '\t(A)Surface #{} NOT A VALID SURFACE!!!'.format(surf_index) )
            elif np.any(spline_tr_magAxis[:,:,1] < 0.0):
                valid_surfs[surf_index][phi_index] = False # not a valid surface for interpolation
                simIO.log.info( '\t(A)Surface #{} NOT A VALID SURFACE!!!'.format(surf_index) )

            ## PLOTTING EACH FLUX SURFACE AT EACH PHI ANGLE
            if PLOT_ALL:
                ## GET R, THETA FOR FLUX SURFACE
                th_in, r_in = flux_surfaces[surf_index]
                r_in = r_in[~np.isnan(r_in)]
                th_in = th_in[~np.isnan(th_in)]
                th_size = th_in.size

                # SHIFT ORIGIN OF R, THETA COORDINATES FROM GEOMETRIC CENTER TO MAGNETIC AXIS
                points_tr_magAxis = axisShift(th_in, r_in, *mag_axis).T
                # REMOVE DUPLICATE THETA VALUES AND SORT DATA IN INCREASING THETA
                #unique_indices = np.unique(points_tr_magAxis[:, 0], return_index=True, return_counts=False)[1:]
                unique_indices = np.unique(points_tr_magAxis[:, 0], return_index=True)[1]
                points_tr_magAxis = points_tr_magAxis[unique_indices]
                th_size = points_tr_magAxis.shape[0]
                points_tr_magAxis = points_tr_magAxis[np.argsort(points_tr_magAxis[:, 0])]

                # PLOT THE DATA POINTS
                ax_rect.scatter(points_tr_magAxis.T[0]*180./np.pi, points_tr_magAxis.T[1], color='k', s=0.25, linewidths=0.0) # mag-axis point
                # plot the spline fit and histogram
                if valid_surfs[surf_index][phi_index]:
                    for i in range(0, num_subsets[surf_index]):
                        ax_rect.scatter(spline_tr_magAxis[i].T[0]*180./np.pi, spline_tr_magAxis[i].T[1], s=0.5, linewidths=0.05)    
                    ax_hist.bar(bin_edges[surf_index][:-1]*180./np.pi, hist[surf_index],
                           width=np.diff(bin_edges[surf_index])*180./np.pi, align='edge', edgecolor='k', linewidth=0.1)

            ## lcfs and current surface: flatten the points for output
            plot_tr_points = points_tr_geoAxis.reshape((total_npts, 2), order='C', copy=True)
            NOW_NPTS = plot_tr_points.shape[0]
            plot_thetas = plot_tr_points.T[0]
            plot_radii = plot_tr_points.T[1]
            if surf_index == LCFS_INDEX:
                LCFSdata_list = plot_tr_points
                if not BIG_MESH:
                    lcfs_flat_point[phi_index][:NOW_NPTS] = LCFSdata_list
            lcfs_radii = axisShift(*LCFSdata_list.T, *mag_axis_rev)[1]

            if num_subsets[surf_index] == MAX_SUBSETS: #if num_subsets[surf_index] > 1:
                plot_centers_geo = centers_array[surf_index][phi_index].T
            else:
                plot_centers_geo = centers_array[surf_index][phi_index][0].T

            # Filter out wild fits
            delta_plot_rs = np.max(lcfs_radii) - np.max(plot_radii)
            if delta_plot_rs > 0.: # Add points to output arrays
                if BIG_MESH: flat_point_meshes[surf_index][phi_index][:NOW_NPTS] = plot_tr_points
                if PLOT_ALL:
                    ax_polar.scatter(plot_thetas, plot_radii, s=0.3, linewidths=0.0)
                    if plot_centers_geo.shape[-1] > 1:
                        ax_polar.scatter(plot_centers_geo[0], plot_centers_geo[1], s=15, color='k', marker='x',  linewidths=0.5)
        ## END SURFACE LOOP

        # PRINTING FLUXES AS OUTPUT:
        for flux_index, flux in enumerate(Fluxes, start=LCFS_INDEX):
            tot_flux_array[flux_index][phi_index] = np.sum(flux, axis=-1)

        if PLOT_ALL: finalize_plotting(fig, ax_rect, ax_hist, ax_polar, PHI_GEN_DEG, num_subsets, MAX_SUBSETS, simIO)

        del flux_surfaces
        if phi_index % 5 == 0:
            gc.collect()
    ## END OF PHI LOOP

    # Set fluxes outside of LCFS to be equal to the LCFS flux, and set range of data to be between 0 and the LCFS flux
    tot_flux_array[:LCFS_INDEX][:] = tot_flux_array[LCFS_INDEX][:]
    tot_flux_array = np.clip(tot_flux_array, 0.0, tot_flux_array[LCFS_INDEX])
    # Set range of data to be between 0 and 1, and set data outside of LCFS to be equal to the 0
    total_flux_norm = 1 - tot_flux_array / tot_flux_array[LCFS_INDEX]
    total_flux_norm = np.clip(total_flux_norm, 0.0, 1.0)
    total_flux_norm[:LCFS_INDEX][:] = 0.0
    ## OUTPUT FLUXES
    filename_fluxes = ANLYS_SUBDIR + '/CalculatedFLuxes.npy'
    filename_fluxNorms = ANLYS_SUBDIR + '/CalculatedFLuxes-normalized.npy'
    filename_validSurfaces = ANLYS_SUBDIR + '/ValidSurfaces.npy'
    simIO.saveNumpyData(tot_flux_array, filename_fluxes)
    simIO.saveNumpyData(total_flux_norm, filename_fluxNorms)
    simIO.saveNumpyData(valid_surfs, filename_validSurfaces)

    # HAVE A BIG ARRAY OF FLUXES, NOW PLOT THEM
    simIO.log.info('Plotting Flux v Surface Index...')
    fig_post = plt.figure()
    axPSI = fig_post.add_subplot(211, ylabel='Toroidal Flux $\psi_{\phi}$\n$[g*m^2]$')
    axNORM = fig_post.add_subplot(212, xlabel='Surface index', ylabel='Surface Parameter $\hat{\psi}_n$')

    for i in range(len(PHI_GENs)):
        axPSI.plot(tot_flux_array[:,i]*10_000, label='{:d}'.format(int(PHI_GENs[i])), linewidth=1)
        axNORM.plot(total_flux_norm.T[i], label='{:d}'.format(int(PHI_GENs[i])), linewidth=1)
    axPSI.set_title('Toroidal Angles $\phi[\degree]$', loc='right', fontsize=8)
    axPSI.set_ylim(0, 1.1*10_000*np.max(tot_flux_array)) 
    axPSI.grid(which='both', linestyle=':', linewidth=0.5)
    axPSI.legend(loc='upper right', fontsize=5,ncols=3)
    axNORM.set_ylim(0, 1.1)
    axNORM.grid(which='both', linestyle=':', linewidth=0.5)

    simIO.saveFig(ANLYS_SUBDIR+'/Flux_v_Surface.png', dpi=300)
    simIO.log.info('Finished, LCFS=#{}, ISLAND AXIS=#{}'.format(LCFS_INDEX, smallest_island_index))

    # SAVE THE NUMPY ARRAYS TO INDIVIDUAL FILES USING SIMIO METHOD
    for surf_index in range(LCFS_INDEX, NSURFACE):
        filename_center = ANLYS_SUBDIR + '/fSurf_{:03d}_center.npy'.format(surf_index)
        simIO.saveNumpyData(centers_array[surf_index], filename_center)

        if BIG_MESH:
            filename_pt_mesh = ANLYS_SUBDIR + '/fSurf_{:03d}_POINTmesh.npy'.format(surf_index)
            simIO.saveNumpyData(flat_point_meshes[surf_index], filename_pt_mesh)
        elif surf_index == LCFS_INDEX:
            filename_pt_mesh = ANLYS_SUBDIR + '/fSurf_{:03d}_POINTmesh.npy'.format(surf_index)
            simIO.saveNumpyData(lcfs_flat_point, filename_pt_mesh)

    return smallest_island_index


def find_Axis(theta_vals: np.ndarray, r_vals: np.ndarray, field: Mesh, alt=False) -> np.ndarray:
    """
    Computes the geometric center (axis) of a set of points in (r, theta) coordinates.

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
    coords_3d = np.array([RTP_to_XYZ(np.array([r, theta, 0.]), field.R0) for r, theta in zip(r_vals, theta_vals)])

    axis_xyz = np.mean(coords_3d, axis=0)
    axis_thetar = XYZ_to_RTP(axis_xyz, field.R0)[1::-1]

    # find the indices of all theta_vals within 20 degrees of axis_thetar[0]
    theta_bins = np.deg2rad(20.)  # 20 degree in radians
    dtheta_to_axis = np.minimum(np.abs(theta_vals - axis_thetar[0]), 2 * np.pi - np.abs(theta_vals - axis_thetar[0]))
    theta_indices = np.where(dtheta_to_axis < theta_bins)[0]

    # Compute the average radius for the selected indices
    axis_thetar[1] = np.mean(r_vals[theta_indices])

    return axis_thetar

def find_subsets(max_subsets, theta_r_pts, mag_axis, field, BINS=120):
    """Function to find contiguous subsets of points in theta-r space
    Args:
        max_subsets (int): Maximum number of subsets to find.
        theta_r_pts (np.ndarray): Array of points in (theta, r) coordinates, wrt to the magnetic Axis.
        mag_axis (np.ndarray): Magnetic axis position in (theta, r) coordinates, wrt to the geometric Axis.
        field (Mesh): Mesh object used for coordinate transformations.
        BINS (int): Number of bins to use for histogramming.

    RETURNS:
        * split_data (list): List of arrays containing the split data points wrt to magnetic axis.
        * found_centers (np.ndarray): Array of found centers wrt to magnetic axis for each subset.
        hist (np.ndarray): Histogram of point density vs theta.
        bin_edges (np.ndarray): Edges of the bins used for the histogram.
        wrapped_flag (bool): Flag indicating if the data wraps around.
    """
    ## HISTOGRAM OF THE POINT DENSITY VS THETA
    hist, bin_edges = np.histogram(theta_r_pts.T[0], bins=BINS, range=(0., 2*np.pi))
    dtheta_bin = bin_edges[1] - bin_edges[0]

    ## FIND CONTIGUOUS SETS OF NON-ZERO BINS
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

    ## SPLIT THE DATA POINTS INTO SUBSETS
    split_data_magAxis = []
    found_centers = np.zeros([num_sets, 2])
    for i, contiguous_set in enumerate(contiguous_sets):
        thisSet_tr = []
        lowerBound = contiguous_set*dtheta_bin
        upperBound = lowerBound + dtheta_bin
        # append data within each bin belonging to the subset
        for lo, hi in zip(lowerBound, upperBound):
            mask = (theta_r_pts[:, 0] >= lo) & (theta_r_pts[:, 0] < hi)
            thisSet_tr += list(theta_r_pts[mask])
        thisSet_tr = np.array(thisSet_tr)
        thisSet_tr = thisSet_tr[np.argsort(thisSet_tr[:, 0])] # sort by theta

        # split if the # of sets is equal to input 'MAX_SUBSETS': 
        if num_sets == max_subsets:
            found_centers[i][:] = find_Axis(thisSet_tr.T[0], thisSet_tr.T[1], field, alt=True)
            split_data_magAxis += [thisSet_tr]
        else: # keep the original magnetic axis
            found_centers[i][:] = mag_axis[:2]
            split_data_magAxis = [theta_r_pts]

    return split_data_magAxis, found_centers, hist, bin_edges, wrapped_flag

def shift_the_subcenters(surf_index, smallest_island_index, subsetCenters, num_subsets, wrap_flag):
    """Function performs tests to see if there is a misalignment of subset centers 
    between the smallest island set and the current island set. If so, it returns the 
    appropriate r and theta values to shift the data set to be relative to the smallest island subcenters """
   
    smallest_centers = subsetCenters[smallest_island_index]
    these_centers = subsetCenters[surf_index]

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

def spline_Data(theta_pts: np.ndarray, rad_pts: np.ndarray, smoothing=1e-5):
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
    theta_clean = theta_pts[valid_mask]
    rad_clean = rad_pts[valid_mask]

    if len(theta_clean) <= 3:
        return None, None, True, "Not enough points for cubic spline (need > 3)"
    # Copy data to both ends for pseudo-periodicity (smooth spline endpoints) 
    append_length = int(len(theta_clean)/2)

    th_A = theta_clean[append_length:-1] - 2*np.pi
    th_B = theta_clean[1:append_length] + 2*np.pi
    theta_spl = np.concatenate((th_A, theta_clean, th_B))

    rad_A = rad_clean[append_length:-1]
    rad_B = rad_clean[1:append_length]
    rad_spl = np.concatenate((rad_A, rad_clean, rad_B))

    # spline parameters
    #spline_rep = make_splrep(theta_spl, rad_spl, k=3, s=smoothing, per=False)
    #spline_rep = splrep(theta_spl, rad_spl, xb=0.0, xe=2*np.pi, k=3, s=smoothing, per=False, full_output=1, quiet=1)

    # append the first point of each array to the end to enforce periodicity in the spline fit
    # theta_spl = np.append(theta_clean, theta_clean[0]+2*np.pi)
    # rad_spl = np.append(rad_clean, rad_clean[0])

    spline_rep = splrep(theta_spl, rad_spl, k=3, s=smoothing, per=False, full_output=1, quiet=1)

    return spline_rep


def integrate_flux(spline_parms, spline_axis, phi, field, err_abs=1e-5, err_rel=1e-3):
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
    # INTEGRATION HELPER FUNCTIONS
    def flux_integrand(r, theta, phi, field, axis):
        """Function to calculate the toroidal field times radius at a given point in space"""
        geo_point = np.array([r+axis[1], theta+axis[0], phi], dtype=np.float64)
        #axis[0] += np.pi
        geo_point = np.array([*axisShift(r, theta, *axis), phi])
        if geo_point[0] < 0.0:
            geo_point[0] *= -1.0
            geo_point[1] += np.pi
        bxy = field.interpField(geo_point, Cart=False)[0][:2]
        #bxy = np.array([1.0, 1.0]) # TESTING

        # Calculate the toroidal flux integrand: r*B_toroidal = r*( -Bx*sin(phi) - By*cos(phi) )
        return -r*( bxy[0]*np.sin(phi) - bxy[1]*np.cos(phi) )

    # Define the upper radial bound of the integration
    def hfun(theta): return splev(theta, spline_parms)

    ## INTEGRATE TOROIDAL FLUX
    PSI, abserr = dblquad(flux_integrand, 0., 2*np.pi,
                          0.0, hfun, args=(phi, field, spline_axis),
                          epsabs=err_abs, epsrel=err_rel)

    return float(PSI)

def first_surface_loop(flux_surfaces, mag_axis, field, start_index, end_index, max_subsets):
    """
    Processes a range of flux surfaces to identify and analyze magnetic islands.

    For each flux surface in the specified range, this function:
      - Shifts the origin of (r, theta) coordinates to the magnetic axis.
      - Sorts the data and removes duplicate theta values.
      - Finds subsets (potential magnetic islands) and their local centers.
      - Computes mean radii for each subset.
      - Identifies the island (subset) with the smallest mean radius among surfaces with multiple subsets.

    Args:
        flux_surfaces (list of tuple): List where each element is a tuple (theta_array, r_array) representing a flux surface.
        mag_axis (array-like): Coordinates of the magnetic axis (e.g., [x, y]).
        field (Mesh): MField Mesh object used in subset finding.
        start_index (int): Starting index of the flux surfaces to process (inclusive).
        end_index (int): Ending index of the flux surfaces to process (exclusive).
        max_subsets (int): Maximum number of subsets (islands) to identify per surface.

    Returns:
        smallest_island_index (int): Index of the flux surface containing the smallest-radius island among those with multiple islands.
        num_subsets (np.ndarray): Array of the number of subsets found for each surface.
        split_data_magAxis (list): List of lists containing subset data wrt to magnetic axis for each surface.
        surface_axes_magAxis (list): List of local centers wrt to magnetic axis for each subset in each surface.
        hist_data (tuple): Tuple containing (hist, bin_edges, wrap_flag) for each surface, useful for diagnostics or plotting.
    """
    num_subsets = np.zeros(end_index, dtype=int)
    set_mean_rads = np.zeros(end_index)
    split_data_magAxis = [0]*end_index
    surface_axes_magAxis = [0]*end_index
    hist = [0]*end_index
    bin_edges = [0]*end_index
    wrap_flag = [0]*end_index
    for surf_index in range(start_index, end_index):
        #print(f'Analyzing flux surface {surf_index} for islands...')
        # get r, theta for flux surface
        th_in, r_in = flux_surfaces[surf_index]
        r_in = r_in[~np.isnan(r_in)]
        th_in = th_in[~np.isnan(th_in)]
        th_size = th_in.size
        # shift origin of r, theta coords from geo center to mag axis
        pts_tr_magAxis = axisShift(th_in, r_in, *mag_axis).T
        # REMOVE DUPLICATE THETA VALUES AND SORT DATA IN INCREASING THETA
        unique_indices = np.unique(pts_tr_magAxis[:, 0], return_index=True, return_counts=False)[1:]
        pts_tr_magAxis = pts_tr_magAxis[unique_indices]
        pts_tr_magAxis = pts_tr_magAxis[np.argsort(pts_tr_magAxis[:, 0])]

        # find subsets of the data, and their local centers,!! DATA RETURNED AS THETA, R RELATIVE TO MAG AXIS
        output_histogram_method = find_subsets(max_subsets, pts_tr_magAxis, mag_axis, field, BINS=HIST_BINS)
        hist[surf_index], bin_edges[surf_index] = output_histogram_method[2:4]
        wrap_flag[surf_index] = output_histogram_method[-1] # use histogram output for wrap_flag
        
        if ISLAND_ALGORITHM == 'histogram':  output = output_histogram_method
        # elif ISLAND_ALGORITHM == 'kmeans':   output = find_subsets_kmeans(max_subsets, pts_tr_magAxis, mag_axis, field)
        # elif ISLAND_ALGORITHM == 'spectral': output = find_subsets_spectral(max_subsets, pts_tr_magAxis, mag_axis, field)
        else: raise ValueError(f"Unknown ISLAND_ALGORITHM: {ISLAND_ALGORITHM}")

        ## initialize split_data with data w.r.t. magnetic axis
        split_data_magAxis[surf_index], surface_axes_magAxis[surf_index] = output[:2]
        num_subsets[surf_index] = len(split_data_magAxis[surf_index])

        # LOOP THROUGH SUBSETS
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

    # FIND THE INDEX OF THE ISLANDS OF SMALLEST RADIUS
    island_indices = np.where(num_subsets > 1)[0]
    if island_indices.size > 0: smallest_island_index = island_indices[np.argmin(set_mean_rads[island_indices])]
    else: smallest_island_index = end_index - 1  #If no islands found, return the last surface index

    if np.any(np.isnan(surface_axes_magAxis[smallest_island_index])):
        print('NaN detected in smallest island centers!!!!')
        print(f'{surface_axes_magAxis[smallest_island_index]=}' )
    hist_data = (hist, bin_edges, wrap_flag)
    return smallest_island_index, num_subsets, split_data_magAxis, surface_axes_magAxis, hist_data

def subset_looper(subsetData, subsetCenters, surf_index,
                  smallest_island_index, num_subsets, wrap_flag,
                  field, mag_axis_rev, NTHETA, PHI_GEN_DEG,
                  INTEGRATE_EPSABS, INTEGRATE_EPSREL, SMOOTH_FCTR, simIO
                  ):
    """
    Loop through subsets of a flux surface, fit spline, calculate flux, and generate
    regularly-spaced points evaluated on the spline fit.

    Args:
        subsetData (list): List of data arrays for each subset of each surface.
        subsetCenters (list): List of center coordinates for each subset of each surface.
        surf_index (int): Index of the current flux surface.
        smallest_island_index (int): Index of the smallest island subset.
        num_subsets (int): Number of subsets for the current flux surface.
        wrap_flag (list): List of flags indicating if wrapping is needed for each surface.
        field: Magnetic field data.
        mag_axis_rev (tuple): Magnetic axis coordinates for reverse axis shift.
        NTHETA (int): Number of theta points for evaluation.
        PHI_GEN_DEG (float): Phi angle in degrees.
        INTEGRATE_EPSABS (float): Absolute error tolerance for integration.
        INTEGRATE_EPSREL (float): Relative error tolerance for integration.
        SMOOTH_FCTR (float): Smoothing factor for spline fitting.
        simIO: Simulation I/O object for logging and saving data.

    Returns:
        splined_tr_GeoAxis (np.ndarray): Regularly-spaced points in geometric coordinates.
        splined_tr_MagAxis (np.ndarray): Regularly-spaced points in magnetic axis coordinates.
        subset_flux_list (list): List of flux values for each subset.
        subCenters_geo (np.ndarray): Centers of subsets in geometric coordinates.
        valid_surface (bool): Flag indicating if the surface is valid for interpolation.
    """
    radius_evals_locAxis = np.zeros([num_subsets, NTHETA])
    splined_tr_MagAxis   = np.zeros([num_subsets, NTHETA, 2])
    splined_tr_GeoAxis   = np.zeros([num_subsets, NTHETA, 2])
    subCenters_geo       = np.zeros([num_subsets, 2])
    THETA_GENs = np.linspace(2*np.pi/NTHETA, 2*np.pi, NTHETA)

    # Subset ordering may have changed due to periodic wraparound of centers
    if num_subsets > 1: 
        subCenters_Shift, shiftint = shift_the_subcenters(surf_index, smallest_island_index, subsetCenters, num_subsets, wrap_flag[surf_index] )
    else: shiftint = 0

    ## LOOP THROUGH SUBSETS TO SPLINE, CALCULATE FLUX, CREATE REGULARLY-SPACED POINTS
    subset_flux_list = []
    for subset_idx in range(num_subsets):
        current_center = subsetCenters[surf_index][subset_idx]
        current_data = np.array(subsetData[surf_index][subset_idx], copy=True)

        # SHIFT DATA POINTS FROM [REL. TO MAGAXIS] TO [REL. TO THE CENTERS OF THE SMALLEST ISLANDS]
        if num_subsets == MAX_SUBSETS:
            current_data = axisShift(current_data[:, 0], current_data[:, 1], *current_center).T
            current_data = axisShift(current_data[:, 0], current_data[:, 1], *subCenters_Shift[subset_idx]).T

            current_data = current_data[np.argsort(current_data[:, 0])]
            current_center = subsetCenters[smallest_island_index][subset_idx-shiftint]
            subCenters_geo[subset_idx][:] = axisShift(*current_center, *mag_axis_rev)
        else:
            subCenters_geo[subset_idx] = current_center

        ## SPLINE FIT
        fSurface_splineParms, res, fail, msg = spline_Data(*current_data.T, smoothing=SMOOTH_FCTR)
        if fail:
            valid_surface = False # not a valid surface for interpolation
            simIO.log.info('\t(B)Surface #{} NOT A VALID SURFACE!!! Residual: {}'.format(surf_index, res) )
            simIO.log.info('\tSurface #{}, fail: {}\tmsg: {}'.format(surf_index, bool(fail), msg) )
            continue
        else:
            simIO.log.info( '\t(B)Surface #{} valid. Residual: {:.3e}'.format(surf_index, res) )
            valid_surface = True # valid surface for interpolation

        ## INTEGRATE AMOUNT OF TOROIDAL FIELD BOUNDED BY FLUX SURFACE [Tesla*m^2]
        this_flux = integrate_flux(fSurface_splineParms, np.array(current_center, copy=True),
                                   PHI_GEN_DEG*np.pi/180, field,
                                   err_abs=INTEGRATE_EPSABS, err_rel=INTEGRATE_EPSREL)
        subset_flux_list += [this_flux]

        # CREATE A SET OF REGULARLY-SPACED POINTS EVALUATED ON THE SPLINE FIT
        current_theta_gens = (THETA_GENs + current_center[0]) % (2*np.pi)
        radius_evals_locAxis[subset_idx] = splev(current_theta_gens, fSurface_splineParms)

        if num_subsets > 1:
            shift_r = current_center[1]
            shift_theta = current_center[0] + np.pi
            splined_tr_MagAxis[subset_idx] = axisShift(
                current_theta_gens,
                radius_evals_locAxis[subset_idx],
                shift_theta,
                shift_r,
            ).T
        else:
            splined_tr_MagAxis[subset_idx, :, 0] = current_theta_gens
            splined_tr_MagAxis[subset_idx, :, 1] = radius_evals_locAxis[subset_idx]

        ## Shift r, theta back relative to geometric axis
        splined_tr_GeoAxis[subset_idx] = axisShift(
            splined_tr_MagAxis[subset_idx, :, 0],
            splined_tr_MagAxis[subset_idx, :, 1],
            *mag_axis_rev,
        ).T

    return splined_tr_GeoAxis, splined_tr_MagAxis, subCenters_geo, subset_flux_list, valid_surface

def init_plotting():
    width_in = 15 / 2.54 # 15 cm in inches
    height_in = width_in * (9/16)  # Maintain 16:9 aspect ratio
    fig = plt.figure(figsize=(width_in, height_in))

    gs = gridspec.GridSpec(2, 2, width_ratios=[1.3, 1], wspace=0.35, hspace=0.0)
    axPolar = fig.add_subplot(gs[:,0], polar=True)
    axRect = fig.add_subplot(gs[0,1], polar=False)
    axHist = fig.add_subplot(gs[1,1], polar=False)

    # Reduce margins around the entire figure
    plt.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.12)

    return fig, axRect, axHist, axPolar

def finalize_plotting(fig, axRect, axHist, axPolar, PHI_GEN_DEG, num_subsets, MAX_SUBSETS, simIO):
    fig.suptitle('Toroidal Angle $\phi={}\degree$'.format(int(PHI_GEN_DEG)), fontsize=12, x=0.18, y=0.98)
    ## r vs theta Cartesian plot
    axRect.set_ylim(0, 0.19)
    axRect.set_xlim(0, 360)
    axRect.tick_params(axis='both', which='major', labelsize=6)
    axRect.set_xticks(np.arange(0, 361, 90))
    axRect.set_xticklabels([])
    axRect.set_yticks(np.arange(0.0, 0.19, 0.025))
    axRect.set_yticklabels(['', '2.5', '5', '7.5', '10', '12.5', '15', '17.5'])
    axRect.grid(linewidth=0.5, linestyle=':', c='grey')
    axRect.set_ylabel('[cm]', fontsize=8)
    ## Histogram plot
    axHist.set_xlim(0, 360)
    axHist.set_xticks(np.arange(0, 361, 90))
    axHist.tick_params(axis='both', which='major', labelsize=6)
    axHist.grid(linewidth=0.5, linestyle=':', c='grey')
    axHist.set_xlabel('Poloidal Angle $\\theta[\degree]$', fontsize=8)
    axHist.set_ylabel('[#]', fontsize=8)
    ## Polar plot
    axPolar.set_rlim(0, 0.19)
    axPolar.tick_params(axis='both', which='major', labelsize=8) #,pad=10)
    axPolar.grid(linewidth=0.5, linestyle=':', c='grey')
    axPolar.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                        labels=['Low\nField', '', '', '', 'High\nField', '', '', ''], fontsize=8)
    axPolar.set_rgrids([0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175],
                    labels=['', '5cm', '', '10cm', '', '15cm', ''], angle=0, fontsize=4)

    simIO.log.info('Plotting Flux Surfaces @ phi={}'.format(PHI_GEN_DEG))
    simIO.saveFig(ANLYS_SUBDIR+'/Splines_{:03d}deg.png'.format(int(PHI_GEN_DEG)), dpi=600)
    plt.close()

if __name__ == '__main__':
    #### DEFINE ANALYSIS PARAMETERS ####
    ## RUN DIRECTORY AND SUBDIRECTORY
    ANLYS_DIR = "DEFAULT"
    ANLYS_SUBDIR = "DEFAULT"

    ## DEFINE FIELDS, default to iota=1/4
    FIELD_FILE_TOR = 'input_files/It486_Ih000_Iv000_1p000_1p000_64bit.npy'
    FIELD_FILE_HEL = 'input_files/It000_Ih900_Iv000_1p000_1p000_64bit.npy'
    CURRENT_TOR = 0.486 #[kA]
    CURRENT_HEL = 0.790 #[kA]
    CONFIG_TOR = 'default_toroidal'
    CONFIG_HEL = 'default_helical_rev'
    ENABLE_ERRFIELD = False

    ## DEFINE LCFS AND ANGLES TO EVALUATE
    LCFS_INDEX = 39 #40 #22 #29?
    NPHI = 360
    NTHETA = 360
    PHI_GENs = np.linspace(360//NPHI, 360, NPHI)

    ## FLUX INTEGRATION PARAMETERS
    MAX_SUBSETS = 4
    SMOOTH_FCTR = 1.0e-5 #7.5e-6 #baseline 1e-6
    INTEGRATE_EPSABS=1e-5
    INTEGRATE_EPSREL=1e-3
    HIST_BINS = 120
    ## PLOTTING FLAG
    ISLAND_ALGORITHM = 'histogram' # 'kmeans', 'spectral
    PLOT_ALL = True
    BIG_MESH = True

    fluxCalculator()


"""## Alternative subset finding methods using clustering (not currently used)
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans, SpectralClustering
# from sklearn.metrics import silhouette_score, calinski_harabasz_score
def find_subsets_kmeans(max_subsets, theta_r_pts, mag_axis, field, BINS=30):
    split_data = []
    wrapped_flag = False
    theta = theta_r_pts.T[0]
    r = theta_r_pts.T[1]
    features = np.column_stack((r, np.sin(theta), np.cos(theta)))

    if max_subsets == 1:
        km = KMeans(n_clusters=1, random_state=0)
        labels = km.fit_predict(features)
    
    best_score = 3000 #0.64
    best_k = 1
    best_labels = 1
    for k in range(max_subsets, max_subsets+1): # !!only doing 1 n!!  (2 -> max_subsets)
        km = KMeans(n_clusters=k, n_init=10, random_state=0, algorithm='elkan')
        labels = km.fit_predict(features)

        sil_score = silhouette_score(features, labels)
        score = calinski_harabasz_score(features, labels)
        #print(f'k={k}, sil_score: {sil_score}, ch_score: {score}')
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    if best_k < 3:
        best_k = 1
        found_centers = np.zeros([best_k, 2])
        split_data = [theta_r_pts]
        found_centers[0][:] = mag_axis[:2]
        wrapped_flag = False
    else:
        found_centers = np.zeros([best_k, 2])
        
        for k in range(0, best_k):
            subset_data = theta_r_pts[best_labels == k]
            found_centers[k][:] = find_Axis(subset_data.T[0], subset_data.T[1], field)
            subset_data_locAxis = np.array([axisShift(theta, r, *found_centers[k]) for theta,r in subset_data])
            subset_data_locAxis = subset_data_locAxis[np.argsort(subset_data_locAxis[:, 0])]
            split_data += [subset_data_locAxis] # no need to sort?
            if subset_data.T[0].max() - subset_data.T[0].min() > np.pi:
                wrapped_flag = True

    hist = bin_edges = [] # leftover returns from histogram subset finding

    return split_data, found_centers, hist, bin_edges, wrapped_flag
def find_subsets_spectral(max_subsets, theta_r_pts, mag_axis, field, BINS=30):
    split_data = []
    wrapped_flag = False
    theta = theta_r_pts.T[0]
    r = theta_r_pts.T[1]
    features = np.column_stack((r, np.sin(theta), np.cos(theta)))

    if max_subsets == 1:
        km = KMeans(n_clusters=1, random_state=0)
        labels = km.fit_predict(features)
    
    best_score = 3000 #0.64
    best_k = 1
    best_labels = 1
    for k in range(max_subsets, max_subsets+1): # !!only doing 1 n!!  (2 -> max_subsets)
        spectral = SpectralClustering(n_clusters=k, n_init=1, random_state=0, assign_labels='kmeans')
        labels = spectral.fit_predict(features)

        sil_score = silhouette_score(features, labels)
        score = calinski_harabasz_score(features, labels)
        #print(f'k={k}, sil_score: {sil_score}, ch_score: {score}')
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    if best_k < 3:
        best_k = 1
        found_centers = np.zeros([best_k, 2])
        split_data = [theta_r_pts]
        found_centers[0][:] = mag_axis[:2]
        wrapped_flag = False
    else:
        found_centers = np.zeros([best_k, 2])
        
        for k in range(0, best_k):
            subset_data = theta_r_pts[best_labels == k]
            found_centers[k][:] = find_Axis(subset_data.T[0], subset_data.T[1], field)
            subset_data_locAxis = np.array([axisShift(theta, r, *found_centers[k]) for theta,r in subset_data])
            subset_data_locAxis = subset_data_locAxis[np.argsort(subset_data_locAxis[:, 0])]
            split_data += [subset_data_locAxis] # no need to sort?
            if subset_data.T[0].max() - subset_data.T[0].min() > np.pi:
                wrapped_flag = True

    hist = bin_edges = [] # leftover returns from histogram subset finding
    return split_data, found_centers, hist, bin_edges, wrapped_flag"""
