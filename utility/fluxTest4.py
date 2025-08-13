import logging
import classes.class_outputHandler as out

import numpy as np
from scipy.interpolate import make_smoothing_spline, spalde, splev, splrep
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from classes.mesh import *
from utility.coordtrans import axisShift, RTP_to_XYZ, XYZ_to_RTP
np.set_printoptions(threshold=np.inf)

def main():
    ## DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
    simIO = out.IOHandler(ANLYS_DIR)
    simIO.startLog()
    simIO.createSubDir(ANLYS_SUBDIR)
    ## DEFINE MESH AND LOAD MAGNETIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField(coilCurrent=CURRENT_TOR, errField=True, att_mult=CONFIG_TOR)
    b_hidra.addFieldPerturbation(coilCurrent=CURRENT_HEL, att_mult=CONFIG_HEL)
    b_hidra.set_nonPer_errField()

    for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):
        ## LOAD POINCARE DATA
        filename = 'Poincare_{:03d}.npy'.format(int(PHI_GEN_DEG))
        flux_surfaces = simIO.loadNumpyData(filename)

        if PLOT_ALL: ax1, ax2, ax4 = init_plotting()
        ## FIND THE MAGNETIC AXIS FROM SMALLEST FLUX SURFACE (ASSUMED LAST FLUX SURFACE IN SET)
        mag_axis = find_Axis(*flux_surfaces[-1], b_hidra)
        mag_axis_rev = np.copy(mag_axis)
        mag_axis_rev[0] += np.pi

        # DECLARE BIG ARRAYS TO STORE DATA OVER SURFACES AND PHI ANGLES
        NSURFACE = len(flux_surfaces)
        if phi_index == 0:
            tot_flux_array = np.zeros([NSURFACE, len(PHI_GENs)])
            total_flux_norm = np.zeros([NSURFACE, len(PHI_GENs)])
            flat_point_meshes = np.full([NSURFACE, len(PHI_GENs), NTHETA*MAX_SUBSETS, 2], np.nan)
            centers_array = np.zeros([NSURFACE, len(PHI_GENs), MAX_SUBSETS, 2])
            plotData_list = [ [0]*NSURFACE for _ in range(len(PHI_GENs)) ]

            valid_surfs = np.zeros(NSURFACE, dtype=bool) # keep track of valid surfaces
            valid_surfs[LCFS_INDEX:] = True  # set all surfaces after LCFS to valid

        ## LOOP THROUGH FLUX SURFACES TO FIND SUBSETS (ISLAND) AND THE SET OF SMALLEST ISLANDS
        first_loop_output = first_surface_loop(flux_surfaces, mag_axis, b_hidra, LCFS_INDEX, NSURFACE, MAX_SUBSETS)
        smallest_island_index, num_subsets, subsetData, subsetCenters, hist_output = first_loop_output
        hist, bin_edges, wrap_flag = hist_output
        simIO.log.info('# of subset in each surface: {}'.format(num_subsets))
        simIO.log.info('Smallest island index: {}'.format(smallest_island_index))

        ## (2ND) LOOP THROUGH FLUX SURFACES TO SHIFT DATA AND SPLINE FIT
        Fluxes = []
        for surf_index in range(LCFS_INDEX, NSURFACE):
            N_subsets = num_subsets[surf_index]
            ## GET R, THETA FOR FLUX SURFACE
            th_in, r_in = flux_surfaces[surf_index]
            r_in = r_in[~np.isnan(r_in)]
            th_in = th_in[~np.isnan(th_in)]
            th_size = th_in.size

            # SHIFT ORIGIN OF R, THETA COORDINATES FROM GEOMETRIC CENTER TO MAGNETIC AXIS
            points_tr_MagAxis = np.empty((th_size, 2))
            points_tr_MagAxis[:] = axisShift(th_in, r_in[:], *mag_axis).T

            # REMOVE DUPLICATE THETA VALUES AND SORT DATA IN INCREASING THETA
            unique_indices = np.unique(points_tr_MagAxis[:, 0], return_index=True)[1]
            points_tr_MagAxis = points_tr_MagAxis[unique_indices]
            th_size = points_tr_MagAxis.shape[0]
            points_tr_MagAxis = points_tr_MagAxis[np.argsort(points_tr_MagAxis[:, 0])]

            ## SUBSET LOOP: LOOP TO SPLINE, CALCULATE FLUX, AND CREATE REGULARLY-SPACED POINTS
            output = subset_looper(subsetData, subsetCenters, surf_index,
                                    smallest_island_index, N_subsets, wrap_flag,
                                    b_hidra, mag_axis_rev, NTHETA, PHI_GEN_DEG,
                                    INTEGRATE_EPSABS, INTEGRATE_EPSREL, SMOOTH_FCTR, simIO)

            points_tr_GeoAxis, spline_tr_MagAxis, subCenters_geo, subset_flux_list, valid_surfs[surf_index] = output
            ## APPENDING LISTS AND RESHAPING ARRAYS
            total_npts = num_subsets[surf_index] * NTHETA
            plotData_list[phi_index][surf_index] = points_tr_GeoAxis.reshape((total_npts, 2), order='C', copy=True)
            Fluxes += [subset_flux_list]
            centers_array[surf_index][phi_index] = subCenters_geo

            ## CHECKING IF SURFACE IS 'VALID'
            if np.any(spline_tr_MagAxis[:,:,1] >= 0.19) or np.any(spline_tr_MagAxis[:,:,1] < 0.0):
                valid_surfs[surf_index] = False # not a valid surface for interpolation
                simIO.log.info( '\tSurface #{} NOT A VALID SURFACE!!!'.format(surf_index) )

            ## PLOTTING EACH FLUX SURFACE AT EACH PHI ANGLE
            if PLOT_ALL:
                # filter out wild fits: if np.all(radpoints_tr_MagAxis < 0.19) and np.all(radpoints_tr_MagAxis > 0.0): 
                # plot the data points
                ax1.scatter(points_tr_MagAxis.T[0]*180./np.pi, points_tr_MagAxis.T[1], color='k', s=0.10, linewidths=0.0) # mag-axis point
                # plot the spline fit
                for i in range(0, num_subsets[surf_index]):
                    ax1.scatter(spline_tr_MagAxis[i].T[0]*180./np.pi, spline_tr_MagAxis[i].T[1], s=0.3, linewidths=0.05)
                # plot the histogram
                if num_subsets[surf_index] > 1:
                   ax2.bar(bin_edges[surf_index][:-1]*180./np.pi, hist[surf_index],
                           width=np.diff(bin_edges[surf_index])*180./np.pi, align='edge', edgecolor='k', linewidth=0.1)

        # PRINTING FLUXES AS OUTPUT:
        for i, flux in enumerate(Fluxes):
            this_surf_index = i + LCFS_INDEX
            tot_flux_array[this_surf_index][phi_index] = np.sum(flux, axis=-1)
            lcfs_flux = tot_flux_array[LCFS_INDEX][phi_index]
            temp_norm = (1 - (tot_flux_array[this_surf_index][phi_index]/lcfs_flux))
            total_flux_norm[this_surf_index][phi_index] = np.copy(max(temp_norm, 0.))

        # FORMATTING DATA TO SAVE AS NUMPY ARRAY
        for surf_index in range(LCFS_INDEX, NSURFACE):
            plot_tr_points = plotData_list[phi_index][surf_index]
            NOW_NPTS = plot_tr_points.shape[0]
            plot_thetas = plot_tr_points.T[0]
            plot_radii = plot_tr_points.T[1]
            lcfs_radii = plotData_list[phi_index][LCFS_INDEX].T[1]

            # Filter out wild fits
            delta_plot_rs = np.max(lcfs_radii) - np.max(plot_radii)
            if delta_plot_rs > 0.: # Add points to output arrays
                flat_point_meshes[surf_index][phi_index][:NOW_NPTS] = plot_tr_points
                if PLOT_ALL: ax4.scatter(plot_thetas, plot_radii, s=0.3, linewidths=0.0)

        if PLOT_ALL: finalize_plotting(ax1, ax2, ax4, PHI_GEN_DEG, surf_index, num_subsets, MAX_SUBSETS, simIO)

    ## OUTPUT FLUXES
    # Set fluxes outside of LCFS to be equal to the LCFS flux
    tot_flux_array[:LCFS_INDEX][:] = tot_flux_array[LCFS_INDEX][:]
    # Set range of data to be between 0 and the LCFS flux
    tot_flux_array = np.maximum(tot_flux_array, 0.0)
    tot_flux_array = np.minimum(tot_flux_array, tot_flux_array[LCFS_INDEX])

    # Set normed fluxes outside of LCFS to be equal to the 0
    total_flux_norm[:LCFS_INDEX][:] = 0
    # Set range of data to be between 0 and 1
    total_flux_norm = np.maximum(total_flux_norm, 0.0)
    total_flux_norm = np.minimum(total_flux_norm, 1.0)

    filename_fluxes = ANLYS_SUBDIR + '/CalculatedFLuxes.npy'
    filename_fluxNorms = ANLYS_SUBDIR + '/CalculatedFLuxes-normalized.npy'
    simIO.saveNumpyData(tot_flux_array, filename_fluxes)
    simIO.saveNumpyData(total_flux_norm, filename_fluxNorms)

    filename_validSurfaces = ANLYS_SUBDIR + '/ValidSurfaces.npy'
    simIO.saveNumpyData(valid_surfs, filename_validSurfaces)

    # HAVE A BIG ARRAY OF FLUXES, NOW PLOT THEM
    fig_post = plt.figure()
    axUP = fig_post.add_subplot(211, title='$\\psi_{\phi}$ at each surface', xlabel='Surface index n', ylabel='Toroidal Flux$\\psi_{\phi},[gauss*m^2]$')
    for i in range(len(PHI_GENs)):
        axUP.plot(tot_flux_array[:,i]*10_000, label='{:d}'.format(int(PHI_GENs[i])), linewidth=0.25)
    axUP.set_ylim(0, 1.1*10_000*np.max(tot_flux_array))
    axUP.grid(which='both', linestyle=':', linewidth=0.25)
    axUP.legend(loc='upper right', fontsize=4,ncols=3)
    # PLOT THE SURFACE PARAMETER: 1 - FLUX/FLUX_LCFS
    axDOWN = fig_post.add_subplot(212, title='Surface Parameter $|\psi_n| = 1-\\frac{ \\psi_{\\phi,n} }{ \psi_{\\phi,LCFS} }$ at each surface', xlabel='Surface index n', ylabel='$|\psi_n|$')
    for i in range(len(PHI_GENs)):
            axDOWN.plot(total_flux_norm.T[i], label='{:d}'.format(int(PHI_GENs[i])), linewidth=0.25)
    axDOWN.set_ylim(0, 1.1)
    axDOWN.grid(which='both', linestyle=':', linewidth=0.25)
    axDOWN.legend(loc='upper left', fontsize=4, ncols=3)

    simIO.saveFig(ANLYS_SUBDIR+'/Flux_v_Surface.png', dpi=300)
    plt.close()

    # SAVE THE NUMPY ARRAYS TO INDIVIDUAL FILES USING SIMIO METHOD
    for surf_index in range(LCFS_INDEX, NSURFACE):
        filename_center = ANLYS_SUBDIR + '/fSurf_{:03d}_center.npy'.format(surf_index)
        simIO.saveNumpyData(centers_array[surf_index], filename_center)
        filename_pt_mesh = ANLYS_SUBDIR + '/fSurf_{:03d}_POINTmesh.npy'.format(surf_index)
        simIO.saveNumpyData(flat_point_meshes[surf_index], filename_pt_mesh)


def find_Axis(theta_vals: np.ndarray, r_vals: np.ndarray, field: Mesh) -> np.ndarray:
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

    theta_size = theta_vals.size
    ## CONVERT TO 2D XZ COORDINATES
    x_in = np.empty(theta_size)
    y_in = np.empty(theta_size)
    z_in = np.empty(theta_size)
    for i, theta, in enumerate(theta_vals):
        x_in[i], y_in[i], z_in[i] = RTP_to_XYZ(np.array([r_vals[i], theta, 0.]), field.R0)

    x_avg = np.average(x_in)
    y_avg = 0.0 # y is always 0 in this case (phi=0)
    z_avg = np.average(z_in)

    axis_xyz = np.array([x_avg, y_avg, z_avg])
    axis_thetar = XYZ_to_RTP(axis_xyz, field.R0)[1::-1]

    return axis_thetar

def find_subsets(max_subsets, theta_r_pts, mag_axis, field, BINS=30):
    """Function to find contiguous subsets of points in theta-r space"""
    # make a histogram of the point density vs theta
    hist, bin_edges = np.histogram(theta_r_pts.T[0], bins=BINS, range=(0., 2*np.pi))
    dtheta_bin = bin_edges[1] - bin_edges[0]

    # find how many contiguous sets of adjacents bins there are
    non_empty_bins = np.where(hist > 0)[0]
    contiguous_sets = np.split(non_empty_bins, np.where(np.diff(non_empty_bins) != 1)[0]+1)
    # if the first and last bins are non-empty, then the first and last sets of bins are contiguous
    if hist[0] > 0 and hist[-1] > 0 and len(contiguous_sets) > 1:
        contiguous_sets[0] = np.concatenate((contiguous_sets[-1], contiguous_sets[0]))
        contiguous_sets.pop()
        wrapped_flag = True
    else:
        wrapped_flag = False
    num_sets = len(contiguous_sets)

    split_data = []
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

        # TESTING: only split if the number of sets is equal to the maximum number of subsets: 
        if num_sets == max_subsets:
            found_centers[i][:] = find_Axis(thisSet_tr.T[0], thisSet_tr.T[1], field)
            # shift the data to be relative to the center of the subset
            thisSetLocAxis = np.array([axisShift(theta, r, *found_centers[i]) for theta, r in thisSet_tr])
            thisSetLocAxis = thisSetLocAxis[np.argsort(thisSetLocAxis[:, 0])]
            split_data += [thisSetLocAxis]
        # if there is only 1 subset, or lots(noisy data), then keep the original magnetic axis
        else:
            found_centers[i][:] = mag_axis[:2]
            thisSetLocAxis = thisSet_tr
            split_data = [theta_r_pts]

    return split_data, found_centers, hist, bin_edges, wrapped_flag

def shift_the_subcenters(surf_index, smallest_island_index, subsetCenters, num_subsets, wrap_flag):
    """Function performs tests to see if there is a misalignment of subset centers between the smallest island set and the current island set.
    If so, it returns the appropriate r and theta values to shift the data set to be relative to the smallest island subcenters """
   
    shifted_data = np.zeros([num_subsets, 2])
    smallest_centers = subsetCenters[smallest_island_index]
    these_centers = subsetCenters[surf_index]

    shifted_data = axisShift(*smallest_centers.T, *these_centers.T).T

    dtheta_this_to_smallest = smallest_centers[0][0] - these_centers[0][0]
    first_set_theta = smallest_centers[0][0]  # first set's dist. to 0
    last_set_theta = 2*np.pi - smallest_centers[-1][0]  # last set's dist. to 0

    proximity_cond = first_set_theta > last_set_theta
    misalign_cond = abs(dtheta_this_to_smallest) > np.pi / 2.  # subset centers don't line up between current and smallest island subset
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
    # Copy data to both ends for pseudo-periodicity (smooth spline endpoints) 
    # Unsure why this seems to work better than setting "per=True" in splrep
    append_length = int(len(theta_pts)/2)

    th_A = theta_pts[append_length:-1] - 2*np.pi
    th_B = theta_pts[1:append_length] + 2*np.pi
    theta_spl = np.concatenate((th_A, theta_pts, th_B))

    rad_A = rad_pts[append_length:-1]
    rad_B = rad_pts[1:append_length]
    rad_spl = np.concatenate((rad_A, rad_pts, rad_B))

    # spline parameters
    return splrep(theta_spl, rad_spl, k=3, s=smoothing, per=False, full_output=1, quiet=1)

def integrate_flux(spline_parms, spline_axis, phi, field, err_abs=1e-5, err_rel=1e-3):
    """
    Integrates the total toroidal flux contained within the given spline, defined relative to the specified center.

    Args:
        spline_parms (tuple): Spline parameters as returned by `scipy.interpolate.splrep`.
        spline_axis (array-like): The (theta, r) coordinates of the spline's center.
        phi (float): Toroidal angle (in radians) at which to evaluate the flux.
        field (Mesh): Mesh object providing the `interpField` method for magnetic field interpolation.
        err_abs (float, optional): Absolute error tolerance for integration. Defaults to 1e-5.
        err_rel (float, optional): Relative error tolerance for integration. Defaults to 1e-3.

    Returns:
        float: Scalar value of the toroidal flux for the (sub)set.
    """
    theta0, r0  = spline_axis

    # INTEGRATION HELPER FUNCTIONS
    def flux_integrand(r, theta, phi, field, axis):
        """Function to calculate the toroidal field times radius at a given point in space"""
        geo_point = np.array([r+axis[1], theta+axis[0], phi])
        #geo_point = np.array([r+r0, theta+theta0, phi])
        if geo_point[0] < 0.0:
            geo_point[0] *= -1.
            geo_point[1] += np.pi

        bxyz = field.interpField(geo_point, Cart=False)[0][:2]
        # Calculate the toroidal flux integrand
        # The integrand is -r*B_toroidal ( Bx*sin(phi) - By*cos(phi) )
        return -r*(bxyz[0]* np.sin(phi) - bxyz[1]*np.cos(phi))
    
    # Define the upper radial bound of the integration, lowerbound=0
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
      - Removes duplicate theta values and sorts the data.
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
        split_data (list): List of lists containing subset data for each surface.
        surface_axes (list): List of local centers for each subset in each surface.
        hist_data (tuple): Tuple containing (hist, bin_edges, wrap_flag) for each surface, useful for diagnostics or plotting.
    """
    num_subsets = np.zeros(end_index, dtype=int)
    set_mean_rads = np.zeros(end_index)
    split_data = [0]*end_index
    surface_axes = [0]*end_index
    hist = [0]*end_index
    bin_edges = [0]*end_index
    wrap_flag = [0]*end_index

    for surf_index in range(start_index, end_index):
        ## GET R, THETA FOR FLUX SURFACE
        th_in, r_in = flux_surfaces[surf_index]
        r_in = r_in[~np.isnan(r_in)]
        th_in = th_in[~np.isnan(th_in)]
        th_size = th_in.size

        # shift origin of r, theta coordinates from geometric center to magnetic axis
        points_tr_MagAxis = np.empty((th_size, 2))

        for j, theta in enumerate(th_in):
            points_tr_MagAxis[j] = axisShift(theta, r_in[j], *mag_axis)

        # Remove duplicate theta values, sort data in increasing theta
        unique_indices = np.unique(points_tr_MagAxis[:, 0], return_index=True, return_counts=False)[1:]
        points_tr_MagAxis = points_tr_MagAxis[unique_indices]
        th_size = points_tr_MagAxis.shape[0]
        points_tr_MagAxis = points_tr_MagAxis[np.argsort(points_tr_MagAxis[:, 0])]

        # find subsets of the data, and their local centers, data returned as theta, r relative to local center
        output = find_subsets(max_subsets, points_tr_MagAxis, mag_axis, field, BINS=120)
        split_data[surf_index], surface_axes[surf_index] = output[:2]
        num_subsets[surf_index] = len(split_data[surf_index])

        hist[surf_index], bin_edges[surf_index], wrap_flag[surf_index] = output[2:]

        subset_mean_rads = np.zeros(num_subsets[surf_index])
        # LOOP THROUGH SUBSETS
        for subset_index in range(num_subsets[surf_index]):
            rad_toSpline = split_data[surf_index][subset_index].T[1]
            subset_mean_rads[subset_index] = np.mean(rad_toSpline)
        set_mean_rads[surf_index] = np.mean(subset_mean_rads)

    # FIND THE INDEX OF THE ISLANDS OF SMALLEST RADIUS
    island_indices = np.where(num_subsets > 1)[0]
    if island_indices.size > 0:
        smallest_island_index = island_indices[np.argmin(set_mean_rads[island_indices])]
    else:
        smallest_island_index = end_index - 1  # If no islands found, return the last surface index

    hist_data = (hist, bin_edges, wrap_flag)

    return smallest_island_index, num_subsets, split_data, surface_axes, hist_data

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
    splined_tr_MagAxis    = np.zeros([num_subsets, NTHETA, 2])
    splined_tr_GeoAxis    = np.zeros([num_subsets, NTHETA, 2])
    subCenters_geo       = np.zeros([num_subsets, 2])
    THETA_GENs = np.linspace(2*np.pi/NTHETA, 2*np.pi, NTHETA)

    # Subset ordering may have changed due to periodic wraparound of centers
    if num_subsets > 1: 
        subCenters_Shift, shiftint = shift_the_subcenters(surf_index, smallest_island_index, subsetCenters, num_subsets, wrap_flag[surf_index] )
    else: shiftint = 0

    ## LOOP THROUGH SUBSETS TO SPLINE, CALCULATE FLUX, AND CREATE REGULARLY-SPACED POINTS
    subset_flux_list = []
    for subset_index in range(num_subsets):
        current_data = subsetData[surf_index][subset_index]
        current_center = subsetCenters[surf_index][subset_index]

        ## SHIFT THE DATA POINTS TO BE RELATIVE TO THE CENTERS OF THE SMALLEST ISLANDS
        if num_subsets > 1:
            for i in range(len(current_data)):
                current_data[i] = axisShift(*current_data[i], *subCenters_Shift[subset_index])
            # sort by theta and set the centers of all island as the centers of the smallest islands
            current_data = current_data[np.argsort(current_data.T[0])]
            current_center = subsetCenters[smallest_island_index][subset_index-shiftint]
            subCenters_geo[subset_index][:] = axisShift(*current_center, *mag_axis_rev)
        else:
            subCenters_geo[subset_index] = current_center

        ## SPLINE FIT
        fSurface_splineParms, res, fail, msg = spline_Data(*current_data.T, smoothing=SMOOTH_FCTR)
        if fail:
            valid_surface = False # not a valid surface for interpolation
            simIO.log.info( '\tSurface #{} NOT A VALID SURFACE!!!'.format(surf_index) )
            simIO.log.info( '\tSurface #{}, fail: {}\tmsg: {}'.format(surf_index, bool(fail), msg) )
        else:
            valid_surface = True # valid surface for interpolation

        ## INTEGRATE AMOUNT OF TOROIDAL FIELD BOUNDED BY FLUX SURFACE [Tesla*m^2]
        this_flux = integrate_flux(fSurface_splineParms, current_center,
                                   PHI_GEN_DEG*np.pi/180, field,
                                   err_abs=INTEGRATE_EPSABS, err_rel=INTEGRATE_EPSREL)
        subset_flux_list += [this_flux]

        # CREATE A SET OF REGULARLY-SPACED POINTS EVALUATED ON THE SPLINE FIT
        current_theta_gens = (THETA_GENs + current_center[0]) % (2*np.pi)
        radius_evals_locAxis[subset_index] = splev(current_theta_gens, fSurface_splineParms)
        # Shift the points to be relative to the magnetic axis
        for theta_index, theta in enumerate(current_theta_gens):
            if num_subsets > 1:
                shift_r = current_center[1]
                shift_theta = current_center[0] + np.pi
                splined_tr_MagAxis[subset_index][theta_index] = axisShift(theta, radius_evals_locAxis[subset_index][theta_index], shift_theta, shift_r)
            else:
               splined_tr_MagAxis[subset_index][theta_index][0] = theta
               splined_tr_MagAxis[subset_index][theta_index][1] = radius_evals_locAxis[subset_index][theta_index]

            ## Shift r, theta back relative to geometric axis
            splined_tr_GeoAxis[subset_index][theta_index] = axisShift(*splined_tr_MagAxis[subset_index][theta_index], *mag_axis_rev)

    return splined_tr_GeoAxis, splined_tr_MagAxis, subCenters_geo, subset_flux_list, valid_surface

def init_plotting():
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])

    ax4 = fig.add_subplot(gs[:,0], polar=True) #projection='3d')
    ax1 = fig.add_subplot(gs[0,1], polar=False)
    ax2 = fig.add_subplot(gs[1,1], polar=False)
    return ax1, ax2, ax4

def finalize_plotting(ax1, ax2, ax4, PHI_GEN_DEG, surf_index, num_subsets, MAX_SUBSETS, simIO):
    num_islandSurfaces= np.where(num_subsets == MAX_SUBSETS)[0].size
    ax1.set_ylim(0, 0.19)
    ax1.set_xticks(np.arange(0, 361, 45))
    ax1.tick_params(axis='both', which='major', labelsize=6)
    ax1.grid(linewidth = 0.25, linestyle=':', c='k')
    #ax1.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, fontsize='xx-small', ncols=2)
    
    # set the axis labels font to be very small
    ax2.set_xticks(np.arange(0, 361, 45))
    ax2.tick_params(axis='both', which='major', labelsize=6)
    ax2.grid(linewidth = 0.25, linestyle=':', c='k')

    ax4.set_title('Flux Surfaces {} @ phi={}\nIsland surfaces detected:{}'.format(surf_index, PHI_GEN_DEG, num_islandSurfaces), fontsize=8)
    ax4.set_rlim(0, 0.19)
    ax4.tick_params(axis='both', which='major', labelsize=6)
    ax4.grid(linewidth = 0.25, linestyle='--', c='grey')
    simIO.saveFig(ANLYS_SUBDIR+'/Flux_at_{:03d}deg.png'.format(int(PHI_GEN_DEG)), dpi=400)
    plt.close()

if __name__ == '__main__':
    #### DEFINE ANALYSIS PARAMETERS ####
    ## RUN DIRECTORY AND SUBDIRECTORY

    #ANLYS_DIR = "AcceptedIota3_1500spins_atole-9"
    # ANLYS_SUBDIR = ""

    # ANLYS_DIR = "AcceptedIota4_1500spins_atole-8_eng"
    # ANLYS_SUBDIR = "LCFS40_3x360x360mesh_UPDATED"

    # ANLYS_DIR = "ChangeToIota3_1500spins_atole-9"
    # ANLYS_SUBDIR = 'LCFS29_3x360x360mesh_CORRECTCURR_lotol'

    ANLYS_DIR = "ChangetoIota4_1500spins_atole-8_eng"
    ANLYS_SUBDIR = "LCFS39_4x360x360mesh_loTol"

    ## DEFINE FIELDS
    FIELD_FILE_TOR = 'input_files/It486_Ih000_Iv000_1p000_1p000_64bit.npy'
    FIELD_FILE_HEL = 'input_files/It000_Ih900_Iv000_1p000_1p000_64bit.npy'
    CURRENT_TOR = 0.486 #[kA]
    CURRENT_HEL = 0.790 #[kA]
    CONFIG_TOR = 'default_toroidal'
    CONFIG_HEL = 'default_helical_rev'

    ## DEFINE LCFS AND ANGLES TO EVALUATE
    LCFS_INDEX = 39 #40 #22 #29?
    NPHI = 360
    NTHETA = 360
    PHI_GENs = np.linspace(360//NPHI, 360, NPHI)

    ## FLUX INTEGRATION PARAMETERS
    MAX_SUBSETS = 4
    SMOOTH_FCTR = 8.0e-6 #7.5e-6 #baseline 1e-6
    # INTEGRATE_EPSABS=1e-5 #1.49e-5
    # INTEGRATE_EPSREL=1e-3 #4.49e-3
    INTEGRATE_EPSABS=1e-3
    INTEGRATE_EPSREL=1e-2

    ## PLOTTING FLAG
    PLOT_ALL = True

    main()