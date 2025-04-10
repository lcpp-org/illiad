import logging
import classes.class_outputHandler as out

import numpy as np
from scipy.interpolate import make_smoothing_spline, spalde, splev, splrep
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from classes.mesh import *
from utility.coordtrans import axisShift, RTP_to_XYZ, XYZ_to_RTP
from utility.anlys_funcs import identifyLCFS #, find_Axis, find_subsets, spline_Data,
np.set_printoptions(threshold=np.inf)


def main():
    ## DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
    simIO = out.IOHandler(ANLYS_DIR)
    simIO.startLog()
    simIO.createSubDir(ANLYS_SUBDIR)

    ## DEFINE MESH AND LOAD FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField(FIELD_FILE_TOR, errField=True, att_mult=FIELD_SCALE_TOR)
    b_hidra.addFieldPerturbation(FIELD_FILE_HEL, att_mult=FIELD_SCALE_HEL)
    b_hidra.set_nonPer_errField(FIELD_ERR_MAG, FIELD_ERR_DIR)
    lcfs_index = identifyLCFS(LCFStype='input', num=LCFS_INPUT, outputHandler=simIO)

    # DEFINE THETAS FOR SPLINE GENERATION
    dtheta = 2*np.pi/NTHETA
    THETA_GENs = np.linspace(dtheta, 2*np.pi, NTHETA)

    ## LOOP THROUGH PHI ANGLES
    for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):
        ## LOAD POINCARE DATA
        filename = 'Poincare_{:03d}.npy'.format(int(PHI_GEN_DEG))
        flux_surfaces = simIO.loadNumpyData(filename)

        # Initialize plotting
        if PLOT_ALL: ax1, ax2, ax4 = init_plotting()

        ## FIND THE MAGNETIC AXIS FROM SMALLEST FLUX SURFACE (ASSUMED LAST FLUX SURFACE IN SET)
        th_small, r_small = flux_surfaces[-1]
        r_small = r_small[~np.isnan(r_small)]
        th_small = th_small[~np.isnan(th_small)]
        th_size = th_small.size
        mag_axis = find_Axis(th_small, r_small, b_hidra)
        mag_axis_rev = np.copy(mag_axis)
        mag_axis_rev[1] = mag_axis_rev[1] + np.pi

        # DECLARE BIG ARRAYS TO STORE DATA OVER SURFACES AND PHI ANGLES
        NSURFACE = len(flux_surfaces)
        if phi_index == 0: #if PHI_GEN_DEG == PHI_GENs[0]:
            tot_flux_array = np.zeros([NSURFACE, len(PHI_GENs)])
            total_flux_norm = np.zeros([NSURFACE, len(PHI_GENs)])
            flat_point_meshes = np.full([NSURFACE, len(PHI_GENs), NTHETA*MAX_SUBSETS, 2], np.nan)
            centers_array = np.zeros([NSURFACE, len(PHI_GENs), MAX_SUBSETS, 2])
            plotData_list = [ [0]*NSURFACE for _ in range(len(PHI_GENs)) ]

        ## LOOP THROUGH FLUX SURFACES TO FIND SUBSETS (ISLAND) AND THE SET OF SMALLEST ISLANDS
        smallest_island_index, num_subsets, subsetData, subsetCenters, hist_output = first_surface_loop(flux_surfaces, mag_axis, b_hidra, lcfs_index, NSURFACE)
        hist, bin_edges, wrap_flag = hist_output

        ## (2ND) LOOP THROUGH FLUX SURFACES TO SHIFT DATA AND SPLINE FIT
        Fluxes = []
        for surf_index in range(lcfs_index, NSURFACE):

            # DECLARE A WHOLE BUNCH OF EMPTY ARRAYS
            N_subsets = num_subsets[surf_index]
            radpoints_tr_LocAxis = np.zeros([N_subsets, NTHETA])
            radpoints_tr_MagAxis = np.zeros([N_subsets, NTHETA])
            theta_evals_MagAxis  = np.zeros([N_subsets, NTHETA])
            points_tr_GeoAxis    = np.zeros([N_subsets, NTHETA, 2])
            subCenters_geo       = np.zeros([N_subsets, 2])
            subCenters_Shift     = np.zeros([N_subsets, 2])
    
            ## GET R, THETA FOR FLUX SURFACE
            th_in, r_in = flux_surfaces[surf_index]
            r_in = r_in[~np.isnan(r_in)]
            th_in = th_in[~np.isnan(th_in)]
            th_size = th_in.size

            # SHIFT ORIGIN OF R, THETA COORDINATES FROM GEOMETRIC CENTER TO MAGNETIC AXIS
            points_tr_MagAxis = np.empty((th_size, 2))
            points_tr_MagAxis[:] = axisShift(r_in[:], th_in, *mag_axis[:2]).T

            # REMOVE DUPLICATE THETA VALUES AND SORT DATA IN INCREASING THETA
            unique_indices = np.unique(points_tr_MagAxis[:, 0], return_index=True)[1]
            points_tr_MagAxis = points_tr_MagAxis[unique_indices]
            th_size = points_tr_MagAxis.shape[0]
            points_tr_MagAxis = points_tr_MagAxis[np.argsort(points_tr_MagAxis[:, 0])]

            # Find the shift in the centers of the smallest islands relative to the current islands
            # Subset ordering may have chnaged due to periodic wraparound of centers
            shiftint = 0
            if N_subsets > 1: subCenters_Shift, shiftint = shift_the_subcenters(surf_index, smallest_island_index, subsetCenters, num_subsets, wrap_flag[surf_index] )

            ## LOOP THROUGH SUBSETS TO SPLINE< CALCULATE FLUX AND AND CREATE REGULARLY-SPACED POINTS
            sub_flux = []
            for subset_index in range(N_subsets):

                current_data = subsetData[surf_index][subset_index]
                current_center = subsetCenters[surf_index][subset_index]

                if N_subsets > 1:
                    # Shift the data points to the centers of the smallest islands
                    for i in range(len(current_data)):
                        current_data[i] = axisShift( current_data[i][1], current_data[i][0], subCenters_Shift[subset_index][1], subCenters_Shift[subset_index][0])
                    # sort by theta and set the centers of all island as the centers of the smallest islands
                    current_data = current_data[np.argsort(current_data.T[0])]
                    current_center = subsetCenters[smallest_island_index][subset_index-shiftint]
                    # Define the data points relative to the geometric axis
                    subCenters_geo[subset_index][:] = axisShift(current_center[0], current_center[1], *mag_axis_rev[:2])

                else: # Define the data points relative to the geometric axis
                    subCenters_geo[subset_index][0] = current_center[1]
                    subCenters_geo[subset_index][1] = current_center[0]

                theta_toSpline = current_data.T[0]
                rad_toSpline = current_data.T[1]
                th_size = theta_toSpline.size

                ## SPLINE FIT #
                fSurface_splineParms, res, fail, msg = spline_Data(theta_toSpline, rad_toSpline, smoothing=SMOOTH_FCTR)
                if fail: simIO.log.info( '\tSurface #{}, fail: {}\n\tmsg: {}'.format(surf_index, bool(fail), msg) )
                else:    pass #simIO.log.info('\tSurface #{}, res: {:.4e}'.format(surf_index, res))

                ## INTEGRATE AMOUNT OF TOROIDAL FIELD BOUNDED BY FLUX SURFACE [Tesla*m^2]
                this_flux= None
                if FLUX_CALC_FLAG:
                    this_flux = integrate_flux( fSurface_splineParms, current_center,
                                                PHI_GEN_DEG*np.pi/180, b_hidra,
                                                del_r=INTGRTE_DR, del_theta=INTGRTE_DTHTA)
                    sub_flux += [this_flux] # append flux to list of subset fluxes

                # CREATE A SET OF REGULARLY-SPACED POINTS EVALUATED ON THE SPLINE FIT
                #current_theta_gens = (THETA_GENs + subCenters_geo[subset_index][0]) % (2*np.pi)
                current_theta_gens = (THETA_GENs + current_center[1]) % (2*np.pi)
                radpoints_tr_LocAxis[subset_index] = splev(current_theta_gens, fSurface_splineParms)

                for th_index, theta in enumerate(current_theta_gens):
                    ## Shift r, theta back relative to overall magnetic axis
                    if N_subsets > 1:
                        shift_r = current_center[0]
                        shift_theta = current_center[1] + np.pi
                        new_vals = axisShift(radpoints_tr_LocAxis[subset_index][th_index], theta, shift_r, shift_theta)
                        theta_evals_MagAxis[subset_index][th_index] = new_vals[0]
                        radpoints_tr_MagAxis[subset_index][th_index] =  new_vals[1]
                    else:
                       theta_evals_MagAxis[subset_index][th_index] = theta
                       radpoints_tr_MagAxis[subset_index][th_index] = radpoints_tr_LocAxis[subset_index][th_index]

                    ## Shift r, theta back relative to geometric axis
                    points_tr_GeoAxis[subset_index][th_index] = axisShift(radpoints_tr_MagAxis[subset_index][th_index], theta_evals_MagAxis[subset_index][th_index], *mag_axis_rev[:2])
            ##### END SUBSET LOOP

            ## DANGER, BETTER DATA STRUCTURES NEEDED!!! ##
            total_npts = num_subsets[surf_index]*NTHETA
            plotData_list[phi_index][surf_index] = points_tr_GeoAxis.reshape((total_npts, 2), order='C', copy=True)
            #simIO.log.info('plotData_list: {}'.format(plotData_list[phi_index][surf_index]))

            #Fluxes.append(sub_flux)
            Fluxes += [sub_flux]

            ## PLOTTING EACH FLUX SURFACE AT EACH PHI ANGLE
            if PLOT_ALL:
                # filter out wild fits: if np.all(radpoints_tr_MagAxis < 0.19) and np.all(radpoints_tr_MagAxis > 0.0): 
                # plot the data points
                ax1.scatter(points_tr_MagAxis.T[0]*180./np.pi, points_tr_MagAxis.T[1], color='k', s=0.15, linewidths=0.0) # mag-axis point
                # plot the spline fit
                for i in range(0, num_subsets[surf_index]):
                    ax1.scatter(theta_evals_MagAxis[i]*180./np.pi, radpoints_tr_MagAxis[i], s=0.15, linewidths=0.0)
                # plot the histogram
                if num_subsets[surf_index] > 1:
                   ax2.bar(bin_edges[surf_index][:-1]*180./np.pi, hist[surf_index], width=np.diff(bin_edges[surf_index])*180./np.pi, align='edge', edgecolor='k', linewidth=0.1)
        ###### END OF SURFACE LOOP

        # PRINTING FLUXES AS OUTPUT:
        if FLUX_CALC_FLAG:
            for i, flux in enumerate(Fluxes):
                this_surf_index = i + lcfs_index

                tot_flux_array[this_surf_index][phi_index] = np.sum(flux, axis=-1)
                lcfs_flux = tot_flux_array[lcfs_index][phi_index] 
                total_flux_norm[this_surf_index][phi_index] =np.copy( max((1 - (tot_flux_array[this_surf_index][phi_index]/lcfs_flux)), 0.) )
                simIO.log.info('Surface {:d}: {:.2e}({:.4f})'.format(this_surf_index, tot_flux_array[this_surf_index][phi_index], total_flux_norm[this_surf_index][phi_index]))

        # FORMATTING DATA TO SAVE AS NUMPY ARRAY
        for surf_index in range(lcfs_index, NSURFACE):
            plot_tr_points = plotData_list[phi_index][surf_index]
            NOW_NPTS = plot_tr_points.shape[0]

            plot_thetas = plot_tr_points.T[0]
            plot_radii = plot_tr_points.T[1]
            lcfs_radii = plotData_list[phi_index][lcfs_index].T[1]

            # Filter out wild fits
            delta_plot_rs = np.max(lcfs_radii) - np.max(plot_radii)
            if delta_plot_rs > 0.:
                # Add points to output arrays
                flat_point_meshes[surf_index][phi_index][:NOW_NPTS] = plot_tr_points
                # if surf_index < 63 and surf_index < 50:
                #     simIO.log.info('flat_point_meshes[surf_index][phi_index][:]: {}'.format(flat_point_meshes[surf_index][phi_index][:]))

                centers_array[surf_index][phi_index] = subCenters_geo   #new array shape, center for each subset    #[0] # only one center for each surface
                if PLOT_ALL: ax4.scatter(plot_thetas, plot_radii, s=0.3, linewidths=0.0)

        # FORMAT AND SAVE PLOTS
        if PLOT_ALL: finalize_plotting(ax1, ax2, ax4, PHI_GEN_DEG, surf_index, num_subsets, simIO)
    ##### END OF PHI LOOP

    ## OUTPUT ANALYSIS
    ##################
    if FLUX_CALC_FLAG:
        # Set fluxes outside of LCFS to be equal to the LCFS flux
        tot_flux_array[:lcfs_index][:] = tot_flux_array[lcfs_index][:]
        # Set range of data to be between 0 and the LCFS flux
        tot_flux_array = np.maximum(tot_flux_array, 0.0)
        tot_flux_array = np.minimum(tot_flux_array, tot_flux_array[lcfs_index])

        # Set normed fluxes outside of LCFS to be equal to the 0
        total_flux_norm[:lcfs_index][:] = 0
        # Set range of data to be between 0 and 1
        total_flux_norm = np.maximum(total_flux_norm, 0.0)
        total_flux_norm = np.minimum(total_flux_norm, 1.0)


        filename_fluxes = ANLYS_SUBDIR + '/CalculatedFLuxes.npy'
        filename_fluxNorms = ANLYS_SUBDIR + '/CalculatedFLuxes-normalized.npy'
        simIO.saveNumpyData(tot_flux_array, filename_fluxes)
        simIO.saveNumpyData(total_flux_norm, filename_fluxNorms)




        # HAVE A BIG ARRAY OF FLUXES, NOW PLOT THEM
        fig_post = plt.figure()

        # PLOT THE FLUX 
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

        # SAVE THE FIGURE
        simIO.saveFig(ANLYS_SUBDIR+'/Flux_v_Surface.png', dpi=300)
        #plt.show()
        plt.close()

    # Save the numpy arrays to individual files for each surf_index using simIO method
    for surf_index in range(lcfs_index, NSURFACE):
        filename_center = ANLYS_SUBDIR + '/fSurf_{:03d}_center.npy'.format(surf_index)
        simIO.saveNumpyData(centers_array[surf_index], filename_center)
        filename_pt_mesh = ANLYS_SUBDIR + '/fSurf_{:03d}_POINTmesh.npy'.format(surf_index)
        simIO.saveNumpyData(flat_point_meshes[surf_index], filename_pt_mesh)
    
## END main()

def find_Axis(theta_vals, r_vals, field):
    """Function to find the geometric center of a set of points in r, theta coordinates"""
    theta_size = theta_vals.size
    ## CONVERT TO 2D XZ COORDINATES
    x_in = np.empty(theta_size)
    y_in = np.empty(theta_size)
    z_in = np.empty(theta_size)
    for i, theta, in enumerate(theta_vals):
        x_in[i], y_in[i], z_in[i] = RTP_to_XYZ(np.array([r_vals[i], theta, 0.]), field.R0)

    ## FIND THE AXIS BY AVERAGING THE POSITIONS
    x_avg = np.average(x_in)
    y_avg = 0.0
    z_avg = np.average(z_in)

    axis_xyz = np.array([x_avg, y_avg, z_avg])
    axis_rtp = XYZ_to_RTP(axis_xyz, field.R0)

    return axis_rtp

def find_subsets(theta_r_pts, mag_axis, field, BINS=30):
    """Function to find contiguous subsets of points in theta-r space"""
    wrapped_flag = False
    rmax = np.max(theta_r_pts.T[1])
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
    num_sets = len(contiguous_sets)

    subsetData = []
    subsetCenters = np.zeros([num_sets, 2])
    # loop throught each contiguous subset
    for i, contiguous_set in enumerate(contiguous_sets):
        thisSet_tr = []
        # calculate bin bounds
        lowerBound = contiguous_set*dtheta_bin
        upperBound = lowerBound + dtheta_bin
        # append data within each bin belonging to the subset
        for lo, hi in zip(lowerBound, upperBound):
            thisSet_tr += [point for point in theta_r_pts if lo <= point[0] < hi]
        thisSet_tr = np.array(thisSet_tr)

        # sort the subset by theta
        thisSet_tr = thisSet_tr[np.argsort(thisSet_tr[:, 0])]

        # TESTING, ONLY CONSIDER 3 SUBSETS!
        # only split if there are between 3 and 5 subsets, treat rest as 1 set
        if num_sets > 2 and num_sets < 4: # and rmax < 0.12: 
            subsetCenters[i][:] = find_Axis(thisSet_tr.T[0], thisSet_tr.T[1], field)[:2]
            # shift the data to be relative to the center of the subset
            thisSetLocAxis = np.array([axisShift(r, theta, *subsetCenters[i][:2]) for theta, r in thisSet_tr])
            thisSetLocAxis = thisSetLocAxis[np.argsort(thisSetLocAxis[:, 0])]
            subsetData += [thisSetLocAxis]
        # if there is only 1 subset, or lots(noisy data), then keep the original magnetic axis
        else:
            subsetCenters[i][:] = mag_axis[:2]
            thisSetLocAxis = thisSet_tr
            subsetData = [theta_r_pts]

    return subsetData, subsetCenters, hist, bin_edges, wrapped_flag

def shift_the_subcenters(surf_index, smallest_island_index, subsetCenters, num_subsets, wrap_flag):
    """Function performs tests to see if there is a misalignment of subset centers between the smallest island set and the current island set.
    If so, it returns the appropriate r and theta values to shift the data set to be relative to the smallest island subcenters """
   
    shifted_data = np.zeros([num_subsets[surf_index], 2])
    smallest_subset_centers = subsetCenters[smallest_island_index]
    this_subset_centers = subsetCenters[surf_index]

    shifted_data = axisShift(smallest_subset_centers.T[0], smallest_subset_centers.T[1], this_subset_centers.T[0], this_subset_centers.T[1]).T

    dtheta_this_to_smallest = smallest_subset_centers[0][1] - this_subset_centers[0][1]
    first_set_theta = smallest_subset_centers[0][1]  # first set's dist. to 0
    last_set_theta = 2*np.pi - smallest_subset_centers[-1][1]  # last set's dist. to 0

    proximity_cond = first_set_theta > last_set_theta 
    misalign_cond = abs(dtheta_this_to_smallest) > np.pi/2. # subset centers don't line up between current and smallest island subset
    shift_flag = wrap_flag and proximity_cond and misalign_cond
    
    if shift_flag:
        #temp_Shift= axisShift(*np.roll(smallest_subset_centers,-1), *this_subset_centers)
        for subset_index in range(num_subsets[surf_index]):
                shifted_data[subset_index] = axisShift(smallest_subset_centers[subset_index-1][0], smallest_subset_centers[subset_index-1][1],
                                                     this_subset_centers[subset_index][0], this_subset_centers[subset_index][1])

    return shifted_data, shift_flag

def spline_Data(theta_pts, rad_pts, smoothing=1e-5):
    """Function to create a smoothing spline fit of the data points."""
    # Copy data to both ends for pseudo-periodicity (smooth spline endpoints) 
    # Unsure why this seems to work better than setting "per=True" in splrep
    th_size = len(theta_pts)
    append_length = int(th_size/2)

    th_A = theta_pts[append_length:-1] - 2*np.pi
    th_B = theta_pts[1:append_length] + 2*np.pi
    theta_spl = np.concatenate((th_A, theta_pts, th_B))

    rad_A = rad_pts[append_length:-1]
    rad_B = rad_pts[1:append_length]
    rad_spl = np.concatenate((rad_A, rad_pts, rad_B))

    # spline parameters
    return splrep(theta_spl, rad_spl, k=3, s=smoothing, per=False, full_output=1, quiet=1)

def integrate_flux(spline_parms, spline_axis, phi, field, del_r=0.001, del_theta=0.001):
    """ Integrate the total toroidal flux contained within the given spline define relative to the given center
     SET delta_theta, delta_r for integration, (calculate N, M)    
     RETURN a scalar value of toroidal flux for the (sub)set """

    # First, calculate the toroidal flux at the local center (not geo center!)
    r0, theta0 = spline_axis

    ## USE SCIPY INTEGRATION METHODS
    ## B_tor(r, theta)
    ## a= 0.
    ## b= 2*np.pi
    ## gfun = lambda y: 0.
    ## hfun = lambda y: r_surf(theta)
    # INTEGRATION HELPER FUNCTIONS
    def B_tor(r, theta, phi, field):
        """Function to calculate the toroidal field at a given point in space"""
        geo_point = np.array([r+r0, theta+theta0, phi])
        if geo_point[0] < 0.0:
            geo_point[0] *= -1.
            geo_point[1] += np.pi

        bxyz = field.interpField(geo_point, Cart=False)[0]
        sphi = np.sin(phi)
        cphi = np.cos(phi)
        btor = -bxyz[0]*sphi - bxyz[1]*cphi
        return btor * r

    def gfun(theta):
        """Function to calculate the lower radial bound of the integration"""
        return 0.0 #r0

    def hfun(theta):
        """Function to calculate the upper radial bound of the integration"""
        return splev(theta, spline_parms)# + r0

    ## INTEGRATE TOROIDAL FLUX
    PSI, abserr = dblquad(B_tor, 0., 2*np.pi, gfun, hfun, args=(phi, field), epsabs=1e-05, epsrel=1e-03)


    return float(PSI)




def first_surface_loop(flux_surfaces, mag_axis, b_hidra, start_index, end_index):
     ## INPUT: LCFS_index, NSURFACE, flux_surfaces[], MAG_AXIS, b_hidra

    num_subsets = np.zeros(end_index, dtype=int)
    set_mean_rads = np.zeros(end_index)
    subsetData = [0]*end_index
    subsetCenters = [0]*end_index
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

        #points_tr_MagAxis = axisShift(r_in, th_in, *mag_axis[:2])
        for j, theta in enumerate(th_in):
            points_tr_MagAxis[j] = axisShift(r_in[j], theta, *mag_axis[:2])

        # Remove duplicate theta values, sort data in increasing theta
        unique_indices = np.unique(points_tr_MagAxis[:, 0], return_index=True, return_counts=False)[1:]
        points_tr_MagAxis = points_tr_MagAxis[unique_indices]
        th_size = points_tr_MagAxis.shape[0]
        points_tr_MagAxis = points_tr_MagAxis[np.argsort(points_tr_MagAxis[:, 0])]

        # find subsets of the data, and their local centers, data returned as theta, r relative to local center
        subsetData[surf_index], subsetCenters[surf_index], hist[surf_index], bin_edges[surf_index], wrap_flag[surf_index] = find_subsets(points_tr_MagAxis, mag_axis, b_hidra, BINS=100)
        num_subsets[surf_index] = len(subsetData[surf_index])


        subset_mean_rads = np.zeros(num_subsets[surf_index])
        # LOOP THROUGH SUBSETS   
        for subset_index in range(num_subsets[surf_index]):
            rad_toSpline = subsetData[surf_index][subset_index].T[1]
            subset_mean_rads[subset_index] = np.mean(rad_toSpline)
        set_mean_rads[surf_index] = np.mean(subset_mean_rads)

    # FIND THE INDEX OF THE ISLANDS OF SMALLEST RADIUS
    island_indices = np.where(num_subsets > 1)[0]
    smallest_island_index = island_indices[np.argmin(set_mean_rads[island_indices])]

    hist_data = (hist, bin_edges, wrap_flag)

    return smallest_island_index, num_subsets, subsetData, subsetCenters, hist_data

# PLOTTING FUNCTIONS
def init_plotting():
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])

    ax4 = fig.add_subplot(gs[:,0], polar=True) #projection='3d')
    ax1 = fig.add_subplot(gs[0,1], polar=False)
    ax2 = fig.add_subplot(gs[1,1], polar=False)
    return ax1, ax2, ax4

def finalize_plotting(ax1, ax2, ax4, PHI_GEN_DEG, surf_index, num_subsets, simIO):
    num_islandSurfaces= np.where(num_subsets == 3)[0].size
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
    #ANLYS_DIR = "Mar14FIT_89at360_2000sing_1p49e12_2p49e9"

    ANLYS_DIR = "AcceptedIota3_1500spins_atole-9"
    ANLYS_SUBDIR = 'LCFS22_3x180x60mesh_FluxTest4-NEW4_epsabs1e-5_epsrel=1e-3'

    #ANLYS_DIR = "ChangeToIota3_1500spins_atole-9"
    #ANLYS_SUBDIR = 'LCFS18_3x60x60mesh_s5e-6'

    ## DEFINE FIELDS
    FIELD_FILE_TOR = 'input_files/It486_Ih000_Iv000_1p000_1p000_64bit.npy'
    FIELD_SCALE_TOR = 0.9452
    FIELD_FILE_HEL = 'input_files/It000_Ih900_Iv000_1p000_1p000_64bit.npy'
    FIELD_SCALE_HEL = -0.955 * FIELD_SCALE_TOR
    FIELD_ERR_MAG = 1.5939e-4 #3.168e-4
    FIELD_ERR_DIR = np.radians(272.)

    ## IDENTIFY LAST-CLOSED FLUX SURFACE
    LCFS_INPUT = 22

    ## DEFINE ANGLES TO EVALUATE AND PLOT
    NPHI = 180
    NTHETA = 60

    PHI_GENs = np.linspace(360//NPHI, 360, NPHI)
    MAX_SUBSETS = 3
    SMOOTH_FCTR = 5e-6

    ## FLUX INTEGRATION PARAMETERS
    FLUX_CALC_FLAG = True
    INTGRTE_DR = 0.0015 #meter
    INTGRTE_DTHTA = 0.1 #rad


    ## PLOTTING FLAG
    PLOT_ALL = True

    main() 