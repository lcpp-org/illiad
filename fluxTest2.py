import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.interpolate import make_smoothing_spline, spalde, splev, splrep
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import classes.class_outputHandler as out
from classes.mesh import *
from utility.coordtrans import axisShift, XYZ_to_RTP, RTP_to_XYZ, RTP_XYZ_JAC
from utility.anlys_funcs import identifyLCFS, find_Axis, spline_Data, find_subsets


def main():

    ## SET UP RUN DIRECTORY
    ## DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
    simIO = out.IOHandler("It486_Ih900_Iv000_0p955_45lines_1500spins_rtol8")
    simIO.startLog()
    anlys_dir = 'FluxTest_pop_1stcenterGTlast_newFred_100BINS'
    simIO.createSubDir(anlys_dir)

    ## DEFINE MESH AND LOAD FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField('input_files/It486_Ih900_Iv000_0p955_hires.npy', errField=True)

    ## IDENTIFY LAST-CLOSED FLUX SURFACE
    #LCFS_index = identifyLCFS(LCFStype='input', num=11, outputHandler=simIO) 
    LCFS_index = 14 #27 #16 
    #NSURFACE = 40
    ## LOOP THROUGH PHI ANGLES
    ##########################
    NTHETA =  360*3
    dtheta = 2*np.pi/NTHETA
    THETA_EVALSOG = np.linspace(dtheta, 2*np.pi, NTHETA)
    PHI_GENs = np.linspace(9, 360, 40)

    plot_all = True
    calc_flux = False

    for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):
        # Option to plot all flux surfaces for each phi degree
        if plot_all:
            fig = plt.figure()
            gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])

            ax4 = fig.add_subplot(gs[:,0], polar=True)
            ax1 = fig.add_subplot(gs[0,1], polar=False)
            ax2 = fig.add_subplot(gs[1,1], polar=False)

        ## LOAD POINCARE DATA
        filename = 'Poincare_{:03d}.npy'.format(int(PHI_GEN_DEG))
        flux_surfaces = simIO.loadNumpyData(filename)

        ## FIND THE MAGNETIC AXIS FROM SMALLEST FLUX SURFACE (ASSUMED LAST FLUX SURFACE IN SET)
        th_small, r_small = flux_surfaces[-1]
        r_small = r_small[~np.isnan(r_small)]
        th_small = th_small[~np.isnan(th_small)]
        th_size = th_small.size

        MAG_AXIS = find_Axis(th_small, r_small, b_hidra)
        MAG_AXIS_rev = np.copy(MAG_AXIS)
        MAG_AXIS_rev[1] = MAG_AXIS_rev[1] + np.pi

        NSURFACE = len(flux_surfaces)
        if PHI_GEN_DEG == PHI_GENs[0]:
            flux = np.zeros([NSURFACE, len(PHI_GENs)])
            plotData_list = [ [0]*NSURFACE for _ in range(len(PHI_GENs)) ]

        ## INSTANTIATE A BUNCHA VARS
        num_subsets = np.zeros(NSURFACE, dtype=int)
        set_mean_rads = np.zeros(NSURFACE)
        subsetData = [0]*NSURFACE
        subsetCenters = [0]*NSURFACE
        hist = [0]*NSURFACE
        bin_edges = [0]*NSURFACE
        testy_flag = [0]*NSURFACE

        ##############################
        ## FIRST LOOP THROUGH FLUX SURFACES TO FIND SUBSETS (ISLAND) AND THE SET OF SMALLEST ISLANDS
        for surf_index in range(LCFS_index, NSURFACE):
        
            ## GET R, THETA FOR FLUX SURFACE
            th_in, r_in = flux_surfaces[surf_index]
            r_in = r_in[~np.isnan(r_in)]
            th_in = th_in[~np.isnan(th_in)]
            th_size = th_in.size

            # shift origin of r, theta coordinates from geometric center to magnetic axis
            points_tr_MagAxis = np.empty((th_size, 2))
            for j, theta, in enumerate(th_in):
                points_tr_MagAxis[j] = axisShift(r_in[j], theta, *MAG_AXIS[:2])

            # Remove duplicate theta values, sort data in increasing theta
            unique_indices = np.unique(points_tr_MagAxis[:, 0], return_index=True, return_counts=False)[1:]
            points_tr_MagAxis = points_tr_MagAxis[unique_indices]
            th_size = points_tr_MagAxis.size
            points_tr_MagAxis = points_tr_MagAxis[np.argsort(points_tr_MagAxis[:, 0])]

            # find subsets of the data, and their local centers, data returned as theta, r relative to local center
            subsetData[surf_index], subsetCenters[surf_index], hist[surf_index], bin_edges[surf_index], testy_flag[surf_index] = find_subsets(points_tr_MagAxis, MAG_AXIS, b_hidra, BINS=100)
            num_subsets[surf_index] = len(subsetData[surf_index])
            # simIO.log.info('Surface #{}, phi={}: Contiguous sets: {}'.format(surf_index, PHI_GEN_DEG, num_subsets))

            subset_mean_rads = np.zeros(num_subsets[surf_index])
            # LOOP THROUGH SUBSETS   
            for subset_index in range(num_subsets[surf_index]):
                rad_toSpline = subsetData[surf_index][subset_index].T[1]
                subset_mean_rads[subset_index] = np.mean(rad_toSpline)
            set_mean_rads[surf_index] = np.mean(subset_mean_rads)

        # FIND THE INDEX OF THE ISLANDS OF SMALLEST RADIUS
        island_indices = np.where(num_subsets > 1)[0]
        smallest_island_index = island_indices[np.argmin(set_mean_rads[island_indices])]


        ##############################
        ## SECOND LOOP THROUGH SURFACES TO SHIFT DATA AND SPLINE FIT
        for surf_index in range(LCFS_index, NSURFACE):

            ## GET R, THETA FOR FLUX SURFACE
            th_in, r_in = flux_surfaces[surf_index]
            r_in = r_in[~np.isnan(r_in)]
            th_in = th_in[~np.isnan(th_in)]
            th_size = th_in.size

            # shift origin of r, theta coordinates from geometric center to magnetic axis
            points_tr_MagAxis = np.empty((th_size, 2))
            for j, theta, in enumerate(th_in):
                points_tr_MagAxis[j] = axisShift(r_in[j], theta, *MAG_AXIS[:2])

            # Remove duplicate theta values
            unique_indices, counts = np.unique(points_tr_MagAxis[:, 0], return_index=True, return_counts=True)[1:]
            points_tr_MagAxis = points_tr_MagAxis[unique_indices]
            th_size = points_tr_MagAxis.size
            # Sort data in increasing theta
            points_tr_MagAxis = points_tr_MagAxis[np.argsort(points_tr_MagAxis[:, 0])]

            radpoints_tr_LocAxis = np.zeros([num_subsets[surf_index], NTHETA])
            radpoints_tr_MagAxis = np.zeros([num_subsets[surf_index], NTHETA])
            theta_evals_MagAxis = np.zeros([num_subsets[surf_index], NTHETA])
            points_tr_GeoAxis = np.zeros([num_subsets[surf_index], NTHETA, 2])
            subCenters_geo = np.zeros([num_subsets[surf_index], 2])
            subCenters_Shift = np.zeros([num_subsets[surf_index], 2])
            
            # Find the shift in the centers of the smallest islands relative to the current islands
            # Subset ordering may have chnaged due to periodic wraparound of centers
            if num_subsets[surf_index] > 1:
                subCenters_Shift, shiftint = shift_the_subcenters(surf_index, smallest_island_index, subsetCenters, num_subsets, testy_flag[surf_index] )
                #simIO.log.info('Surface #{}, shiftint: {}, subsetCenters: {}'.format(surf_index, shiftint, subsetCenters[surf_index]))
                simIO.log.info('Surface #{}, shiftint: {}'.format(surf_index, shiftint))
            ## LOOP THROUGH SUBSETS
            for subset_index in range(num_subsets[surf_index]): 

                if num_subsets[surf_index] > 1:

                    # Shift the data points to the centers of the smallest islands
                    for i in range(len(subsetData[surf_index][subset_index])):
                        subsetData[surf_index][subset_index][i] = axisShift( subsetData[surf_index][subset_index][i][1], subsetData[surf_index][subset_index][i][0],
                                                                             subCenters_Shift[subset_index][1], subCenters_Shift[subset_index][0])
                    # sort by theta
                    subsetData[surf_index][subset_index] = subsetData[surf_index][subset_index][np.argsort(subsetData[surf_index][subset_index].T[0])]
                    # Set the centers of all island to the centers of the smallest islands
                    subsetCenters[surf_index][subset_index] = subsetCenters[smallest_island_index][subset_index-shiftint]
                    # Define the data points relative to the geometric axis
                    subCenters_geo[subset_index][:] = axisShift(subsetCenters[surf_index][subset_index][0], subsetCenters[surf_index][subset_index][1], *MAG_AXIS_rev[:2])
                
                else:
                    # Define the data points relative to the geometric axis
                    subCenters_geo[subset_index][0] = subsetCenters[surf_index][subset_index][1]
                    subCenters_geo[subset_index][1] = subsetCenters[surf_index][subset_index][0]
                
                theta_toSpline = subsetData[surf_index][subset_index].T[0]
                rad_toSpline = subsetData[surf_index][subset_index].T[1]
                th_size = theta_toSpline.size

                ## SPLINE FIT #
                fSurface_splineParms, res, fail, msg = spline_Data(theta_toSpline, rad_toSpline)
                if fail:
                    simIO.log.info( 'Surface #{}, fail: {}'.format(surf_index, bool(fail)) )
                    simIO.log.info('msg: {}'.format(msg))
                #else:
                #    simIO.log.info('Surface #{}, res: {}'.format(surf_index, res))

                # Create a set of regularly-spaced points evaluated on the spline fit
                radpoints_tr_LocAxis[subset_index] = splev(THETA_EVALSOG, fSurface_splineParms)

                ## LOOP THROUGH THETA POINTS
                for th_index, theta in enumerate(THETA_EVALSOG):
                    
                    ## Shift r, theta back relative to overall magnetic axis
                    if num_subsets[surf_index] > 1:
                        shift_r = subsetCenters[surf_index][subset_index][0]
                        shift_theta = subsetCenters[surf_index][subset_index][1] + np.pi
                        new_vals = axisShift(radpoints_tr_LocAxis[subset_index][th_index], theta, shift_r, shift_theta)
                        theta_evals_MagAxis[subset_index][th_index] = new_vals[0]
                        radpoints_tr_MagAxis[subset_index][th_index] =  new_vals[1]
                    else:
                       theta_evals_MagAxis[subset_index][th_index] = theta
                       radpoints_tr_MagAxis[subset_index][th_index] = radpoints_tr_LocAxis[subset_index][th_index]

                    ## Shift r, theta back relative to geometric axis
                    points_tr_GeoAxis[subset_index][th_index] = axisShift(radpoints_tr_MagAxis[subset_index][th_index], theta_evals_MagAxis[subset_index][th_index], *MAG_AXIS_rev[:2])

                    if calc_flux:
                        ## Calculate B-field at the point
                        bxyz = b_hidra.interpField(np.array([*points_tr_GeoAxis[subset_index][th_index], PHI_GEN_DEG*np.pi/180]), Cart=False)[0]
                        ## Transform B-field vectors to Br, Btheta, Bphi components
                        btor = RTP_XYZ_JAC(np.array([*points_tr_GeoAxis[subset_index][th_index], PHI_GEN_DEG*np.pi/180]), bxyz)[2]
                        ## 'INTEGRATE' THE FLUX OVER RANGE OF THETAs
                        flux[surf_index][phi_index] += btor * radpoints_tr_MagAxis[subset_index][th_index] * dtheta  # distance from magnetic axis

            plotData_list[phi_index][surf_index] = points_tr_GeoAxis.reshape(num_subsets[surf_index]*NTHETA, 2)


            ## PLOTTING EACH FLUX SURFACE AT EACH PHI ANGLE
            if plot_all:

                # # filter out wild fits
                # if np.all(radpoints_tr_MagAxis < 0.19) and np.all(radpoints_tr_MagAxis > 0.0): 
                # plot the data points
                ax1.scatter(points_tr_MagAxis.T[0]*180./np.pi, points_tr_MagAxis.T[1], color='k', s=0.2, linewidths=0.0) # mag-axis point
                # plot the spline fit
                for i in range(0, num_subsets[surf_index]):
                    #ax1.plot(theta_evals_MagAxis[i], radpoints_tr_MagAxis[i], '-', linewidth=0.4)#, label='$\psi=${:.4e}'.format(flux[surf_index][phi_index]) ) 
                    ax1.scatter(theta_evals_MagAxis[i]*180./np.pi, radpoints_tr_MagAxis[i], s=0.25, linewidths=0.0)


                # plot the histogram
                if num_subsets[surf_index] > 1:
                    ax2.bar(bin_edges[surf_index][:-1]*180./np.pi, hist[surf_index], width=np.diff(bin_edges[surf_index])*180./np.pi, align='edge', edgecolor='k', linewidth=0.1)
                
                #filter out wild fits
                #if np.all( plotData_list[phi_index][surf_index].T[1] < 0.19):# and np.all( plotData_list[phi_index][surf_index].T[1] > 0.0):
                # plot polar plot of fitted points and local centers
                ax4.scatter(plotData_list[phi_index][surf_index].T[0], plotData_list[phi_index][surf_index].T[1], s=0.2, linewidths=0.0)
                ax4.scatter(subCenters_geo.T[0], subCenters_geo.T[1], color='k', s=2, linewidths=0.0, zorder=5) # mag-axis point


        if plot_all:
            #plt.tight_layout()
            num_islandSurfaces= np.where(num_subsets == 3)[0].size

            # format and Save Plot
            ax1.set_ylim(0, 0.19)
            ax1.set_xticks(np.arange(0, 361, 45))
            ax1.tick_params(axis='both', which='major', labelsize=6)
            ax1.grid(linewidth = 0.25, linestyle=':', c='k')
            #ax1.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, fontsize='xx-small', ncols=2)

            ax2.set_xticks(np.arange(0, 361, 45))
            # set the axis labels font to be very small
            ax2.tick_params(axis='both', which='major', labelsize=6)

            ax4.set_title('Flux Surfaces {} @ phi={}\nIsland surfaces detected:{}'.format(surf_index, PHI_GEN_DEG, num_islandSurfaces), fontsize=8)
            ax4.set_rlim(0, 0.19)
            ax4.grid(linewidth = 0.25, linestyle='--', c='grey')

            simIO.saveFig(anlys_dir+'/Flux_at_{:03d}deg.png'.format(int(PHI_GEN_DEG)), dpi=400)
            plt.close()



def shift_the_subcenters(surf_index, smallest_island_index, subsetCenters, num_subsets, testy_flag):
    """Function performs tests to see if there is a misalignment of subset centers between the smallest island set and the current island set.
    If so, it returns the appropriate r and theta values to shift the data set to be relative to the smallest island subcenters """
    temp_Shift = np.zeros([num_subsets[surf_index], 2])
    shift_int = 0

    for subset_index in range(num_subsets[surf_index]):
            temp_Shift[subset_index] = axisShift(subsetCenters[smallest_island_index][subset_index][0], subsetCenters[smallest_island_index][subset_index][1],
                                                 subsetCenters[surf_index][subset_index][0], subsetCenters[surf_index][subset_index][1])
    fred = subsetCenters[smallest_island_index].T[1] - subsetCenters[surf_index].T[1]

    temp1 = subsetCenters[smallest_island_index][0][1] 
    temp2 = 2*np.pi - subsetCenters[smallest_island_index][-1][1] 
    cond1 = testy_flag #dataset has already been concatenated (contiguous across periodic boundary)
    cond2 = temp1 > temp2 # the last set is closer to 0 than the first set
    cond3 = abs(fred[0]) > 90*np.pi/180. # subset centers don't line up between current and smallest island subset

    if cond1 and cond2 and cond3:
        for subset_index in range(num_subsets[surf_index]):
                temp_Shift[subset_index] = axisShift(subsetCenters[smallest_island_index][subset_index-1][0], subsetCenters[smallest_island_index][subset_index-1][1],
                                                     subsetCenters[surf_index][subset_index][0], subsetCenters[surf_index][subset_index][1])
        shift_int = 1

    return temp_Shift, shift_int




if __name__ == '__main__':
    main()