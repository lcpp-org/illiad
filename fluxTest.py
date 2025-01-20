import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.interpolate import make_smoothing_spline, spalde, splev, splrep
import matplotlib.pyplot as plt

import classes.class_outputHandler as out
from classes.mesh import *
from utility.coordtrans import axisShift, XYZ_to_RTP, RTP_to_XYZ, RTP_XYZ_JAC
from utility.anlys_funcs import identifyLCFS, find_Axis, spline_Data, find_subsets


def main():

    ## SET UP RUN DIRECTORY
    ## DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
    #simIO = out.IOHandler("It486_Ih900_Iv000_0p955_42lines_1000spins_rtol8") 
    #simIO = out.IOHandler("It486_Ih900_Iv000_0p955_45lines_800spins_rtol7")
    simIO = out.IOHandler("It486_Ih900_Iv000_0p955_89lines_400spinsDOUBLED_rtol7")
    simIO.startLog()
    anlys_dir = 'FluxTest_1'
    simIO.createSubDir(anlys_dir)

    ## DEFINE MESH AND LOAD FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField('input_files/It486_Ih900_Iv000_0p955_hires.npy', errField=True)

    ## IDENTIFY LAST-CLOSED FLUX SURFACE
    #LCFS_index = identifyLCFS(LCFStype='input', num=11, outputHandler=simIO) 
    LCFS_index = 27#16 
    #NSURFACE = 40
    ## LOOP THROUGH PHI ANGLES
    ##########################
    NTHETA = 360 #24
    dtheta = 2*np.pi/NTHETA
    THETA_EVALSOG = np.linspace(dtheta, 2*np.pi, NTHETA)
    PHI_GENs = np.linspace(9, 360, 40)
    #PHI_GENs = np.linspace(1, 360, 360)

    plot_all = True
    calc_flux = False

    for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):

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

        if plot_all:
            fig = plt.figure()
            ax1 = fig.add_subplot(221, polar=False)  
            ax2 = fig.add_subplot(222)
            #ax3 = fig.add_subplot(223, polar=True)  
            ax4 = fig.add_subplot(224, polar=True)

        ## LOOP THROUGH FLUX SURFACES   
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

            # find subsets of the data, and their local centers, data returned as theta, r relative to local center
            subsetData, subsetCenters, hist, bin_edges = find_subsets(points_tr_MagAxis, MAG_AXIS, b_hidra)

            num_subsets = len(subsetData)
            simIO.log.info('Surface #{}, phi={}: Contiguous sets: {}'.format(surf_index, PHI_GEN_DEG, num_subsets))
            radpoints_tr_LocAxis = np.zeros([num_subsets, NTHETA])
            radpoints_tr_MagAxis = np.zeros([num_subsets, NTHETA])
            theta_evals_MagAxis = np.zeros([num_subsets, NTHETA])
            points_tr_GeoAxis = np.zeros([num_subsets, NTHETA, 2])
            subCenters_geo = np.zeros([num_subsets, 2])

            
            ## LOOP THROUGH SUBSETS     
            for subset_index in range(num_subsets):
                if num_subsets > 1:
                    subCenters_geo[subset_index][:] = axisShift(subsetCenters[subset_index][0], subsetCenters[subset_index][1], *MAG_AXIS_rev[:2])
                else:
                    subCenters_geo[subset_index][0] = subsetCenters[subset_index][1]
                    subCenters_geo[subset_index][1] = subsetCenters[subset_index][0]
                
                theta_toSpline = subsetData[subset_index].T[0]
                rad_toSpline = subsetData[subset_index].T[1]
                th_size = theta_toSpline.size

                ## SPLINE FIT #
                fSurface_splineParms, res, fail, msg = spline_Data(theta_toSpline, rad_toSpline)
                if fail:
                    simIO.log.info( 'Surface #{}, fail: {}'.format(surf_index, bool(fail)) )
                    simIO.log.info('msg: {}'.format(msg))
                else:
                    pass#simIO.log.info('Surface #{}, res: {}'.format(surf_index, res))

                # Create a set of regularly-spaced points evalutaed on the spline fit
                radpoints_tr_LocAxis[subset_index] = splev(THETA_EVALSOG, fSurface_splineParms)

                ## LOOP THROUGH THETA POINTS
                for th_index, theta in enumerate(THETA_EVALSOG):
                    
                    ## Shift r, theta back relative to overall magnetic axis
                    if num_subsets > 1:
                        shift_r = subsetCenters[subset_index][0]
                        shift_theta = subsetCenters[subset_index][1] + np.pi
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
                        bxyz, dum_ = b_hidra.interpField(np.array([*points_tr_GeoAxis[subset_index][th_index], PHI_GEN_DEG*np.pi/180]), Cart=False)
                        ## Transform B-field vectors to Br, Btheta, Bphi components
                        br, bpol, btor = RTP_XYZ_JAC(np.array([*points_tr_GeoAxis[subset_index][th_index], PHI_GEN_DEG*np.pi/180]), bxyz)
                        ## 'INTEGRATE' THE FLUX OVER RANGE OF THETAs
                        flux[surf_index][phi_index] += btor * radpoints_tr_MagAxis[subset_index][th_index] * dtheta  # distance from magnetic axis

                #/END theta
            #/END subset
            plotData_list[phi_index][surf_index] = points_tr_GeoAxis.reshape(num_subsets*NTHETA, 2)

            ## PLOTTING EACH FLUX SURFACE AT EACH PHI ANGLE
            ###########
            if plot_all:
            #     fig = plt.figure()
            #     ax1 = fig.add_subplot(221, polar=False)  
            #     ax2 = fig.add_subplot(222)
            #     #ax3 = fig.add_subplot(223, polar=True)  
            #     ax4 = fig.add_subplot(224, polar=True)

                # plot the spline fit
                if np.all(radpoints_tr_MagAxis < 0.19) and np.all(radpoints_tr_MagAxis > 0.0): #filter out wild fits
                    for i in range(0, num_subsets):
                        #if np.any(theta_evals_MagAxis[i] < 355.*np.pi/180.) and np.any(theta_evals_MagAxis[i] < 5.*np.pi/180.):
                        #    theta_evals_MagAxis[i] = np.where(theta_evals_MagAxis[i] < np.pi, theta_evals_MagAxis[i]+ 2*np.pi, theta_evals_MagAxis[i])
                        ax1.plot(theta_evals_MagAxis[i], radpoints_tr_MagAxis[i], '-', linewidth=0.4)#, label='$\psi=${:.4e}'.format(flux[surf_index][phi_index]) ) 
                # plot the data points
                ax1.scatter(points_tr_MagAxis.T[0], points_tr_MagAxis.T[1], color='k', s=0.25, linewidths=0.0) # mag-axis point

                # Format and Save Plot
                ax1.set_ylim(0, 0.19)
                ax1.grid(linewidth = 0.25, linestyle=':', c='k')
                ax1.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, fontsize='xx-small', ncols=2)

                # Plot the histogram
                ax2.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge', edgecolor='k')
                ax2.set_title('Spline fit to Last Flux Surface {} @ phi={}'.format(surf_index, PHI_GEN_DEG), fontsize=8)

                # Plot something else
                #ax3.scatter(theta_evals_MagAxis, radpoints_tr_MagAxis, s=2, linewidths=0.3) # mag-axis point

                # Plot polar plot of fitted points and local centers
                if np.all( plotData_list[phi_index][surf_index].T[1] < 0.19) and np.all( plotData_list[phi_index][surf_index].T[1] > 0.0): #filter out wild fits
                    ax4.plot(plotData_list[phi_index][surf_index].T[0], plotData_list[phi_index][surf_index].T[1], markersize=0.1, linewidth=0.4)
                    #ax4.scatter(subsetCenters.T[1], subsetCenters.T[0], color='k', s=3, linewidths=0.0, zorder=5) # mag-axis point
                    ax4.scatter(subCenters_geo.T[0], subCenters_geo.T[1], color='k', s=3, linewidths=0.0, zorder=5) # mag-axis point

                # simIO.saveFig(anlys_dir+'/Flux{:03d}_at_{:03d}deg.png'.format(surf_index, int(PHI_GEN_DEG)), dpi=300)
                # plt.close()
        if plot_all:
            simIO.saveFig(anlys_dir+'/Flux_at_{:03d}deg.png'.format(int(PHI_GEN_DEG)), dpi=400)
            plt.close()

        #/END surface
    #/END phi

    ## PLOT FLUX SURFACES
    #####################
    plot_surfIndex = [67, 41] #21
    plotData_XYZ_list = []

    for surf_index in plot_surfIndex:
        onesurf_plotData_XYZ_list = []
        #print('surf_index = ', surf_index)
        for phi_index, phi in enumerate(PHI_GENs):
            phi_radians = phi * np.pi/180
            for theta_index in range(len(plotData_list[phi_index][surf_index])):
                r = plotData_list[phi_index][surf_index][theta_index][1]
                theta = plotData_list[phi_index][surf_index][theta_index][0]

                xyz_point = RTP_to_XYZ([r, theta, phi_radians], b_hidra.R0)
                onesurf_plotData_XYZ_list += [xyz_point]
            
        plotData_XYZ_list += [onesurf_plotData_XYZ_list]

    plot1_array = np.array(plotData_XYZ_list[0])
    plot2_array = np.array(plotData_XYZ_list[1])


    fig2 = plt.figure()
    ax5 = fig2.add_subplot(projection='3d')

    # ## PLOT VACUUM VESSEL TORUS
    # ptheta = np.linspace(-np.pi, np.pi, 91) #np.linspace(-np.pi, 0, 46) #to plot only bottom half of torus
    # pphi   = np.linspace(0, np.pi, 91) #np.linspace(0, 2.*np.pi, 91) 
    # ptheta, pphi = np.meshgrid(ptheta, pphi)
    # px = (b_hidra.R0 + b_hidra.a*np.cos(ptheta)) * np.cos(pphi)
    # py = (b_hidra.R0 + b_hidra.a*np.cos(ptheta)) * np.sin(pphi)
    # pz = b_hidra.a * np.sin(ptheta)
    # ax5.plot_wireframe(px, py, pz, rstride=1, cstride=1, edgecolor='k', linewidth=0.1)

    ## PLOT FITTED FLUX SURFACE POINTS
    ax5.plot(plot1_array.T[0], plot1_array.T[1], plot1_array.T[2],
                        '#13294b', linestyle='none', marker='.', markersize=2.0, markeredgewidth=0.0)
    
    setlength = plot2_array.shape[0]
     #plot 1 subset (1 phi rotation)
    # for i in range(NTHETA*2, setlength-NTHETA*3, NTHETA*3):
    #     ax5.plot(plot2_array.T[0][i+2*NTHETA:i+3*NTHETA], plot2_array.T[1][i+2*NTHETA:i+3*NTHETA], plot2_array.T[2][i+2*NTHETA:i+3*NTHETA],
    #                     '#ff5f05', linestyle='none', marker='.', markersize=1.0, markeredgewidth=0.0)
        
    # plot whole set
    ax5.plot(plot2_array.T[0], plot2_array.T[1], plot2_array.T[2],
                        '#ff5f05', linestyle='none', marker='.', markersize=1.0, markeredgewidth=0.0)


    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_aspect('equal')
    ax5.set_title('Flux Surfaces')
    simIO.saveFig(anlys_dir+'/SelectedFluxSurfaces.png', dpi=400)
    plt.show()


    if calc_flux:
        ## SUM FLUX OF EACH SURFACE OVER PHIs
        #####################################
        total_flux = np.sum(flux, axis=1)
        simIO.log.info('Total flux: {}'.format(total_flux))
        ## PLOT FLUX VS. FLUX SURFACE FOR EVERY PHI
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=False)
        for ind in range(flux.shape[1]):
            ax.plot(flux.T[ind], '-o')
        ax.set_ylim(0,3)
        plt.title('Flux vs. Flux Surface for every phi')
        simIO.saveFig(anlys_dir+'/Flux_vs_Flux_Surface.png')

        ## PLOT TOTAL FLUX
        ##################
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=False)  
        ax.plot(total_flux[::-1], '-o')
        ax.set_ylim(0,3)
        plt.title('Total Flux')
        simIO.saveFig(anlys_dir+'/Total_Flux.png')

    ## END RUN ##



if __name__ == '__main__':
    main()