import numpy as np
from scipy.interpolate import make_smoothing_spline, spalde, splev, splrep

import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 10})
#plt.rcParams.update({'figure.autolayout':True})

import classes.class_outputHandler as out
from classes.mesh import *

from utility.coordtrans import axisShift, XYZ_to_RTP, RTP_to_XYZ
from utility.anlys_funcs import identifyLCFS, find_Axis


def main():

    ## SET UP RUN DIRECTORY
    simIO = out.IOHandler("It486_Ih900_Iv000_0p955_rtol6_49lines_2x300spins") #DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
    simIO.startLog()

    ## DEFINE MESH AND LOAD FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField('input_files/It486_Ih900_Iv000_0p955_hires.npy', errField=True)

    ## LOOP THROUGH PHI ANGLES
    PHI_GENs = np.linspace(9, 360, 40)
    
    ## IDENTIFY LAST-CLOSED FLUX SURFACE
    LCFS_index = identifyLCFS(LCFStype='input', num=13, outputHandler=simIO) 

    for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):

        ## LOAD POINCARE DATA
        filename = 'Poincare_{:03d}.npy'.format(int(PHI_GEN_DEG))
        flux_surfaces = simIO.loadNumpyData(filename)

        ## GET r, theta FOR SMALLEST FLUX SURFACE (ASSUMED LAST FLUX SURFACE IN SET)
        th_small, r_small = flux_surfaces[-1]
        r_small = r_small[~np.isnan(r_small)]
        th_small = th_small[~np.isnan(th_small)]
        th_size = th_small.size

        ## FIND THE MAGNETIC AXIS
        RTP_delta = find_Axis(th_small, r_small, b_hidra)
        RTP_delta_rev = np.copy(RTP_delta)
        RTP_delta_rev[1] = RTP_delta_rev[1] + np.pi

        ## SET UP PLOTTING
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=False)  

        ## LOOP THROUGH FLUX SURFACES
        NSURFACE = len(flux_surfaces)
        if PHI_GEN_DEG == PHI_GENs[0]:
            flux = np.zeros([NSURFACE, len(PHI_GENs)])

        for surf_index in range(LCFS_index, NSURFACE):

            ## GET r, theta FOR FLUX SURFACE
            th_in, r_in = flux_surfaces[surf_index]
            r_in = r_in[~np.isnan(r_in)]
            th_in = th_in[~np.isnan(th_in)]
            th_size = th_in.size

            # shift origin of r, theta coordinates from geometric center to magnetic axis
            magCenterCoords = np.empty((th_size, 2))
            for j, theta, in enumerate(th_in):
                magCenterCoords[j] = axisShift(r_in[j], theta, *RTP_delta[:2])

            # Sort data in increasing theta
            sortedMagCenter = magCenterCoords[np.argsort(magCenterCoords[:,0])]
            theta_pts = sortedMagCenter.T[0]
            rad_pts = sortedMagCenter.T[1]

            # # make a histogram of the point density in thet_pts  with binns at every 2*pi/360 radians
            # fig2 = plt.figure()
            # ax2 = fig2.add_subplot(111, polar=False)
            # ax2.hist(theta_pts, bins=90, range=(0, 2*np.pi))
            # plt.show()

            # fit a periodic spline to the data
            #fSurface_splineParms, residual, success, msg = splrep(theta_pts, rad_pts, xb=0.0, xe=2*np.pi, k=3, s=1e-6, per=True, full_output=1, quiet=1)

            # Copy data to both ends for pseudo-periodicity (smooth spline endpoints)
            append_length = int(th_size/2)
            th_A = theta_pts[append_length:-1] - 2*np.pi
            th_B = theta_pts[1:append_length] + 2*np.pi
            theta_spl = np.concatenate((th_A, theta_pts, th_B))
            rad_A = rad_pts[append_length:-1]
            rad_B = rad_pts[1:append_length]
            rad_spl = np.concatenate((rad_A, rad_pts, rad_B))

            fSurface_splineParms, residual, success, msg = splrep(theta_spl, rad_spl, k=1, s=1e-5, per=False, full_output=1, quiet=1)

            # Create a set of regularly-spaced points evalutaed on the spline fit
            NTHETA = 360*5
            dtheta = 2*np.pi/NTHETA
            theta_evals = np.linspace(dtheta, 2*np.pi, NTHETA)

            reg_points = splev(theta_evals, fSurface_splineParms)
            #derivs =  splev(theta_evals, fSurface_splineParms, der=1)
 
            for th_index, theta in enumerate(theta_evals):

                ## Shift fitted data back to geometric coordinates
                r_geo, th_geo = axisShift(reg_points[th_index], theta, *RTP_delta_rev[:2])
            
                ## Calculate B-field at the point
                bxyz, dum_ = b_hidra.interpField(np.array([r_geo, th_geo, PHI_GEN_DEG*np.pi/180]), Cart=False)

                ## Transform B-field vectors to Br, Btheta, Bphi components
                ctheta = np.cos(theta)
                stheta = np.sin(theta)
                cphi = np.cos(PHI_GEN_DEG*np.pi/180)
                sphi = np.sin(PHI_GEN_DEG*np.pi/180)

                Xform = np.array([[ctheta*cphi, -ctheta*sphi, stheta],
		        				[ -stheta*cphi,  stheta*sphi, ctheta],
		        				[ -sphi, -cphi, 0]])
                
                br, bpol, btor = np.dot(Xform, bxyz)

                # 'INTEGRATE' THE FLUX OVER RANGE OF THETAs
                flux[surf_index][phi_index] += btor * reg_points[th_index] * dtheta  # distance from magnetic axis

                # /END THETA LOOP #

            # plot the spline fit
            if np.all(reg_points < 0.15) and np.all(reg_points > 0.0):
                plt.plot(theta_evals, reg_points, '-o', markersize='0.3', linewidth=0.75, label='$\psi=${:.4e}'.format(flux[surf_index][phi_index]) )

            plt.scatter(theta_pts, rad_pts, color='k', s=1, linewidths=0.0) # mag-axis point
            #plt.scatter(th_in, r_in, color='k', s=1) # geo-axis points

            # /END FLUX SURFACE LOOP #

        # Format cartesian plot
        ax.set_ylim(0, 0.19)
        ax.grid(linewidth = 0.25, linestyle=':', c='k')
        ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, fontsize='xx-small', ncols=2)
        # Save Plot
        plt.title('Spline fit to Last Closed Flux Surface @ phi={}'.format(PHI_GEN_DEG))
        plt.savefig('Flux_at_{:03d}deg.png'.format(int(PHI_GEN_DEG)), dpi=300)
        plt.close()

        # /END PHI LOOP #

    ## CALCULATE TOTAL FLUX OF EACH FLUX SURFACE
    total_flux = np.sum(flux, axis=1)
    simIO.log.info('Total flux: {}'.format(total_flux))

    ## PLOT FLUX VS. FLUX SURFACE FOR EVERY PHI
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=False)
    for ind in range(flux.shape[1]):
        ax.plot(flux.T[ind], '-o')
    #ax.set_ylim(0, 50)
    plt.title('Flux vs. Flux Surface')
    plt.savefig('Flux_vs_Flux_Surface.png', dpi=300)


    ## PLOT TOTAL FLUX
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=False)  
    ax.plot(total_flux[::-1], '-o')
    plt.title('Total Flux')
    plt.savefig('Total_Flux.png', dpi=300)

    ## END RUN ##



if __name__ == '__main__':
    main()