import numpy as np
import logging
from scipy.interpolate import make_smoothing_spline, spalde, splev, splrep

from coordtrans import *

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout':True})

def generateSeedShells(drList, Ntheta, r_in, th_in, phi, field, outputHandler, filename):
    outputHandler.createSubDir(filename)

    r_in = r_in[~np.isnan(r_in)]
    th_in = th_in[~np.isnan(th_in)]
    th_size = th_in.size

    # find the centroid(?) by average positions
    # two for loops, wow
    x_in = np.empty(th_size)
    y_in = np.empty(th_size)
    z_in = np.empty(th_size)
    for i, theta, in enumerate(th_in):
        x_in[i], y_in[i], z_in[i] = RTP_to_XYZ(np.array([r_in[i], theta, 0.]), field.R0)

    x_avg = (np.max(x_in) + np.min(x_in))/2
    y_avg = 0
    z_avg = (np.max(z_in) + np.min(z_in))/2
    XYZ_delta = np.array([x_avg, y_avg, z_avg])

    #XYZ_delta = np.mean(np.array([x_in, y_in, z_in]), axis=1)



    # shift origin of r, theta coordinates from geometric center to magnetic axis
    # then sort points on theta
    magCenterCoords = np.empty((th_size, 2))
    #r_delta, th_delta, dum = XYZ_to_RTP(XYZ_delta, field.R0)
    RTP_delta = XYZ_to_RTP(XYZ_delta, field.R0)
    RTP_delta_rev = np.copy(RTP_delta)
    RTP_delta_rev[1] = RTP_delta_rev[1] + np.pi
    #print(f'{RTP_delta=}')


    for i, theta, in enumerate(th_in):
        magCenterCoords[i] = axisShift(r_in[i], theta, *RTP_delta[:2])
    #print(f'{magCenterCoords.shape=}')

    # Sort data in increasing theta
    sortedMagCenter = magCenterCoords[np.argsort(magCenterCoords[:,0])]
    theta_spl = sortedMagCenter.T[0]
    rad_spl = sortedMagCenter.T[1]


    # Append data to either end for smooth spline endpoints
    theta_spl= np.append(theta_spl, theta_spl[:15]+2*np.pi)
    rad_spl= np.append(rad_spl, rad_spl[:15])

    theta_spl= np.append(theta_spl[:15]-2*np.pi, theta_spl)
    rad_spl= np.append(rad_spl[:15], rad_spl)


    ## SPLINING
    # perform curve-fitting (smoothing spline)
    # function and derivative continuity not enforced across periodic boundary (would need fancier spline)
    fSurface_splineParms = splrep(theta_spl, rad_spl, s=1e-6, k=3, per=True, quiet=1)

    theta_evals = np.linspace(0, 2*np.pi*(1 - 1/Ntheta), Ntheta)
    seedPts_0 = splev(theta_evals, fSurface_splineParms)
    derivs =  splev(theta_evals, fSurface_splineParms, der=1)


    ## PLOTTING THE SPLINE FIT
    thetaPlot = np.linspace(0., 2*np.pi, 5000)
    rPlot = splev(thetaPlot, fSurface_splineParms)
    geoCenterCoords = axisShift(rPlot, thetaPlot, *RTP_delta_rev[:2])

    rPlotGeo = geoCenterCoords[1]
    thetaPlotGeo = geoCenterCoords[0]
    #rPlotGeo = rPlot
    #thetaPlotGeo = thetaPlot

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    plt.scatter(th_in, r_in, s=1) # geo-axis points
    plt.scatter(theta_spl, rad_spl, s=1) # mag-axis points

    #plt.plot(thetaPlot, splev(thetaPlot, fSurface_splineParms), '-k', linewidth=0.5) # fitted spline curve
    plt.plot(thetaPlotGeo, rPlotGeo, '-k', linewidth=0.5) # fitted spline curve
    
    ax.set_rmax(field.a)
    ax.set_rticks(np.arange(0.0, 0.19, 0.02))
    ax.yaxis.set_tick_params(labelsize=5)
    ax.grid(linewidth = 0.25, linestyle=':', c='k')

    plt.title('Spline fit to Last Closed Flux Surface @ phi={}'.format(phi*180/np.pi))
    spline_name = filename+'/'+'LCFS_phi={:03.0f}_splineFit.png'.format(phi*180/np.pi)
    outputHandler.saveFig(spline_name)
    plt.close()




    ## Calculating (and plotting) the seed points
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    #plt.plot(thetaPlot, splev(thetaPlot, fSurface_splineParms), '-k', linewidth=1) # fitted spline curve
    plt.plot(thetaPlotGeo, rPlotGeo, '-k', linewidth=0.5) # fitted spline curve
    output_ind_geo = np.zeros((Ntheta, 2))
    output_ind     = np.zeros((Ntheta, 3))
    output_ind_XYZ = np.zeros((Ntheta, 3))
    outData = []

    #adj_dr = dr
    #seedPt = seedPts_0 + adj_dr


    for dr in drList:
        for i, theta in enumerate(theta_evals):
            # scale delta-r to achieve uniform expansion normal to surface
            #adj_dr = dr
            adj_dr = dr * np.sqrt(1 + derivs[i]**2)
            #seedPt = min(field.a, seedPts_0[i] + adj_dr)
            seedPt = seedPts_0[i] + adj_dr
            
            # shift back to geometric axis
            output_ind_geo[i] = axisShift(seedPt, theta, *RTP_delta_rev[:2])
            output_ind_geo[i][1] = min(field.a, output_ind_geo[i][1])

            # convert rtp vector to xyz
            output_ind[i] = np.array([output_ind_geo[i][1], output_ind_geo[i][0], phi])
            output_ind_XYZ[i] = RTP_to_XYZ(output_ind[i], field.R0)
        
        plt.plot(output_ind_geo[:,0], output_ind_geo[:,1], '--o', linewidth=0.25, markersize=0.50)
        
        outData.extend(output_ind_XYZ)



    ax.set_rmax(field.a)
    ax.set_rticks(np.arange(0.0, 0.19, 0.02))
    ax.yaxis.set_tick_params(labelsize=5)
    ax.grid(linewidth = 0.25, linestyle=':', c='k')

    plt.title(r'Generated Seed Points, $\phi$={:02.0f}$\degree$'.format(phi*180/np.pi))
    plot_name = filename+'/'+'InitConds_phi={:03.0f}.png'.format(phi*180/np.pi)
    outputHandler.saveFig(plot_name)
    plt.close()


    outArray = np.asarray(outData)
    outputHandler.saveNumpyData(outArray, filename)

    outputHandler.log.info('Seed Shell Generated')
    return outArray
