import numpy as np
import logging
from scipy.interpolate import make_smoothing_spline, spalde, splev, splrep

from coordtrans import *

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout':True})

def generateSeedShells(drList, Ntheta, r_in, th_in, phi, field, outputHandler, filename):
    
    outputHandler.createSubDir(filename)

    theta_evals = np.linspace(0, 2*np.pi*(1 - 1/Ntheta), Ntheta)
    magCenterCoords = np.zeros((th_in.size, 2))

    r_delta = np.average(r_in)

    # shift origin of r, theta coordinates from geometric center to magnetic axis
    # then sort points on theta
    for i, theta, in enumerate(th_in):
        magCenterCoords[i] = axisShift(r_in[i], theta, r_delta)

    sortedMagCenter = magCenterCoords[np.argsort(magCenterCoords[:,0])]

    theta_spl = sortedMagCenter.T[0]
    rad_spl = sortedMagCenter.T[1]

    np.append(theta_spl, theta_spl[0])
    np.append(rad_spl, rad_spl[0])


    ## SPLINING
    # perform curve-fitting (smoothing spline)
    # function and derivative continuity not enforced across periodic boundary (would need fancier spline)
    fSurface_splineParms = splrep(theta_spl, rad_spl, s=1e-4, k=3, per=False, quiet=1)

    seedPts_0 = splev(theta_evals, fSurface_splineParms)
    derivs =  splev(theta_evals, fSurface_splineParms, der=1)


    ## Plotting the spline fit
    thetaPlot = np.linspace(0., 2*np.pi, 5000)
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    plt.scatter(th_in, r_in, s=1) # geo-axis points
    plt.plot(thetaPlot, splev(thetaPlot, fSurface_splineParms), '-k', linewidth=0.5) # fitted spline curve
    
    ax.set_rmax(field.a)
    ax.set_rticks(np.arange(0.0, 0.19, 0.02))
    ax.yaxis.set_tick_params(labelsize=5)
    ax.grid(linewidth = 0.25, linestyle=':', c='k')

    plt.title(f'Spline fit to Last Closed Flux Surface @ phi={phi*180/np.pi}')
    
    outputHandler.saveFig( 'LCFS_phi={:03.0f}_splineFit.png'.format(phi*180/np.pi) )
    plt.close()


    ## Calculating (and plotting) the seed points
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    plt.plot(thetaPlot, splev(thetaPlot, fSurface_splineParms), '-k', linewidth=1) # fitted spline curve

    output_ind_geo = np.zeros((Ntheta, 2))
    output_ind     = np.zeros((Ntheta, 3))
    output_ind_XYZ = np.zeros((Ntheta, 3))
    outData = []

    for dr in drList:
        for i, theta in enumerate(theta_evals):
            # scale delta-r to achieve uniform expansion normal to surface
            #adj_dr = dr
            adj_dr = np.sqrt(dr**2 + (dr*derivs[i])**2)
            seedPt = min(field.a - 1e-3, seedPts_0[i] + adj_dr)
            
            # shift back to geometric axis
            output_ind_geo[i] = axisShift(seedPt, theta, (-1)*r_delta)
            
            # convert rtp vector to xyz
            output_ind[i] = np.array([output_ind_geo[i][1], output_ind_geo[i][0], phi])
            output_ind_XYZ[i] = RTP_to_XYZ(output_ind[i], field.R0)
        
        plt.plot(theta_evals, output_ind_geo[:,1], '--o', linewidth=0.25, markersize=0.50)
        
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
