import numpy as np
import logging
from scipy.interpolate import make_smoothing_spline, spalde

from coordtrans import *

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout':True})

def generateSeedShells(drList, Ntheta, r_in, th_in, phi, field, outputHandler, filename):

    theta_evals = np.linspace(0, 2*np.pi*(1 - 1/Ntheta), Ntheta)
    magCenterCoords = np.zeros((th_in.size, 2))

    r_delta = np.average(r_in)
    #th_delta = np.average(th_in)

    # shift origin of r, theta coordinates from geometric center to magnetic axis
    for i, theta, in enumerate(th_in):
        magCenterCoords[i] = axisShift(r_in[i], theta, r_delta)
    
    ## SPLINING
    # sort points on theta, then perform curve-fitting (smoothing spline)
    # function and derivative continuity not enforced across periodic boundary (would need fancier spline)
    sortedMagCenter = magCenterCoords[np.argsort(magCenterCoords[:,0])]
    fSurface_spline = make_smoothing_spline(sortedMagCenter.T[0], sortedMagCenter.T[1], lam=1E-6)
    # seed Points on the LCFS
    seedPts_0 = np.array(fSurface_spline(theta_evals))

    # Spline parameters needed to find derivative
    fSurface_splineParms = (fSurface_spline.t, fSurface_spline.c, fSurface_spline.k)
    print(f'{fSurface_splineParms=}')

    ## Plotting the spline fit
    thetaPlot = np.linspace(0., 2*np.pi, 5000)
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    plt.scatter(th_in, r_in, s=1) # geo-axis points
    plt.plot(thetaPlot, fSurface_spline(thetaPlot), '-k', linewidth=0.25) # fitted spline curve
    ax.set_rmax(field.a)
    plt.title('Spline fit to Last Closed Flux Surface')
    outputHandler.saveFig( 'LCFS_phi={:03.0f}_splineFit.png'.format(phi*180/np.pi) )
    plt.close()


    # find the derivative at each theta value
    derivs = spalde(theta_evals, fSurface_splineParms)

    ## Calculating (and plotting) the seed points
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    # thetaPlot = np.linspace(0., 2*np.pi, 5000)
    plt.plot(thetaPlot, fSurface_spline(thetaPlot), '-k', linewidth=1) # fitted spline curve

    output_ind_geo = np.zeros((Ntheta, 2))
    output_ind     = np.zeros((Ntheta, 3))
    output_ind_XYZ = np.zeros((Ntheta, 3))
    outData = []

    for dr in drList:
        for i, theta in enumerate(theta_evals):
            # scale delta-r to achieve uniform expansion normal to surface
            adj_dr = dr * np.sqrt(1 + pow(derivs[i][0], 2))
            seedPt = seedPts_0[i] + adj_dr
            
            # shift back to geometric axis
            output_ind_geo[i] = axisShift(seedPt, theta, (-1)*r_delta)
            
            # convert rtp vector to xyz
            output_ind[i] = np.array([output_ind_geo[i][1], output_ind_geo[i][0], phi])
            output_ind_XYZ[i] = RTP_to_XYZ(output_ind[i], field.R0)
        
        plt.plot(theta_evals, output_ind_geo[:,1], '-o', linewidth=0.25, markersize=0.50)
        
        outData.extend(output_ind_XYZ)

    ax.set_rmax(field.a)
    plt.title(r'Generated Seed Points, $\phi$={:02.0f}$\degree$'.format(phi*180/np.pi))
    plot_name = 'SeedShellPoints_phi={:03.0f}.png'.format(phi*180/np.pi)
    outputHandler.saveFig(plot_name)
    
    outArray = np.asarray(outData)
    #print(outArray)
    #print(outArray.shape)

    #fname = 'SeedShell_test1'
    outputHandler.saveNumpyData(outArray, filename)
    outputHandler.log.info('Seed Shell Generated')
    return outArray
