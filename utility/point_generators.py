import numpy as np
import logging
from scipy.interpolate import splev, splrep
#from scipy.interpolate import make_smoothing_spline, spalde, splev, splrep
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout':True})

from utility.coordtrans import RTP_to_XYZ, XYZ_to_RTP, RTP_XYZ_JAC, axisShift, align_z_to_vector
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generateSeedShells(drList, Ntheta, r_in, th_in, phi, Bfield, outputHandler, filename, genNormals=False, Efield=None):
    outputHandler.createSubDir(filename)
    r_in = r_in[~np.isnan(r_in)]
    th_in = th_in[~np.isnan(th_in)]

    # hack solution, need to determine why an extra 30 copies of 1 initial condition are being appended to this event
    phi_deg = int(phi*180/np.pi)
    if phi_deg == 324:
        r_in = r_in[30:]
        th_in = th_in[30:]
    th_size = th_in.size

    # find the centroid(?) by average positions
    x_in = np.empty(th_size)
    y_in = np.empty(th_size)
    z_in = np.empty(th_size)
    for i, theta, in enumerate(th_in):
        x_in[i], y_in[i], z_in[i] = RTP_to_XYZ(np.array([r_in[i], theta, 0.]), Bfield.R0)

    x_avg = (np.max(x_in) + np.min(x_in))/2
    y_avg = 0
    z_avg = (np.max(z_in) + np.min(z_in))/2
    XYZ_delta = np.array([x_avg, y_avg, z_avg])

    # shift origin of r, theta coords from geo center to magnetic axis, sort pts on theta
    RTP_delta = XYZ_to_RTP(XYZ_delta, Bfield.R0)[1::-1]
    RTP_delta_rev = np.copy(RTP_delta)
    RTP_delta_rev[0] += np.pi
    magCenterCoords = np.empty((th_size, 2))
    for i, theta, in enumerate(th_in):
        magCenterCoords[i] = axisShift(theta, r_in[i], *RTP_delta)

    # Sort data in increasing theta
    sortedMagCenter = magCenterCoords[np.argsort(magCenterCoords[:,0])]
    theta_pts = sortedMagCenter.T[0]
    rad_pts = sortedMagCenter.T[1]

    # Append data to either end for pseudo-periodicity (smooth spline endpoints)
    append_length = int(th_size/2)
    th_A = np.copy(theta_pts[append_length:-1]) - 2*np.pi
    rad_A = np.copy(rad_pts[append_length:-1])
    th_B = np.copy(theta_pts[1:append_length]) + 2*np.pi
    rad_B = np.copy(rad_pts[1:append_length])

    theta_spl = np.concatenate((th_A, theta_pts, th_B))
    rad_spl = np.concatenate((rad_A, rad_pts, rad_B))

    ## SPLINING
    fSurface_splineParms = splrep(theta_spl, rad_spl, s=1e-4, k=3, per=False, quiet=1)
    theta_evals = np.linspace(0, 2*np.pi*(1 - 1/Ntheta), Ntheta)
    seedPts_0 = splev(theta_evals, fSurface_splineParms)
    derivs =  splev(theta_evals, fSurface_splineParms, der=1)

    ## PLOTTING THE SPLINE FIT
    thetaPlot = np.linspace(0., 2*np.pi, 5000)
    rPlot = splev(thetaPlot, fSurface_splineParms)
    geoCenterCoords = axisShift(thetaPlot, rPlot, *RTP_delta_rev)
    rPlotGeo = geoCenterCoords[1]
    thetaPlotGeo = geoCenterCoords[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    plt.scatter(th_in, r_in, s=1) # geo-axis points
    plt.scatter(theta_spl, rad_spl, s=1) # mag-axis points
    plt.plot(thetaPlotGeo, rPlotGeo, '-k', linewidth=0.5) # fitted spline curve
    ax.set_rmax(Bfield.a)
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
    plt.plot(thetaPlotGeo, rPlotGeo, '-k', linewidth=0.5) # fitted spline curve
    output_ind     = np.zeros((Ntheta, 3))
    output_ind_geo = np.zeros((Ntheta, 3))
    output_ind_XYZ = np.zeros((Ntheta, 3))
    output_ind_normal = np.zeros((Ntheta, 3))
    plot_norm_rtp = np.zeros((Ntheta, 3))
    outData = []
    outNormals = []
    tensor_ind_XYZ = torch.zeros((Ntheta, 3), dtype=torch.float32, device=device)

    for dr in drList:
        for i, theta in enumerate(theta_evals):
            # scale delta-r to achieve uniform expansion normal to surface
            adj_dr = dr * np.sqrt(1 + derivs[i]**2)
            seedPt = seedPts_0[i] + adj_dr
            
            # shift back to geometric axis
            output_ind_geo[i][:2] = axisShift(theta, seedPt, *RTP_delta_rev)
            output_ind_geo[i][1] = min(Bfield.a, output_ind_geo[i][1])
            output_ind_geo[i][2] = phi # keep phi constant for all points in this shell
            # convert rtp vector to xyz
            output_ind[i] = np.array([output_ind_geo[i][1], output_ind_geo[i][0], phi])
            output_ind_XYZ[i] = RTP_to_XYZ(output_ind[i], Bfield.R0)
            output_ind_XYZ[i] = RTP_to_XYZ(output_ind[i], Bfield.R0)
            tensor_ind_XYZ[i] = torch.tensor(output_ind_XYZ[i], dtype=torch.float32, device=device)

            # HERE WE CAN GENERATE UNIT VECTOR NORMALS
            if genNormals:
                output_ind_normal[i] = Efield.interpField(tensor_ind_XYZ[i], Cart=True).cpu().numpy()
                output_ind_normal[i] /= np.linalg.norm(output_ind_normal[i]) # normalize the vector

        plt.plot(output_ind_geo[:,0], output_ind_geo[:,1], '--o', linewidth=0.25, markersize=0.50)

        outData.extend(np.copy(output_ind_XYZ))
        if genNormals:
            outNormals.extend(np.copy(output_ind_normal))

    ## Plot the surface normals
    if genNormals:
        for i in range(len(output_ind_normal)):
            plot_norm_rtp[i] = RTP_XYZ_JAC(output_ind_geo[i], output_ind_normal[i], form='xyz2rtp')
            #norm_r, norm_theta, norm_phi = RTP_XYZ_JAC(output_ind_geo[i], output_ind_normal[i], form='xyz2rtp')
            #norm_r, norm_theta, norm_phi = XYZ_to_RTP(output_ind_XYZ[i], Bfield.R0)
            #ax.quiver(norm_theta, norm_r, output_ind_normal[i][0], output_ind_normal[i][1], color='red', scale=10, width=0.001)
        plt.quiver(output_ind_geo[:,0], output_ind_geo[:,1], plot_norm_rtp[:,0], plot_norm_rtp[:,1],  color='red', scale=20, width=0.002, angles='uv')

    ax.set_rmax(Bfield.a)
    ax.set_rticks(np.arange(0.0, 0.19, 0.02))
    ax.yaxis.set_tick_params(labelsize=5)
    ax.grid(linewidth = 0.25, linestyle=':', c='k')

    plt.title(r'Generated Seed Points, $\phi$={:02.0f}$\degree$'.format(phi*180/np.pi))
    plot_name = filename+'/'+'InitConds_phi={:03.0f}.png'.format(phi*180/np.pi)
    outputHandler.saveFig(plot_name)
    plt.close()

    outArray = np.asarray(outData)
    outputHandler.saveNumpyData(outArray, filename)

    if genNormals:
        outNormalsArray = np.asarray(outNormals)
        return outArray, outNormalsArray
    else:
        return outArray

def generate_MB_velocities(N_particles, normals_list, ion_temp, ion_mass, nparticles_per_emitter=1):
    """
    Generates initial velocities for particles following a Maxwell-Boltzmann energy distribution.

    The function samples random directions uniformly over a hemisphere and scales the velocities
    according to the specified ion temperature. The resulting velocity vectors are rotated to align
    with the provided surface normals.

    Args:
        N_particles (int): Number of particles to generate velocities for.
        normals_list (array-like): List or array of normal vectors to align velocities with.
        ion_temp (float): Ion temperature in eV.
        ion_mass (float): Ion mass in atomic mass units (amu).
        nparticles_per_emitter (int, optional): Number of particles per emitter. Defaults to 1.

    Returns:
        np.ndarray: Array of shape (N_particles, 3) containing the generated velocity vectors.
    """
    kg_per_amu = 1.660_539_068E-27
    kboltz = 1.602_176_634E-19 # Joules/eV
    r = np.random.uniform(0, 1, N_particles)

    z = np.sqrt(1 - r**2)
    phi = np.random.uniform(0, 2 * np.pi, N_particles)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    velocity_array = np.stack([x, y, z], axis=1) # shape (N, 3)

    # GENERATE NORMAL DISTRIBUTION OF SPEEDS
    # Calculate the root mean square velocities
    v_rms1d = np.sqrt( kboltz*ion_temp / (ion_mass*kg_per_amu) )
    initSpeeds = v_rms1d * np.sqrt(np.random.chisquare(df=3, size=N_particles))

    # APPLY INIT SPEEDS TO THE RANDOM UNIT VECTORS
    velocity_array *= initSpeeds[:, None]

    # ROTATE TO ALIGN POLE WITH NORMAL VECTOR
    for i,normal in enumerate(normals_list):
        Rotater = align_z_to_vector(normal)
        velocity_array[::nparticles_per_emitter] = velocity_array[::nparticles_per_emitter] @ Rotater.T

    return velocity_array