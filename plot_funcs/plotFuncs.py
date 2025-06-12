import matplotlib.pyplot as plt
from matplotlib import patches, colors, cm, colormaps
import copy
import numpy as np
import logging
from utility.coordtrans import RTP_to_XYZ
#import class_outputHandler as out

## PORT PLOTTING CONVENIENCE FUNCTION
def plotPorts(ax_, simIO):
    #simIO = logging.getLogger()
    #ax_ = figure.add_subplot()

    # Import data on HIDRA port size/locations for plotting
    ports = simIO.loadPorts_fromCSV('input_files/HIDRA_ports.csv')
    for port in ports.T:
        port_plot = patches.Ellipse((port[0], port[1]), port[2], port[3],
                                    fill=True, alpha=0.2, facecolor='black', edgecolor='black', linewidth=0.0)
        ax_.add_patch(port_plot)


def plotWallHist(wallPtArray, runString, simIO):
    simIO.log.info('Plotting wall hits, total events = {}...'.format(wallPtArray[0].size))

    # extract theta and phi
    theta_plot = wallPtArray[1]
    # convert to phi= +CCW (as if viewing from outside the vaccum vessel)
    phi_plot = wallPtArray[2]*(-1) + 2*np.pi

    # shift theta domain to -180 to 180
    for i in range(len(theta_plot)):
        if theta_plot[i]>np.pi: theta_plot[i] -= 2*np.pi

    # convert to degrees
    # shift to physical phi=0 at at the South-side split
    a_phi = -18. # degrees, phi_comp is 18 CW from south-side split
    phi_plot_deg = (phi_plot*(180/np.pi) + 180. + a_phi) % 360.
    theta_plot_deg = theta_plot*(180/np.pi)

    # define bin edges for 2d histogram
    phi_edges = np.linspace(0, 360, 361)
    theta_edges = np.linspace(-180, 180, 181)

    ## CREATE HISTOGRAM
    H, phi_edges, theta_edges = np.histogram2d(phi_plot_deg, theta_plot_deg, bins=[phi_edges, theta_edges], density=True)
    H = H.T # histogram reverse axes for some reason; transpose

    ## PLOT HISTOGRAM
    plt.rcParams.update({'font.size': 8})
    plt.rcParams.update({'figure.autolayout':True})

    w, h = plt.figaspect(0.4)
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(polar=False, aspect=0.2)

    plt.grid(which='both', linewidth=0.25)
    plotPorts(ax, simIO)

    plt.imshow( H, interpolation='nearest', origin='lower',
                extent=[phi_edges[0], phi_edges[-1], theta_edges[0], theta_edges[-1]],
                cmap='Blues', norm=colors.LogNorm(vmin=1E-6, vmax=1E-3),
                aspect=0.2 )
    
    plt.colorbar(location='bottom', shrink=0.6)
    
    ax.set_xlabel('Toroidal Angle (+CCW from South-Side Split), $\phi[\degree]$')
    ax.set_xlim(0, 360)
    ax.set_xticks(np.linspace(9, 360, 40))
    ax.xaxis.set_tick_params(labelsize=6)

    ax.set_ylabel('Poloidal Location')
    ax.set_ylim(-180, 180)
    ax.set_yticks(np.linspace(-180, 180, 5))
    ax.set_yticklabels(['Inner   \nMidplane', 'Bottom', 'Outer   \nMidplane', 'Top', 'Inner   \nMidplane'])
    ax.yaxis.set_tick_params(labelsize=5)

    plotname = 'Wall_Histogram_' + runString + '.png'
    simIO.saveFig(plotname, dpi=200)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()


def plotWallPoints(phi_plot_deg, theta_plot_deg, color_data=None, colorRange=None, colorLabel=None, runString='default', simIO=None):
    #log = logging.getLogger()
    plt.rcParams.update({'font.size': 6})
    plt.rcParams.update({'figure.autolayout':True})
    fig = plt.figure()
    ax = fig.add_subplot(polar=False, aspect=0.2)
    plotPorts(ax, simIO)

    # plot wall event locations
    #plt.scatter(phi_plot_deg, theta_plot_deg, s=0.25, c='k', linewidths=0.0)
    # plot wall event locations
    if color_data is not None:
        if colorRange is None:
            sc = plt.scatter(phi_plot_deg, theta_plot_deg, linewidths=0.0, s=0.05, c=color_data, cmap='viridis', vmin=0., vmax=2*np.mean(color_data))
        else:
            sc = plt.scatter(phi_plot_deg, theta_plot_deg, linewidths=0.0, s=0.05, c=color_data, cmap='viridis', vmin=colorRange[0], vmax=colorRange[1])
        if colorLabel is None:    
            plt.colorbar(sc, ax=ax, label='Color Data', shrink=0.6)
        else:
            plt.colorbar(sc, ax=ax, label=colorLabel, shrink=0.6)

    else:
        plt.scatter(phi_plot_deg, theta_plot_deg, s=0.25, c='k', linewidths=0.0)

    ax.grid(linewidth = 0.25, linestyle=':', c='grey')

    ax.set_xlabel('Toroidal Angle, $\phi$, $[\degree]$')
    ax.set_xlim(0, 360)
    ax.set_xticks(np.linspace(9, 360, 40))
    ax.xaxis.set_tick_params(labelsize=3.5)

    ax.set_ylabel('Poloidal Location')
    ax.set_ylim(-180, 180)
    ax.set_yticks(np.linspace(-180, 180, 5))
    ax.set_yticklabels(['Inner Midplane', 'Bottom', 'Outer Midplane', 'Top', 'Inner Midplane'])
    ax.yaxis.set_tick_params(labelsize=5)

    ax.set_title('Distribution of Field Line Intersections with HIDRA Wall\n' + runString )

    plotname = 'Wallpoints_BorisPts_' + runString +  '.png'
    simIO.saveFig(plotname, dpi=700)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()


def plotWallPoints3D(phi_plot_deg, theta_plot_deg, b_hidra, runString, simIO):
    #log = logging.getLogger()
    #simIO.log.info('Attempting 3D plot...')
    ntheta = int(360*1 + 1)
    nphi = int(360*2 + 1)

    fig = plt.figure()
    #ax2 = fig.add_subplot(projection='3d', computed_zorder=True)
    ax2 = fig.add_subplot(projection='3d')
    ## PLOT VACUUM VESSEL TORUS
    ptheta = np.linspace(-np.pi, np.pi, ntheta)
    #pphi = np.linspace(0, 2.*np.pi, nphi)
    pphi = np.linspace(0, np.pi, int(np.ceil(nphi/2)) ) # only plot half the torus, since it is symmetric

    ptheta, pphi = np.meshgrid(ptheta, pphi)

    px = (b_hidra.R0 + b_hidra.a*np.cos(ptheta)) * np.cos(pphi)
    py = (b_hidra.R0 + b_hidra.a*np.cos(ptheta)) * np.sin(pphi)
    pz = b_hidra.a * np.sin(ptheta)

    ## CREATE HISTOGRAM 2
    phi_edges = np.linspace(0, 360, nphi)
    theta_edges = np.linspace(-180, 180, ntheta)
    H_2, phi_edges, theta_edges = np.histogram2d(phi_plot_deg, theta_plot_deg, bins=[phi_edges, theta_edges], density=True) #density=False

    ## Set up histogram output as colormap data
    color_dimension = H_2
    minn = 1E-6 #1E-8
    maxx = 1E-3
    norm = colors.LogNorm(vmin=minn, vmax=maxx)

    #my_cmap = copy.copy(colormaps['Blues'])
    my_cmap = copy.copy(colormaps['bone'])
    my_cmap.set_bad(my_cmap(0))

    m = plt.cm.ScalarMappable(norm=norm, cmap=my_cmap)
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)

    ## PLOT WALL POINTS 3D SURFACE WITH HISTOGRAM FACECOLORS
    ax2.plot_surface(px, py, pz, rstride=1, cstride=1,
                     vmin=minn, vmax=maxx,
                     facecolors=fcolors,
                     edgecolor='grey', linewidth=0.1,
                     alpha=1.0, shade=False)

    # set a camera projection with 24mm focal length
    my_fov = 85 # degrees
    focal_length = 1 / np.tan(np.radians(my_fov) / 2)
    ax2.set_proj_type('persp', focal_length=focal_length)

    ax2.set_xlim3d(0.52,  0.93)
    ax2.set_ylim3d(-0.03,  0.03)
    ax2.set_zlim3d(-0.20, 0.16)
    ax2.set_axis_off()
    ax2.elev = 1
    ax2.azim = -87
    #ax2.dist = -5
    plt.title('Distribution of Field Line Intersections with HIDRA Wall\n' + runString)

    plotname = 'WallHist3D_' + runString + '.png'
    simIO.saveFig(plotname)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    #plt.close()
    plt.show()

def plotInitEnergies(init_file, mass, runString='default', simIO=None):
    ## SOME PHYSICAL CONSTANTS
    kg_per_amu = 1.66054E-27
    kboltz = 1.602E-19 # Joules/eV

    init_conds = simIO.loadNumpyData(init_file)
    v0s = init_conds[:,0:3].T

    ## calculate initial energies in eV
    E0s = 0.5 * mass * kg_per_amu * (v0s[0]**2 + v0s[1]**2 + v0s[2]**2) / kboltz #eV

    ## create a 1d histogram of initial energies using numpy hist
    dist, bin_edges= np.histogram(E0s, bins=500, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    startfit_i = np.where(dist == np.max(dist))[0][0]
    stopfit_i = np.where(dist < 1)[0][0]
    simIO.log.info('Fitting initial energy distribution from index {} to {}.'.format(startfit_i, stopfit_i))

    lnE = np.log(dist)
    slope, intercept = np.polyfit(bin_centers[startfit_i:stopfit_i], lnE[startfit_i:stopfit_i], 1)

    Te_calc = -1/slope
    print(f'Calculated Ion Temperature: {Te_calc} eV')
    plt.figure()

    #plt.plot(bin_centers, lnE)
    #plt.plot(bin_centers[startfit_i:stopfit_i], fit, '--k', linewidth=3)
    plt.hist(E0s, bins=500, density=False)#histtype='step',
    plt.xlabel('Initial Energy (eV)')
    plt.ylabel('Number of Particles')
    plt.xlim(0, Te_calc*4) # limit x-axis to 5 times the calculated temperature
    #plt.yscale('log')
    plt.title('Initial Energy Distribution, $T_{{calc}}$ = {:.2f} eV'.format(Te_calc))

    plotname = 'E0_Dist_' + runString + '.png'
    simIO.saveFig(plotname, dpi=300)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()

def plotFinalEnergies(energy_array, mass, runString='default', simIO=None):
    Efs = energy_array

    ## create a 1d histogram of initial energies using numpy hist
    dist, bin_edges= np.histogram(Efs, bins=500, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    startfit_i = np.where(dist == np.max(dist))[0][0]
    stopfit_i = np.where(dist < 1)[0][0]

    lnE = np.log(dist)
    slope, intercept = np.polyfit(bin_centers[startfit_i:stopfit_i], lnE[startfit_i:stopfit_i], 1)

    Te_calc = -1/slope
    print(f'Calculated Ion Temperature: {Te_calc} eV')
    plt.figure()

    #plt.plot(bin_centers, lnE)
    #plt.plot(bin_centers[startfit_i:stopfit_i], fit, '--k', linewidth=3)
    plt.hist(Efs, bins=500, density=False)#histtype='step',
    plt.xlabel('Deposition Energy (eV)')
    plt.ylabel('Number of Particles')
    #plt.yscale('log')
    plt.xlim(0, Te_calc*4)
    plt.title('Final Energy Distribution, $T_{{calc}}$ = {:.2f} eV'.format(Te_calc))

    plotname = 'Ef_Dist_' + runString + '.png'
    simIO.saveFig(plotname, dpi=300)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()


def plotParticlesOverTime(maxN_array, tot_particles, tmax, dt, runString='default', simIO=None):
    # maxN_array is an array of maximum timestep for each particle. create a plot showing the number of particles running over time

    # Calculate the number of particles running over time
    time_steps = np.arange(0, tmax, dt)
    maxTime_array = maxN_array * dt
    particles_running = np.array([np.sum(maxTime_array > t) for t in time_steps])
    pct_running = 100 * particles_running / tot_particles
    # Plot the number of particles running over time
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, pct_running, label='Particles Running')
    plt.xlabel('Time (s)')
    plt.ylabel('Percent of Particles')
    plt.title('Particles Running Over Time')
    #plt.legend()
    plt.grid(True)

    plotname = 'IonsVtime_' + runString + '.png'
    simIO.saveFig(plotname, dpi=300)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    #plt.savefig(simIO.outputDir + '/ParticlesRunningOverTime_' + cond_string + TAG + '.png')
    plt.close()
