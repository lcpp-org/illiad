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
    phi_plot = wallPtArray[2]*(-1) + 2*np.pi

    # shift theta domain to -180 to 180
    for i in range(len(theta_plot)):
        if theta_plot[i]>np.pi: theta_plot[i] -= 2*np.pi

    # convert to degrees
    phi_plot_deg = (phi_plot*(180/np.pi) + 180. - 18.) % 360.
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
    
    ax.set_xlabel('Toroidal Angle, $\phi[\degree]$')
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


def plotWallPoints(phi_plot_deg, theta_plot_deg, runString, simIO):
    #log = logging.getLogger()
    plt.rcParams.update({'font.size': 6})
    plt.rcParams.update({'figure.autolayout':True})
    fig = plt.figure()
    ax = fig.add_subplot(polar=False, aspect=0.2)
    plotPorts(ax, simIO)

    # plot wall event locations
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
    simIO.saveFig(plotname)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()


def plotWallPoints3D(phi_plot_deg, theta_plot_deg, b_hidra, runString, simIO):
    #log = logging.getLogger()
    simIO.log.info('Attempting 3D plot...')

    fig = plt.figure()
    ax2 = fig.add_subplot(projection='3d', computed_zorder=True)
    ## PLOT VACUUM VESSEL TORUS
    ptheta = np.linspace(-np.pi, np.pi, 181)
    pphi = np.linspace(0, 2.*np.pi, 181)
    ptheta, pphi = np.meshgrid(ptheta, pphi)

    px = (b_hidra.R0 + b_hidra.a*np.cos(ptheta)) * np.cos(pphi)
    py = (b_hidra.R0 + b_hidra.a*np.cos(ptheta)) * np.sin(pphi)
    pz = b_hidra.a * np.sin(ptheta)

    ## CREATE HISTOGRAM 2
    phi_edges = np.linspace(0, 360, 181)
    theta_edges = np.linspace(-180, 180, 181)
    H_2, phi_edges, theta_edges = np.histogram2d(phi_plot_deg, theta_plot_deg, bins=[phi_edges, theta_edges], density=False)

    ## Set up histogram output as colormap data
    color_dimension = H_2
    minn = 1E-8
    maxx = 1E-3
    norm = colors.LogNorm()
    my_cmap = copy.copy(colormaps['Blues'])
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

    ax2.set_xlim3d(-1, 1)
    ax2.set_ylim3d(-1, 1)
    ax2.set_zlim3d(-0.7, 0.7)
    ax2._axis3don = False
    ax2.elev -= 12
    ax2.azim += 10
    plt.title('Distribution of Field Line Intersections with HIDRA Wall\n' + runString)

    plotname = 'WallHist3D_' + runString + '.png'
    simIO.saveFig(plotname)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()