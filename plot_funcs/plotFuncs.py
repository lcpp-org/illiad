import gc

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches, colors, cm, colormaps
plt.rcParams['animation.ffmpeg_path'] = '/home/sgula/miniforge/envs/testenv/bin/ffmpeg'
import copy
import numpy as np
import logging
from PIL import Image
from tqdm import tqdm
from illiad.utilities.coordtrans import XYZ_to_RTP2

# UIUC branding color palette
UIUC = {
    'il_blue': '#13294B',
    'il_orange': '#FF5F05',
    'il_storm': '#707372',
    'il_stormdark1': '#4A4C4B',
    'il_stormdark2': '#252525',
    'il_stormlight1': '#8D8F8E',
    }

## PORT PLOTTING CONVENIENCE FUNCTION
def global_plotPorts(ax_, simIO):
    """Plots the ports on the given axis."""
    # Import data on HIDRA port size/locations for plotting
    ports = simIO.loadPorts_fromCSV('input_files/HIDRA_ports.csv')
    for port in ports.T:
        port_plot = patches.Ellipse((port[0], port[1]), port[2], port[3],
                                    fill=True, alpha=0.2, facecolor='black', edgecolor='black', linewidth=0.0)
        ax_.add_patch(port_plot)

#def boris_plotWallHist(wallPtArray, runString, simIO):
def boris_plotWallHist(wallPtArray, runString, simIO, cond_string):
    """ Plots a histogram of wall intersection points from the simulation."""
    simIO.log.info('Plotting wall hits, total events = {}...'.format(wallPtArray[0].size))

    # cond string decoder
    parts = cond_string.split('_')
    dr_mm = parts[0]
    LCFS_index = parts[1][4:]  # Remove 'LCFS' prefix
    ion_temp_eV = parts[2][:-2]  # Remove 'eV' suffix
    electric_field_V = parts[3][:-1]  # Remove 'V' suffix
    charge_num_Z = parts[4][1:]  # Remove 'Z' prefix

    phi_plot = wallPtArray[2]*(-1) + 2*np.pi # convert to phi= +CCW (viewing from outside VV)
    theta_plot = wallPtArray[1]
    theta_plot[theta_plot>np.pi] -= 2*np.pi #shift so that (theta=0) is centered in the plot

    # shift to physical phi=0 at at the South-side split, convert to deg.
    a_phi = 18. # (deg), phi_comp 18 CW from south-split
    phi_plot_deg = (phi_plot*(180/np.pi) + 180. + a_phi) % 360.
    theta_plot_deg = theta_plot*(180/np.pi)

    # define bin edges for 2d histogram
    phi_edges = np.linspace(0, 360, 361)
    theta_edges = np.linspace(-180, 180, 181)
    H, phi_edges, theta_edges = np.histogram2d(phi_plot_deg, theta_plot_deg, bins=[phi_edges, theta_edges], density=True)
    H = H.T # histogram reverse axes for some reason; transpose

    ## PLOT HISTOGRAM
    plt.rcParams.update({'font.size': 8})
    w, h = plt.figaspect(0.40)
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(polar=False, aspect=0.2)

    plt.grid(which='both', linewidth=0.5)
    global_plotPorts(ax, simIO)

    plt.imshow( H, interpolation='nearest', origin='lower',
                extent=[phi_edges[0], phi_edges[-1], theta_edges[0], theta_edges[-1]],
                cmap=plt.get_cmap('Blues', 6), norm=colors.LogNorm(vmin=1E-6, vmax=1E-3),
                aspect=0.2 )
 
    #cbar = plt.colorbar(boundaries=levels, location='top', shrink=0.6)
    cbar = plt.colorbar(location='top', shrink=0.6)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('$\\hat{\\Gamma}_{depo}=\\frac{N_{depo}}{N_{total}}$', fontsize=12)
 
    ax.set_xlabel('$\phi~\\mathit{(\\degree CCW~from~South\\text{-}Split)}$', fontsize=14)
    ax.set_ylabel('Poloidal Location', fontsize=14)   
    ax.set_xlim(0, 360)
    ax.set_ylim(-180, 180)

    phi_spacing = 18. # degrees
    xticks = np.arange(phi_spacing, 361-phi_spacing, phi_spacing) 
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{int(tick)}$\degree$' if i % 2 != 0 else '' for i, tick in enumerate(xticks)])
    ax.xaxis.set_tick_params(labelsize=12)

    ax.set_yticks(np.linspace(-180, 180, 5))
    ax.set_yticklabels(['', 'Bottom', 'Outer', 'Top', ''])
    ax.yaxis.set_tick_params(labelsize=12, labelrotation=0)

    #ax.text(0.995, 0.975, f'$\\mathrm{{{ion_temp_eV}eV, {electric_field_V}V, Z{charge_num_Z}}}$',
    #ax.text(0.9955, 0.9755, f'$\\mathbf{{ T_i = {ion_temp_eV}eV}}$',
    ax.text(0.9945, 0.974, f'$\\mathbf{{ T_i = {ion_temp_eV}eV}}$',
    transform=ax.transAxes,
    ha='right', va='top',
    fontsize=14,
    bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.9))



    plt.tight_layout()
    plotname = 'Wall_Histogram.png'
    simIO.saveFig(plotname, dpi=600)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()

def boris_plotWallPoints(phi_plot_deg, theta_plot_deg, color_data=None, colorRange=None, colorLabel=None, runString='default', simIO=None):
    """Plots the discrete wall intersection points from the simulation."""
    #log = logging.getLogger()
    plt.rcParams.update({'font.size': 6})
    #plt.rcParams.update({'figure.autolayout':True})
    fig = plt.figure()
    ax = fig.add_subplot(polar=False, aspect=0.2)
    global_plotPorts(ax, simIO)

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
    plt.tight_layout()
    plotname = 'Wallpoints_BorisPts.png'
    simIO.saveFig(plotname, dpi=700)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()

def boris_plotWallPoints3D(phi_plot_deg, theta_plot_deg, b_hidra, runString, simIO):
    """Plots the discrete wall intersection points in 3D."""
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
    minn = 2.5E-7 #3E-7 #6 #1E-8
    maxx = 3E-3
    norm = colors.LogNorm(vmin=minn, vmax=maxx)

    #my_cmap = copy.copy(colormaps['Blues'])
    #my_cmap = copy.copy(colormaps['bone'])
    my_cmap = copy.copy(plt.get_cmap('Greys_r'))
    my_cmap.set_bad(my_cmap(39)) #33

    m = plt.cm.ScalarMappable(norm=norm, cmap=my_cmap)
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)

    ## PLOT WALL POINTS 3D SURFACE WITH HISTOGRAM FACECOLORS
    ax2.plot_surface(px, py, pz, rstride=1, cstride=1,
                     vmin=minn, vmax=maxx,
                     facecolors=fcolors,
                     edgecolor='grey', linewidth=0.05,
                     alpha=1.0, shade=False)

    # set a camera projection with prescribed FOV
    my_fov = 85 # degrees
    focal_length = 1 / np.tan(np.radians(my_fov) / 2)
    ax2.set_proj_type('persp', focal_length=focal_length)

    #ax2.set_xlim3d(0.53,  0.94)
    ax2.set_xlim3d(-0.94, -0.53)
    ax2.set_ylim3d(-0.03,  0.03)
    ax2.set_zlim3d(-0.20, 0.16)
    ax2.set_axis_off()
    ax2.elev = 2 #2
    ax2.azim = -94
    #plt.tight_layout()
    plt.title('Distribution of Field Line Intersections with HIDRA Wall\n' + runString)

    plotname = 'WallHist3D.png'
    simIO.saveFig(plotname, dpi=600)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()
    #plt.show()

def boris_plotInitEnergies(init_file, mass, runString='default', simIO=None):
    """Plots the initial energy distribution of particles to validate Maxwellian profile and ion temperature."""
    ## SOME PHYSICAL CONSTANTS
    kg_per_amu = 1.66054E-27
    kboltz = 1.602E-19 # Joules/eV

    # if simIO:
    #     init_conds = simIO.loadNumpyData(init_file)
    # else:
    init_conds = simIO.loadNumpyData(init_file)

    # extract initial velocities, calculate initial energies in eV
    v0s = init_conds[:,0:3].T
    E0s = 0.5 * mass * kg_per_amu * (v0s[0]**2 + v0s[1]**2 + v0s[2]**2) / kboltz #eV

    ## create a 1d histogram of initial energies using numpy hist
    counts, bin_edges= np.histogram(E0s, bins=500, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    lnE = np.log(counts[counts > 0])  # take log of only positive values to avoid log(0)
    startfit_i = np.where(counts == np.max(counts))[0][0] + 100 # start fitting 100 bins after the maximum for a better slope fit
    stopfit_i = np.where(counts < 1)[0][0]
    if stopfit_i > startfit_i:
        slope, intercept = np.polyfit(bin_centers[startfit_i:stopfit_i], lnE[startfit_i:stopfit_i], 1)
    else:
        slope, intercept = np.polyfit(bin_centers, lnE, 1)
    Te_calc = -1/slope

    plt.figure()
    plt.grid(which='both', zorder=0)
    plt.hist(E0s, bins=500, density=False, color=UIUC['il_orange'], edgecolor=UIUC['il_blue'], zorder=2)  # histtype='step',
    plt.xlabel('Initial Energy (eV)', fontsize=12)
    plt.ylabel('# of Particles', fontsize=12)
    plt.xlim(0, min(Te_calc*6,5000)) # limit x-axis to 6 times the calculated temperature
    plt.xticks(fontsize=12)  # Increase x-tick label size
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'))
    plt.yticks(fontsize=12)
    #plt.yscale('log')

    plt.title('Initial Energy Distribution, $T_{{est}}$ = {:.2f} eV'.format(Te_calc))
    plt.tight_layout()

    plotname = 'E0_Dist.png'
    simIO.saveFig(plotname, dpi=300)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()

def boris_plotFinalEnergies(energy_array, mass, runString='default', simIO=None):
    """Plots the final energy distribution of particles."""
    ## create a 1d histogram of initial energies using numpy hist
    counts, bin_edges= np.histogram(energy_array, bins=500, range=(0., 4000.), density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    startfit_i = np.where(counts == np.max(counts))[0][0]
    # find the first index where counts is less than 1
     # if there are no values less than 1, we stop at the end of the distribution    
    if np.any(counts < 1):
        stopfit_i = np.where(counts < 1)[0][0]
    else:
        stopfit_i = len(counts) - 1

    # take log of only positive values to avoid log(0)
    lnE = np.log(counts, out=np.zeros_like(counts, dtype=np.float64), where=(counts > 0))  

    if stopfit_i > startfit_i:
        slope, intercept = np.polyfit(bin_centers[startfit_i:stopfit_i], lnE[startfit_i:stopfit_i], 1)
    else:
        slope, intercept = np.polyfit(bin_centers, lnE, 1)

    Te_calc = -1/slope

    plt.figure()
    plt.grid(which='both', zorder=0)
    plt.hist(energy_array, bins=500, range=(0., 4000.), density=False, color=UIUC['il_orange'], edgecolor=UIUC['il_blue'], linewidth=0.3, zorder=2)
    plt.xlabel('Deposition Energy (eV)')
    plt.ylabel('Number of Particles')
    if not np.isnan(Te_calc) and not np.isinf(Te_calc):
        plt.xlim(0, Te_calc*4)
    else:
        plt.xlim(0, 4000)
    #plt.yscale('log')

    plt.title('Final Energy Distribution, $T_{{est}}$ = {:.2f} eV'.format(Te_calc))
    plt.tight_layout()

    plotname = 'Ef_Dist.png'
    simIO.saveFig(plotname, dpi=300)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()

def boris_plotDepoAngles(angle_array, runString='default', simIO=None):
    """Plots the distribution of deposition angles from the simulation."""
    plt.figure()
    plt.grid(which='both', zorder=0)
    plt.hist(angle_array, bins=90, density=False, color=UIUC['il_blue'], edgecolor=UIUC['il_orange'], linewidth=0.3, zorder=2)

    plt.xlabel('Deposition Angle (degrees from normal)')
    plt.ylabel('Number of Particles')
    plt.xlim(0, 90)

    plt.title('Ion Angle Distribution (degrees from normal)')
    plt.tight_layout()

    plotname = 'Angle_Dist.png'
    simIO.saveFig(plotname, dpi=300)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()

def boris_plotCombined(phi_plot_deg, theta_plot_deg, data, colorRange=None, colorLabel=None, myColormap='viridis', runString='default', simIO=None, cond_string=None):
    """Plots the combined 2D histogram and 1D statistical distribution of the given data."""
    plt.rcParams.update({'font.size': 6})
    #plt.rcParams.update({'figure.autolayout':True})

    # cond string decoder
    parts = cond_string.split('_')
    dr_mm = parts[0]
    LCFS_index = parts[1][4:]  # Remove 'LCFS' prefix
    ion_temp_eV = parts[2][:-2]  # Remove 'eV' suffix
    electric_field_V = parts[3][:-1]  # Remove 'V' suffix
    charge_num_Z = parts[4][1:]  # Remove 'Z' prefix

    height = 0.75

    width_left = 5/6 * height #fig_height / aspect_left
    width_right = 1/6 * height  # or choose a different one
    h_buffer = 0.01  # horizontal buffer between left and right plots

    # Total width for the figure
    total_width = width_left + width_right + h_buffer
    left_start = (1 - total_width) / 2
    bottom_start = (1 - total_width) * 2 / 3
    right_start = left_start + width_left + h_buffer

    fig = plt.figure(figsize=(24, 4))
    #fig = plt.figure(figsize=(20, 4))

    axWall = fig.add_axes([left_start, bottom_start, width_left, height])
    axWall.set_aspect(0.2)  # height/width

    axDist = fig.add_axes([right_start, bottom_start, width_right, height])
    global_plotPorts(axWall, simIO)

    if colorRange is None:
        colorRange = np.array([0, 3*np.mean(data)])  # default color range if not provided
        axDist.set_xlim(colorRange)
    else:
        axDist.set_xlim(colorRange)
        nxticks = 5
        lower_xtick = colorRange[0] + (colorRange[1] - colorRange[0]) / (nxticks+1)
        upper_xtick = colorRange[1] - (colorRange[1] - colorRange[0]) / (nxticks+1)
        axDist.set_xticks(np.linspace(lower_xtick, upper_xtick, nxticks))

    sc = axWall.scatter(phi_plot_deg, theta_plot_deg, linewidths=0.0, s=0.05, c=data, cmap=myColormap, vmin=colorRange[0], vmax=colorRange[1])
    norm = colors.Normalize(vmin=colorRange[0], vmax=colorRange[1])


    axWall.grid(linewidth = 0.5)#, linestyle=':', c='grey')

    #axWall.set_xlabel('$\phi$ (+CCW from South-Side Split)', fontsize=12)
    axWall.set_xlabel('$\phi~\\mathit{(\\degree CCW~from~South\\text{-}Split)}$', fontsize=20)
    axWall.set_xlim(0, 360)
    phi_spacing = 18. # degrees
    xticks = np.arange(phi_spacing, 361-phi_spacing, phi_spacing) 
    axWall.set_xticks(xticks)
    #axWall.set_xticklabels([f'{int(tick)}' if i % 2 != 0 else '' for i, tick in enumerate(xticks)])
    axWall.set_xticklabels([f'{int(tick)}$\degree$' if i % 2 != 0 else '' for i, tick in enumerate(xticks)])
    axWall.xaxis.set_tick_params(labelsize=16)

    axWall.set_ylabel('Poloidal Location', fontsize=18)
    axWall.set_ylim(-180, 180)
    axWall.set_yticks(np.linspace(-180, 180, 5))
    axWall.set_yticklabels(['', 'Bottom', 'Outer', 'Top', ''])
    #axWall.yaxis.set_tick_params(labelsize=8, labelrotation=45)
    axWall.yaxis.set_tick_params(labelsize=16, labelrotation=0)

    axWall.text(0.9945, 0.974, f'$\\mathbf{{ T_i = {ion_temp_eV}eV}}$',
    transform=axWall.transAxes,
    ha='right', va='top',
    fontsize=18,
    bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.9))

    n, bins, patches = axDist.hist(data, bins=90, range=colorRange, density=False, linewidth=0.3, zorder=2)
    # Color each bar
    cmap = cm.get_cmap(myColormap)
    for bin_left, patch in zip(bins[:-1], patches):
        color = cmap(norm(bin_left))  # or use bin center: (bin_left + bin_right)/2
        patch.set_facecolor(color)

    axDist.grid(which='both', zorder=0)
    axDist.set_xlabel(colorLabel, fontsize=20)
    axDist.xaxis.set_tick_params(labelsize=16)
    axDist.set_yticklabels([])
    axDist.yaxis.set_tick_params(color='white')

    # Remove all padding between subplots
    #plt.subplots_adjust(wspace=0, hspace=0)
    plotname = 'WallPts_' + runString +  '.png'
    simIO.saveFig(plotname, dpi=400)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()

def boris_plotParticlesOverTime(maxN_array, tot_particles, tmax, dt, runString='default', simIO=None):
    """Plots the percent of particles running over time."""
    # maxN_array is an array of maximum timestep for each particle. 
    # create a plot showing the number of particles running over time

    # Calculate the number of particles running over time (efficiently)
    time_steps = np.arange(0, tmax, dt)
    time_ms = time_steps * 1000
    sorted_maxTime = np.sort(maxN_array) * dt

    # Use searchsorted to find how many particles have maxTime > t for each t
    particles_running = len(sorted_maxTime) - np.searchsorted(sorted_maxTime, time_steps, side='right')
    frac_running = particles_running / tot_particles
    frac_running += 1 - frac_running[0]  # adjust so that it starts at 100% at t=0

    unnormalized_frac_running = particles_running / tot_particles
    frac_running = unnormalized_frac_running + 1 - unnormalized_frac_running[0]  # adjust so that it starts at 100% at t=0


    # Estimate residence time using trapezoidal integration of the fraction running over time
    tau_res = np.trapz(frac_running, dx=dt)
    if frac_running[-1] > 0:
        slope = (np.log(frac_running[-1]) - np.log(frac_running[-101])) / 100 / dt # use wider range for slope to reduce noise
        slope = min(slope, -1e-1)  # prevent division by zero or very small slope
        tau_res_corr = -frac_running[-1] / slope
    else:
        tau_res_corr = 0.0

    tau_res_est = tau_res + tau_res_corr
    frac_at_tau_res = frac_running[np.argmin(np.abs(time_steps - tau_res_est))]

    # Plot the number of particles running over time
    plt.figure(figsize=(8, 5))

    plt.plot(time_ms, frac_running, color=UIUC['il_blue'], label='Particles Running', linewidth=1.5)

    plt.fill_between(time_ms, frac_running, color=UIUC['il_blue'], alpha=0.3)

    plt.plot(tau_res_est*1000, frac_at_tau_res, 'ro', label='Estimated Residence Time Point')

    plt.axvline(tau_res_est*1000, 0.0, frac_at_tau_res,
                 color='k', linestyle='--', label='Estimated Residence Time', zorder=0)

    plt.xlabel('$t~[ms]$', fontsize=12)
    plt.ylabel('$\\dfrac{N_{active}}{N_{total}}$', fontsize=12, rotation=0, labelpad=22)
    plt.title('Estimated Residence Time = {:.3f}$\\mathit{{(+{:.3f}ms~correction) }}$'
              .format(tau_res_corr*1000, tau_res_corr*1000), fontsize=12)
    
    plt.xlim(0, time_ms[-1])    
    plt.ylim(0, 1.05)
    plt.xticks(np.arange(0, time_ms[-1]+0.1, 0.1), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(which='both')#, linestyle=':', linewidth=0.5)
    #plt.tight_layout()

    plotname = 'IonsVtime.png'
    simIO.saveFig(plotname, dpi=300)
    simIO.log.info('OUTPUT PLOT: {}, residence time = {:.3f}ms, corr = {:.3f}ms, slope = {:.3f}'
                   .format(plotname, tau_res*1000, tau_res_corr*1000, slope))
    plt.close()

def boris_plotCombined_Hist(wallPtArray, maxN_array, tot_particles, tmax, dt, runString, simIO):
    """Plots a combined histogram of wall intersection points and the percent of particles running over time."""
    simIO.log.info('Plotting Combined Histogram...')

    ## CREATE HISTOGRAM
    # extract theta and phi
    theta_plot = wallPtArray[1]
    # convert to phi= +CCW (as if viewing from outside the vaccum vessel)
    phi_plot = wallPtArray[2]*(-1) + 2*np.pi

    # shift theta domain to -180 to 180
    # for i in range(len(theta_plot)):
    #     if theta_plot[i]>np.pi: theta_plot[i] -= 2*np.pi
    theta_plot[theta_plot>np.pi] -= 2*np.pi #shift so that (theta=0) is centered in the plot

    # convert to degrees
    # shift to physical phi=0 at at the South-side split
    a_phi = 18 #positive for consitency w/ histogram func! -18. # degrees, phi_comp is 18 CW from south-side split
    phi_plot_deg = (phi_plot*(180/np.pi) + 180. + a_phi) % 360.
    theta_plot_deg = theta_plot*(180/np.pi)

    # define bin edges for 2d histogram
    phi_edges = np.linspace(0, 360, 361)
    theta_edges = np.linspace(-180, 180, 181)

    H, phi_edges, theta_edges = np.histogram2d(phi_plot_deg, theta_plot_deg, bins=[phi_edges, theta_edges], density=True)
    H = H.T # histogram reverse axes for some reason; transpose


    ## CREATE %PARTICLES V TIME
    # calculate the number of particles running over time (efficiently)
    time_steps = np.arange(0, tmax, dt)
    maxTime_array = maxN_array * dt
    # sort maxTime_array once
    sorted_maxTime = np.sort(maxTime_array)
    # use searchsorted to find how many particles have maxTime > t for each t
    particles_running = len(maxTime_array) - np.searchsorted(sorted_maxTime, time_steps, side='right')
    pct_running = 100 * particles_running / tot_particles
    pct_running += 100 - pct_running[0]


    ## PLOT HISTOGRAM
    plt.rcParams.update({'font.size': 8})
    #plt.rcParams.update({'figure.autolayout':True})

    tot_scale = 0.8
    width_left = 5/6 * tot_scale #fig_height / aspect_left
    width_right = 1/6 * tot_scale  # or choose a different one
    h_buffer = 0.01  # horizontal buffer between left and right plots

    # Total width for the figure
    total_width = (width_left + width_right)
    left_start = (1 - total_width) / 2
    bottom_start = (1 - total_width) * 2 / 3
    right_start = left_start + width_left + h_buffer

    fig = plt.figure(figsize=(24, 4))
    axWall = fig.add_axes([left_start, bottom_start, width_left, tot_scale])
    axWall.set_aspect(0.2)  # height/width
    global_plotPorts(axWall, simIO)
    # axRight = fig.add_axes([right_start, bottom_start, width_right, tot_scale])

    axWall.imshow( H, interpolation='nearest', origin='lower',
                extent=[phi_edges[0], phi_edges[-1], theta_edges[0], theta_edges[-1]],
                cmap='Blues', norm=colors.LogNorm(vmin=1E-6, vmax=1E-3),
                aspect=0.2 )
    
    # axWall.colorbar(location='bottom', shrink=0.6)
    axWall.grid(linewidth = 0.25, linestyle=':', c='grey')
    axWall.set_xlabel('Toroidal Angle, $\phi$, $[\degree]$', fontsize=16)
    axWall.set_xlim(0, 360)
    xticks = np.linspace(9, 351, 39)
    axWall.set_xticks(xticks)
    axWall.set_xticklabels([f'{int(tick)}' if i % 2 == 0 else '' for i, tick in enumerate(xticks)])
    axWall.xaxis.set_tick_params(labelsize=14)

    axWall.set_ylabel('Poloidal Location', fontsize=14)
    axWall.set_ylim(-180, 180)
    axWall.set_yticks(np.linspace(-90, 90, 3))
    axWall.set_yticklabels(['Bottom', 'Outer\nMidplane', 'Top'])
    axWall.yaxis.set_tick_params(labelsize=12)


    ## PLOT %PARTICLES V TIME
    axRight = fig.add_axes([right_start, bottom_start, width_right, tot_scale])
    # Convert time_steps from seconds to milliseconds for plotting
    axRight.plot(time_steps*1000, pct_running, 'k')#, label='Particles Running')
    #axRight.plot(time_steps, pct_running, 'k')#, label='Particles Running')

    axRight.set_xlabel('Simulation Time (ms)', fontsize=12)
    ## Set major ticks every 0.0001 and minor ticks every 0.00005 on the x-axis
    axRight.set_xticks(np.linspace(0.0, 1.0, 11))
    #axRight.set_xticks(np.linspace(0.0, 0.001, 11))
    axRight.set_xlim(0, 1.0)
    axRight.xaxis.set_tick_params(labelsize=12)

    axRight.set_ylabel('% of Particles Running', fontsize=12)

    axRight.set_yticks(np.linspace(10, 100, 10))
    axRight.yaxis.set_label_position("right")
    axRight.yaxis.tick_right()
    axRight.set_ylim(0, 100)
    #axRight.yaxis.set_tick_params(color='white')

    # Set minor tick gridlines to be dashed and smaller width
    axRight.grid(which='minor', linestyle=':', linewidth=0.5)
    axRight.grid(which='major', linestyle='-', linewidth=1)

    plotname = 'CombinedHistogram.png'
    simIO.saveFig(plotname, dpi=400)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()

def boris_plotTraces(ion_traces, b_hidra, runString='default', simIO=None):
    """Plots the ion traces in 3D."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Ion Traces')

    ## PLOT VACUUM VESSEL BOTTOM-HALF TORUS
    nphi = ntheta = 180
    ptheta = np.linspace(-np.pi, 0, int(np.ceil(ntheta/2)) )
    pphi = np.linspace(0, 2.*np.pi, nphi)
    ptheta, pphi = np.meshgrid(ptheta, pphi)
    px = (b_hidra.R0 + b_hidra.a*np.cos(ptheta)) * np.cos(pphi)
    py = (b_hidra.R0 + b_hidra.a*np.cos(ptheta)) * np.sin(pphi)
    pz = b_hidra.a * np.sin(ptheta)
    ax.plot_surface(px, py, pz, rstride=9, cstride=9,
                     facecolor='lightgrey',
                     edgecolor='k', linewidth=0.1,
                     alpha=1.0, shade=True, zorder=1)

    for i in range(ion_traces.shape[1]):  # Loop over particles
        this_ion = ion_traces[:, i, :]
        # filter rows containing all zeros
        this_ion = this_ion[~np.all(this_ion == 0, axis=1)]
    
        this_X = this_ion[:,0] #ion_traces[:, i, 0]
        this_Y = this_ion[:,1] #ion_traces[:, i, 1]
        this_Z = this_ion[:,2] #ion_traces[:, i, 2]
        #ax.plot(ion_traces[:, i, 0], ion_traces[:, i, 1], ion_traces[:, i, 2])

        skip_indices = [0,1,3,5,6,7,8,10]
        if i not in skip_indices:
            ax.plot(this_X, this_Y, this_Z, linewidth=0.5, zorder=5)

    ax.set_xlim([-0.61, 0.61])#[-1, 1])
    ax.set_ylim([-0.61, 0.61])#[-1, 1])
    ax.set_zlim([-0.61, 0.61])#[-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_axis_off()  # Remove bounding box and grid
    plt.tight_layout()

    plotname = 'IonTraces.png'
    simIO.saveFig(plotname, dpi=600)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()

def boris_plotTracesPoincare(ion_traces, b_hidra, runString='default', simIO=None):
    """Plots the ion traces in polar coordinates (Poincare plot)."""
    print('ion_traces shape:', ion_traces.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.set_title('Ion Traces')

    for i in range(ion_traces.shape[1]):  # Loop over particles
        this_ion = ion_traces[:, i, :]
        # filter rows containing all zeros
        this_ion = this_ion[~np.all(this_ion == 0, axis=1)]
    
        this_ion_rtp = XYZ_to_RTP2(this_ion, b_hidra.R0).cpu().numpy()
        this_r = this_ion_rtp[:,0] #ion_traces[:, i, 0]
        this_theta = this_ion_rtp[:,1] #ion_traces[:, i, 1]

        skip_indices = [1] #[0,1,3,5,6,7,8,10]
        if i not in skip_indices:
            ax.plot(this_theta, this_r, linewidth=0.5, zorder=5)

    ax.set_rlim([0., b_hidra.a])#[-1, 1])
    ax.set_rgrids([0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175],
                labels=['', '', '', '', '', '', ''], angle=0, fontsize=4)
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                #labels=['Low\nField', '', '', '', 'High\nField', '', '', ''], fontsize=12)
                labels=['', '', '', '', '', '', '', ''], fontsize=12)
    
    plotname = 'IonTracesPoin.png'
    simIO.saveFig(plotname, dpi=600)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()
    #plt.show()

def poincare_plotPoincareBW(radtheta_pts, point_total, phi_deg, b_hidra, analysis_name='default', simIO=None, plot_args=None):
    """Plots a black and white Poincare plot of the magnetic field lines."""

    if plot_args:
        title_on = plot_args['title_on']
        dpi = plot_args['dpi']
    else:        
        title_on = True
        dpi = 400
    
    rho_max = b_hidra.a
    num_sets = len(radtheta_pts)

    plt.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    for i in range(num_sets):
        plt.scatter(radtheta_pts[i][0][:point_total[i]], radtheta_pts[i][1][:point_total[i]],
                     marker='.', s=1.00, c='k', linewidths=0.0)
        # #check if more efficient for marge datasets!?
        # plt.plot(radtheta_pts[i][0][:point_total[i]], radtheta_pts[i][1][:point_total[i]],
        #      marker='.', markersize=1.00, color='k', linestyle='none')
    ax.set_rmax(rho_max)
    # ax.set_rticks(np.arange(0.0, rho_max, 0.02))
    # ax.yaxis.set_tick_params(labelsize=5)
    ax.grid(linewidth = 0.25, linestyle=':', c='k')
 
    ax.set_rgrids([0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175],
                labels=['', '', '', '', '', '', ''], angle=0, fontsize=4)

    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                    #labels=['Low\nField', '', '', '', 'High\nField', '', '', ''], fontsize=12)
                    labels=['', '', '', '', '', '', '', ''], fontsize=12)

    #ax.grid(False)


    phi_phys_deg = (phi_deg + 198.) % 360.
    phi_phy_string = '$\phi_{{phy}}$={:02.0f}$\degree$ CW from North Split\n'.format(phi_phys_deg)
    phy_comp_string = '$\phi_c$={:02.0f}$\degree$'.format(phi_deg)
    if title_on: ax.set_title(phi_phy_string + phy_comp_string, loc='left')

    plot_name = analysis_name +'/'+ analysis_name + '_phi={:03.0f}.png'.format(phi_deg)
    plt.tight_layout()
    simIO.saveFig(plot_name, dpi=dpi)
    plt.close(fig)
    del fig, ax, radtheta_pts
    gc.collect()

    #simIO.log.info('\tPHI: {:.2f} degrees'.format(phi_deg))


# ANIMATION & SUPPORT FUNCTIONS
_TRACE_ANIM_STYLE_PRESETS = {
    'classic': {
        'background_color': '#FFFFFF',
        'torus_facecolor': 'lightgrey',
        'torus_edgecolor': '#000000',
        'torus_linewidth': 0.10,
        'torus_alpha': 1.00,
        'torus_shade': True,
        'camera_dist': 3.0,
        'camera_elev': None,
        'camera_azim': None,
        'title_color': '#13294B',
        'title_fontsize': 18,
    },
    'research_clean': {
        'background_color': '#F4F6F8',
        'torus_facecolor': '#D8DEE6',
        'torus_edgecolor': '#9AA6B2',
        'torus_linewidth': 0.045,
        'torus_alpha': 0.36,
        'torus_shade': True,
        'camera_dist': 3.0,
        'camera_elev': None,
        'camera_azim': None,
        'title_color': '#1F2933',
        'title_fontsize': 20,
    },
    'conference_slide': {
        'background_color': '#111821',
        'torus_facecolor': '#AAB7C4',
        'torus_edgecolor': '#65717D',
        'torus_linewidth': 0.045,
        'torus_alpha': 0.22,
        'torus_shade': True,
        'camera_dist': 3.0,
        'camera_elev': None,
        'camera_azim': None,
        'title_color': '#E7EEF5',
        'title_fontsize': 20,
    },
    'cinematic_uiuc': {
        'background_color': '#081A29',
        'torus_facecolor': '#C7D0D9',
        'torus_edgecolor': '#5F6C79',
        'torus_linewidth': 0.035,
        'torus_alpha': 0.18,
        'torus_shade': True,
        'camera_dist': 3.0,
        'camera_elev': None,
        'camera_azim': None,
        'title_color': '#F2F6FA',
        'title_fontsize': 20,
    },
    'poster_overdriveplus': {
        'background_color': '#02070D',
        'torus_facecolor': '#C7D0D9',
        'torus_edgecolor': '#2D3E4A',
        'torus_linewidth': 0.013,
        'torus_alpha': 0.06,
        'torus_shade': True,
        'camera_dist': 1.22,
        'camera_elev': 34,
        'camera_azim': -86,
        'camera_fov_deg': 132,
        'axes_zoom': 1.32,
        'allow_scene_clip': True,
        'limits_scale': 0.72,
        'limits_offset': (0.12, 0.12, 0.00),
        'title_color': '#FFF4EA',
        'title_fontsize': 32,
    },
    'poster_manual_1': {
        'background_color': '#02070D',
        'torus_facecolor': '#C7D0D9',
        'torus_edgecolor': '#334450',
        'torus_linewidth': 0.014,
        'torus_alpha': 0.07,
        'torus_shade': True,
        'camera_dist': 0.80,
        'camera_elev': 10,
        'camera_azim': -85,
        'camera_fov_deg': 160,
        'axes_zoom': 1.39,
        'allow_scene_clip': True,
        'limits_scale': 1.36,
        'limits_offset': (-0.23, 0.27, 0.23),
        'title_color': '#FFF4EA',
        'title_fontsize': 30,
    },
}


def _resolve_trace_anim_style(style='classic', style_overrides=None):
    """Resolves a named trace-animation style and merges any explicit overrides."""
    if style is None:
        style = 'classic'

    style_config = dict(_TRACE_ANIM_STYLE_PRESETS['classic'])

    if isinstance(style, dict):
        style_config.update(style)
    else:
        style_key = str(style).lower()
        if style_key not in _TRACE_ANIM_STYLE_PRESETS:
            valid_styles = ', '.join(sorted(_TRACE_ANIM_STYLE_PRESETS))
            raise ValueError(f'Unknown trace animation style "{style}". Valid styles: {valid_styles}')
        style_config.update(_TRACE_ANIM_STYLE_PRESETS[style_key])

    if style_overrides:
        style_config.update(style_overrides)

    return style_config


def _normalize_resolution(resolution):
    """Normalizes resolution names so preset lookup is case-insensitive."""
    if resolution is None:
        return '1080p'
    return str(resolution).strip().lower()


def _normalize_trace_sources(ion_traces=None, trace_sources=None, skip_indices=None,
                             linecolor=None, markercolor=None):
    """Normalizes animation inputs into a list of trace-source dictionaries."""
    if trace_sources is None:
        if ion_traces is None:
            raise ValueError('boris_plotTraceAnim requires ion_traces or trace_sources.')
        trace_sources = [{'ion_traces': ion_traces}]
    elif not isinstance(trace_sources, (list, tuple)):
        trace_sources = [trace_sources]

    default_skip = list(skip_indices or [])
    normalized = []
    for source in trace_sources:
        spec = dict(source) if isinstance(source, dict) else {'ion_traces': source}
        if spec.get('ion_traces') is None and spec.get('path') is None:
            raise ValueError('Each trace source requires either ion_traces or path.')
        spec.setdefault('skip_indices', default_skip)
        spec['skip_indices'] = sorted(set(spec.get('skip_indices', [])))
        spec.setdefault('linecolor', linecolor)
        spec.setdefault('markercolor', markercolor)
        if spec.get('path') is not None and 'mmap_mode' not in spec:
            spec['mmap_mode'] = 'r'
        normalized.append(spec)

    return normalized


def _get_trace_source_array(spec):
    """Returns the ndarray or memmap backing a trace source."""
    if '_ion_traces' in spec:
        return spec['_ion_traces']

    ion_traces = spec.get('ion_traces')
    if ion_traces is None:
        ion_traces = np.load(spec['path'], mmap_mode=spec.get('mmap_mode', 'r'))
    spec['_ion_traces'] = ion_traces
    return ion_traces


def _infer_valid_lengths(ion_traces):
    """Counts non-zero samples per particle without building trimmed copies."""
    valid_lengths = np.zeros(ion_traces.shape[1], dtype=np.int64)
    for particle_idx in range(ion_traces.shape[1]):
        particle_trace = ion_traces[:, particle_idx, :]
        valid_lengths[particle_idx] = int(np.count_nonzero(np.any(particle_trace != 0, axis=1)))
    return valid_lengths


def _get_source_valid_lengths(spec):
    """Gets or lazily infers valid particle lengths for a source."""
    if '_valid_lengths' in spec:
        return spec['_valid_lengths']

    valid_lengths = spec.get('valid_lengths')
    if valid_lengths is None:
        valid_lengths = _infer_valid_lengths(_get_trace_source_array(spec))
    valid_lengths = np.asarray(valid_lengths, dtype=np.int64)
    spec['_valid_lengths'] = valid_lengths
    return valid_lengths


def _build_trace_entries(trace_sources, stride):
    """Builds per-particle metadata while keeping trace data file-backed."""
    trace_entries = []
    for source_idx, spec in enumerate(trace_sources):
        ion_traces = _get_trace_source_array(spec)
        valid_lengths = _get_source_valid_lengths(spec)
        skip_indices = set(spec.get('skip_indices', []))

        for particle_idx in range(ion_traces.shape[1]):
            if particle_idx in skip_indices:
                continue
            valid_length = int(valid_lengths[particle_idx])
            if valid_length < 2:
                continue
            strided_length = (valid_length + stride - 1) // stride
            if strided_length < 2:
                continue
            trace_entries.append({
                'source_idx': source_idx,
                'particle_idx': particle_idx,
                'ion_traces': ion_traces,
                'valid_length': valid_length,
                'strided_length': strided_length,
                'linecolor': spec.get('linecolor'),
                'markercolor': spec.get('markercolor'),
            })

    return trace_entries


def _setup_trace_anim_axes(fig, R0, a_radius, style_config=None):
    """Creates the shared 3D vessel scene for trace animations."""
    style_config = _resolve_trace_anim_style(style_config)

    ax = fig.add_subplot(111, projection='3d')
    background_color = style_config['background_color']
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    pane_color = colors.to_rgba(background_color, alpha=1.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.set_pane_color(pane_color)
        except AttributeError:
            pass

    nphi = ntheta = 180
    ptheta = np.linspace(-np.pi, 0, int(np.ceil(ntheta / 2)))
    pphi = np.linspace(0, 2. * np.pi, nphi)
    ptheta, pphi = np.meshgrid(ptheta, pphi)
    px = (R0 + a_radius * np.cos(ptheta)) * np.cos(pphi)
    py = (R0 + a_radius * np.cos(ptheta)) * np.sin(pphi)
    pz = a_radius * np.sin(ptheta)
    ax.plot_surface(px, py, pz, rstride=9, cstride=9,
                    facecolor=style_config['torus_facecolor'],
                    edgecolor=style_config['torus_edgecolor'],
                    linewidth=style_config['torus_linewidth'],
                    alpha=style_config['torus_alpha'],
                    shade=style_config['torus_shade'], zorder=1)

    limits_scale = float(style_config.get('limits_scale', 1.0) or 1.0)
    if limits_scale <= 0.0:
        raise ValueError('limits_scale must be > 0')
    allow_scene_clip = bool(style_config.get('allow_scene_clip', False))
    effective_limits_scale = limits_scale if allow_scene_clip else max(1.0, limits_scale)

    limits_offset = style_config.get('limits_offset', (0.0, 0.0, 0.0))
    if limits_offset is None:
        limits_offset = (0.0, 0.0, 0.0)
    elif np.isscalar(limits_offset):
        limits_offset = (float(limits_offset),) * 3
    else:
        limits_offset = tuple(float(value) for value in limits_offset)
    if len(limits_offset) != 3:
        raise ValueError('limits_offset must be a scalar or a length-3 sequence.')
    axis_offsets = dict(zip(('xlim', 'ylim', 'zlim'), limits_offset))

    xy_extent = max(0.61, float(R0 + a_radius + 0.02))
    z_extent = max(0.41, float(a_radius + 0.02))
    default_limits = {
        'xlim': (-xy_extent, xy_extent),
        'ylim': (-xy_extent, xy_extent),
        'zlim': (-z_extent, z_extent),
    }
    for limit_name, base_limits in default_limits.items():
        limits = style_config.get(limit_name)
        if limits is None:
            center = 0.5 * (base_limits[0] + base_limits[1])
            half_span = 0.5 * (base_limits[1] - base_limits[0]) * effective_limits_scale
            offset = axis_offsets[limit_name]
            center += offset
            if not allow_scene_clip:
                half_span += abs(offset)
            limits = (center - half_span, center + half_span)
        getattr(ax, f'set_{limit_name}')([limits[0], limits[1]])

    box_aspect = style_config.get('box_aspect', (1.0, 1.0, 0.67))
    if box_aspect is not None:
        axes_zoom = style_config.get('axes_zoom')
        try:
            if axes_zoom is None:
                ax.set_box_aspect(box_aspect)
            else:
                ax.set_box_aspect(box_aspect, zoom=axes_zoom)
        except TypeError:
            ax.set_box_aspect(box_aspect)

    camera_fov_deg = style_config.get('camera_fov_deg')
    camera_focal_length = style_config.get('camera_focal_length')
    if camera_fov_deg is not None:
        if not 0.0 < float(camera_fov_deg) < 180.0:
            raise ValueError('camera_fov_deg must be between 0 and 180 degrees.')
        camera_focal_length = 1.0 / np.tan(np.radians(float(camera_fov_deg)) / 2.0)
    if camera_focal_length is not None:
        try:
            ax.set_proj_type('persp', focal_length=float(camera_focal_length))
        except TypeError:
            ax.set_proj_type('persp')

    ax.set_axis_off()
    if style_config.get('camera_dist') is not None:
        ax.dist = style_config['camera_dist']
    if style_config.get('camera_elev') is not None:
        ax.elev = style_config['camera_elev']
    if style_config.get('camera_azim') is not None:
        ax.azim = style_config['camera_azim']
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return ax


def _trace_frame_to_rgba(fig):
    """Draws the current figure canvas and returns an RGBA pixel buffer."""
    fig.canvas.draw()
    return np.array(fig.canvas.buffer_rgba(), copy=True)


def _measure_trace_frame_bbox(fig, background_color, padding_frac=0.08):
    """Measures a tight scene bbox against the solid figure background color."""
    rgba = _trace_frame_to_rgba(fig)
    bg_rgb = np.rint(np.array(colors.to_rgba(background_color)[:3]) * 255.0).astype(np.uint8)
    diff_mask = np.any(rgba[..., :3] != bg_rgb, axis=2)

    if not np.any(diff_mask):
        return (0, 0, rgba.shape[1], rgba.shape[0])

    ys, xs = np.nonzero(diff_mask)
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    pad_px = int(np.ceil(max(x1 - x0, y1 - y0) * float(padding_frac)))
    if pad_px > 0:
        x0 = max(0, x0 - pad_px)
        x1 = min(rgba.shape[1], x1 + pad_px)
        y0 = max(0, y0 - pad_px)
        y1 = min(rgba.shape[0], y1 + pad_px)

    return (x0, y0, x1, y1)


def _compose_trace_frame(fig, target_size_px, background_color, crop_bbox):
    """Crops the active scene and fits it into the requested output frame."""
    target_w, target_h = target_size_px
    rgba = _trace_frame_to_rgba(fig)
    x0, y0, x1, y1 = crop_bbox
    cropped = rgba[y0:y1, x0:x1, :]

    if cropped.size == 0:
        bg_rgba = tuple(int(round(channel * 255.0)) for channel in colors.to_rgba(background_color))
        return np.full((target_h, target_w, 4), bg_rgba, dtype=np.uint8)

    resampling = getattr(Image, 'Resampling', Image).LANCZOS
    crop_img = Image.fromarray(cropped, mode='RGBA')
    scale = min(target_w / crop_img.width, target_h / crop_img.height)
    new_w = max(1, int(round(crop_img.width * scale)))
    new_h = max(1, int(round(crop_img.height * scale)))
    resized = crop_img.resize((new_w, new_h), resampling)

    bg_rgba = tuple(int(round(channel * 255.0)) for channel in colors.to_rgba(background_color))
    canvas = Image.new('RGBA', (target_w, target_h), bg_rgba)
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.alpha_composite(resized, dest=offset)
    return np.asarray(canvas)


def _attach_trace_artists(ax, trace_entries, linewidth, line_alpha, trail_length, markersize):
    """Allocates line and trail artists for each trace entry."""
    for entry in trace_entries:
        if entry['linecolor']:
            line = ax.plot([], [], [], linewidth=linewidth, color=entry['linecolor'], alpha=line_alpha, zorder=5)[0]
        else:
            line = ax.plot([], [], [], linewidth=linewidth, alpha=line_alpha, zorder=5)[0]
        line_color = entry['linecolor'] if entry['linecolor'] else line.get_color()
        marker_color = entry['markercolor'] if entry['markercolor'] else line_color
        trail_dots = [
            ax.plot([], [], [], '.', markersize=markersize, markeredgewidth=0,
                    color=marker_color, alpha=0.0, zorder=6)[0]
            for _ in range(trail_length)
        ]
        entry['line'] = line
        entry['line_alpha'] = line_alpha
        entry['trail_dots'] = trail_dots


def _set_artist_empty(artist):
    """Hides a 3D artist without dropping it from the animation graph."""
    artist.set_data_3d([0.], [0.], [0.])
    artist.set_alpha(0.0)


def _update_trace_artists(frame_idx, trace_entries, steps_per_frame, stride,
                          line_window, trail_length, trail_alphas):
    """Updates all trace artists for a specific animation frame."""
    all_artists = []
    for entry in trace_entries:
        strided_end = min(frame_idx * steps_per_frame + 1, entry['strided_length'])
        line = entry['line']
        trail_dots = entry['trail_dots']

        if strided_end <= 0:
            _set_artist_empty(line)
            all_artists.append(line)
            for dot in trail_dots:
                _set_artist_empty(dot)
                all_artists.append(dot)
            continue

        strided_start = max(0, strided_end - line_window) if line_window else 0
        raw_start = strided_start * stride
        raw_stop = min(entry['valid_length'], (strided_end - 1) * stride + 1)
        trace_segment = entry['ion_traces'][raw_start:raw_stop:stride, entry['particle_idx'], :]

        line.set_data_3d(trace_segment[:, 0], trace_segment[:, 1], trace_segment[:, 2])
        line.set_alpha(entry['line_alpha'])
        all_artists.append(line)

        for k, dot in enumerate(trail_dots):
            strided_idx = strided_end - trail_length + k
            if strided_idx < 0:
                _set_artist_empty(dot)
            else:
                raw_idx = strided_idx * stride
                point = entry['ion_traces'][raw_idx, entry['particle_idx'], :]
                dot.set_data_3d([point[0]], [point[1]], [point[2]])
                dot.set_alpha(float(trail_alphas[k]))
            all_artists.append(dot)

    return all_artists


def _serialize_trace_sources(trace_sources):
    """Strips runtime caches before passing sources to worker processes."""
    serialized = []
    for spec in trace_sources:
        source_out = {
            'skip_indices': list(spec.get('skip_indices', [])),
            'linecolor': spec.get('linecolor'),
            'markercolor': spec.get('markercolor'),
        }
        if spec.get('path') is not None:
            source_out['path'] = spec['path']
            source_out['mmap_mode'] = spec.get('mmap_mode', 'r')
        else:
            source_out['ion_traces'] = spec.get('ion_traces', spec.get('_ion_traces'))

        valid_lengths = spec.get('valid_lengths', spec.get('_valid_lengths'))
        if valid_lengths is not None:
            source_out['valid_lengths'] = np.asarray(valid_lengths, dtype=np.int64)
        serialized.append(source_out)

    return serialized


def boris_saveTracePreviewFrame(ion_traces, b_hidra, frame_idx, save_path,
                                trace_sources=None, skip_indices=None, stride=1, steps_per_frame=1,
                                linewidth=1.0, linecolor=None, line_alpha=1.0, line_window=None,
                                trail_length=10, markersize=4, markercolor=None,
                                resolution='720p', render_dpi=100,
                                style='classic', style_overrides=None,
                                title=None, title_kwargs=None):
    """Renders a single still frame using the same lazy-loading path as the animation."""
    if skip_indices is None:
        skip_indices = []
    if stride < 1:
        raise ValueError('stride must be >= 1')
    if steps_per_frame < 1:
        raise ValueError('steps_per_frame must be >= 1')

    trace_sources = _normalize_trace_sources(
        ion_traces=ion_traces,
        trace_sources=trace_sources,
        skip_indices=skip_indices,
        linecolor=linecolor,
        markercolor=markercolor,
    )
    trace_entries = _build_trace_entries(trace_sources, stride)
    if len(trace_entries) == 0:
        raise ValueError('boris_saveTracePreviewFrame: no valid traces to render.')

    resolution_key = _normalize_resolution(resolution)
    w_px, h_px = _RESOLUTION_MAP.get(resolution_key, _RESOLUTION_MAP['1080p'])
    scene_figsize = (h_px / render_dpi, h_px / render_dpi)
    figsize = (w_px / render_dpi, h_px / render_dpi)
    trail_alphas = np.linspace(0.0, 1.0, trail_length + 1)[1:]
    style_config = _resolve_trace_anim_style(style, style_overrides)

    fig = plt.figure(figsize=scene_figsize, dpi=render_dpi)
    ax = _setup_trace_anim_axes(fig, b_hidra.R0, b_hidra.a, style_config=style_config)
    _attach_trace_artists(ax, trace_entries, linewidth, line_alpha, trail_length, markersize)
    _update_trace_artists(int(frame_idx), trace_entries, steps_per_frame, stride,
                          line_window, trail_length, trail_alphas)
    crop_bbox = _measure_trace_frame_bbox(fig, style_config['background_color'])
    frame_image = _compose_trace_frame(fig, (w_px, h_px), style_config['background_color'], crop_bbox)
    plt.close(fig)

    final_fig = plt.figure(figsize=figsize, dpi=render_dpi)
    final_fig.patch.set_facecolor(style_config['background_color'])
    final_ax = final_fig.add_axes([0.0, 0.0, 1.0, 1.0])
    final_ax.imshow(frame_image)
    final_ax.axis('off')

    if title:
        title_defaults = {
            'x': 0.03,
            'y': 0.94,
            's': title,
            'color': style_config['title_color'],
            'fontsize': style_config['title_fontsize'],
            'fontweight': 'bold',
            'ha': 'left',
            'va': 'top',
        }
        if title_kwargs:
            title_defaults.update(title_kwargs)
        final_fig.text(**title_defaults)

    final_fig.savefig(save_path, dpi=render_dpi, facecolor=final_fig.get_facecolor())
    plt.close(final_fig)
    return save_path


def _anim_render_chunk(args):
    """Renders a consecutive frame chunk using one figure and file-backed traces."""
    (frame_start, frame_stop, frame_dir, trace_sources, steps_per_frame, stride,
     line_window, trail_length, trail_alphas, linewidth, line_alpha, markersize,
    scene_figsize, target_size_px, dpi, style_config, R0, a_radius, bbox_frame_idx) = args

    import os

    plt.switch_backend('Agg')

    local_sources = [dict(spec) for spec in trace_sources]
    trace_entries = _build_trace_entries(local_sources, stride)
    if len(trace_entries) == 0:
        return 0

    fig = plt.figure(figsize=scene_figsize, dpi=dpi)
    ax = _setup_trace_anim_axes(fig, R0, a_radius, style_config=style_config)
    _attach_trace_artists(ax, trace_entries, linewidth, line_alpha, trail_length, markersize)
    _update_trace_artists(bbox_frame_idx, trace_entries, steps_per_frame, stride,
                          line_window, trail_length, trail_alphas)
    crop_bbox = _measure_trace_frame_bbox(fig, style_config['background_color'])

    for frame_idx in range(frame_start, frame_stop):
        _update_trace_artists(frame_idx, trace_entries, steps_per_frame, stride,
                              line_window, trail_length, trail_alphas)
        frame_image = _compose_trace_frame(fig, target_size_px, style_config['background_color'], crop_bbox)
        Image.fromarray(frame_image).save(os.path.join(frame_dir, f'frame_{frame_idx:06d}.png'))

    plt.close(fig)
    return frame_stop - frame_start


_RESOLUTION_MAP = {
    '480p':  (854,  480),
    '720p':  (1280, 720),
    '1080p': (1920, 1080),
    '1440p': (2560, 1440),
    '4k':    (3840, 2160),
}

def boris_plotTraceAnim(ion_traces, b_hidra, runString='default', simIO=None,
                        interval=50, skip_indices=None, stride=1, max_frames=None, steps_per_frame=1,
                        linewidth=1.0, linecolor=None, line_alpha=1.0, line_window=None,
                        trail_length=10, markersize=4, markercolor=None,
                        parallel=True, n_workers=None, resolution='1080p',
                        trace_sources=None, parallel_chunk_size=120,
                        style='classic', style_overrides=None):
    """Animates the ion traces in 3D, growing each trace step by step.

    Speed control:
      - steps_per_frame (int): data points advanced per frame; faster playback,
        no data loss. Prefer over stride.
      - stride (int): sub-samples stored trace (memory reduction only; loses accuracy).
      - interval (ms): ms between frames (GIF capped ~20ms; MP4 handles any value).
      - line_window (int|None): cap displayed line to last N points. Avoids the O(N)
        data copy per frame that makes later frames slow. None = show full trail.
      - parallel (bool): render frames as PNGs in parallel across CPU cores, then
        stitch with ffmpeg. Dramatically faster for large frame counts. Requires ffmpeg.
      - n_workers (int|None): number of worker processes. Defaults to cpu_count().
            - resolution (str): output resolution preset. One of '480p', '720p', '1080p',
                '1440p', '4K'. Lookup is case-insensitive. Default: '1080p'.
            - trace_sources (list|None): optional list of source specs. Each spec may
                provide `path` or `ion_traces`, plus optional `valid_lengths`,
                `skip_indices`, `linecolor`, `markercolor`, and `mmap_mode`.
            - parallel_chunk_size (int): frames rendered per worker task in the parallel
                path. Larger chunks reduce process overhead and repeated scene setup.
            - style (str|dict): named torus/background style preset, or a style dict.
            - style_overrides (dict|None): optional explicit style overrides applied on
                top of the selected preset.
    Trail effect:
      - trail_length (int): number of recent positions shown as scatter dots with
        alpha fading from 0 (oldest) to 1 (newest).
    """
    import os

    if skip_indices is None:
        skip_indices = []
    if stride < 1:
        raise ValueError('stride must be >= 1')
    if steps_per_frame < 1:
        raise ValueError('steps_per_frame must be >= 1')

    trace_sources = _normalize_trace_sources(
        ion_traces=ion_traces,
        trace_sources=trace_sources,
        skip_indices=skip_indices,
        linecolor=linecolor,
        markercolor=markercolor,
    )

    trace_entries = _build_trace_entries(trace_sources, stride)
    if len(trace_entries) == 0:
        simIO.log.info('boris_plotTraceAnim: no valid traces to animate.')
        return

    max_data_pts = max(entry['strided_length'] for entry in trace_entries)
    num_frames = (max_data_pts + steps_per_frame - 1) // steps_per_frame
    if max_frames:
        num_frames = min(num_frames, max_frames)

    trail_alphas = np.linspace(0.0, 1.0, trail_length + 1)[1:]
    fps = max(1, round(1000 / interval))
    resolution_key = _normalize_resolution(resolution)
    w_px, h_px = _RESOLUTION_MAP.get(resolution_key, _RESOLUTION_MAP['1080p'])
    render_dpi = 100
    scene_figsize = (h_px / render_dpi, h_px / render_dpi)
    figsize = (w_px / render_dpi, h_px / render_dpi)
    style_config = _resolve_trace_anim_style(style, style_overrides)
    plotname_mp4 = 'IonTraceAnim.mp4'
    plotname_gif = 'IonTraceAnim.gif'

    ## ── PARALLEL PATH ────────────────────────────────────────────────────────
    if parallel:
        import multiprocessing, tempfile, subprocess, shutil
        frame_dir = tempfile.mkdtemp(prefix='boris_anim_')
        try:
            n = n_workers or multiprocessing.cpu_count()
            chunk_size = max(1, int(parallel_chunk_size))
            serialized_sources = _serialize_trace_sources(trace_sources)
            worker_args = [
                (frame_start, min(frame_start + chunk_size, num_frames), frame_dir,
                 serialized_sources, steps_per_frame, stride, line_window,
                 trail_length, trail_alphas, linewidth, line_alpha, markersize,
                 scene_figsize, (w_px, h_px), render_dpi, style_config,
                 b_hidra.R0, b_hidra.a, num_frames - 1)
                for frame_start in range(0, num_frames, chunk_size)
            ]
            simIO.log.info(
                f'boris_plotTraceAnim: rendering {num_frames} frames in {len(worker_args)} chunks on {n} workers...'
            )
            ctx = multiprocessing.get_context('fork')   # fork avoids re-importing on Linux
            with ctx.Pool(n) as pool:
                with tqdm(total=num_frames, desc=f'Frame generation: {runString}', unit='frame') as pbar:
                    for frames_done in pool.imap_unordered(_anim_render_chunk, worker_args):
                        pbar.update(frames_done)

            save_path = simIO._outputPath(simIO.plot_dir, plotname_mp4)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            ffmpeg_bin = animation.FFMpegWriter.bin_path()
            ffmpeg_cmd = [ffmpeg_bin, '-y',
                '-framerate', str(fps),
                '-i', os.path.join(frame_dir, 'frame_%06d.png'),
                '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '18', save_path]
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            
            simIO.log.info('OUTPUT PLOT: {} (parallel, steps_per_frame={}, fps={})'.format(plotname_mp4, steps_per_frame, fps))
        except subprocess.CalledProcessError as e:
            if isinstance(e.stderr, bytes):
                stderr = e.stderr.decode('utf-8', errors='replace').strip()
            else:
                stderr = str(e.stderr).strip() if e.stderr else 'ffmpeg produced no stderr output.'
            simIO.log.warning(f'boris_plotTraceAnim parallel path failed ({e}); ffmpeg stderr: {stderr}; falling back to serial.')
            parallel = False   # fall through to serial below
        except Exception as e:
            simIO.log.warning(f'boris_plotTraceAnim parallel path failed ({e}); falling back to serial.')
            parallel = False   # fall through to serial below
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)
        if parallel:
            return

    ## ── SERIAL PATH (FuncAnimation) ──────────────────────────────────────────
    fig = plt.figure(figsize=figsize, dpi=render_dpi)
    ax = _setup_trace_anim_axes(fig, b_hidra.R0, b_hidra.a, style_config=style_config)
    _attach_trace_artists(ax, trace_entries, linewidth, line_alpha, trail_length, markersize)

    def update_lines(num):
        return _update_trace_artists(num, trace_entries, steps_per_frame, stride,
                                     line_window, trail_length, trail_alphas)

    ani = animation.FuncAnimation(
        fig, update_lines, frames=num_frames, interval=interval, blit=True)

    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=-1)
        save_path = simIO._outputPath(simIO.plot_dir, plotname_mp4)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.save(save_path, writer=writer)
        simIO.log.info('OUTPUT PLOT: {} (serial, steps_per_frame={}, fps={})'.format(
            plotname_mp4, steps_per_frame, fps))
    except Exception:
        simIO.log.warning('FFMpeg not available; falling back to GIF (fps capped at ~50).')
        writer = animation.PillowWriter(fps=fps)
        save_path = simIO._outputPath(simIO.plot_dir, plotname_gif)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.save(save_path, writer=writer)
        simIO.log.info('OUTPUT PLOT: {} (serial GIF, steps_per_frame={}, fps={})'.format(
            plotname_gif, steps_per_frame, fps))
    plt.close()
