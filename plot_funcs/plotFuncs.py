import gc

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches, colors, cm, colormaps
plt.rcParams['animation.ffmpeg_path'] = '/home/sgula/miniforge/envs/testenv/bin/ffmpeg'
import copy
import numpy as np
import logging
from utility.coordtrans import XYZ_to_RTP2

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
    plotname = 'Wall_Histogram_' + runString + '.png'
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
    plotname = 'Wallpoints_BorisPts_' + runString +  '.png'
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

    plotname = 'WallHist3D_' + runString + '.png'
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

    plotname = 'E0_Dist_' + runString + '.png'
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

    plotname = 'Ef_Dist_' + runString + '.png'
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

    plotname = 'Angle_Dist_' + runString + '.png'
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

    plotname = 'IonsVtime_' + runString + '.png'
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
    a_phi = -18. # degrees, phi_comp is 18 CW from south-side split
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

    plotname = 'CombinedHistogram_' + runString + '.png'
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

    plotname = 'IonTraces_' + runString + '.png'
    simIO.saveFig(plotname, dpi=600)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()

def _anim_render_frame(args):
    """Module-level worker for parallel animation frame rendering (must be picklable).

    Renders a single animation frame to a PNG. Each subprocess creates its own
    matplotlib figure, so this is safe to call from a multiprocessing Pool.
    """
    (frame_idx, out_path, traces_xyz, steps_per_frame, trail_length, trail_alphas,
     line_window, linewidth, linecolor, alpha_line, markersize, markercolor, R0, a_radius, figsize, dpi) = args

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Static torus surface
    nphi = ntheta = 180
    ptheta = np.linspace(-np.pi, 0, ntheta // 2)
    pphi   = np.linspace(0, 2. * np.pi, nphi)
    ptheta, pphi = np.meshgrid(ptheta, pphi)
    px = (R0 + a_radius * np.cos(ptheta)) * np.cos(pphi)
    py = (R0 + a_radius * np.cos(ptheta)) * np.sin(pphi)
    pz = a_radius * np.sin(ptheta)
    ax.plot_surface(px, py, pz, rstride=9, cstride=9,
                    facecolor='lightgrey', edgecolor='k', linewidth=0.1,
                    alpha=1.0, shade=True, zorder=1)

    for i, trace in enumerate(traces_xyz):
        end   = min(frame_idx * steps_per_frame + 1, trace.shape[0])
        start = max(0, end - line_window) if line_window else 0
        color = linecolor if linecolor else f'C{i % 10}'
        ax.plot(trace[start:end, 0], trace[start:end, 1], trace[start:end, 2],
                linewidth=linewidth, color=color, alpha=alpha_line, zorder=5)
        for k in range(trail_length):
            pt_idx = end - trail_length + k
            if pt_idx >= 0:
                pt = trace[pt_idx]
                ax.plot([pt[0]], [pt[1]], [pt[2]], '.', markersize=markersize,
                        color=markercolor if markercolor else color, markeredgewidth=0, alpha=float(trail_alphas[k]), zorder=6)

    ax.set_xlim([-0.61, 0.61]); ax.set_ylim([-0.61, 0.61]); ax.set_zlim([-0.41, 0.41])
    ax.set_axis_off()
    ax.dist = 3
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)


_RESOLUTION_MAP = {
    '480p':  (854,  480),
    '720p':  (1280, 720),
    '1080p': (1920, 1080),
    '1440p': (2560, 1440),
    '4K':    (3840, 2160),
}

def boris_plotTraceAnim(ion_traces, b_hidra, runString='default', simIO=None,
                        interval=50, skip_indices=None, stride=1, max_frames=None, steps_per_frame=1,
                        linewidth=1.0, linecolor=None, line_alpha=1.0, line_window=None,
                        trail_length=10, markersize=4, markercolor=None,
                        parallel=True, n_workers=None, resolution='1080p'):
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
        '1440p', '4K'. All presets use a 16:9 aspect ratio. Default: '1080p'.
    Trail effect:
      - trail_length (int): number of recent positions shown as scatter dots with
        alpha fading from 0 (oldest) to 1 (newest).
    """
    import os
    import math

    if skip_indices is None:
        skip_indices = []

    ## PRE-PROCESS TRACES: filter zero rows, apply striding, and collect valid traces
    traces_xyz = []
    for i in range(ion_traces.shape[1]):
        if i in skip_indices:
            continue
        this_ion = ion_traces[:, i, :]
        this_ion = this_ion[~np.all(this_ion == 0, axis=1)]
        if stride > 1:
            this_ion = this_ion[::stride]
        if this_ion.shape[0] < 2:
            continue
        traces_xyz.append(this_ion)  # shape (nsteps_i, 3)

    if len(traces_xyz) == 0:
        simIO.log.info('boris_plotTraceAnim: no valid traces to animate.')
        return

    max_data_pts = max(t.shape[0] for t in traces_xyz)
    num_frames = math.ceil(max_data_pts / steps_per_frame)
    if max_frames:
        num_frames = min(num_frames, max_frames)

    trail_alphas = np.linspace(0.0, 1.0, trail_length + 1)[1:]
    fps = max(1, round(1000 / interval))
    w_px, h_px = _RESOLUTION_MAP.get(resolution, (1920, 1080))
    render_dpi = 100
    figsize = (w_px / render_dpi, h_px / render_dpi)
    plotname_mp4 = 'IonTraceAnim_' + runString + '.mp4'
    plotname_gif = 'IonTraceAnim_' + runString + '.gif'

    ## ── PARALLEL PATH ────────────────────────────────────────────────────────
    if parallel:
        import multiprocessing, tempfile, subprocess, shutil
        frame_dir = tempfile.mkdtemp(prefix='boris_anim_')
        try:
            worker_args = [(f, os.path.join(frame_dir, f'frame_{f:06d}.png'),
                             traces_xyz, steps_per_frame, trail_length, trail_alphas,
                             line_window, linewidth, linecolor, line_alpha, markersize, markercolor,
                             b_hidra.R0, b_hidra.a, figsize, render_dpi)
                            for f in range(num_frames)]
            n = n_workers or multiprocessing.cpu_count()
            simIO.log.info(f'boris_plotTraceAnim: rendering {num_frames} frames on {n} workers...')
            ctx = multiprocessing.get_context('fork')   # fork avoids re-importing on Linux
            with ctx.Pool(n) as pool:
                pool.map(_anim_render_frame, worker_args)

            save_path = os.path.join(simIO.plot_dir, plotname_mp4)
            ffmpeg_bin = animation.FFMpegWriter.bin_path()
            subprocess.run([ffmpeg_bin, '-y',
                '-framerate', str(fps),
                '-i', os.path.join(frame_dir, 'frame_%06d.png'),
                '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '18', save_path], check=True, capture_output=True)
            
            simIO.log.info('OUTPUT PLOT: {} (parallel, steps_per_frame={}, fps={})'.format(plotname_mp4, steps_per_frame, fps))
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode('utf-8', errors='replace').strip() if e.stderr else 'ffmpeg produced no stderr output.'
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
    ax = fig.add_subplot(111, projection='3d')

    nphi = ntheta = 180
    ptheta = np.linspace(-np.pi, 0, int(np.ceil(ntheta / 2)))
    pphi = np.linspace(0, 2. * np.pi, nphi)
    ptheta, pphi = np.meshgrid(ptheta, pphi)
    px = (b_hidra.R0 + b_hidra.a * np.cos(ptheta)) * np.cos(pphi)
    py = (b_hidra.R0 + b_hidra.a * np.cos(ptheta)) * np.sin(pphi)
    pz = b_hidra.a * np.sin(ptheta)
    ax.plot_surface(px, py, pz, rstride=9, cstride=9,
                    facecolor='lightgrey', edgecolor='k', linewidth=0.1,
                    alpha=1.0, shade=True, zorder=1)

    if linecolor:
        lines = [ax.plot([], [], [], linewidth=linewidth, color=linecolor, alpha=line_alpha, zorder=5)[0] for _ in traces_xyz]
    else:
        lines = [ax.plot([], [], [], linewidth=linewidth, alpha=line_alpha, zorder=5)[0] for _ in traces_xyz]

    trail_dots = []
    for line in lines:
        color = linecolor if linecolor else line.get_color()
        trail_dots.append([
            ax.plot([], [], [], '.', markersize=markersize, markeredgewidth=0,
                    color=markercolor if markercolor else color, alpha=0.0, zorder=6)[0]
            for _ in range(trail_length)
        ])

    ax.set_xlim([-0.61, 0.61]); ax.set_ylim([-0.61, 0.61]); ax.set_zlim([-0.41, 0.41])
    ax.set_axis_off()
    ax.dist = 3
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def update_lines(num, traces, lines, trail_dots):
        all_artists = []
        for line, t_dots, trace in zip(lines, trail_dots, traces):
            end   = min(num * steps_per_frame + 1, trace.shape[0])
            start = max(0, end - line_window) if line_window else 0
            line.set_data_3d(trace[start:end, :].T)
            all_artists.append(line)
            for k, dot in enumerate(t_dots):
                pt_idx = end - trail_length + k
                if pt_idx < 0:
                    dot.set_alpha(0.0)
                    dot.set_data_3d([[0.], [0.], [0.]])
                else:
                    pt = trace[pt_idx]
                    dot.set_data_3d([[pt[0]], [pt[1]], [pt[2]]])
                    dot.set_alpha(float(trail_alphas[k]))
                all_artists.append(dot)
        return all_artists

    ani = animation.FuncAnimation(
        fig, update_lines, frames=num_frames,
        fargs=(traces_xyz, lines, trail_dots), interval=interval, blit=True)

    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=-1)
        save_path = os.path.join(simIO.plot_dir, plotname_mp4)
        ani.save(save_path, writer=writer)
        simIO.log.info('OUTPUT PLOT: {} (serial, steps_per_frame={}, fps={})'.format(
            plotname_mp4, steps_per_frame, fps))
    except Exception:
        simIO.log.warning('FFMpeg not available; falling back to GIF (fps capped at ~50).')
        writer = animation.PillowWriter(fps=fps)
        save_path = os.path.join(simIO.plot_dir, plotname_gif)
        ani.save(save_path, writer=writer)
        simIO.log.info('OUTPUT PLOT: {} (serial GIF, steps_per_frame={}, fps={})'.format(
            plotname_gif, steps_per_frame, fps))
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

        skip_indices = [0,1,3,5,6,7,8,10]
        if i not in skip_indices:
            ax.plot(this_theta, this_r, linewidth=0.5, zorder=5)

    ax.set_rlim([0., b_hidra.a])#[-1, 1])

    plotname = 'IonTracesPoin_' + runString + '.png'
    simIO.saveFig(plotname, dpi=600)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()
    #plt.show()

# POINCARE PLOT FUNCTIONS:
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