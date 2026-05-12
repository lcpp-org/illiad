import os
import sys
# Allow running from any subdirectory: resolve the project root relative to this file
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from classes.iohandler import IOHandler
from classes.mesh import Mesh
from utility.coordtrans import axisShift, RTP_to_XYZ, XYZ_to_RTP

mpl.rcParams.update({
    # --- fonts & text (IOP-friendly, ~8–12 pt at final size) ---
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})
# ANLYS_DIR = "LSODA-2p49e8_Iota4FWD_1200spins_320Lines"
# ANLYS_SUBDIR = "LCFS100_4x36x360mesh_histogram_7"
ANLYS_DIR = "AcceptedIota4_1500spins_atole-8_eng"
ANLYS_SUBDIR = "LCFS35_360x90_tol_5e1_5e2_LOMEM"

##DEFINE FIELDS
FIELD_FILE_TOR = 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy'
FIELD_FILE_HEL = 'input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy'
CURRENT_TOR = 0.486 #[kA]
CURRENT_HEL = 0.790 #[kA]
CONFIG_TOR = 'default_toroidal'
CONFIG_HEL = 'default_helical'

##DEFINE LCFS AND ANGLES TO EVALUATE
LCFS_INDEX = 35 #100 #40 #22 #29?
NPHI = 360
NTHETA = 36
PHI_GENs = np.linspace(360//NPHI, 360, NPHI)

##FLUX INTEGRATION PARAMETERS
MAX_SUBSETS = 4
SMOOTH_FCTR = 1e-5 #7.5e-6 #baseline 1e-6
INTEGRATE_EPSABS=5e-1
INTEGRATE_EPSREL=5e-2

##PLOTTING FLAG
ISLAND_ALGORITHM = 'histogram' # 'kmeans', 'spectral'
HIST_BINS = 120
PLOT_ALL = True




def find_Axis(theta_vals: np.ndarray, r_vals: np.ndarray, field: Mesh) -> np.ndarray:
    """Find the magnetic axis from theta and r values"""
    r_vals = r_vals[~np.isnan(r_vals)]
    theta_vals = theta_vals[~np.isnan(theta_vals)]

    theta_size = theta_vals.size
    ## CONVERT TO 2D XZ COORDINATES
    x_in = np.empty(theta_size)
    y_in = np.empty(theta_size)
    z_in = np.empty(theta_size)
    for i, theta, in enumerate(theta_vals):
        x_in[i], y_in[i], z_in[i] = RTP_to_XYZ(np.array([r_vals[i], theta, 0.]), field.R0)

    x_avg = np.average(x_in)
    y_avg = 0.0 # y is always 0 in this case (phi=0)
    z_avg = np.average(z_in)

    axis_xyz = np.array([x_avg, y_avg, z_avg])
    axis_thetar = XYZ_to_RTP(axis_xyz, field.R0)[1::-1]

    return axis_thetar

def calculate_histogram_data(theta_r_pts, BINS=120):
    """Calculate histogram data for a given set of theta-r points"""
    hist, bin_edges = np.histogram(theta_r_pts.T[0], bins=BINS, range=(0., 2*np.pi))
    return hist, bin_edges

# Initialize IOHandler
simIO = IOHandler(ANLYS_DIR)
simIO.startLog()

# Initialize the field for magnetic axis calculation
b_hidra = Mesh(R0=0.72, a=0.19)
b_hidra.loadCartesianField(coilCurrent=CURRENT_TOR, errField=True, att_mult=CONFIG_TOR)
b_hidra.addFieldPerturbation(coilCurrent=CURRENT_HEL, att_mult=CONFIG_HEL)
b_hidra.set_nonPer_errField()

# Load the saved flux data
tot_flux_array = simIO.loadNumpyData(ANLYS_SUBDIR + '/CalculatedFLuxes.npy')
total_flux_norm = simIO.loadNumpyData(ANLYS_SUBDIR + '/CalculatedFLuxes-normalized.npy')

NSURFACE = tot_flux_array.shape[0]
good_phi_index = 1
# Reproduce the main flux vs surface plot
simIO.log.info('Plotting Flux v Surface Index...')
fig_post = plt.figure()
axUP = fig_post.add_subplot(211)
#axUP.set_title('Toroidal Angles $\phi[\degree]$', loc='right', fontsize=8)
axUP.set_ylabel('Toroidal Flux $\psi_{\phi}\,[g \cdot m^2]$', fontsize=8)
axUP.plot(tot_flux_array[:,good_phi_index]*10_000, label='{:d}'.format(int(PHI_GENs[good_phi_index])), linewidth=1)
# for i in range(len(PHI_GENs)):
#     axUP.plot(tot_flux_array[:,i]*10_000, label='{:d}'.format(int(PHI_GENs[i])), linewidth=1)

axUP.set_ylim(0, 1.1*10_000*np.max(tot_flux_array)) 
axUP.grid(which='both', linestyle=':', linewidth=0.5)
#axUP.legend(loc='upper right', fontsize=5,ncols=3)
# PLOT THE SURFACE PARAMETER: 1 - FLUX/FLUX_LCFS
axDOWN = fig_post.add_subplot(212, xlabel='Surface index')
axDOWN.set_ylabel('Surface Parameter $\hat{\psi}_n$', fontsize=8)
axDOWN.plot(total_flux_norm.T[good_phi_index], label='{:d}'.format(int(PHI_GENs[good_phi_index])), linewidth=1)
# for i in range(len(PHI_GENs)):
#         axDOWN.plot(total_flux_norm.T[i], label='{:d}'.format(int(PHI_GENs[i])), linewidth=1)

axDOWN.set_ylim(0, 1.1)
axDOWN.grid(which='both', linestyle=':', linewidth=0.5)
simIO.saveFig(ANLYS_SUBDIR+'/NEW_Flux_v_Surface.png', dpi=300)
simIO.log.info('Saved file: {}'.format(ANLYS_SUBDIR+'/NEW_Flux_v_Surface.png'))
plt.close()


exit()

# Reproduce the individual phi angle plots
def init_plotting():
    width_in = 15 / 2.54 # 15 cm in inches
    height_in = width_in * (9/16)  # Maintain 16:9 aspect ratio
    fig = plt.figure(figsize=(width_in, height_in))

    gs = gridspec.GridSpec(2, 2, width_ratios=[1.3, 1], wspace=0.35, hspace=0.0)
    ax4 = fig.add_subplot(gs[:,0], polar=True)
    ax1 = fig.add_subplot(gs[0,1], polar=False)
    ax2 = fig.add_subplot(gs[1,1], polar=False)

    # Reduce margins around the entire figure
    plt.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.12)

    return fig, ax1, ax2, ax4

def finalize_plotting(fig, ax1, ax2, ax4, PHI_GEN_DEG, surf_index, num_subsets, MAX_SUBSETS, simIO):
    num_islandSurfaces = np.where(num_subsets == MAX_SUBSETS)[0].size

    ## set overall figure title with phi angle
    fig.suptitle('Toroidal Angle $\phi={}\degree$'.format(int(PHI_GEN_DEG)), fontsize=12, x=0.18, y=0.98)

    ## r vs theta Cartesian plot
    ax1.set_ylim(0, 0.19)
    ax1.set_xlim(0, 360)
    ax1.tick_params(axis='both', which='major', labelsize=6)
    ax1.set_xticks(np.arange(0, 361, 90))
    ax1.set_xticklabels([])
    ax1.set_yticks(np.arange(0.0, 0.19, 0.025))
    ax1.set_yticklabels(['', '2.5', '5', '7.5', '10', '12.5', '15', '17.5'])
    ax1.grid(linewidth=0.5, linestyle=':', c='grey')
    ax1.set_ylabel('[cm]', fontsize=8)

    ## Histogram plot
    ax2.set_xlim(0, 360)
    ax2.set_xticks(np.arange(0, 361, 90))
    ax2.tick_params(axis='both', which='major', labelsize=6)
    ax2.grid(linewidth=0.5, linestyle=':', c='grey')
    ax2.set_xlabel('Poloidal Angle $\\theta[\degree]$', fontsize=8)
    ax2.set_ylabel('[#]', fontsize=8)

    ## Polar plot
    #print('Plotting Flux Surfaces {} @ phi={}'.format(surf_index, PHI_GEN_DEG))
    ax4.set_rlim(0, 0.19)
    ax4.tick_params(axis='both', which='major', labelsize=8) #,pad=10)
    ax4.grid(linewidth=0.5, linestyle=':', c='grey')
    ax4.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                        labels=['Low\nField', '', '', '', 'High\nField', '', '', ''], fontsize=8)
    ax4.set_rgrids([0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175],
                    labels=['', '5cm', '', '10cm', '', '15cm', ''], angle=0, fontsize=4)
    ## Alternative r-grid settings 
    # ax4.set_rgrids([0.05, 0.1, 0.15],
    #                 labels=['5cm', '10cm', '15cm'], angle=0, fontsize=5)

    simIO.saveFig(ANLYS_SUBDIR+'/NEW_Flux_at_{:03d}deg.png'.format(int(PHI_GEN_DEG)), dpi=600)
    plt.close()

# Loop through each phi angle and recreate the plots
for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):
    if PLOT_ALL: 
        fig, ax1, ax2, ax4 = init_plotting()
        
        # Load flux surface data to find magnetic axis
        filename = 'Poincare_{:03d}.npy'.format(int(PHI_GEN_DEG))
        flux_surfaces = simIO.loadNumpyData(filename)
        mag_axis = find_Axis(*flux_surfaces[-1], b_hidra)
        
        # Load point mesh data for this phi angle
        num_subsets = np.zeros(NSURFACE, dtype=int)
        hist_data = {}
        bin_edges_data = {}

        for surf_index in range(LCFS_INDEX, NSURFACE):
            try:
                # Load the point mesh data
                point_mesh = simIO.loadNumpyData(ANLYS_SUBDIR + '/fSurf_{:03d}_POINTmesh.npy'.format(surf_index))
                centers = simIO.loadNumpyData(ANLYS_SUBDIR + '/fSurf_{:03d}_center.npy'.format(surf_index))
                
                # Get data for this phi angle
                phi_data = point_mesh[phi_index]
                phi_centers = centers[phi_index]
                
                # Filter out NaN values
                valid_points = ~np.isnan(phi_data).any(axis=1)
                phi_data_valid = phi_data[valid_points]
                
                if len(phi_data_valid) > 0:
                    # Count number of subsets (non-zero centers)
                    valid_centers = ~np.isnan(phi_centers).any(axis=1)
                    num_subsets[surf_index] = np.sum(valid_centers)
                    
                    # Calculate histogram data for surfaces with multiple subsets
                    if num_subsets[surf_index] > 1:
                        # Convert to magnetic axis coordinates for histogram calculation
                        th_in, r_in = flux_surfaces[surf_index]
                        r_in = r_in[~np.isnan(r_in)]
                        th_in = th_in[~np.isnan(th_in)]
                        th_size = th_in.size
                        
                        # Shift origin to magnetic axis
                        pts_tr_magAxis = np.empty((th_size, 2))
                        for j, theta in enumerate(th_in):
                            pts_tr_magAxis[j] = axisShift(theta, r_in[j], *mag_axis)
                        
                        # Remove duplicates and sort
                        unique_indices = np.unique(pts_tr_magAxis[:, 0], return_index=True)[1]
                        pts_tr_magAxis = pts_tr_magAxis[unique_indices]
                        pts_tr_magAxis = pts_tr_magAxis[np.argsort(pts_tr_magAxis[:, 0])]
                        
                        # Calculate histogram
                        hist_data[surf_index], bin_edges_data[surf_index] = calculate_histogram_data(pts_tr_magAxis, HIST_BINS)
                    
                    # Plot the data points in polar coordinates
                    theta_vals = phi_data_valid[:, 0]
                    r_vals = phi_data_valid[:, 1]
                    ax4.scatter(theta_vals, r_vals, s=0.5, linewidths=0.0)
                    
                    # Plot centers
                    valid_center_coords = phi_centers[valid_centers]
                    if len(valid_center_coords) > 0:
                        ax4.scatter(valid_center_coords[:, 0], valid_center_coords[:, 1], s=20, color='k', marker='x', linewidths=1)

                    # Also plot in Cartesian coordinates for ax1
                    shifted_theta = []
                    shifted_r = []
                    for j in range(len(theta_vals)):
                        shifted_point = axisShift(theta_vals[j], r_vals[j], *mag_axis)
                        shifted_theta.append(shifted_point[0])
                        shifted_r.append(shifted_point[1])
                    ax1.scatter(np.array(shifted_theta)*180./np.pi, shifted_r, s=0.5, linewidths=0.0)

            except FileNotFoundError:
                print(f"Warning: Data not found for surface {surf_index}")
                continue

        # Plot histogram for surfaces with multiple subsets
        for surf_index in range(LCFS_INDEX, NSURFACE):
            if num_subsets[surf_index] > 1 and surf_index in hist_data:
                ax2.bar(bin_edges_data[surf_index][:-1]*180./np.pi, hist_data[surf_index],
                        width=np.diff(bin_edges_data[surf_index])*180./np.pi, align='edge', edgecolor='k', linewidth=0.1)
                        
        finalize_plotting(fig, ax1, ax2, ax4, PHI_GEN_DEG, surf_index, num_subsets, MAX_SUBSETS, simIO)
    #break  # Remove this break to plot all angles



print("Plot reproduction complete!")