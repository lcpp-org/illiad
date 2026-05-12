#plot_fig4.py
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
LCFS_INDEX = 35 
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
GOOD_PHI_INDEX = 1

# Load the additional data files
big_grid_linear = simIO.loadNumpyData(ANLYS_SUBDIR + '/big_grid_linear.npy')
efield_data = simIO.loadNumpyData(ANLYS_SUBDIR + '/Efield_AcceptedIota4_lcfs25.npy')

# Create meshgrid for plotting
b_hidra = Mesh(R0=0.72, a=0.19)
b_hidra.loadCartesianField(coilCurrent=CURRENT_TOR, errField=True, att_mult=CONFIG_TOR)
b_hidra.addFieldPerturbation(coilCurrent=CURRENT_HEL, att_mult=CONFIG_HEL)
b_hidra.set_nonPer_errField()

RADS = np.linspace(b_hidra.r_min, b_hidra.r_max, b_hidra.nr)
THETAS = np.linspace(b_hidra.theta_min, b_hidra.theta_max, b_hidra.ntheta)
grid_theta, grid_rad = np.meshgrid(THETAS, RADS, indexing='ij')

# PLOT THE linear flux norm data in middle subplot
PHI_PLOT_INDEX = GOOD_PHI_INDEX  # Use the same phi index as the left plot
flux_data_middle = big_grid_linear[PHI_PLOT_INDEX]

# Add periodic boundary for plotting
flux_data_plot = np.vstack((flux_data_middle[-1], flux_data_middle))
grid_rad_plot = np.vstack((grid_rad[-1], grid_rad))
grid_theta_plot = np.vstack((grid_theta[-1], grid_theta))
grid_theta_plot[0] = 0



# Reproduce the main flux vs surface plot
simIO.log.info('Plotting Flux v Surface Index...')
fig_post = plt.figure()

# Subplot layout
gs = gridspec.GridSpec(4, 2, width_ratios=[1.3, 1], height_ratios=[0.4, 1, 1, 0.4], wspace=0.2, hspace=0.)
axLEFT = fig_post.add_subplot(gs[1:3,0], polar=False)
axMIDDLE = fig_post.add_subplot(gs[:2,1], polar=True)
axRIGHT = fig_post.add_subplot(gs[2:,1], polar=True)
#plt.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.12)
plt.subplots_adjust(left=0.09, right=0.97, top=0.95, bottom=0.05)


# Plot surface parameter vs. surface number
axLEFT.set_xlabel('Surface index')
axLEFT.plot(total_flux_norm.T[GOOD_PHI_INDEX], label='{:d}'.format(int(PHI_GENs[GOOD_PHI_INDEX])), linewidth=1)
axLEFT.set_ylabel('Surface Parameter $\hat{\psi}_n$', fontsize=8)
axLEFT.set_ylim(0, 1.1)
#axLEFT.set_title('Flux Surface Parameter vs. Surface Index', fontsize=10)
axLEFT.grid(which='both', linestyle=':', linewidth=0.5)

PHI_PLOT_DEG = int(PHI_GENs[GOOD_PHI_INDEX])


# PLOT THE linear flux norm data
#axMIDDLE = fig_post.add_subplot(132, polar=True)
c_middle = axMIDDLE.pcolormesh(grid_theta_plot, grid_rad_plot, flux_data_plot, 
                              shading='gouraud', cmap='Blues', vmin=0.0, vmax=1.0)
#axMIDDLE.set_title('Normalized Flux, $\\hat{\psi}$', fontsize=10)
axMIDDLE.set_rmax(0.19)
axMIDDLE.set_rgrids([0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175],
                labels=['', ' ', '', ' ', '', ' ', ''], angle=0, fontsize=4)

axMIDDLE.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                    labels=['Low\nField', '', '', '', 'High\nField', '', '', ''], fontsize=6)

axMIDDLE.grid(False)

# PLOT THE electric field magnitude data in right subplot
# Calculate E-field magnitude from the 3-component data
Ex = efield_data[0][:, :, PHI_PLOT_INDEX]  # X-component
Ey = efield_data[1][:, :, PHI_PLOT_INDEX]  # Y-component  
Ez = efield_data[2][:, :, PHI_PLOT_INDEX]  # Z-component

# Calculate magnitude
efield_magnitude = np.sqrt(Ex**2 + Ey**2 + Ez**2)
efield_plot_data = efield_magnitude.T
# Add periodic boundary for plotting
efield_data_plot = np.vstack((efield_plot_data[-1], efield_plot_data))


# PLOT THE flux grad magnitude data
c_right = axRIGHT.pcolormesh(grid_theta_plot, grid_rad_plot, efield_data_plot, 
                            shading='gouraud', cmap='afmhot_r', vmin=0.0, vmax=400.0)
axRIGHT.set_rmax(0.19)
axRIGHT.set_rgrids([0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175],
                labels=['', ' ', '', ' ', '', ' ', ''], angle=0, fontsize=4)

axRIGHT.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                    labels=['Low\nField', '', '', '', 'High\nField', '', '', ''], fontsize=6)

axRIGHT.grid(False)

# Add colorbars
cbarMID =fig_post.colorbar(c_middle, ax=axMIDDLE, shrink=0.5, pad=0.2)
cbarRIGHT = fig_post.colorbar(c_right, ax=axRIGHT, shrink=0.5, pad=0.2)
cbarMID.ax.tick_params(labelsize=6)
cbarRIGHT.ax.tick_params(labelsize=6)
cbarMID.ax.set_title('$\\hat{\psi}$', fontsize=8)
cbarRIGHT.ax.set_title('$|\\nabla \hat{\psi}|$', fontsize=8)

# Adjust layout and save
#plt.tight_layout()
simIO.saveFig(ANLYS_SUBDIR + '/Figure_4_Combined_Analysis.png', dpi=300)
simIO.log.info('Saved combined figure: Figure_4_Combined_Analysis.png')
#plt.show()