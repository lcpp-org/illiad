## IMPORTS
import numpy as np
from time import perf_counter

import classes.class_outputHandler as out
from classes.meshNew import *
from utility.coordtrans import *
from utility.anlys_funcs import *
from utility.point_generators import generateSeedShells
from classes.particle import *
import plot_funcs.plotFuncs as plotFuncs

import matplotlib.pyplot as plt
from matplotlib import patches, colors, cm, colormaps

## SOME PHYSICAL CONSTANTS
kg_per_amu = 1.660_539_068E-27
kboltz = 1.602_176_634E-19 # Joules/eV
Li_mass = 6.941 #amu
He_mass = 4.002602 #amu

## Plotting helper function
def plotPorts(ax_, simIO):
    """Plots the ports on the given axis."""
    # Import data on HIDRA port size/locations for plotting
    ports = simIO.loadPorts_fromCSV('input_files/HIDRA_ports.csv')
    for port in ports.T:
        port_plot = patches.Ellipse((port[0], port[1]), port[2], port[3],
                                    fill=True, alpha=0.2, facecolor='black', edgecolor='black', linewidth=0.0)
        ax_.add_patch(port_plot)


## INPUT #1 PARAMETERS
######################
INPUT1_DIRECTORY_NAME = "AcceptedIota3_1500spins_atole-9"
TAG1 = 'APS25_Li-1ms'
# ION PROPERTIES
ION_TEMP = 1.0 #eV 
ION_MASS = Li_mass #amu
CHARGE_NUM = 1 # Z
FIELD_SCALE_ELECTRIC = 60.0 # [Volts]
# INITIAL CONDITIONS
LCFS_INDEX = 37 # from Poincare output (simIO.log)
NPHI = 60
NTHETA = 72 #90
DELTRS = [0.000]
NPARTICLES_PER_EMITTER = 15 #300
# SIMULATION PARAMETERS
DT = 1e-8
TMAX = 0.001
NSTEPS = int(TMAX / DT)
## DEFINE STRING (FOR FILE NAME)
delimiter = '-'
dr_String = delimiter.join(str(int(dr*1000)) for dr in DELTRS)
cond_string1 = dr_String + 'mm_LCFS{}_{}eV_{}V_Z{}_'.format(int(LCFS_INDEX), int(ION_TEMP),
                                                           int(FIELD_SCALE_ELECTRIC), int(CHARGE_NUM))


## INPUT #2 PARAMETERS
######################
INPUT2_DIRECTORY_NAME = "AcceptedIota3_1500spins_atole-9"
TAG2 = 'APS25_He-1ms'
#output/AcceptedIota3_1500spins_atole-9/plots/WallPts_0mm_LCFS37_1eV_60V_Z1_APS25_He-1ms_EnergyCombined.png
# ION PROPERTIES
ION_TEMP = 1.0 #eV 
ION_MASS = He_mass #amu
CHARGE_NUM = 1 # Z
# INITIAL CONDITIONS
LCFS_INDEX = 37 #40 # from Poincare output (simIO.log)
NPHI = 60
NTHETA = 72 #90
DELTRS = [0.000]
NPARTICLES_PER_EMITTER = 15 #300
# SIMULATION PARAMETERS
DT = 1e-8
TMAX = 0.001
NSTEPS = int(TMAX / DT)
## DEFINE STRING (FOR FILE NAME)
delimiter = '-'
dr_String = delimiter.join(str(int(dr*1000)) for dr in DELTRS)
cond_string2 = dr_String + 'mm_LCFS{}_{}eV_{}V_Z{}_'.format(int(LCFS_INDEX), int(ION_TEMP),
                                                           int(FIELD_SCALE_ELECTRIC), int(CHARGE_NUM))


# UNIQUE OUTPUT TAG
OUTPUT_DIRECTORY_NAME = "Test_Output"
TAGOUT = 'TESTING_i3_60V_1eV-Z1_Li-He_Union'



## SET UP RUN DIRECTORY AND LOGGING
## DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simIO = out.IOHandler(OUTPUT_DIRECTORY_NAME) 
simIO.startLog()

sim_IN1 = out.IOHandler(INPUT1_DIRECTORY_NAME) 
sim_IN2 = out.IOHandler(INPUT2_DIRECTORY_NAME)


## CALCULATE SOME CONSTANTS
N_emitters = len(DELTRS) * NTHETA * NPHI
N_particles = NPARTICLES_PER_EMITTER * N_emitters

## DEFINE MESH
b_hidra = Mesh(R0=0.72, a=0.19)


# ####################
# ## PREPARE OUTPUT ##
# ####################
## LOAD WALL POINTS
filename1 = 'Wallpt_OUTPUT_' + cond_string1+TAG1
outputArray1 = sim_IN1.loadNumpyData(filename1+'.npy')
wallPtArray = outputArray1[:3, :]  # r, theta, phi

filename2 = 'Wallpt_OUTPUT_' + cond_string2+TAG2
outputArray2 = sim_IN2.loadNumpyData(filename2+'.npy')
print(f'Loaded wall point data: {outputArray1.shape=}, {outputArray2.shape=}')
wallPtArray2 = outputArray2[:3, :]  # r, theta, phi


# COORDINATE FLIIPING & CONVERSION
a_phi = 18. #-36. # degrees, phi_comp is 18 CW from south-side split

phi_plot = wallPtArray[2]*(-1) + 2*np.pi # flip phi for the perspective outside the vacuum vessel
phi_plot_deg = (phi_plot*(180/np.pi) + a_phi) % 360.

theta_plot = wallPtArray[1]
theta_plot[theta_plot>np.pi] -= 2*np.pi #shift so that (theta=0) is centered in the plot
theta_plot_deg = theta_plot*(180/np.pi)

phi_plot2 = wallPtArray2[2]*(-1) + 2*np.pi # flip phi for the perspective outside the vacuum vessel
phi_plot_deg2 = (phi_plot2*(180/np.pi) + a_phi) % 360.

theta_plot2 = wallPtArray2[1]
theta_plot2[theta_plot2>np.pi] -= 2*np.pi #shift so that (theta=0) is centered in the plot
theta_plot_deg2 = theta_plot2*(180/np.pi)


# ##############
# ## PLOTTING ##
# ##############

# define bin edges for 2d histogram
phi_edges = np.linspace(0, 360, 361)
theta_edges = np.linspace(-180, 180, 181)

## CREATE HISTOGRAMS
H1, phi_edges, theta_edges = np.histogram2d(phi_plot_deg, theta_plot_deg, bins=[phi_edges, theta_edges], density=True)
H1 = H1.T # histogram reverse axes for some reason; transpose

H2, phi_edges2, theta_edges2 = np.histogram2d(phi_plot_deg2, theta_plot_deg2, bins=[phi_edges, theta_edges], density=True)
H2 = H2.T # histogram reverse axes for some reason; transpose

# multiplicative overlay
#H = H1 * H2
# logical overlay
#H = (H1 > 0) & (H2 > 0)
# minimum union
H = np.minimum(H1, H2)

# # red-blue overlay
# r = H1 / H1.max()
# b = H2 / H2.max()
# H = np.stack([r, np.zeros_like(r), b], axis=-1)

## PLOT HISTOGRAM
plt.rcParams.update({'font.size': 8})
w, h = plt.figaspect(0.4)
fig = plt.figure(figsize=(w, h))
ax = fig.add_subplot(polar=False, aspect=0.2)
plt.grid(which='both', linewidth=0.25)
plotPorts(ax, simIO)

plt.imshow(H, interpolation='nearest', origin='lower',
            extent=[phi_edges[0], phi_edges[-1], theta_edges[0], theta_edges[-1]],
            cmap='Purples', norm=colors.LogNorm(vmin=1E-6, vmax=1E-3),
            aspect=0.2 )
plt.colorbar(location='bottom', shrink=0.6)

ax.set_xlabel('$\phi$ (+CCW from South-Side Split)', fontsize=12)
ax.set_xlim(0, 360)
xticks = np.linspace(9, 351, 39)
ax.set_xticks(xticks)
ax.set_xticklabels([f'{int(tick)}$\degree$' if i % 2 != 0 else '' for i, tick in enumerate(xticks)])
ax.xaxis.set_tick_params(labelsize=10)

ax.set_ylabel('Poloidal Location', fontsize=12)
ax.set_ylim(-180, 180)
ax.set_yticks(np.linspace(-180, 180, 5))
ax.set_yticklabels(['', 'Bottom', 'Low-\nField', 'Top', ''])
ax.yaxis.set_tick_params(labelsize=8, labelrotation=45)
plt.tight_layout()

plotname = 'Wall_Histogram_' + TAGOUT + '.png'
simIO.saveFig(plotname, dpi=400)
simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
plt.close()





## END RUN ##
simIO.log.info('## SIM FINISHED! ##\n\n\n')