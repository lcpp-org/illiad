import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import patches

import class_outputHandler as out
from mesh import *
from coordtrans import *
from anlys_funcs import *
from poincare_gen import Gen_Poincare
from point_generators import generateSeedShells
from particle import *



## SET UP RUN DIRECTORY
simOut = out.IOHandler("HIDRA-1q3-ERR_particles_5") #DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simOut.startLog()

## DEFINE MESH AND LOAD FIELD
BX, BY, BZ = np.load('input_files/Bxyz_i-1q3_hires_5Period_IH-95p5pct.npy')
mesh_prd = np.array([0, 1, 5], dtype=np.int32)
b_hidra = Mesh(R0=0.72, a=0.19)
b_hidra.loadCartesianField(BX, BY, BZ, mesh_prd, errField=True)

## IDENTIFY LAST-CLOSED FLUX SURFACE
LCFS_index = identifyLCFS(LCFStype='input', num=7, outputHandler=simOut)

## DEFINE RUN CONDITIONS (PARTICLE TYPE, # OF PARTICLES, INIT. POSITION AND VELOCITY, ETC)
simOut.log.info('GENERATING SEED POINTS:\n')
phiGen_list = np.linspace(9, 360, 40, dtype=int).tolist() # list of phi angles to generated shells
ntheta      = 30                                          # number of equally-spaced theta points for each shell
expand_dr   = [0.020]                                     # define number of 'shells' (delta-r) to generate

delimiter = '-'
dr_String = delimiter.join(str(int(dr*1000)) for dr in expand_dr)

max_life = 0.25 # seconds
kg_per_amu = 1.66054E-27
kboltz = 1.602E-19 # Joules/eV
Li_mass = 6.941 #amu
He_mass = 4.002602 #amu
ion_mass = Li_mass
charge_num = 1

# calculate v_(most probable) from ion temperature & mass
ion_temp_eV = 10.
init_v_phi = np.sqrt(2 * kboltz * ion_temp_eV / (ion_mass * kg_per_amu)) #initial velocity in the phi direction

## GENERATE LIST OF INITIAL CONDITIONS FOR IONS
seed_subset = []
seed_list = []
initVel_subset =[]
initVel_list =[]
for phi_gen_deg in phiGen_list:
    filename = 'Poincare_{:03d}.npy'.format(phi_gen_deg)
    th_in, r_in = simOut.loadNumpyData(filename)[LCFS_index]
    # generate list of initial positions as 'shells' expanded from the LCFS
    phi_gen = phi_gen_deg*(np.pi/180)
    seed_subset = generateSeedShells(expand_dr, ntheta, r_in, th_in, phi_gen, 
        b_hidra, simOut, 'IonSeedPts_{}mm'.format(dr_String))
    seed_list.extend(seed_subset)
    # generate list of initial velocities (uniformly pointed in the 'phi' direction)
    initVel_subset = [init_v_phi * np.array([-np.sin(phi_gen_deg), -np.cos(phi_gen_deg), 0.])] * ntheta
    initVel_list.extend(initVel_subset)

## INSTANTIATE IONS AND SET THEIR INITIAL STATES
seed_array = [Ion(seed_pt, ion_mass, charge_num, max_life) for seed_pt in seed_list]
for particle, v_0 in zip(seed_array, initVel_list):
    particle.initialize_velocity(v_0)

## RE-RUN 'POINCARE' WITH IONS
simOut.log.info('###########################################################################')
simOut.log.info('RE-RUNNING POINCARE PLOT GENERATOR WITH NEW !ION! SEED POINTS:\n')
simOut.log.info('Initial Conditions:\t{} points'.format(len(seed_array)))
simOut.log.info('Ions:\tmass={}[amu], Z={}, ion temp.={}eV, initial velocity={:.0f} [m/s]'.format(ion_mass, charge_num, ion_temp_eV, init_v_phi))
simOut.log.info('Shells generated at delta-r(s) of {}mm from LCFS'.format(dr_String))
simOut.log.info('###########################################################################\n')
subName = 'IonSeedPts_{}mm'.format(dr_String)
tMax2, Poincare_output2, wallPt_output2 = Gen_Poincare(b_hidra, seed_array, simOut, subName, 'RK45', 1e-6, 1e-32, saveData=False)



## POST-SOLVER OUTPUT (WALL PLOT)
wallPtArray = np.transpose(np.array(wallPt_output2)) 
simOut.saveNumpyData(wallPtArray, 'Wallpoints_{}mm'.format(dr_String))
simOut.log.info('Plotting wall hits. Total events = {}:\n'.format(wallPtArray[0].size))

phi_plot = wallPtArray[2]*(-1) + 2*np.pi
theta_plot = wallPtArray[1]
for i in range(len(theta_plot)):
    if theta_plot[i]>np.pi: theta_plot[i] -= 2*np.pi

plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'figure.autolayout':True})
fig = plt.figure()
ax = fig.add_subplot(polar=False, aspect=0.2)

# Plot HIDRA ports
## Import data on port size/locations for plotting
#ports = simOut.loadPorts_fromCSV('input_files/HIDRA_ports.csv')
#for port in ports.T:
#    port_plot = patches.Ellipse((port[0], port[1]), port[2], port[3], fill=True, alpha=0.3, facecolor='black')
#    ax.add_patch(port_plot)

# plot wall event locations
plt.scatter(phi_plot*(180/np.pi), theta_plot*(180/np.pi), s=0.75, c='k', linewidths=0.0)
ax.grid(linewidth = 0.25, linestyle=':', c='grey')
plt.xlabel('Toroidal Angle, $\phi$, $[\degree]$')
plt.xlim(0, 360)
plt.xticks(np.linspace(9, 360, 40))
ax.xaxis.set_tick_params(labelsize=3.5)
plt.ylabel('Poloidal Location')
plt.ylim(-180, 180)
plt.yticks(np.linspace(-180, 180, 5), ['Inner Midplane', 'Bottom', 'Outer Midplane', 'Top', 'Inner Midplane'])
ax.yaxis.set_tick_params(labelsize=5)
plt.title('Distribution of Field Line Intersections with HIDRA Wall')

simOut.saveFig('Wallpoints_IonSeedPts_{}mm'.format(dr_String))


## END RUN ##
simOut.log.info('## SIM FINISHED ##\n\n\n\n')