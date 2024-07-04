import numba as nb
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

from functools import partial
import concurrent.futures as cf
from time import perf_counter

## SET UP RUN DIRECTORY
#simOut = out.IOHandler("HIDRA-1q3-ERR_particles_5") #DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simOut = out.IOHandler("HIDRA-1q4-ERR_N21-1000spins") #DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simOut.startLog()
## DEFINE MESH AND LOAD FIELD
BX, BY, BZ = np.load('input_files/Bxyz_i-1q4_hires_5Period_IH-95p5pct.npy')
mesh_prd = np.array([0, 1, 5], dtype=np.int32)
b_hidra = Mesh(R0=0.72, a=0.19)
b_hidra.loadCartesianField(BX, BY, BZ, mesh_prd, errField=True)
## IDENTIFY LAST-CLOSED FLUX SURFACE
LCFS_index = identifyLCFS(LCFStype='input', num=13, outputHandler=simOut)



kg_per_amu = 1.66054E-27
kboltz = 1.602E-19 # Joules/eV
Li_mass = 6.941 #amu
He_mass = 4.002602 #amu

## DEFINE RUN CONDITIONS (PARTICLE TYPE, # OF PARTICLES, INIT. POSITION AND VELOCITY, ETC)
simOut.log.info('GENERATING SEED POINTS:\n')
phiGen_list = np.linspace(9, 360, 40, dtype=int).tolist() # list of phi angles to generated shells
ntheta      = 90                                          # number of equally-spaced theta points for each shell
expand_dr   = [0.000]                                     # define number of 'shells' (delta-r) to generate
delimiter = '-'
dr_String = delimiter.join(str(int(dr*1000)) for dr in expand_dr)

## GENERATE LIST OF INITIAL CONDITIONS FOR IONS
ion_mass = Li_mass
charge_num = 1
ion_temp_eV = 10.
init_v_phi = np.sqrt(2 * kboltz * ion_temp_eV / (ion_mass * kg_per_amu)) # calculate v_(most probable) from ion temperature & mass

seed_subset = []
seed_list = []
initVel_subset = []
initVel_list = []
for phi_gen_deg in phiGen_list:
    filename = 'Poincare_{:03d}.npy'.format(phi_gen_deg)
    th_in, r_in = simOut.loadNumpyData(filename)[LCFS_index]
    # generate list of initial positions as 'shells' expanded from the LCFS
    phi_gen = phi_gen_deg*(np.pi/180)
    seed_subset = generateSeedShells(expand_dr, ntheta, r_in, th_in, phi_gen, 
        b_hidra, simOut, 'IonSeedPts_{}mm'.format(dr_String))
    seed_list.extend(seed_subset)
    # generate list of initial velocities (uniformly pointed in the 'phi' direction)
    initVel_subset = [init_v_phi * np.array([-np.sin(phi_gen_deg), -np.cos(phi_gen_deg), 0.])] * ntheta #initial velocity in the phi direction
    initVel_list.extend(initVel_subset)

## INSTANTIATE IONS AND SET THEIR INITIAL STATES
dt = 1E-7
N = 1E5
tmax = dt*N
ion_list = [Ion(seed_pt, ion_mass, charge_num, tmax) for seed_pt in seed_list]
for ion, v_0 in zip(ion_list, initVel_list):
    ion.initialize_velocity(v_0)
    ion.initialize_output(dt, tmax)
simOut.log.info('tmax={}, dt={}, #ofpts={}'.format(tmax, dt, N))

## RUN BORIS SOLVER FOR PARTICLES
simOut.log.info('###########################################################################')
simOut.log.info('RUNNING BORIS-BUNEMAN SOLVER WITH NEW !ION! SEED POINTS:\n')
simOut.log.info('Initial Conditions:\t{} points'.format(len(ion_list)))
simOut.log.info('Ions:\tmass={}[amu], Z={}, ion temp.={}eV, initial velocity={:.0f} [m/s]'
                .format(ion_mass, charge_num, ion_temp_eV, init_v_phi))
simOut.log.info('Shells generated at delta-r(s) of {}mm from LCFS'.format(dr_String))
simOut.log.info('SOLVER SETTINGS: tmax: {}sec., dt: {}sec., N={}pts'.format(tmax, dt, N))
simOut.log.info('###########################################################################\n')
subName = 'IonSeedPts_{}mm'.format(dr_String)

## PARALLELIZATION WITH CONCURRENT FUTURES 'MAP' OVER EACH PARTICLE
boris_x = partial(boris_solver, dt=dt, tmax=tmax, Bfield=b_hidra)
t_start = perf_counter()
with cf.ProcessPoolExecutor(max_workers=40) as executor:
    boris_output = executor.map(boris_x, ion_list)
t_stop = perf_counter()
tot_elapsed_time = t_stop - t_start
simOut.log.info('ALL SOLVERS FINISHED IN {} seconds\n###############\n\n'.format(tot_elapsed_time))


## PREPARE OUTPUT
wallPt_output1 = []
for i, data in enumerate(boris_output):
    wall_point, path = data
    
    ion_list[i].pos_XYZ = path
    notZero = [True if x!=0. else False for x in wall_point]
    if any(notZero):
        wallPt_output1 += [XYZ_to_RTP(wall_point, b_hidra.R0)]
    else:
        pass
wallPt_output2 = np.asarray(wallPt_output1)


## POST-SOLVER OUTPUT (WALL PLOT)
wallPtArray = np.transpose(np.array(wallPt_output2)) 
simOut.saveNumpyData(wallPtArray, 'Wallpoints_{}mm'.format(dr_String))
simOut.log.info('Plotting wall hits. Total events = {}:\n'.format(wallPtArray[0].size))

phi_plot = wallPtArray[2]*(-1) + 2*np.pi
theta_plot = wallPtArray[1]
for i in range(len(theta_plot)):
    if theta_plot[i]>np.pi: theta_plot[i] -= 2*np.pi

phi_plot_deg = (phi_plot*(180/np.pi) + 180.) % 360.
theta_plot_deg = theta_plot*(180/np.pi)

plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'figure.autolayout':True})
fig = plt.figure()
ax = fig.add_subplot(polar=False, aspect=0.2)

# Import data on HIDRA port size/locations for plotting
ports = simOut.loadPorts_fromCSV('input_files/HIDRA_ports.csv')
for port in ports.T:
    port_plot = patches.Ellipse((port[0], port[1]), port[2], port[3], fill=True, alpha=0.3, facecolor='black')
    ax.add_patch(port_plot)

# plot wall event locations
plt.scatter(phi_plot_deg, theta_plot_deg, s=0.75, c='k', linewidths=0.0)
ax.grid(linewidth = 0.25, linestyle=':', c='grey')

plt.xlabel('Toroidal Angle, $\phi$, $[\degree]$')
plt.xlim(0, 360)
plt.xticks(np.linspace(9, 360, 40))
ax.xaxis.set_tick_params(labelsize=3.5)

plt.ylabel('Poloidal Location')
plt.ylim(-180, 180)
plt.yticks(np.linspace(-180, 180, 5), ['Inner Midplane', 'Bottom', 'Outer Midplane', 'Top', 'Inner Midplane'])
ax.yaxis.set_tick_params(labelsize=5)

plt.title('Distribution of Field Line Intersections with HIDRA Wall\n'
          +'Particle: $Li^+, T={}eV$'.format(ion_temp_eV))

simOut.saveFig('Wallpoints_BorisPts_{}mm'.format(dr_String))
plt.close()


## LET"S PLOT THE PATH OF A SINGLE TEST ION!!!!!!!!!!!!
simOut.log.info('PLOTTING TRACE')
# choose particle from list
test_ion = ion_list[27]
simOut.log.info('Choosing Particle #{}'.format(test_ion.particleID))
path_XYZ = test_ion.pos_XYZ
path_RTP = np.zeros(path_XYZ.shape)
for i, xyz in enumerate(path_XYZ):
    path_RTP[i][:] = XYZ_to_RTP(xyz, b_hidra.R0)
r_f = path_RTP.T[0]
th_f = path_RTP.T[1]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
plt.plot(th_f, r_f)
ax.set_rmax(b_hidra.a)
ax.set_rticks(np.arange(0.0, 0.19, 0.02))
ax.yaxis.set_tick_params(labelsize=5)
ax.grid(linewidth = 0.25, linestyle=':', c='k')
plt.title('Single Particle(#{}) Trace, Boris-Buneman Solver'.format(test_ion.particleID))
plot_name = 'SingleTrace_Boris.png'
simOut.saveFig(plot_name)
plt.close()

## END RUN ##
simOut.log.info('## SIM FINISHED ##\n\n\n\n')