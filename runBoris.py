import numpy as np
import numba as nb

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
simIO = out.IOHandler("HIDRA_1q4ERR_interpFieldErr") #DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simIO.startLog()

## DEFINE MESH AND LOAD FIELD
BX, BY, BZ = np.load('input_files/HIDRA_i4ERR_hires.npy')
mesh_prd = np.array([0, 1, 5], dtype=np.int32)
b_hidra = Mesh(R0=0.72, a=0.19)
b_hidra.loadCartesianField(BX, BY, BZ, mesh_prd, errField=True)

## IDENTIFY LAST-CLOSED FLUX SURFACE
LCFS_index = identifyLCFS(LCFStype='input', num=6, outputHandler=simIO)

kg_per_amu = 1.66054E-27
kboltz = 1.602E-19 # Joules/eV
Li_mass = 6.941 #amu
He_mass = 4.002602 #amu

## DEFINE RUN CONDITIONS (PARTICLE TYPE, # OF PARTICLES, INIT. POSITION AND VELOCITY, ETC)
simIO.log.info('GENERATING SEED POINTS:\n')
phiGen_list = np.linspace(9, 360, 40, dtype=int).tolist() # list of phi angles to generated shells
ntheta      = 90                                          # number of equally-spaced theta points for each shell
expand_dr   = [0.010, 0.015, 0.020, 0.025, 0.030]         # define number of 'shells' (delta-r) to generate

delimiter = '-'
dr_String = delimiter.join(str(int(dr*1000)) for dr in expand_dr)

## GENERATE LIST OF INITIAL CONDITIONS FOR IONS
ion_mass = Li_mass
charge_num = 1
ion_temp_eV = 2.
init_v_phi = np.sqrt(2 * kboltz * ion_temp_eV / (ion_mass * kg_per_amu)) # calculate v_(most probable) from ion temperature & mass

theta_list = np.linspace(0, 2*np.pi*(1 - 1/ntheta), ntheta)

seed_subset = []
seed_list = []
initVel_subset = []
initVel_list = []
for phi_gen_deg in phiGen_list:
    filename = 'Poincare_{:03d}.npy'.format(phi_gen_deg)
    th_in, r_in = simIO.loadNumpyData(filename)[LCFS_index]

    phi_gen = phi_gen_deg*(np.pi/180)
    seed_subset = generateSeedShells(expand_dr, ntheta, r_in, th_in, phi_gen,
        b_hidra, simIO, 'IonSeedPts_{}mm'.format(dr_String))
    seed_list.extend(seed_subset)
    for dr in expand_dr:
        for theta_gen in theta_list:
            initVel_subset = init_v_phi * np.array([cos(theta_gen)*cos(phi_gen), -cos(theta_gen)*sin(phi_gen), sin(theta_gen)])  #initial velocity in the +radial direction
            initVel_list.append(initVel_subset)
simIO.log.info('FINISHED LOADING NUMPY DATA\n')


## INSTANTIATE IONS AND SET THEIR INITIAL STATES
dt = 1.2E-7 #2E-7
N = 2E5
tmax = dt*N
ion_list = [Ion(seed_pt, ion_mass, charge_num, tmax) for seed_pt in seed_list]
for ion, v_0 in zip(ion_list, initVel_list):
    ion.initVelocity(v_0)
    ion.initOutput(dt, tmax)

##################################
## RUN BORIS SOLVER FOR PARTICLES
boris_output = boris_wrapper(ion_list, b_hidra, ion_temp_eV, dt, tmax, dr_String)

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

wallPtArray = np.transpose(np.array(wallPt_output2)) 
simIO.saveNumpyData(wallPtArray, 'Wallpoints_{}mm'.format(dr_String))



##################################
## POST-SOLVER OUTPUT (WALL PLOT)
simIO.log.info('Plotting wall hits, total events = {}...'.format(wallPtArray[0].size))

phi_plot = wallPtArray[2]*(-1) + 2*np.pi
theta_plot = wallPtArray[1]
for i in range(len(theta_plot)):
    if theta_plot[i]>np.pi: theta_plot[i] -= 2*np.pi

phi_plot_deg = ((phi_plot*(180/np.pi) + 180.) % 360.) #+ 18.
theta_plot_deg = theta_plot*(180/np.pi)

plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'figure.autolayout':True})
fig = plt.figure()
ax = fig.add_subplot(polar=False, aspect=0.2)

# Import data on HIDRA port size/locations for plotting
ports = simIO.loadPorts_fromCSV('input_files/HIDRA_ports.csv')
for port in ports.T:
    port_plot = patches.Ellipse((port[0], port[1]), port[2], port[3], fill=True, alpha=0.3, facecolor='black')
    ax.add_patch(port_plot)

# plot wall event locations
plt.scatter(phi_plot_deg, theta_plot_deg, s=1, c='k', linewidths=0.0)
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

simIO.saveFig('Wallpoints_BorisPts_{}mm'.format(dr_String))
plt.close()
simIO.log.info('...finished\n')


##################################
## POST-SOLVER OUTPUT ( *3D* WALL PLOT)
simIO.log.info('Attempting 3D plot...')

rad_plot = wallPtArray[0]
theta_plot = wallPtArray[1]
phi_plot = wallPtArray[2] + 2*np.pi

simIO.log.info('r{}, th{}, ph{}'.format(len(rad_plot), len(theta_plot), len(phi_plot)))
xyz_plt = np.zeros(shape=(len(theta_plot), 3))

for i in range(len(theta_plot)):
    if theta_plot[i]>np.pi: theta_plot[i] -= 2*np.pi
    xyz_plt[i] = RTP_to_XYZ( np.array([rad_plot[i], theta_plot[i], phi_plot[i]]), b_hidra.R0 )

fig = plt.figure()
ax2 = fig.add_subplot(projection='3d')
ax2.scatter(xyz_plt.T[0], xyz_plt.T[1], xyz_plt.T[2], s=0.25, c='k', linewidths=0.0)
ax2.set_xlim3d(-1, 1)
ax2.set_ylim3d(-1, 1)
ax2.set_zlim3d(-1, 1)

plt.title('Distribution of Field Line Intersections with HIDRA Wall\n'
          +'Particle: $Li^+, T={}eV$'.format(ion_temp_eV))

simIO.saveFig('Wallpoints3D_BorisPts_{}mm'.format(dr_String))
plt.close()
simIO.log.info('...finished\n')


##############################################
## POST-SOLVER OUTPUT (SINGLE PARTICLE TRACE)
# choose particle from list
ionIndex_list = [47, 300, 700, 1000, 1800, 1986, 2350, 5000, 6000, 7000]
for ionIndex in ionIndex_list:
    test_ion = ion_list[ionIndex] #choose particle from list
    simIO.log.info('Plotting trace, Particle #{}..'.format(test_ion.particleID))
    path_XYZ = np.array(test_ion.pos_XYZ)
    path_RTP = np.zeros(path_XYZ.shape)
    for i, xyz in enumerate(path_XYZ):
        path_RTP[i][:] = XYZ_to_RTP(xyz, b_hidra.R0)
    r_f = path_RTP.T[0]
    th_f = path_RTP.T[1]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    plt.plot(th_f, r_f, linewidth = 0.25)
    ax.set_rmax(b_hidra.a)
    ax.set_rticks(np.arange(0.0, 0.19, 0.02))
    ax.yaxis.set_tick_params(labelsize=5)
    ax.grid(linewidth = 0.25, linestyle=':', c='k')
    plt.title('Single Particle(#{}) Trace, Boris-Buneman Solver'.format(test_ion.particleID))
    plot_name = 'BorisTrace_{}.png'.format(ionIndex)
    simIO.saveFig(plot_name)
    plt.close()
    simIO.log.info('...finished\n')

## END RUN ##
simIO.log.info('## SIM FINISHED! ##\n\n\n\n')