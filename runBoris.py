#runBoris.py
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
#import particle
import phi_events

from functools import partial
import concurrent.futures as cf
from time import perf_counter

kg_per_amu = 1.66054E-27
kboltz = 1.602E-19 # Joules/eV
Li_mass = 6.941 #amu
He_mass = 4.002602 #amu



def boris_solver(ion, dt, tmax, Bfield):
    log = logging.getLogger()
    log.info('Start IC: {}, {}'.format(ion.particleID, ion.pos0_XYZ))
    t_startInd = perf_counter()
    B = np.empty(3, dtype=np.float64)

    wallPt = np.zeros(3)
    N: np.int32v
    N = int((tmax // dt) + 1)
    # Need particle parms: qdt2m, v0, p0
    qdt2m = ion.charge_mass_ratio * dt/2
    ion.set_pos(0, ion.pos0_XYZ)
    ion.set_vel(0, ion.vel0_XYZ)
    #log.info('pos_XYZ[0]={}'.format(ion.pos_XYZ[0]))

    ## Stepping through dt's until tmax (!!OR TERMINAL CONDITION!!):
    for k in range(N-1):
        B, dum_ = Bfield.interpField(ion.pos_XYZ[k])
        
        tvec = qdt2m * B# tvec given by (4-4, Eq11)
        vprime = ion.vel_XYZ[k] + np.cross(ion.vel_XYZ[k], tvec)# vminus is incremented (4-4, Eq10), get vprime
        svec = 2*tvec / ( 1 + (np.linalg.norm(tvec)*np.linalg.norm(tvec)) )# svec given by (4-4, Eq13)
        #log.info('tvec:{}, vprime:{}, svec:{}'.format(tvec, vprime, svec))
        #ion.vel_XYZ[k+1] = ion.vel_XYZ[k] + np.cross(vprime, svec)# from vminus, vprime, svec (4-4, Eq12), get vplus 
        #ion.pos_XYZ[k+1] = np.copy(ion.pos_XYZ[k] + ion.vel_XYZ[k+1] * dt)# from vplus, dt, get xplus
        vplus = ion.vel_XYZ[k] + np.cross(vprime, svec)# from vminus, vprime, svec (4-4, Eq12), get vplus 
        xplus = ion.pos_XYZ[k] + vplus * dt# from vplus, dt, get xplus

        ion.set_vel(k+1, vplus)
        ion.set_pos(k+1, xplus)
                    
        ion.maxLife = (k+1)*dt
        #log.info('ion.pos_XYZ[k+1]={}'.format(ion.pos_XYZ[k+1]))
        if phi_events.inVV(1, ion.pos_XYZ[k+1], Bfield) < 0.0:
            ion.terminated = True
            wallPt = ion.pos_XYZ[k+1]
            break

    t_stopInd = perf_counter()
    elapsed_timeInd = t_stopInd - t_startInd

    #ion.pos_XYZ = ion.pos_XYZ[~np.isnan(ion.pos_XYZ)]
    #ion.vel_XYZ = ion.vel_XYZ[~np.isnan(ion.vel_XYZ)]
    log.info('pos_XYZ: {}'.format(ion.pos_XYZ))

    if ion.terminated:
        log.info('Success!: Particle {} of {} took {:.5f} sec.\tWall Event at t={:.5f}, k={}'
                 .format(ion.particleID, ion.particleCount, elapsed_timeInd, ion.maxLife, ion.maxLife//dt))
    else:
        log.info('Success!: Particle {} of {} took {:.5f} sec.\tWall Event at t='
                 .format(ion.particleID, ion.particleCount, elapsed_timeInd))
    return wallPt








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
ntheta      = 12                                          # number of equally-spaced theta points for each shell
expand_dr   = [0.010]                                     # define number of 'shells' (delta-r) to generate

delimiter = '-'
dr_String = delimiter.join(str(int(dr*1000)) for dr in expand_dr)

## GENERATE LIST OF INITIAL CONDITIONS FOR IONS
ion_mass = Li_mass
charge_num = 1

# calculate v_(most probable) from ion temperature & mass
ion_temp_eV = 10.
init_v_phi = np.sqrt(2 * kboltz * ion_temp_eV / (ion_mass * kg_per_amu)) #initial velocity in the phi direction

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
dt = 1E-8
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
simOut.log.info('Ions:\tmass={}[amu], Z={}, ion temp.={}eV, initial velocity={:.0f} [m/s]'.format(ion_mass, charge_num, ion_temp_eV, init_v_phi))
simOut.log.info('Shells generated at delta-r(s) of {}mm from LCFS'.format(dr_String))
simOut.log.info('SOLVER SETTINGS: tmax: {}sec., dt: {}sec., N={}pts'.format(tmax, dt, N))
simOut.log.info('###########################################################################\n')
subName = 'IonSeedPts_{}mm'.format(dr_String)

## PARALLELIZATION WITH CONCURRENT FUTURES 'MAP' OVER EACH PARTICLE
boris_x = partial(boris_solver, dt=dt, tmax=tmax, Bfield=b_hidra)
t_start = perf_counter()
with cf.ProcessPoolExecutor(max_workers=40) as executor:
    boris_output = executor.map(boris_x, ion_list)
    #boris_output = executor.submit(boris_x, ion_list)
    simOut.log.info('ping')
t_stop = perf_counter()
tot_elapsed_time = t_stop - t_start
simOut.log.info('ALL SOLVERS FINISHED IN {} seconds\n###############\n\n'.format(tot_elapsed_time))

## PREPARE OUTPUT
wallPt_output1 = []
for wall_point in boris_output:
    #:simOut.log.info('wall_point(XYZ)={}'.format(wall_point))
    notZero = [True if x!=0. else False for x in wall_point]
    #simOut.log.info('notZero={}'.format(notZero))
    if any(notZero):
        wallPt_output1 += [XYZ_to_RTP(wall_point, b_hidra.R0)]
    else:
        pass
wallPt_output2 = np.asarray(wallPt_output1)
simOut.log.info('preFilt wallPt_output2 shape: {}'.format(wallPt_output2.shape))



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

simOut.saveFig('Wallpoints_BorisPts_{}mm'.format(dr_String))
plt.close()

## LET"S PLOT THE PATH OF A SINGLE TEST ION!!!!!!!!!!!!
simOut.log.info('PLOTTING TRACE')

# choose particle from list
test_ion = ion_list[28]
simOut.log.info('Choosing Particle #{}'.format(test_ion.particleID))
path_XYZ = test_ion.pos_XYZ
simOut.log.info('path_XYZ: {}'.format(path_XYZ))

test_ion = ion_list[26]
simOut.log.info('Choosing Particle #{}'.format(test_ion.particleID))
path_XYZ = test_ion.pos_XYZ
simOut.log.info('path_XYZ: {}'.format(path_XYZ))

test_ion = ion_list[27]
simOut.log.info('Choosing Particle #{}'.format(test_ion.particleID))
path_XYZ = test_ion.pos_XYZ
simOut.log.info('path_XYZ: {}'.format(path_XYZ))


path_RTP = np.zeros([int(path_XYZ.size), 3])
for xyz, rtp in zip(path_XYZ, path_RTP):
    rtp = XYZ_to_RTP(xyz, b_hidra.R0)
simOut.log.info('path_RTP: {}'.format(path_RTP))

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)

r_f = path_RTP[:,0]
th_f = path_RTP[:,1]

r_f = r_f[~np.isnan(r_f)]
th_f = th_f[~np.isnan(th_f)]

simOut.log.info('r_f: {}'.format(r_f))
simOut.log.info('th_f: {}'.format(th_f))

plt.scatter(th_f, r_f, '-')
ax.set_rmax(b_hidra.a)
ax.set_rticks(np.arange(0.0, 0.19, 0.02))
ax.yaxis.set_tick_params(labelsize=5)
ax.grid(linewidth = 0.25, linestyle=':', c='k')
plt.title('Single Particle Trace, Boris-Buneman Solver')
plot_name = 'SingleTrace_Boris.png'
simOut.saveFig(plot_name)
plt.close()

## END RUN ##
simOut.log.info('## SIM FINISHED ##\n\n\n\n')