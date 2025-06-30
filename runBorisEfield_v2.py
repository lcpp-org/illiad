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

## SOME PHYSICAL CONSTANTS
kg_per_amu = 1.660_539_068E-27
kboltz = 1.602_176_634E-19 # Joules/eV
Li_mass = 6.941 #amu
He_mass = 4.002602 #amu

############################
## SET SIMULATION INPUTS: ##
############################

# TOROIDAL AND HELICAL MAGNETIC FIELDS
TOROIDAL_CURRENT = 486. #[Amps]
HELICAL_CURRENT = 900. #[Amps]

# ELECTRIC FIELD
FIELD_FILE_ELECTRIC = 'input_files/Efield_SOFE2.npy'
FIELD_SCALE_ELECTRIC = 60.0 # [Volts]

# ION PROPERTIES
ION_TEMP = 2.0 #eV 
ION_MASS = Li_mass #amu
CHARGE_NUM = 1 # Z

# INITIAL CONDITIONS
LCFS_INDEX = 37 # from Poincare output (simIO.log)
NPHI = 60
NTHETA = 72 #90
DELTRS = [0.000]
NPARTICLES_PER_EMITTER = 15 #300

# SIMULATION PARAMETERS
DT = 1e-8
TMAX = 0.0006
NSTEPS = int(TMAX / DT)

# UNIQUE OUTPUT TAG
TAG= '60V_Z1_TraceTest'
OUTPUT_DIRECTORY_NAME = "AcceptedIota3_1500spins_atole-9"



#####################
## RUN SIMULATION: ##
#####################
## SET UP RUN DIRECTORY AND LOGGING
## DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simIO = out.IOHandler(OUTPUT_DIRECTORY_NAME) 
simIO.startLog()
simIO.borisBoilerplate()

## DEFINE STRING (FOR FILE NAME)
delimiter = '-'
dr_String = delimiter.join(str(int(dr*1000)) for dr in DELTRS)
cond_string = dr_String + 'mm_{}eV_LCFS{}_'.format(int(ION_TEMP), int(LCFS_INDEX))

## CALCULATE SOME CONSTANTS
N_emitters = len(DELTRS) * NTHETA * NPHI
N_particles = NPARTICLES_PER_EMITTER * N_emitters

## DEFINE MESH AND LOAD FIELD
b_hidra = Mesh(R0=0.72, a=0.19)
b_hidra.loadCartesianField(coilCurrent=TOROIDAL_CURRENT, errField=True, att_mult='default_toroidal')
b_hidra.addFieldPerturbation(coilCurrent=HELICAL_CURRENT, att_mult='default_helical')
b_hidra.set_nonPer_errField()

e_hidra = Mesh(R0=0.72, a=0.19)
e_hidra.loadCartesianField(FIELD_FILE_ELECTRIC, period_=np.array([0, 1, 1]),
                                att_mult=FIELD_SCALE_ELECTRIC)


##########################
## POSITION GENERATION: ##
##########################
## DEFINE ARRAYS FOR SEED POINT GENERATION
phiGen_arr = np.arange(360//NPHI, 361, 360//NPHI, dtype=int).tolist()
theta_arr = np.linspace(0, 2*np.pi*(1 - 1/NTHETA), NTHETA)
seed_subset = []
seed_list = []
normals_list = []
initVel_list = []
## GENERATE SEED POINTS
simIO.log.info('GENERATING SEED POINTS:')
for phi_gen_deg in phiGen_arr:
    filename = 'Poincare_{:03d}.npy'.format(phi_gen_deg)
    th_in, r_in = simIO.loadNumpyData(filename)[LCFS_INDEX]

    phi_gen = phi_gen_deg*np.pi/180
    seed_subset, normals = generateSeedShells(DELTRS, NTHETA, r_in, th_in, phi_gen,
        b_hidra, simIO, 'IonSeedPts_{}mm'.format(dr_String), genNormals=True, Efield=e_hidra)

    seed_list.extend(seed_subset)
    normals_list.extend(normals)
print('\n')
simIO.log.info('FINISHED LOADING NUMPY DATA & GENERATING INIT. POSITIONS\n')

##########################
## VELOCITY GENERATION: ##
##########################
# GENERATE RANDOM UNIT VECTORS, UNIFORMLY DISTRIBUTED IN A HEMISPHERE, POLE AT +Z
r = np.random.uniform(0, 1, N_particles)
z = np.sqrt(1 - r**2)
phi = np.random.uniform(0, 2 * np.pi, N_particles)
x = r * np.cos(phi)
y = r * np.sin(phi)
initVel_array = np.stack([x, y, z], axis=1) # shape (N, 3)

# GENERATE NORMAL DISTRIBUTION OF SPEEDS
# Calculate the root mean square velocities
v_rms1d = np.sqrt( kboltz*ION_TEMP / (ION_MASS*kg_per_amu) )
initSpeeds = v_rms1d * np.sqrt(np.random.chisquare(df=3, size=N_particles))

# APPLY INIT SPEEDS TO THE RANDOM UNIT VECTORS
initVel_array *= initSpeeds[:, None]

# ROTATE TO ALIGN POLE WITH NORMAL VECTOR
for i,normal in enumerate(normals_list):
    Rotater = align_z_to_vector(normal)
    initVel_array[::NPARTICLES_PER_EMITTER] = initVel_array[::NPARTICLES_PER_EMITTER] @ Rotater.T

tic = perf_counter()

initVelPos = np.zeros((NPARTICLES_PER_EMITTER*N_emitters, 6))
ion_list = []
for i in range(NPARTICLES_PER_EMITTER):
    # chunking
    starti = i*N_emitters
    stopi = (i+1)*N_emitters
    # setting velocities nad positions  in array as chunks
    initVelPos[starti:stopi, 0:3] = initVel_array[starti:stopi]
    initVelPos[starti:stopi, 3:6] = np.array(seed_list)
    # instantiating ions in a list
    ion_list += [Ion(seed_pt, ION_MASS, CHARGE_NUM, TMAX) for seed_pt in seed_list]

toc = perf_counter()
simIO.log.info('Velocities generated in {}sec'.format(toc-tic))
simIO.log.info('initVel_array shape={}'.format(initVel_array.shape) )

## SAVE THE INITIAL VELOCITIES AND POSITIONS AS COMBINED ARRAY
IC_filename = 'initVelPos_' + cond_string+TAG
simIO.saveNumpyData(initVelPos, IC_filename)
simIO.log.info('OUTPUT IC DATA: {}'.format(IC_filename))

## SET INITIAL STATES AND OUTPUT(?necessary?)
for ion, v_0 in zip(ion_list, initVel_array):
    ion.initVelocity(v_0)
    ion.initOutput(DT, TMAX)



##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## RUN BORIS SOLVER FOR PARTICLES ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
particle_tracker_list = [10,13,20,500,2346, 13130, 29777, 33333, 40266, 50000]
# It returns the wall intersection points and their indices.
#wallPt_output, velocity_output, max_timeStep = boris_solver2(ion_list, DT, TMAX, b_hidra, e_hidra)
wallPt_output, velocity_output, max_timeStep, ion_traces = boris_solver2(ion_list, DT, TMAX, b_hidra, e_hidra, particle_tracker_list)
simIO.log.info('PYTORCH STATS:\n' + torch.cuda.memory_summary())



####################
## PREPARE OUTPUT ##
####################
tic = perf_counter()

wallPt_output = wallPt_output.cpu().numpy()
velocity_output = velocity_output.cpu().numpy()
max_timeStep = max_timeStep.cpu().numpy()
ion_traces = ion_traces.cpu().numpy()


## SAVE WALL POINTS AND VELOCITIES
filenameTrac = 'Ion_traces_' + cond_string+TAG
simIO.saveNumpyData(ion_traces, filenameTrac)
simIO.log.info('OUTPUT ION TRACES: {}'.format(filename))



# filter out rows containing all zeros
wallPt_output = wallPt_output[~np.all(wallPt_output == 0, axis=1)]
# Filter velocity_output and get the indices of nonzero rows
nonzero_indices = ~np.all(velocity_output == 0, axis=1)
velocity_output = velocity_output[nonzero_indices]
max_timeStep = max_timeStep[nonzero_indices]

speed_output = np.linalg.norm(velocity_output, axis=1)
energy_output = 0.5 * ION_MASS * kg_per_amu * speed_output**2 / kboltz #convert speed to energy in eV
simIO.log.info('Energy output stats: min={:.2f} eV, max={:.2f} eV, avg={:.2f} eV'.format(
    np.min(energy_output), np.max(energy_output), np.mean(energy_output)))

wallPtArray = np.asarray( [XYZ_to_RTP(wall_point, b_hidra.R0) for wall_point in wallPt_output] ).T
outputArray = np.vstack((wallPtArray, velocity_output.T, max_timeStep[None, :]))

## CALCULATE ANGLE FROM NORMAL
unit_vec_xyz = velocity_output/speed_output[:, None]  # Normalize the velocity vectors to get unit vectors
radial_vec_xyz = np.asarray( [RTP_XYZ_JAC(wall_point, np.array([1,0,0]), form='rtp2xyz') for wall_point in wallPtArray.T] )# Convert unit vectors to RTP coordinates
deposition_angles = np.arccos(np.einsum('ij,ij->i', unit_vec_xyz, radial_vec_xyz))  # Calculate angles between unit vectors and radial vectors
deposition_angles_deg = np.degrees(deposition_angles)  # Convert angles to degrees

simIO.log.info('deposition_angles_deg min: {:.2f} deg, max: {:.2f} deg, avg: {:.2f} deg'.format(
    np.min(deposition_angles_deg), np.max(deposition_angles_deg), np.mean(deposition_angles_deg)))

toc = perf_counter()
simIO.log.info('Output sent to cpu and converted to rtp in {}sec'.format(toc-tic))


## SAVE WALL POINTS AND VELOCITIES
filename = 'Wallpt_OUTPUT_' + cond_string+TAG
simIO.saveNumpyData(outputArray, filename)
simIO.log.info('OUTPUT RESULT DATA: {}'.format(filename))

# COORDINATE FLIIPING & CONVERSION
phi_plot = (-1)*wallPtArray[2] + 2*np.pi # flip phi for the perspective outside the vacuum vessel
theta_plot = wallPtArray[1]
theta_plot[theta_plot>np.pi] -= 2*np.pi #shift so that (theta=0) is centered in the plot

a_phi = -18. # degrees, phi_comp is 18 CW from south-side split
phi_plot_deg = (phi_plot*(180/np.pi) + a_phi) % 360.
theta_plot_deg = theta_plot*(180/np.pi)

# ##############
# ## PLOTTING ##
# ##############

plotFuncs.plotTraces(ion_traces, b_hidra, runString=cond_string+TAG, simIO=simIO)

# ## PLOT HISTOGRAM OF WALL POINTS
# plotFuncs.plotWallHist(wallPtArray, cond_string+TAG, simIO=simIO)
# ## PLOT *3D* HISTOGRAM
# plotFuncs.plotWallPoints3D(phi_plot_deg, theta_plot_deg, b_hidra, runString=cond_string+TAG, simIO=simIO)


# ## PLOT DISCRETE WALL POINTS
# plotFuncs.plotWallPoints(phi_plot_deg, theta_plot_deg, runString=cond_string+TAG, simIO=simIO)
# ## PLOT DISCRETE WALL POINTS with Energy Colorscale
# plotFuncs.plotWallPoints(phi_plot_deg, theta_plot_deg, color_data=energy_output, colorLabel='Ion Deposition Energy (eV)',
#                           runString=cond_string+TAG+'_EnergyDepo', simIO=simIO)
# ## PLOT DISCRETE WALL POINTS with Angle Colorscale
# plotFuncs.plotWallPoints(phi_plot_deg, theta_plot_deg, color_data=deposition_angles_deg, colorRange=[0, 90], colorLabel='Ion Deposition Angle (deg. from normal)',
#                           runString=cond_string+TAG+'_AngleDepo', simIO=simIO)


# ## PLOT INITIAL ENERGY DISTRIBUTION TO VALIDATE MAXWELLIAN PROFILE & ION TEMPERATURE
# plotFuncs.plotInitEnergies(IC_filename+'.npy', ION_MASS, runString=cond_string+TAG, simIO=simIO)
# # PLOT FINAL ENERGY DISTRIBUTION
# plotFuncs.plotFinalEnergies(energy_output, ION_MASS, runString=cond_string+TAG, simIO=simIO)
# # Plot # of perticles running over time
# plotFuncs.plotParticlesOverTime(max_timeStep, N_particles, TMAX, DT, runString=cond_string+TAG, simIO=simIO)

# # PLOT DEPOSITION ANGLE DISTRIBUTION
# plotFuncs.plotDepoAngles(deposition_angles_deg, runString=cond_string+TAG, simIO=simIO)


# plotFuncs.plotCombined(phi_plot_deg, theta_plot_deg, deposition_angles_deg, colorRange=[0, 90], 
#                             colorLabel='Ion Deposition Angle (deg. from normal)', myColormap='viridis',
#                             runString=cond_string+TAG+'_AngleCombined', simIO=simIO)

# plotFuncs.plotCombined(phi_plot_deg, theta_plot_deg, energy_output, 
#                             colorLabel='Ion Deposition Energy (eV)', myColormap='magma',
#                             runString=cond_string+TAG+'_EnergyCombined', simIO=simIO)


## END RUN ##
simIO.log.info('## SIM FINISHED! ##\n\n\n')