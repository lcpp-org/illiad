## IMPORTS
import numpy as np
from time import perf_counter

import classes.class_outputHandler as out
from classes.meshNew import *
from utility.coordtrans import *
from utility.anlys_funcs import *
from utility.point_generators import generateSeedShells
from classes.particle import *

## SOME PHYSICAL CONSTANTS
kg_per_amu = 1.660_539_068E-27
kboltz = 1.602_176_634E-19 # Joules/eV
Li_mass = 6.941 #amu
He_mass = 4.002602 #amu

############################
## SET SIMULATION INPUTS: ##
############################
FIELD_FILE_TOR = 'input_files/It486_Ih000_Iv000_1p000_1p000_64bit.npy'
FIELD_SCALE_TOR = 0.9448
FIELD_FILE_HEL = 'input_files/It000_Ih900_Iv000_1p000_1p000_64bit.npy'
FIELD_SCALE_HEL = -0.955 * FIELD_SCALE_TOR
ERRFIELD_MAG = 1.5654e-4 # [Tesla]
ERRFIELD_DIR_DEG = 271.5 # [degrees]

# SCALE BY PEAK PLASMA POTENTIAL
FIELD_SCALE_ELECTRIC = 60.0 # [VOLTS]
#FIELD_FILE_ELECTRIC = 'input_files/Efield_accepted_linear.npy'
FIELD_FILE_ELECTRIC = 'input_files/Efield_acceptedSmoothed_linear.npy'
OUTPUT_DIRECTORY_NAME = "AcceptedIota3_1500spins_atole-9"
# FIELD_FILE_ELECTRIC = 'input_files/Efield_changeto_linear.npy'
# OUTPUT_DIRECTORY_NAME = "ChangeToIota3_1500spins_atole-9"

LCFS_INDEX = 61 #61 # from Poincare output (simIO.log)

NPHI = 180 #180
NTHETA = 90 #45
DELTRS = [0.000]
NPARTICLES_PER_EMITTER = 200

ION_TEMP = 2.0 #eV 
ION_MASS = Li_mass
CHARGE_NUM = 3

DT = 5e-9 #5E-8 #1E-7 #2E-7
NSTEPS = 100E3 #2E4 #1E4 #5 #2E5 #5E3

#TAG= 'Efield-linear180_LCFS37_180Nphi_1e-8DT_25e3NT_Z3_10eV_NEWV0s-2'
TAG= 'LCFS61_60V_Li_2eV_Z3_lots_5e9DT_100e3NSTEPS'


## SET UP RUN DIRECTORY AND LOGGING
## DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simIO = out.IOHandler(OUTPUT_DIRECTORY_NAME) 
simIO.startLog()
simIO.log.info('\n|==========================================================================|'
              +'\n| LOADED TOROIDAL FIELD DATA FROM: {}'.format(FIELD_FILE_TOR)
              +'\n| LOADED TOROIDAL FIELD SCALING FACTOR: {}'.format(FIELD_SCALE_TOR)
              +'\n| LOADED HELICAL FIELD DATA FROM: {}'.format(FIELD_FILE_HEL)
              +'\n| LOADED HELICAL FIELD SCALING FACTOR: {}'.format(FIELD_SCALE_HEL)
              +'\n| LOADED ERRFIELD MAG: {}'.format(ERRFIELD_MAG)
              +'\n| LOADED ERRFIELD DIR: {}'.format(ERRFIELD_DIR_DEG)
              +'\n| LOADED ELECTRIC FIELD DATA FROM: {}'.format(FIELD_FILE_ELECTRIC)
              +'\n| LOADED ELECTRIC FIELD SCALING FACTOR: {}'.format(FIELD_SCALE_ELECTRIC)
              +'\n|--------------------------------------------------------------------------|'
              +'\n| LAST-CLOSED FLUX SURFACE INDEX: {}'.format(LCFS_INDEX)
              +'\n| ION TEMPERATURE: {} eV'.format(ION_TEMP)
              +'\n| ION MASS: {} amu'.format(ION_MASS)
              +'\n| ION CHARGE: {}'.format(CHARGE_NUM)
              +'\n|--------------------------------------------------------------------------|'
              +'\n| RUNNING {} EMITTERS WITH {} PARTICLES PER EMITTER'.format( len(DELTRS)*NPHI*NTHETA, NPARTICLES_PER_EMITTER )
              +'\n| TOTAL PARTICLES: {}'.format( len(DELTRS)*NPHI*NTHETA*NPARTICLES_PER_EMITTER )
              +'\n|--------------------------------------------------------------------------|'
              +'\n| TIME STEP: {} sec'.format(DT)
              +'\n| # OF TIME STEPS: {}'.format(NSTEPS)
              +'\n| TOTAL TIME: {:.6f} sec'.format(DT*NSTEPS)
              +'\n|==========================================================================|\n\n\n')

#####################
## RUN SIMULATION: ##
#####################
## DEFINE MESH AND LOAD FIELD
b_hidra = Mesh(R0=0.72, a=0.19)
b_hidra.loadCartesianField(FIELD_FILE_TOR, att_mult=FIELD_SCALE_TOR, errField=True )
b_hidra.addFieldPerturbation(FIELD_FILE_HEL, att_mult=FIELD_SCALE_HEL)
b_hidra.set_nonPer_errField(ERRFIELD_MAG, ERRFIELD_DIR_DEG*np.pi/180.)

if FIELD_FILE_ELECTRIC:
    e_hidra = Mesh(R0=0.72, a=0.19)
    e_hidra.loadCartesianField(FIELD_FILE_ELECTRIC, period_=np.array([0, 1, 1]),
                                att_mult=FIELD_SCALE_ELECTRIC, errField=False )
else:
    e_hidra = None

## DEFINE ARRAYS FOR SEED POINT GENERATION
phiGen_arr = np.arange(360//NPHI, 361, 360//NPHI, dtype=int).tolist() # phi angles to generated shells
theta_arr = np.linspace(0, 2*np.pi*(1 - 1/NTHETA), NTHETA) # theta angles to generated sh

## DEFINE STRING (FOR FILE NAME)
delimiter = '-'
dr_String = delimiter.join(str(int(dr*1000)) for dr in DELTRS)
cond_string = dr_String + 'mm_{}eV_LCFS{}_'.format(int(ION_TEMP), int(LCFS_INDEX))

# INSTANTIATE LISTS (FASTER THAN APPENDING TO NUMPY ARRAYS?)
seed_subset = []
seed_list = []
normals_list = []
initVel_list = []
## GENERATE SEED POINTS
simIO.log.info('GENERATING SEED POINTS:\n')
for phi_gen_deg in phiGen_arr:
    filename = 'Poincare_{:03d}.npy'.format(phi_gen_deg)
    th_in, r_in = simIO.loadNumpyData(filename)[LCFS_INDEX]

    phi_gen = phi_gen_deg*np.pi/180
    seed_subset, normals = generateSeedShells(DELTRS, NTHETA, r_in, th_in, phi_gen,
        b_hidra, simIO, 'IonSeedPts_{}mm'.format(dr_String), genNormals=True, Efield=e_hidra)

    seed_list.extend(seed_subset)
    normals_list.extend(normals)

simIO.log.info('FINISHED LOADING NUMPY DATA & GENERATING INIT. POSITIONS\n')

## CONVERT LISTS TO ARRAYS
N_emitters = len(DELTRS) * NTHETA * NPHI
N_particles = NPARTICLES_PER_EMITTER * N_emitters

# Calculate the 1D and 3D root mean square velocities
v_rms1d = np.sqrt( kboltz*ION_TEMP / (ION_MASS*kg_per_amu) )
v_rms3d = np.sqrt(3*kboltz*ION_TEMP / (ION_MASS*kg_per_amu) )

# GENERATE NORMAL DISTRIBUTION OF SPEEDS
initSpeeds = np.random.normal(0, v_rms3d, N_particles)

# GENERATE RANDOM UNIT VECTORS, UNIFORMLY DISTRIBUTED IN A HEMISPHERE, POLE AT +Z
z = np.random.uniform(0, 1, N_particles)
phi = np.random.uniform(0, 2 * np.pi, N_particles)
r = np.sqrt(1 - z**2)
x = r * np.cos(phi)
y = r * np.sin(phi)
initVel_array = np.stack([x, y, z], axis=1) # shape (N, 3)

# APPLY INIT SPEEDS TO THE RANDOM UNIT VECTORS
initVel_array *= initSpeeds[:, None]

# ROTATE TO ALIGN POLE WITH NORMAL VECTOR
def align_z_to_vector(v):
    z_axis = np.array([0, 0, 1])
    #v = v / np.linalg.norm(v)
    if np.allclose(v, z_axis):
        return np.eye(3)
    if np.allclose(v, -z_axis):
        # 180 degree rotation around any perpendicular axis
        return np.array([[-1,  0,  0],
                         [ 0, -1,  0],
                         [ 0,  0,  1]])
    axis = np.cross(z_axis, v)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.dot(z_axis, v))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R

for i,normal in enumerate(normals_list):
    Rotater = align_z_to_vector(normal)
    initVel_array[::NPARTICLES_PER_EMITTER] = initVel_array[::NPARTICLES_PER_EMITTER] @ Rotater.T

tic = perf_counter()
tmax = DT*NSTEPS
initVelPos = np.zeros((NPARTICLES_PER_EMITTER*N_emitters, 6))
ion_list = []

for i in range(NPARTICLES_PER_EMITTER):
    # chunking
    starti = i*N_emitters
    stopi = (i+1)*N_emitters

    # setting velocities in array as chunks
    #initVelPos[starti:stopi, 0:3] = initVelXYZ_list[starti:stopi]
    initVelPos[starti:stopi, 0:3] = initVel_array[starti:stopi]
    # set positions, repeating in array as chunks
    initVelPos[starti:stopi, 3:6] = np.array(seed_list)
    # instantiating ions in a list
    ion_list += [Ion(seed_pt, ION_MASS, CHARGE_NUM, tmax) for seed_pt in seed_list]

toc = perf_counter()
simIO.log.info('Velocities generated in {}sec'.format(toc-tic))
simIO.log.info('initVel_array shape={}'.format(initVel_array.shape) )


#using simIO, save the initial velocities and positions as one array
filename = 'initVelPos_' + cond_string+TAG
simIO.saveNumpyData(initVelPos, filename)
simIO.log.info('OUTPUT DATA: {}'.format(filename))

## SET INITIAL STATES AND OUTPUT(?necessary?)
for ion, v_0 in zip(ion_list, initVel_array):
    ion.initVelocity(v_0)
    ion.initOutput(DT, tmax)

####################################
## RUN BORIS SOLVER FOR PARTICLES ##
####################################
# It returns the wall intersection points and their indices.
wallPt_output, index_wallPts, velocity_output = boris_solver2(ion_list, DT, tmax, b_hidra, e_hidra)
wallPt_output = wallPt_output.cpu().numpy()
velocity_output = velocity_output.cpu().numpy()

speed_output = np.linalg.norm(velocity_output, axis=1)
#convert speed to energy in eV
energy_output = 0.5 * ION_MASS * kg_per_amu * speed_output**2 / kboltz


## PRINT MEMORY USAGE
simIO.log.info('PYTORCH STATS:\n' + torch.cuda.memory_summary())

## PRINT ENERGY STATS
simIO.log.info('ENERGY OUTPUT: min={:.2f} eV, max={:.2f} eV, avg={:.2f} eV'.format(
    np.min(energy_output), np.max(energy_output), np.mean(energy_output)))
####################
## PREPARE OUTPUT ##
####################
tic = perf_counter()
wallPtArray = np.asarray( [XYZ_to_RTP(wall_point, b_hidra.R0) for wall_point in wallPt_output] ).T
toc = perf_counter()
simIO.log.info('WALLPT ARRAY SENT TO CPU AND CONVERTED TO RTP IN {}sec'.format(toc-tic))

## SAVE WALL POINTS
filename = 'Wallpoints_' + cond_string+TAG
simIO.saveNumpyData( wallPtArray, filename )
simIO.log.info('OUTPUT DATA: {}'.format(filename))

## PLOTTING ##
import plot_funcs.plotFuncs as plotFuncs

# flip phi for the perspective outside the vacuum vessel
phi_plot = (-1)*wallPtArray[2] + 2*np.pi

theta_plot = wallPtArray[1]
# shift so that the outer midplane (theta=0) is centered in the plot
theta_plot[theta_plot>np.pi] -= 2*np.pi

a_phi = -18. #-18. # degrees, phi_comp is 18 CW from south-side split
phi_plot_deg = (phi_plot*(180/np.pi) + a_phi) % 360.
theta_plot_deg = theta_plot*(180/np.pi)

## PLOT HISTOGRAM OF WALL POINTS
plotFuncs.plotWallHist(wallPtArray, cond_string+TAG, simIO=simIO)

## PLOT DISCRETE WALL POINTS
plotFuncs.plotWallPoints(phi_plot_deg, theta_plot_deg, runString=cond_string+TAG, simIO=simIO)
## PLOT DISCRETE WALL POINTS with color
plotFuncs.plotWallPoints(phi_plot_deg, theta_plot_deg, color_data=energy_output,
                          runString=cond_string+TAG+'_color', simIO=simIO)

## *3D* WALL PLOT)
#plotFuncs.plotWallPoints3D(phi_plot_deg, theta_plot_deg, b_hidra, runString=cond_string+TAG, simIO=simIO)

## END RUN ##
simIO.log.info('## SIM FINISHED! ##\n\n\n')