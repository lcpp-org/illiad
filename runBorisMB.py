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
INPUT_FILE_LOCATION = 'input_files/It486_Ih900_Iv000_0p955_hires.npy'
OUTPUT_DIRECTORY_NAME = 'It486_Ih900_Iv000_0p955_21lines_rtol6_500spins'
TAG= 'ONEPHI'
LCFS_INDEX = 31

NPHI = 40
NTHETA = 60
DELTRS = [0.010, 0.015]
NPARTICLES_PER_EMITTER = 500

ION_TEMP = 2.0 #eV
ION_MASS = Li_mass
CHARGE_NUM = 1

DT = 2E-7
NSTEPS = 5E5 #5E3


## SET UP RUN DIRECTORY AND LOGGING
## DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simIO = out.IOHandler(OUTPUT_DIRECTORY_NAME) 
simIO.startLog()
simIO.log.info('\n|==========================================================================|'
              +'\n| LOADED FIELD DATA FROM: {}'.format(INPUT_FILE_LOCATION)
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
b_hidra.loadCartesianField('input_files/It486_Ih900_Iv000_0p955_hires.npy', errField=True)
simIO.log.info('Loaded field data from input_files/It486_Ih900_Iv000_0p955_hires.npy')

## DEFINE ARRAYS FOR SEED POINT GENERATION
phiGen_arr = np.arange(360//NPHI, 361, 360//NPHI, dtype=int).tolist() # phi angles to generated shells
#phiGen_arr = np.array([360], dtype=int).tolist() # phi angles to generated shells
theta_arr = np.linspace(0, 2*np.pi*(1 - 1/NTHETA), NTHETA) # theta angles to generated sh

## DEFINE STRING (FOR FILE NAME)
delimiter = '-'
dr_String = delimiter.join(str(int(dr*1000)) for dr in DELTRS)
cond_string = dr_String + 'mm_{}eV_'.format(int(ION_TEMP))

# INSTANTIATE LISTS (FASTER THAN APPENDING TO NUMPY ARRAYS?)
seed_subset = []
seed_list = []

r_hat_list = []
theta_hat_list = []
phi_hat_list = []

initVel_list = []
## GENERATE SEED POINTS
simIO.log.info('GENERATING SEED POINTS:\n')
for phi_gen_deg in phiGen_arr:
    filename = 'Poincare_{:03d}.npy'.format(phi_gen_deg)
    th_in, r_in = simIO.loadNumpyData(filename)[LCFS_INDEX]

    phi_gen = phi_gen_deg*(np.pi/180)
    seed_subset = generateSeedShells(DELTRS, NTHETA, r_in, th_in, phi_gen,
        b_hidra, simIO, 'IonSeedPts_{}mm'.format(dr_String))
    seed_list.extend(seed_subset)

    # CONSIDER MOVING THIS TO FUNCTION generateSeedShells(), MAKE USE OF DERIVATIVE/(GRADIENT) RO FIND PERPENDICULAR UNIT VECTORS
    for dr in DELTRS:
        for theta_gen in theta_arr:
            # GENERATE UNIT VECTORS FOR PARTICLE VELOCITIES (THESE ARE NOT PERPENDICULAR TO FLUX SURFACES!)
            r_hats = np.array([cos(theta_gen)*cos(phi_gen), -cos(theta_gen)*sin(phi_gen), sin(theta_gen)])  #xyz vectors in the +radial direction
            theta_hats = np.array([-sin(theta_gen)*cos(phi_gen), sin(theta_gen)*sin(phi_gen), cos(theta_gen)])  #xyz vectors in the +theta direction
            iphi_hats = np.array([-sin(phi_gen), -cos(phi_gen), 0.])  #xyz vectors in the +phi direction 

            r_hat_list.append(r_hats)
            theta_hat_list.append(theta_hats)
            phi_hat_list.append(iphi_hats)

simIO.log.info('FINISHED LOADING NUMPY DATA & GENERATING INIT. POSITIONS\n')

## CONVERT LISTS TO ARRAYS
N_emitters = len(r_hat_list)
r_phat_arr = np.array(r_hat_list)
theta_hat_arr = np.array(theta_hat_list)
phi_hat_arr = np.array(phi_hat_list)


tic = perf_counter()

# calculate the 1D and 3D root mean square velocities
v_rms1d = np.sqrt( kboltz*ION_TEMP / (ION_MASS*kg_per_amu) )
#v_rms3d = np.sqrt(3*kboltz*ION_TEMP / (ION_MASS*kg_per_amu) )

# initialize velocity vectors of a maxwell-boltzmann energy distribution
initVel_list = np.zeros((NPARTICLES_PER_EMITTER*N_emitters, 3) )
initVel_list[:,0] = np.random.normal(0, v_rms1d, NPARTICLES_PER_EMITTER*N_emitters)
initVel_list[:,1] = np.random.normal(0, v_rms1d, NPARTICLES_PER_EMITTER*N_emitters)
initVel_list[:,2] = np.random.normal(0, v_rms1d, NPARTICLES_PER_EMITTER*N_emitters)
initVel_list[:,0] = np.abs(initVel_list[:,0]) # ensure that it is pointed radially outward


tmax = DT*NSTEPS
initVelXYZ_list = np.zeros(initVel_list.shape)
initVelPos = np.zeros((NPARTICLES_PER_EMITTER*N_emitters, 6))
ion_list = []
for i in range(NPARTICLES_PER_EMITTER):
    # chunking
    starti = i*N_emitters
    stopi = (i+1)*N_emitters
    # create list of velocities in xyz coordinates
    initVelXYZ_list[starti:stopi] = (initVel_list[starti:stopi,0][:,None] * r_phat_arr + 
                                     initVel_list[starti:stopi,1][:,None] * theta_hat_arr + 
                                     initVel_list[starti:stopi,2][:,None] * phi_hat_arr)

    # setting velocities in array as chunks
    initVelPos[starti:stopi, 0:3] = initVelXYZ_list[starti:stopi]
    # set positions, repeating in array as chunks
    initVelPos[starti:stopi, 3:6] = np.array(seed_list)
    # instantiating ions in a list
    ion_list += [Ion(seed_pt, ION_MASS, CHARGE_NUM, tmax) for seed_pt in seed_list]

toc = perf_counter()
simIO.log.info('Velocities generated in {}sec'.format(toc-tic))
simIO.log.info('initVelXYZ_list shape={}'.format(initVelXYZ_list.shape) )

#using simIO, save the initial velocities and positions as one array
filename = 'initVelPos_' + cond_string+TAG
simIO.saveNumpyData(initVelPos, filename)
simIO.log.info('OUTPUT DATA: {}'.format(filename))

## SET INITIAL STATES AND OUTPUT(?necessary?)
for ion, v_0 in zip(ion_list, initVelXYZ_list):
    ion.initVelocity(v_0)
    ion.initOutput(DT, tmax)


####################################
## RUN BORIS SOLVER FOR PARTICLES ##
####################################
# It returns the wall intersection points and their indices.
wallPt_output, index_wallPts = boris_solver2(ion_list, DT, tmax, b_hidra)
wallPt_output = wallPt_output.cpu().numpy()
#wallPt_indices = index_wallPts.detach().cpu().numpy()

## PRINT MEMORY USAGE
simIO.log.info('PYTORCH STATS:\n' + torch.cuda.memory_summary())


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

phi_plot = wallPtArray[2]*(-1) + 2*np.pi
theta_plot = wallPtArray[1]
for i in range(len(theta_plot)):
    if theta_plot[i]>np.pi: theta_plot[i] -= 2*np.pi

phi_plot_deg = (phi_plot*(180/np.pi) + 180. + 0.) % 360.
theta_plot_deg = theta_plot*(180/np.pi)

## PLOT HISTOGRAM OF WALL POINTS
plotFuncs.plotWallHist(wallPtArray, cond_string+TAG, simIO)

## PLOT DISCRETE WALL POINTS
plotFuncs.plotWallPoints(phi_plot_deg, theta_plot_deg, cond_string+TAG, simIO)

## *3D* WALL PLOT)
plotFuncs.plotWallPoints3D(phi_plot_deg, theta_plot_deg, b_hidra, cond_string+TAG, simIO)

## END RUN ##
simIO.log.info('## SIM FINISHED! ##\n\n\n')
