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
# # TOROIDAL AND HELICAL MAGNETIC FIELDS
# FIELD_FILE_TOR = 'input_files/It486_Ih000_Iv000_1p000_1p000_64bit.npy'
# FIELD_SCALE_TOR = 0.9448
# FIELD_FILE_HEL = 'input_files/It000_Ih900_Iv000_1p000_1p000_64bit.npy'
# FIELD_SCALE_HEL = -0.955 * FIELD_SCALE_TOR
# # ELECTRIC FIELD
# FIELD_FILE_ELECTRIC = 'input_files/Efield_SOFE2.npy'
# FIELD_SCALE_ELECTRIC = 60.0 # [Volts]



## INPUT #1 PARAMETERS
######################
INPUT1_DIRECTORY_NAME = "AcceptedIota3_1500spins_atole-9"
#TAG1 = '60V_Z1_TraceTest'
#TAG1 = '60V_Li_Z1_SOFE25-2'
TAG1 = 'APS25_Li-1ms'
#output/AcceptedIota3_1500spins_atole-9/plots/Wall_Histogram_0mm_LCFS37_1eV_60V_Z1_APS25_Li-1ms.png
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
TMAX = 0.0006
NSTEPS = int(TMAX / DT)

## DEFINE STRING (FOR FILE NAME)
delimiter = '-'
dr_String = delimiter.join(str(int(dr*1000)) for dr in DELTRS)
# cond_string1 = dr_String + 'mm_{}eV_LCFS{}_'.format(int(ION_TEMP), int(LCFS_INDEX))
cond_string1 = dr_String + 'mm_LCFS{}_{}eV_{}V_Z{}_'.format(int(LCFS_INDEX), int(ION_TEMP),
                                                           int(FIELD_SCALE_ELECTRIC), int(CHARGE_NUM))


## INPUT #2 PARAMETERS
######################
# INPUT2_DIRECTORY_NAME = "AcceptedIota4_1500spins_atole-8_eng"
# TAG2 = '60V_Z1_1ms'

INPUT2_DIRECTORY_NAME = "AcceptedIota3_1500spins_atole-9"
TAG2 = 'APS25_He-1ms'
#output/ChangetoIota4_1500spins_atole-8_eng/plots/Wall_Histogram_0mm_LCFS39_2eV_60V_Z1_1ms.png

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
TMAX = 0.0006
NSTEPS = int(TMAX / DT)

## DEFINE STRING (FOR FILE NAME)
delimiter = '-'
dr_String = delimiter.join(str(int(dr*1000)) for dr in DELTRS)
cond_string2 = dr_String + 'mm_LCFS{}_{}eV_{}V_Z{}_'.format(int(LCFS_INDEX), int(ION_TEMP),
                                                           int(FIELD_SCALE_ELECTRIC), int(CHARGE_NUM))

# UNIQUE OUTPUT TAG
OUTPUT_DIRECTORY_NAME = "TEST_HISTOGRAM"
TAGOUT = 'article'



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

## LOAD WALL POINTS
filename1 = 'Wallpt_OUTPUT_' + cond_string1+TAG1
outputArray1 = sim_IN1.loadNumpyData(filename1+'.npy')

filename2 = 'Wallpt_OUTPUT_' + cond_string2+TAG2
outputArray2 = sim_IN2.loadNumpyData(filename2+'.npy')
print(f'Loaded wall point data: {outputArray1.shape=}, {outputArray2.shape=}')

# JUST #1
wallPtArray = outputArray1[:3, :]  # r, theta, phi
velocity_output = outputArray1[3:6, :].T  # velocity vectors
max_timeStep = outputArray1[6, :]  # max time step for each particle
cond_string = cond_string1
IC_filename = 'initVelPos_' + cond_string1+TAG1

# # JUST #2
# wallPtArray = outputArray2[:3, :]  # r, theta, phi
# velocity_output = outputArray2[3:6, :].T  # velocity vectors
# max_timeStep = outputArray2[6, :]  # max time step for each particle

# COMBINE the data from outputArray1 and outputArray2
# wallPtArray = np.hstack((outputArray1[:3, :], outputArray2[:3, :]))  # r, theta, phi
# velocity_output = np.hstack((outputArray1[3:6, :], outputArray2[3:6, :])).T  # velocity vectors
# max_timeStep = np.hstack((outputArray1[6, :], outputArray2[6, :]))  # max time step for each particle
# print(f'Combined wall point data shape: {wallPtArray.shape=}')

## CALCULATE ENERGY FROM FINAL VELOCITY
speed_output = np.linalg.norm(velocity_output, axis=1)
energy_output = 0.5 * ION_MASS * kg_per_amu * speed_output**2 / kboltz #convert speed to energy in eV
simIO.log.info('Energy output stats: min={:.2f} eV, max={:.2f} eV, avg={:.2f} eV'.format(
    np.min(energy_output), np.max(energy_output), np.mean(energy_output)))

## CALCULATE ANGLE FROM NORMAL
unit_vec_xyz = velocity_output/speed_output[:, None]  # Normalize the velocity vectors to get unit vectors
radial_vec_xyz = np.asarray( [RTP_XYZ_JAC(wall_point, np.array([1,0,0]), form='rtp2xyz') for wall_point in wallPtArray.T] )# Convert unit vectors to RTP coordinates
toroidal_vec_xyz = np.asarray( [RTP_XYZ_JAC(wall_point, np.array([0,0,1]), form='rtp2xyz') for wall_point in wallPtArray.T] )# Convert unit vectors to RTP coordinates


deposition_angles = np.arccos(np.einsum('ij,ij->i', unit_vec_xyz, radial_vec_xyz))  # Calculate angles between unit vectors and radial vectors
deposition_angles_deg = np.degrees(deposition_angles)  # Convert angles to degrees

cos_toroidal_angles = np.einsum('ij,ij->i', unit_vec_xyz, toroidal_vec_xyz)
toroidal_angles = np.arccos(cos_toroidal_angles)
toroidal_angles_deg = np.degrees(toroidal_angles)  # Convert angles to degrees



simIO.log.info('deposition_angles_deg min: {:.2f} deg, max: {:.2f} deg, avg: {:.2f} deg'.format(
    np.min(deposition_angles_deg), np.max(deposition_angles_deg), np.mean(deposition_angles_deg)))

# COORDINATE FLIIPING & CONVERSION
#phi_plot = (-1)*wallPtArray[2] + 2*np.pi # flip phi for the perspective outside the vacuum vessel
phi_plot = wallPtArray[2]*(-1) + 2*np.pi # flip phi for the perspective outside the vacuum vessel
theta_plot = wallPtArray[1]
theta_plot[theta_plot>np.pi] -= 2*np.pi #shift so that (theta=0) is centered in the plot

a_phi = 18. #-36. # degrees, phi_comp is 18 CW from south-side split
phi_plot_deg = (phi_plot*(180/np.pi) + a_phi) % 360.
theta_plot_deg = theta_plot*(180/np.pi)



## PLOT HISTOGRAM OF WALL POINTS
plotFuncs.boris_plotWallHist(wallPtArray, TAGOUT, simIO=simIO, cond_string=cond_string1)

## PLOT *3D* HISTOGRAM
# plotFuncs.boris_plotWallPoints3D(phi_plot_deg, theta_plot_deg, b_hidra, TAGOUT, simIO=simIO)

# ## PLOT DISCRETE WALL POINTS
# plotFuncs.plotWallPoints(phi_plot_deg, theta_plot_deg, runString=cond_string+TAG, simIO=simIO)
# ## PLOT DISCRETE WALL POINTS with Energy Colorscale
# plotFuncs.plotWallPoints(phi_plot_deg, theta_plot_deg, color_data=energy_output, colorLabel='Ion Deposition Energy (eV)',
#                           runString=cond_string+TAG+'_EnergyDepo', simIO=simIO)
# ## PLOT DISCRETE WALL POINTS with Angle Colorscale
# plotFuncs.plotWallPoints(phi_plot_deg, theta_plot_deg, color_data=deposition_angles_deg, colorRange=[0, 90], colorLabel='Ion Deposition Angle (deg. from normal)',
#                           runString=cond_string+TAG+'_AngleDepo', simIO=simIO)

# ## PLOT INITIAL ENERGY DISTRIBUTION TO VALIDATE MAXWELLIAN PROFILE & ION TEMPERATURE
# plotFuncs.boris_plotInitEnergies(IC_filename+'.npy', ION_MASS, runString=cond_string+TAGOUT, simIO=simIO, sim_in=sim_IN1)
# # PLOT FINAL ENERGY DISTRIBUTION
# plotFuncs.plotFinalEnergies(energy_output, ION_MASS, runString=cond_string+TAG, simIO=simIO) 
# # Plot # of perticles running over time
# plotFuncs.plotParticlesOverTime(max_timeStep, N_particles, TMAX, DT, runString=cond_string+TAG, simIO=simIO)
# # PLOT DEPOSITION ANGLE DISTRIBUTION
# plotFuncs.plotDepoAngles(deposition_angles_deg, runString=cond_string+TAG, simIO=simIO)


plotFuncs.boris_plotCombined(phi_plot_deg, theta_plot_deg, deposition_angles_deg, colorRange=[0, 90], 
                            colorLabel='$\\theta_{depo}~[\\degree]$', myColormap='viridis',
                            runString=cond_string+TAGOUT+'_AngleCombined', simIO=simIO, cond_string=cond_string1)
plotFuncs.boris_plotCombined(phi_plot_deg, theta_plot_deg, energy_output, 
                            colorLabel='$E_{depo}~[eV]$', myColormap='viridis',
                            runString=cond_string+TAGOUT+'_EnergyCombined', simIO=simIO, cond_string=cond_string1)
plotFuncs.boris_plotCombined(phi_plot_deg, theta_plot_deg, toroidal_angles_deg,  colorRange=[0, 180], 
                            colorLabel='$\\theta_{tor}~[\\degree]$', myColormap='magma',
                            runString=cond_string+TAGOUT+'_ToroidalAngleCombined', simIO=simIO, cond_string=cond_string1)

plotFuncs.boris_plotCombined(phi_plot_deg, theta_plot_deg, cos_toroidal_angles,  colorRange=[-1, 1], 
                            colorLabel='$\cos(\\theta_{tor})$', myColormap='magma',
                            runString=cond_string+TAGOUT+'_CosToroidalAngleCombined', simIO=simIO, cond_string=cond_string1)
# plotFuncs.plotCombined_Hist(wallPtArray, max_timeStep, N_particles, TMAX, DT, runString=cond_string+TAG, simIO=simIO)
cos_toroidal_angles

## END RUN ##
simIO.log.info('## SIM FINISHED! ##\n\n\n')