## IMPORTS
import os
import sys
from pathlib import Path

# Allow running from any subdirectory: resolve the project root relative to this file
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

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
# FIELD_FILE_TOR = 'input_files/It486_Ih000_Iv000_1p000_1p000_64bit.npy'
# FIELD_SCALE_TOR = 0.9448
# FIELD_FILE_HEL = 'input_files/It000_Ih900_Iv000_1p000_1p000_64bit.npy'
# FIELD_SCALE_HEL = -0.955 * FIELD_SCALE_TOR
# ERRFIELD_MAG = 1.5654e-4 # [Tesla]
# ERRFIELD_DIR_DEG = 271.5 # [degrees]

# ELECTRIC FIELD
# FIELD_FILE_ELECTRIC = 'input_files/Efield_acceptedSmoothed_linear_3.npy'
# FIELD_FILE_ELECTRIC = 'input_files/Efield_acceptedSOFE1.npy'
# FIELD_FILE_ELECTRIC = 'input_files/Efield_SOFE2.npy'
FIELD_SCALE_ELECTRIC = 40.0 # [Volts]
# ION PROPERTIES
ION_MASS = Li_mass # [amu]
ION_TEMP = 2.0 # [eV]
CHARGE_NUM = 1 # [Z]
# INITIAL CONDITIONS
LCFS_INDEX = 40 #30 #29 #40 (from Poincare output (simIO.log))
DELTRS = [0.000] # [m]
NPHI = 180
NTHETA = 120
NPARTICLES_PER_EMITTER = 50
# SIMULATION PARAMETERS
DT = 1e-8 # [s]
TMAX = 0.0010 # [s]
NSTEPS = int(TMAX / DT)

# UNIQUE OUTPUT TAG
OUTPUT_DIRECTORY_NAME = "It-0486_Ih-0900_noErr_1500sp_LSODA1e8"
TAG = "Lithium_FS40_1p0ms_PHANGLE2"


#####################
## RUN SIMULATION: ##
#####################
## SET UP RUN DIRECTORY AND LOGGING
## DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simIO = out.IOHandler(OUTPUT_DIRECTORY_NAME) 
simIO.startLog()

## DEFINE STRING (FOR FILE NAME)
delimiter = '-'
dr_String = delimiter.join(str(int(dr*1000)) for dr in DELTRS)
cond_string = dr_String + 'mm_LCFS{}_{}eV_{}V_Z{}_'.format(int(LCFS_INDEX), int(ION_TEMP),
                                                           int(FIELD_SCALE_ELECTRIC), int(CHARGE_NUM))
## CALCULATE SOME CONSTANTS
N_emitters = len(DELTRS) * NTHETA * NPHI
N_particles = NPARTICLES_PER_EMITTER * N_emitters

####################
## PREPARE OUTPUT ##
####################
filenameTrac = 'Ion_traces_' + cond_string+TAG
ion_traces = simIO.loadNumpyData(filenameTrac+'.npy')

tmax = DT*NSTEPS
IC_filename = 'initVelPos_' + cond_string+TAG

## LOAD WALL POINTS
filename = 'Wallpt_OUTPUT_' + cond_string+TAG
outputArray = simIO.loadNumpyData(filename+'.npy')
wallPtArray = outputArray[:3, :]  # r, theta, phi
velocity_output = outputArray[3:6, :].T  # velocity vectors
max_timeStep = outputArray[6, :]  # max time step for each particle

## CALCULATE ENERGY FROM FINAL VELOCITY
speed_output = np.linalg.norm(velocity_output, axis=1)
energy_output = 0.5 * ION_MASS * kg_per_amu * speed_output**2 / kboltz #convert speed to energy in eV
simIO.log.info('Energy output stats: min={:.2f} eV, max={:.2f} eV, avg={:.2f} eV'.format(
    np.min(energy_output), np.max(energy_output), np.mean(energy_output)))

## CALCULATE ANGLE FROM NORMAL
vf_hat_xyz = velocity_output/speed_output[:, None]  # Normalize the velocity vectors to get unit vectors
radial_vec_xyz = np.asarray( [RTP_XYZ_JAC(wall_point, np.array([1,0,0]), form='rtp2xyz') for wall_point in wallPtArray.T] )# Convert unit vectors to RTP coordinates
deposition_angles = np.arccos(np.einsum('ij,ij->i', vf_hat_xyz, radial_vec_xyz))  # Calculate angles between unit vectors and radial vectors
deposition_angles_deg = np.degrees(deposition_angles)  # Convert angles to degrees

vf_hat_rtp = np.asarray( [RTP_XYZ_JAC(wall_point, vf_hat_xyz[i], form='xyz2rtp') for i, wall_point in enumerate(wallPtArray.T)] ) # Convert velocity unit vector to RTP coordinates

# This is the angle between the projection of the velocity vector onto the theta-phi plane and the -phi_hat direction (i.e. CCW)
theta_phi_angle_rad = np.abs(np.atan2(vf_hat_rtp[:, 1], vf_hat_rtp[:, 2])) 
# shift theta_phi_angle_rad so that 0 degrees is centered on the poloidal direction
theta_phi_angle_rad -= np.pi/2

simIO.log.info('deposition_angles_deg min: {:.2f} deg, max: {:.2f} deg, avg: {:.2f} deg'.format(
    np.min(deposition_angles_deg), np.max(deposition_angles_deg), np.mean(deposition_angles_deg)))

# COORDINATE FLIIPING & CONVERSION
phi_plot = wallPtArray[2]*(-1) + 2*np.pi # flip phi for the perspective outside the vacuum vessel
a_phi = 18. #-36. # degrees, phi_comp is 18 CW from south-side split
phi_plot_deg = (phi_plot*(180/np.pi) + a_phi) % 360.

theta_plot = wallPtArray[1]
theta_plot[theta_plot>np.pi] -= 2*np.pi #shift so that (theta=0) is centered in the plot
theta_plot_deg = theta_plot*(180/np.pi)

##############
## PLOTTING ##
##############
## DEFINE MESH
b_hidra = Mesh(R0=0.72, a=0.19)

plotFuncs.plotTraces(ion_traces, b_hidra, runString=cond_string+TAG, simIO=simIO)

#plotFuncs.plotTracesPoincare(ion_traces, b_hidra, runString=cond_string+TAG, simIO=simIO)

## PLOT HISTOGRAM OF WALL POINTS
# plotFuncs.plotWallHist(wallPtArray, cond_string+TAG, simIO=simIO)

## PLOT *3D* HISTOGRAM
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
plotFuncs.plotInitEnergies(IC_filename+'.npy', ION_MASS, runString=cond_string+TAG, simIO=simIO)
# # PLOT FINAL ENERGY DISTRIBUTION
# plotFuncs.plotFinalEnergies(energy_output, ION_MASS, runString=cond_string+TAG, simIO=simIO)
# # Plot # of perticles running over time
plotFuncs.plotParticlesOverTime(max_timeStep, N_particles, TMAX, DT, runString=cond_string+TAG, simIO=simIO)

# # PLOT DEPOSITION ANGLE DISTRIBUTION
# plotFuncs.plotDepoAngles(deposition_angles_deg, runString=cond_string+TAG, simIO=simIO)

# # plotFuncs.plotCombined(phi_plot_deg, theta_plot_deg, deposition_angles_deg, colorRange=[0, 90], 
# #                             colorLabel='Ion Deposition Angle (deg. from normal)', myColormap='viridis',
# #                             runString=cond_string+TAG+'_AngleCombined', simIO=simIO)
# # plotFuncs.plotCombined(phi_plot_deg, theta_plot_deg, energy_output,
# #                             colorLabel='Ion Deposition Energy (eV)', myColormap='magma',
# #                             runString=cond_string+TAG+'_EnergyCombined', simIO=simIO)

plotFuncs.plotCombined(phi_plot_deg, theta_plot_deg, theta_phi_angle_rad*(180/np.pi), colorRange=[-90, 90], 
                            colorLabel='Ion Deposition Toroidal Angle (deg. from $\\hat{\\theta}$)', myColormap='cividis',
                            runString=cond_string+TAG+'_PHIAngleCombined', simIO=simIO)

# plotFuncs.plotCombined(phi_plot_deg, theta_plot_deg, deposition_angles_deg, colorRange=[0, 90], 
#                             colorLabel='Ion Deposition Angle (deg. from normal)', myColormap='viridis',
#                             runString=cond_string+TAG+'_AngleCombined', simIO=simIO)
# plotFuncs.plotCombined(phi_plot_deg, theta_plot_deg, energy_output, 
#                             colorLabel='Ion Deposition Energy (eV)', myColormap='viridis',
#                             runString=cond_string+TAG+'_EnergyCombined', simIO=simIO)
# plotFuncs.plotCombined_Hist(wallPtArray, max_timeStep, N_particles, TMAX, DT, runString=cond_string+TAG, simIO=simIO)


## END RUN ##
simIO.log.info('## SIM FINISHED! ##\n\n\n')