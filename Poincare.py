import numpy as np
import scipy as sp
import glob

import class_outputHandler as out
from mesh import *
from coordtrans import *
from poincare_gen import Gen_Poincare
#from ode import blines, solvePoincare
import matplotlib.pyplot as plt
#from matplotlib import cm
from matplotlib import patches


## SET UP RUN DIRECTORY ##
##======================##
# RIGHT NOW, DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simOut = out.IOHandler("i3ideal_9-72_4cm")
simOut.startLog()


## DEFINE MESH AND LOAD FIELD ##
##============================##
b_hidra = CartesianField()
b_hidra.setToroidalGeometry(0.72, 0.19)

#b_hidra.loadCartesianField_fromFile('Bxyz_i-1q4_hires_5Period.npy', 0, 1, 5)
b_hidra.loadCartesianField_fromFile('Bxyz_negY_i-1q3_hires_5Period.npy', 0, 1, 5)
#b_hidra.loadCartesianField_fromFile('Bxyz_i-1q3_hires_5Period_IH-95p5pct.npy', 0, 1, 5)


## SET UP POINCARE SEED POINTS ##
##=============================##
## NEED A BETTER WAY TO SET UP INITIAL POINT-9S!
Nlines = 15 #25
spins = 700

fl_R0     = np.array(np.linspace( 0.130, 0.095, Nlines))
fl_THETA0 = np.zeros(Nlines)
fl_PHI0   = np.ones(Nlines) * (np.pi/5.)

ICs_RTP = np.transpose(np.vstack([fl_R0, fl_THETA0, fl_PHI0]))
ICs_XYZ = np.zeros(shape=(Nlines, 3))
for i in range(Nlines):
    ICs_XYZ[i] = RTP_to_XYZ(ICs_RTP[i], b_hidra.R0)

length = (2*np.pi * b_hidra.R0) * spins
simOut.log.info(f'Initial Conditions (RTP):\n{ICs_RTP}')


## GENERATE POINCARE PLOTS/DATA ##
##==============================##
tMax, Poincare_output, wallPt_output = Gen_Poincare(b_hidra, ICs_XYZ, length, simOut, 'Poincare')
simOut.log.info(f'{tMax=}')

plt.figure()
plt.plot(fl_R0, tMax, '-k')
plt.title('Connection length vs. r(IC)')
simOut.saveFig('connectLengths')
plt.close()

## IDENTIFY LAST-CLOSED FLUX SURFACE ##
##===================================##
maxTime = max(tMax)
LCFS_index = tMax.index(maxTime)


#LCFS_index = 10
simOut.log.info(f'{LCFS_index=}')

## IMPORT CORRESOPONDING POINCARE SURFACE ##
## GENERATE 'SEED SHELLS' OF IC's         ##
## EXPANDING CO-AXIALLY WITH LCFS         ##
##========================================##
simOut.log.info('GENERATING SEED POINTS:\n')

from point_generators import generateSeedShells

# generate seeds from the same flux surface at different phi angles

# list of phi angles to generated shells
phiGen_list = [9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117, 126, 135, 144, 153, 162, 171, 180,
            189, 198, 207, 216, 225, 234, 243, 252, 261, 270, 279, 288, 297, 306, 315, 324, 333, 342, 351]
# define number of 'shells' (dr) to generate
expand_dr = [ 0.020]#, 0.010, 0.014]  
# number of equally-spaced theta points for each shell
ntheta = 90 

seed_subset = []
seed_list = []

for phi_gen_deg in phiGen_list:

    phi_gen = phi_gen_deg*(np.pi/180)

    filename = f'Poincare_{phi_gen_deg:03d}_{LCFS_index:d}.npy'
    th_in, r_in = simOut.loadNumpyData(filename)

    seed_subset = generateSeedShells(expand_dr, ntheta, r_in, th_in, phi_gen, b_hidra, simOut, 'SeedPts_360Phi_2cm')
    seed_list.extend(seed_subset)

seed_array = np.array(seed_list)


## RE-RUN 'POINCARE' WITH SEED ARRAY ##
##===================================##
simOut.log.info('RE-RUNNING POINCARE PLOT GENERATOR WITH NEW SEED POINTS:\n')
spins = 400
length = (2*np.pi * b_hidra.R0) * spins

simOut.log.info(f'Initial Conditions (XYZ):\n{len(seed_array)} points')
tMax2, Poincare_output2, wallPt_output2 = Gen_Poincare(b_hidra, seed_array, length, simOut, 'SeedPts_360Phi_2cm')


## POST-SOLVER OUTPUT
####################
wallPtArray = np.transpose( np.array(wallPt_output2) )
simOut.saveNumpyData(wallPtArray, 'Wallpoints_360Phi_2cm')

#wallPtArray = simOut.loadNumpyData('Wallpoints_i4err.npy')

phi_plot = wallPtArray[2]
phi_plot = phi_plot*(-1) + 2*np.pi
theta_plot = wallPtArray[1]
for i in range(len(theta_plot)):
    if theta_plot[i]>np.pi: theta_plot[i] -= 2*np.pi

simOut.log.info(f'Plotting wall hits. Total events = {wallPtArray[0].size}:\n')

## Import data on port size/locations for plotting
ports = simOut.loadPorts_fromCSV('input_files/HIDRA_ports.csv')

plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'figure.autolayout':True})

fig = plt.figure()
ax = fig.add_subplot(polar=False, aspect=0.2)
for port in ports.T:
    port_plot = patches.Ellipse((port[0], port[1]), port[2], port[3], fill=True, alpha=0.3, facecolor='black')
    ax.add_patch(port_plot)

plt.scatter(phi_plot*(180/np.pi), theta_plot*(180/np.pi), s=0.2, c='k')
plt.grid(True) 
plt.xlim(0, 360)
plt.xticks(np.linspace(9, 360, 40))
plt.ylim(-180, 180)
plt.yticks(np.linspace(-180, 180, 5), ['Inner Midplane', 'Bottom', 'Outer Midplane', 'Top', 'Inner Midplane'])

plt.title('Distribution of Field Line Intersections with Vacuum-Vessel Wall')
simOut.saveFig('Wallpoints_SeedPts_allPhi_2cm')


## END RUN ##
simOut.log.info('## SIM FINISHED ##\n\n\n\n')