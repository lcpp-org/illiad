import numpy as np
import scipy as sp
import glob

import class_outputHandler as out
from mesh import *
from coordtrans import *
#from ode import blines, solvePoincare

## SET UP RUN DIRECTORY ##
##======================##
# RIGHT NOW, DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simOut = out.IOHandler("FullTest_i3-errField_10")
simOut.startLog()
Nlines = 6+5
spins = 800


## DEFINE MESH AND LOAD FIELD ##
##============================##
b_hidra = CartesianField()
b_hidra.setToroidalGeometry(0.72, 0.19)
b_hidra.loadCartesianField_fromFile('Bxyz_i-1q4_hires_5Period.npy', 0, 1, 5)
#b_hidra.loadCartesianField_fromFile('Bxyz_negY_i-1q3_hires_5Period.npy', 0, 1, 5)
#b_hidra.loadCartesianField_fromFile('Bxyz_i-1q3_hires_5Period_IH-95p5pct.npy', 0, 1, 5)


## SET UP POINCARE SEED POINTS ##
##=============================##
## NEED A BETTER WAY TO SET UP INITIAL POINTS!
fl_R0     = np.array(np.linspace( 0.120, 0.070, Nlines))
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
from poincare_gen import Gen_Poincare
tMax, Poincare_output, wallPt_output = Gen_Poincare(b_hidra, ICs_XYZ, length, simOut, 'Poincare')
print(f'{tMax=}')

## IDENTIFY LAST-CLOSED FLUX SURFACE ##
##===================================##
maxTime = max(tMax)
LCFS_index = tMax.index(maxTime)

## IMPORT CORRESOPONDING POINCARE SURFACE ##
## GENERATE 'SEED SHELLS' OF IC's         ##
## EXPANDING CO-AXIALLY WITH LCFS         ##
##========================================##
simOut.log.info('GENERATING SEED POINTS:\n')

from point_generators import generateSeedShells

# structured to eventually generate seeds from the same flux surface at different phi angles
phiGen_list = [36] # list of phi angles to generated shells
#expand_dr = [ 0.002, 0.004, 0.008, 0.010]  # define number of 'shells' (dr) to generate
expand_dr = [ 0.004, 0.008, 0.010, 0.014]  # define number of 'shells' (dr) to generate
ntheta = 90 # number of equally-spaced theta points for each shell

seed_subset = []
seed_list = []
#seed_array = numpy.zeros(( 3, ntheta*len(phiGen_list) ))
#seed_array = np.empty(3)
for phi_gen_deg in phiGen_list:
    #phi_gen_deg = 36
    phi_gen = phi_gen_deg*(np.pi/180)

    filename = f'Poincare_output_{phi_gen_deg:03d}_{LCFS_index:d}.npy'
    th_in, r_in = simOut.loadNumpyData(filename)

    seed_subset = generateSeedShells(expand_dr, ntheta, r_in, th_in, phi_gen, b_hidra, simOut, 'SeedShell_test1')
    seed_list.extend(seed_subset)
    #np.vstack((seed_array, seed_subset))
seed_array = np.array(seed_list)
# need to +=, hstack, whatever into single list of IC's

## RE-RUN 'POINCARE' WITH SEED ARRAY ##
##===================================##
simOut.log.info('RE-RUNNING POINCARE PLOT GENERATOR WITH NEW SEED POINTS:\n')
spins = 500
length = (2*np.pi * b_hidra.R0) * spins

simOut.log.info(f'Initial Conditions (XYZ):\n{len(seed_array)} points')
tMax2, Poincare_output2, wallPt_output2 = Gen_Poincare(b_hidra, seed_array, length, simOut, 'SeedPts')

## POST-SOLVER OUTPUT
####################
wallPtArray = np.transpose( np.array(wallPt_output2) )
simOut.saveNumpyData(wallPtArray, 'Wallpoints')

phi_plot = wallPtArray[2]
theta_plot = wallPtArray[1]
simOut.log.info(f'Plotting wall hits. Total events = {wallPtArray[0].size}:\n')

import matplotlib.pyplot as plt
from matplotlib import cm

plt.figure()
plt.scatter(phi_plot, theta_plot, s=1)
plt.grid(True) 

plt.xlim(0., 2*np.pi)
plt.ylim(0., 2*np.pi)
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\theta$')
plt.xticks([1/2*np.pi, 2/2*np.pi, 3/2*np.pi, 4/2*np.pi], 
           [r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'] )
plt.yticks([1/2*np.pi, 2/2*np.pi, 3/2*np.pi, 4/2*np.pi], 
           [r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'] )

plt.title('Distribution of Field Line Intersection with Vacuum-Vessel Wall')
simOut.saveFig('Wall_Points')
#plt.show()

## END RUN
simOut.log.info('## SIM FINISHED ##\n\n\n\n')