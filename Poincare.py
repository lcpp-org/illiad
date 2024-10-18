import numpy as np
import scipy as sp
#import glob
import matplotlib.pyplot as plt
from matplotlib import patches

import class_outputHandler as out
from mesh import *
from coordtrans import *
from anlys_funcs import *
from poincare_gen import Gen_Poincare


## SET UP RUN DIRECTORY ##
##======================##
simOut = out.IOHandler("1q3ERR_wtfTesting1") #DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simOut.startLog()


## DEFINE MESH AND LOAD FIELD ##
##============================##
#b_hidra = CartesianField()
b_hidra = Mesh()
b_hidra.setToroidalGeometry(0.72, 0.19)

BX, BY, BZ = np.load('input_files/Bxyz_i-1q3_hires_5Period_IH-95p5pct.npy')
simOut.log.info('Input  data type:{}'.format(type(BX)))
simOut.log.info('Input Array data type:{}'.format(BX.dtype))

b_hidra.loadCartesianField(BX, BY, BZ, np.array([0, 1, 5], dtype=np.int32))


## SET UP POINCARE SEED POINTS ##
##=============================##
Nlines = 41
spins = 1500

#l_R0     = np.array(np.linspace( 0.140, 0.065, Nlines))
fl_R0     = np.array(np.linspace( 0.120, 0.020, Nlines))
fl_THETA0 = np.zeros(Nlines)
fl_PHI0   = np.ones(Nlines) * (2*np.pi - (np.pi/5.))

ICs_RTP = np.transpose(np.vstack([fl_R0, fl_THETA0, fl_PHI0]))
ICs_XYZ = np.zeros(shape=(Nlines, 3))
for i in range(Nlines):
    ICs_XYZ[i] = RTP_to_XYZ(ICs_RTP[i], b_hidra.R0)

length = (2*np.pi * b_hidra.R0) * spins
simOut.log.info('Initial Conditions (RTP):\n{}'.format(ICs_RTP))


## GENERATE POINCARE DATA ##
##==============================##
tMax, Poincare_output, wallPt_output = Gen_Poincare(b_hidra, ICs_XYZ, length, simOut, True, 'Poincare', 'LSODA', 1e-7, 1e-14)


## IDENTIFY LAST-CLOSED FLUX SURFACE ##
##===================================##
LCFS_index = identifyLCFS(LCFStype='inner', 
                          iconds=fl_R0, 
                          t_maxs=tMax, 
                          outputHandler=simOut)

#LCFS_index = identifyLCFS(LCFStype='input', 
#                          num=14, 
#                          outputHandler=simOut)

simOut.log.info('LCFS_index = {}'.format(LCFS_index))

## GENERATE 'SEED SHELLS' OF IC's ##
## EXPANDING CO-AXIALLY WITH LCFS ##
##================================##
simOut.log.info('GENERATING SEED POINTS:\n')
from point_generators import generateSeedShells

phiGen_list = np.linspace(9, 360, 40, dtype=int).tolist() # list of phi angles to generated shells
expand_dr   = [0.030]                                     # define number of 'shells' (delta-r) to generate
ntheta      = 90                                          # number of equally-spaced theta points for each shell

# generate seeds from the same flux surface at different phi angles
seed_subset = []
seed_list = []
for phi_gen_deg in phiGen_list:
    filename = 'Poincare_{:03d}.npy'.format(phi_gen_deg)
    th_in, r_in = simOut.loadNumpyData(filename)[LCFS_index]

    phi_gen = phi_gen_deg*(np.pi/180)
    seed_subset = generateSeedShells(expand_dr, ntheta, r_in, th_in, phi_gen, 
        b_hidra, simOut, 'SeedPts_{:.0f}mm'.format(expand_dr[0]*1000))
    seed_list.extend(seed_subset)

seed_array = np.array(seed_list)


## RE-RUN 'POINCARE' WITH SEED ARRAY ##
##===================================##
simOut.log.info('RE-RUNNING POINCARE PLOT GENERATOR WITH NEW SEED POINTS:\n')
spins = 350
length = (2*np.pi * b_hidra.R0) * spins

simOut.log.info('Initial Conditions (XYZ):\n{} points'.format(len(seed_array)))

subName = 'SeedPts_{:.0f}mm'.format(expand_dr[0]*1000)
tMax2, Poincare_output2, wallPt_output2 = Gen_Poincare(b_hidra, seed_array, length, simOut, False, subName, 'RK45', 1e-8, 1e-14)



## ================== ##
## POST-SOLVER OUTPUT ##
## ================== ##
wallPtArray = np.transpose( np.array(wallPt_output2) ) 
simOut.saveNumpyData(wallPtArray, 'Wallpoints_{:.0f}mm'.format(expand_dr[0]*1000))

phi_plot = wallPtArray[2]*(-1) + 2*np.pi

theta_plot = wallPtArray[1]
for i in range(len(theta_plot)):
    if theta_plot[i]>np.pi: theta_plot[i] -= 2*np.pi

simOut.log.info('Plotting wall hits. Total events = {}:\n'.format(wallPtArray[0].size))

## Import data on port size/locations for plotting
#ports = simOut.loadPorts_fromCSV('input_files/HIDRA_ports.csv')

plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'figure.autolayout':True})

fig = plt.figure()
ax = fig.add_subplot(polar=False, aspect=0.2)

# Plot HIDRA ports
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

simOut.saveFig('Wallpoints_SeedPts_{:.0f}mm'.format(expand_dr[0]*1000))

## END RUN ##
simOut.log.info('## SIM FINISHED ##\n\n\n\n')