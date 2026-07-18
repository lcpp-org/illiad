## IMPORTS
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

import classes.class_outputHandler as out
from classes.mesh import *

############################
## SET SIMULATION INPUTS: ##
############################
OUTPUT_DIRECTORY_NAME = "AcceptedIota3_1500spins_atole-9"

TAG = 'RLP_fitting_110625_Newprofile2'
# LCFS_INDEX = 61 # from Poincare output (simIO.log)

## SET UP RUN DIRECTORY AND LOGGING
## DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simIO = out.IOHandler(OUTPUT_DIRECTORY_NAME) 
simIO.startLog()

## DEFINE SCALAR FIELDS
Psi_hidra_linear = Mesh(R0=0.72, a=0.19)
#Psi_hidra_linear.loadScalarField('input_files/Efieldaccpt_0p965_linear.npy', period_=np.array([0, 1, 1]), att_mult=1.0, errField=False )
#Psi_hidra_linear.loadScalarField('input_files/FluxParm_acceptedSmoothed_linear3.npy', period_=np.array([0, 1, 1]), att_mult=1.0, errField=False )

Psi_hidra_linear.loadScalarField('input_files/big_grid_linearTEST.npy', period=np.array([0, 1, 1]))
#Psi_hidra_linear.loadScalarField('output/AcceptedIota3_1500spins_atole-9/data/LCFS19_360x180_tol_5e1_5e2_APS2025/big_grid_linear.npy', period=np.array([0, 1, 1]))

## LOAD RLP DATA FROM 'input_files/DataforSteve.csv'
RLP_DATA = np.genfromtxt('input_files/DataforSteve.csv', delimiter=',', skip_header=1)[:,1:]
RLP_DATA[:, 0] += 120 # RLP data starts at 120mm from the wall
RLP_DATA[:, 0] *= 1e-3 # convert from mm to m
RLP_DENSITY = RLP_DATA[:,1]
# print(RLP_DATA[:, 0])
# print(RLP_DENSITY)

## LEAST-SQUARES FIT
from numpy.polynomial import Polynomial
p = Polynomial.fit(RLP_DATA[:,0], RLP_DENSITY, 9)
PolyFit = p(RLP_DATA[:, 0])


PHI_GEN_RAD = np.radians(306.) # RLP phi-location
DIST_PLOT = np.arange(0.0, 0.38, 0.005) # RLP radius location
linear_profile = np.zeros(len(DIST_PLOT))
for i, dist in enumerate(DIST_PLOT):
    if dist < Psi_hidra_linear.a:
        theta = 0.0
        rad = Psi_hidra_linear.a - dist
    else:
        theta = np.pi
        rad = dist - Psi_hidra_linear.a

    rtp_point = np.array([rad, theta, PHI_GEN_RAD])
    linear_profile[i] = Psi_hidra_linear.interpScalarField(rtp_point, Cart=False)[0]


linear_data = np.array([DIST_PLOT, linear_profile])
scale_factor = np.max(PolyFit) # scale psi profiles to peak density
#plot_profile = linear_profile*scale_factor
#plot_profile = np.maximum(1e17, linear_profile*scale_factor)
plot_profile_adj = scale_factor*(1 - (1 - linear_profile)**(0.85))  # adjusted fit to better match RLP data shape
plot_profile_lin = scale_factor* linear_profile  # adjusted fit to better match RLP data shape
#save data as csv
np.savetxt("linear_profile.csv", linear_data.T, delimiter=",", header="Distance from Outer Wall [m], Psi", comments="")

plt.figure()
plt.plot(RLP_DATA[:,0] * 100, RLP_DENSITY, ':o', label='RLP Data', color='lightgrey', markersize=2.5, linewidth=0.8)
plt.plot(RLP_DATA[:,0] * 100, PolyFit, '-k', linewidth=2, label='RLP Polynomial Fit')  
plt.plot(DIST_PLOT * 100, plot_profile_lin, '--b', linewidth=1.5, label='Linear Prediction ($\\alpha=1$)')
plt.plot(DIST_PLOT * 100, plot_profile_adj, '--y', linewidth=1.5,  label='Prediction ($\\alpha=0.85$)')


plt.xlabel('Distance from Outer Wall [cm]', fontsize=10)
plt.xticks(np.arange(0, 39, 2))
plt.ylabel('Plasma Density [m$^{-3}$]', fontsize=10)
#plt.yscale('log')
plt.ylim(1e16, 5e18)
plt.xlim(0., 38)
#plt.title('Plasma Density, RLP Data and Psi Profiles')
plt.legend(loc='upper right', fontsize=8)
plt.grid(which='both')
plt.tick_params(axis='x', labelsize=8)
plt.tick_params(axis='y', labelsize=8)
simIO.saveFig(TAG + '.png', dpi=300)
#simIO.saveFig('RLP_Psi-fitting_profiles.png', dpi=300)
#plt.show()
