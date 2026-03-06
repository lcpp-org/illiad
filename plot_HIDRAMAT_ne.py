## IMPORTS
import numpy as np
from numpy.polynomial import Polynomial
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import classes.class_outputHandler as out
from classes.mesh import *


def main():
    ## SET UP RUN DIRECTORY AND LOGGING
    simIO = out.IOHandler(OUTPUT_DIRECTORY_NAME) 
    simIO.startLog()
    ## LOAD SCALAR FIELDS FOR ALL CONDITIONS
    ne_mesh_period = np.array([0, 1, 1]) # meshes defined over full phi domain, so period in phi is 1 (not 5)
    psi_iota3_dflt = Mesh(R0=0.72, a=0.19)
    psi_iota3_dflt.loadScalarField('input_files/big_grid_linearTEST.npy', period=ne_mesh_period)
    psi_iota3_rev = Mesh(R0=0.72, a=0.19)
    psi_iota3_rev.loadScalarField('input_files/psiNorm_i3rev.npy', period=ne_mesh_period)
    psi_iota4_dflt = Mesh(R0=0.72, a=0.19)
    psi_iota4_dflt.loadScalarField('input_files/psiNorm_i4dflt.npy', period=ne_mesh_period)
    psi_iota4_rev = Mesh(R0=0.72, a=0.19)
    psi_iota4_rev.loadScalarField('input_files/psiNorm_i4rev.npy', period=ne_mesh_period)

    ## GENERATE PSI PROFILES ALONG RLP PATH
    PHI_GEN_RAD = np.radians(90.) # HIDRA-MAT location, # np.radians(306.) # RLP location

    DIST_PLOT = np.arange(0.0, 0.38, 0.005) * 100. # RLP radius location
    iota3_dflt_profile = np.zeros(len(DIST_PLOT))
    iota3_rev_profile = np.zeros(len(DIST_PLOT))
    iota4_dflt_profile = np.zeros(len(DIST_PLOT))
    iota4_rev_profile = np.zeros(len(DIST_PLOT))
    for i, dist in enumerate(DIST_PLOT):
        if (dist/100.) < psi_iota3_dflt.a:
            theta = 0.0
            rad = psi_iota3_dflt.a - (dist/100.)
        else:
            theta = np.pi
            rad = (dist/100.) - psi_iota3_dflt.a

        rtp_point = np.array([rad, theta, PHI_GEN_RAD])

        iota3_dflt_profile[i] = psi_iota3_dflt.interpScalarField(rtp_point, Cart=False)[0]
        iota3_rev_profile[i] = psi_iota3_rev.interpScalarField(rtp_point, Cart=False)[0]
        iota4_dflt_profile[i] = psi_iota4_dflt.interpScalarField(rtp_point, Cart=False)[0]
        iota4_rev_profile[i] = psi_iota4_rev.interpScalarField(rtp_point, Cart=False)[0]

    # plotting
    for cond in CONDITIONS:
        if cond == 'iota3_dflt':
            this_profile = iota3_dflt_profile
            this_label = 'iota3_dflt Psi Profile'
        elif cond == 'iota3_rev':
            this_profile = iota3_rev_profile
            this_label = 'iota3_rev Psi Profile'
        elif cond == 'iota4_dflt':
            this_profile = iota4_dflt_profile
            this_label = 'iota4_dflt Psi Profile'
        elif cond == 'iota4_rev':
            this_profile = iota4_rev_profile
            this_label = 'iota4_rev Psi Profile'

        reshaped_psi = (1 - (1 - this_profile)**(INPUT_ALPHA)) 
        #print(f'{reshaped_psi=}' )
        print(f'{PEAK_NE=}')
        scaled_profile = reshaped_psi * PEAK_NE

        plt.figure()
        plt.plot(DIST_PLOT, scaled_profile, ':b', linewidth=1.5, label=this_label)
        plt.xticks(np.arange(0, 39, 2))
        plt.xlabel('Distance from Outer Wall [cm]', fontsize=10)
        plt.ylabel('$n_e$ [m$^{-3}$]', fontsize=10)
        plt.legend(loc='upper right', fontsize=8)
        plt.grid(which='both')
        plt.tick_params(axis='both', labelsize=8)
        simIO.saveFig(cond + '_' + TAG + '_psi_profile.png', dpi=300)

        # save data as csv
        output_data = np.hstack((DIST_PLOT.reshape(-1, 1), scaled_profile.reshape(-1, 1)))
        simIO.saveCSV(output_data, cond+'_output.csv', header='d [cm], ne_norm')


if __name__ == "__main__":
    ## SET SIMULATION INPUTS:
    CONDITIONS = ['iota3_dflt', 'iota3_rev', 'iota4_dflt', 'iota4_rev']
    DATA_PATH = Path("input_files") / "HIDRAMAT_Results"
    POS_COL = "Position (cm)"
    NE_COL = "ne (m-3)"
    TOTAL_POS_COL = "Total distance (cm)"

    PEAK_NE = 1e18
    INPUT_ALPHA = 1.0 # manual value of alpha psi profile scaling exponent
    OUTPUT_DIRECTORY_NAME = "HIDRAMAT_density_profiles"
    TAG = 'alpha1p0'

    main()