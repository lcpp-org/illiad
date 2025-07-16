####------------------------------------------------------####
#### GENERATING POINCARE PLOTS FOR HIDRA'S MAGNETIC FIELD ####
####------------------------------------------------------####
import os
import numpy as np
import matplotlib.pyplot as plt

import classes.class_outputHandler as out
from classes.mesh import Mesh
from classes.poincare import Poincare
from utility.anlys_funcs import identifyLCFS

#-------------------------#
# DEFINE OUTPUT DIRECTORY #
#-------------------------#
OUTPUT_DIR = "AcceptedIota5_400spins_49Lines"

#---------------#
# DEFINE FIELDS #
#---------------#
CURRENT_TOR = 0.486 #[kA]
CURRENT_HEL = 0.710 #[kA]
CONFIG_TOR = 'default_toroidal'
CONFIG_HEL = 'default_helical'

#---------------------------#
# DEFINE INITIAL CONDITIONS #
#---------------------------#
#IC_PHI_DEG = 324. #Accepted iota1/3
#IC_PHI_DEG = 180. #Accepted iota1/4
IC_PHI_DEG = 360. #Accepted iota1/5

#IC_PHI_DEG = ?? #ChangeTo iota1/3
#IC_PHI_DEG = 144. #ChangeTo iota1/4
#IC_PHI_DEG = ?? #ChangeTo iota1/5

IC_THETA_DEG = 180.
START_RADIUS = 0.140
END_RADIUS = 0.020
NLINES = 13 + 12 + 24 #+ 48
SPINS = 400 # max length, SPIN = 2pi*R0 [meters]

#-------------------#
# SOLVER ARGUMENTS #
#-------------------#
"""
# NTHREADS:
##  N > 0: use N threads
##  N = 0: use all available threads
##  N < 0: use all but the last N threads
# DOUBLE_LINE:
##  True: run each fieldline in both directions from the init pos 
##        !ONLY USE WHEN (NTHREADS > NLINES)!
##  False: run each fieldline in +B direction from the init pos
"""
SOLVER = 'LSODA'
RTOL = 2.49e-12
ATOL = 2.49e-8
NTHREADS = 31 #-1
DOUBLE_LINE = False


def main():
    """Main function to set up the mesh, load magnetic field data, and generate Poincare plots."""

    ## SET UP RUN DIRECTORY (*DATA AND PLOTS WILL BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!*)
    simIO = out.IOHandler(OUTPUT_DIR) 
    simIO.startLog()

    ## DEFINE MESH AND LOAD MAGNETIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField(coilCurrent=CURRENT_TOR, errField=True, att_mult=CONFIG_TOR)
    b_hidra.set_nonPer_errField()
    b_hidra.addFieldPerturbation(coilCurrent=CURRENT_HEL, att_mult=CONFIG_HEL)

    ## SET UP INITIAL CONDITIONS
    ic_rad = np.array(np.linspace(START_RADIUS, END_RADIUS, NLINES))
    ic_theta = IC_THETA_DEG * np.pi/180.
    ic_phi = IC_PHI_DEG * np.pi/180.
    init_conds_rtp = np.array([[R, ic_theta, ic_phi] for R in ic_rad])

    ## GENERATE POINCARE PLOTS
    solver_args = [SOLVER, RTOL, ATOL, NTHREADS, DOUBLE_LINE]

    PoinCare = Poincare(simIO, *solver_args)
    PoinCare.set_conditions(init_conds_rtp, SPINS, b_hidra)
    tMax = PoinCare.run()[0]

    ## IDENTIFY LAST-CLOSED FLUX SURFACE
    identifyLCFS(LCFStype='inner', iconds=ic_rad, t_maxs=tMax, outputHandler=simIO)

    ## END RUN ##
    simIO.log.info('## SIM FINISHED ##\n\n\n\n')

if __name__ == '__main__': main()