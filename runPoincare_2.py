####------------------------------------------------------####
#### GENERATING POINCARE PLOTS FOR HIDRA'S MAGNETIC FIELD ####
####------------------------------------------------------####
import os
import numpy as np

import classes.class_outputHandler as out
from classes.mesh import Mesh
from utility.anlys_funcs import identifyLCFS
from solver.poincare_gen import Gen_Poincare

#---------------#
# DEFINE FIELDS #
#---------------#
FIELD_FILE_TOR = 'input_files/It486_Ih000_Iv000_1p000_1p000_64bit.npy'
FIELD_SCALE_TOR = 0.9448
FIELD_FILE_HEL = 'input_files/It000_Ih900_Iv000_1p000_1p000_64bit.npy'
#FIELD_FILE_HEL = 'input_files/It000_Ih790_Iv000_1p000_1p000_64bit.npy'

# TOROIDAL AND HELICAL MAGNETIC FIELDS
TOROIDAL_CURRENT = 0.486 #[kA]
HELICAL_CURRENT = 0.790 #[kA]

FIELD_SCALE_HEL = -0.955 * FIELD_SCALE_TOR
ERRFIELD_MAG = 1.5654e-4 #[Tesla]
ERRFIELD_DIR_DEG = 271.5 #[degrees]




#---------------------------#
# DEFINE INITIAL CONDITIONS #
#---------------------------#
#IC_PHI_DEG = 324. #Accepted iota1/3
IC_PHI_DEG = 180. #Accepted iota1/4
IC_THETA_DEG = 180.
START_RAD = 0.140
END_RAD = 0.020
NLINES = 13 + 12 + 24 # + 44
SPINS = 250 # max length, SPIN = 2pi*R0 [meters]

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
NTHREADS = 28 #-1
DOUBLE_LINE = False

#-------------------------#
# DEFINE OUTPUT DIRECTORY #
#-------------------------#
OUTPUT_DIR = "AcceptedIota4_200spins_atole-9"
#OUTPUT_DIR = "DELETEME"


def main():

    ## SET UP RUN DIRECTORY (*DATA AND PLOTS WILL BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!*)
    simIO = out.IOHandler(OUTPUT_DIR) 
    simIO.startLog()

    ## DEFINE MESH AND LOAD FIELD
    # b_hidra = Mesh(R0=0.72, a=0.19)
    # b_hidra.loadCartesianField(FIELD_FILE_TOR, att_mult=FIELD_SCALE_TOR, errField=True )
    # b_hidra.addFieldPerturbation(FIELD_FILE_HEL, att_mult=FIELD_SCALE_HEL)
    # b_hidra.set_nonPer_errField(ERRFIELD_MAG, ERRFIELD_DIR_DEG*np.pi/180.)

    ## DEFINE MESH AND LOAD MAGNETIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField(coilCurrent=TOROIDAL_CURRENT, errField=True, att_mult='default_toroidal')
    b_hidra.addFieldPerturbation(coilCurrent=HELICAL_CURRENT, att_mult='default_helical')
    b_hidra.set_nonPer_errField()



    ## SET UP INIT CONDS
    ic_rad = np.array(np.linspace( START_RAD, END_RAD, NLINES))
    ic_theta = IC_THETA_DEG * np.pi/180.
    ic_phi = IC_PHI_DEG * np.pi/180.
    init_conds_rtp = np.array([[R, ic_theta, ic_phi] for R in ic_rad])

    ## GENERATE POINCARE PLOTS
    NTHREADS = 31 #os.cpu_count() - 1
    # if NTHREADS <= 0:
    #     NTHREADS = os.cpu_count() + NTHREADS
    # else:
    #     NTHREADS = NTHREADS
    
    solver_args = [SOLVER, RTOL, ATOL, NTHREADS, DOUBLE_LINE]
    tMax = Gen_Poincare(init_conds_rtp, SPINS, b_hidra, simIO, 'Poincare', *solver_args)[0]

    ## IDENTIFY LAST-CLOSED FLUX SURFACE
    identifyLCFS(LCFStype='inner', iconds=ic_rad, t_maxs=tMax, outputHandler=simIO)

    ## END RUN ##
    simIO.log.info('## SIM FINISHED ##\n\n\n\n')

if __name__ == '__main__':

    main()