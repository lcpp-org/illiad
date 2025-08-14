"""
####------------------------------------------------------####
#### GENERATING POINCARE PLOTS FOR HIDRA'S MAGNETIC FIELD ####
####------------------------------------------------------####
"""
import numpy as np
import matplotlib.pyplot as plt

from classes.iohandler import IOHandler
from classes.mesh import Mesh
from classes.poincare import Poincare

# DEFINE OUTPUT DIRECTORY #
OUTPUT_DIR = "Iota4-dflt_1500spins_89Lines_RK45"

# DEFINE FIELDS #
"""
# COIL CURRENTS NORMALLY RUN ON HIDRA
###########################################
#      | [Amp]| [Amp]| [Amp]| [deg]       #
# IOTA |  I_T |  I_H |  I_V | PHI FWD/REV #
# 1/3  |  486 |  900 |   00 | 324/???     #
# 1/4  |  486 |  790 |   00 | 180/144     #
# 1/5  |  486 |  710 |   00 | 360/???     #
# 1/7  |  581 |  581 |   00 | ???/???     #
# MAX. | 3500 | 7000 |   ?? | ???/???     #
###########################################
"""
CURRENT_TOR = 0.486 #[kA]
CURRENT_HEL = 0.790 #[kA]
CONFIG_TOR = 'default_toroidal'
CONFIG_HEL = 'default_helical'

# DEFINE INITIAL CONDITIONS #
IC_PHI_DEG = 180. #Accepted iota1/4
IC_THETA_DEG = 180.
START_RADIUS = 0.130
END_RADIUS = 0.020
#NLINES = 14 + 13 + 26 + 52 + 104
NLINES = 12 + 11 + 22 + 44 #+ 88
SPINS = 1500 # max length, SPIN = 2pi*R0 [meters]

# DEFINE SOLVER PARAMETERS #
"""
##  NTHREADS > 0: use N threads
##  NTHREADS = 0: use all available threads
##  NTHREADS < 0: use all but the last N threads
# DOUBLE_LINE:
##  True: run each fieldline in both directions from the init pos 
##        !ONLY USE WHEN (NTHREADS > NLINES)!
##  False: run each fieldline in +B direction from the init pos
"""
#SOLVER = 'LSODA'
SOLVER = 'RK45'
RTOL = 2.49e-12
ATOL = 2.49e-8
NTHREADS = -1
DOUBLE_LINE = False


def main():
    """
    Main function to set up the mesh, load magnetic field data, and generate Poincare plots.
    """
    ## SET UP RUN DIRECTORY (*DATA AND PLOTS WILL BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!*)
    simIO = IOHandler(OUTPUT_DIR) 
    simIO.startLog()

    ## DEFINE MESH AND LOAD MAGNETIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField(coilCurrent=CURRENT_TOR, errField=True, att_mult=CONFIG_TOR)
    b_hidra.set_nonPer_errField()
    b_hidra.addFieldPerturbation(coilCurrent=CURRENT_HEL, att_mult=CONFIG_HEL)

    ## SET UP INITIAL CONDITIONS
    ic_radii = np.array(np.linspace(START_RADIUS, END_RADIUS, NLINES))
    ic_theta = IC_THETA_DEG * np.pi/180.
    ic_phi = IC_PHI_DEG * np.pi/180.
    init_conds_rtp = np.array([[ic_r, ic_theta, ic_phi] for ic_r in ic_radii])

    ## GENERATE POINCARE PLOTS
    solver_args = [SOLVER, RTOL, ATOL, NTHREADS, DOUBLE_LINE]
    PoinCare = Poincare(simIO, *solver_args)
    PoinCare.set_conditions(init_conds_rtp, SPINS, b_hidra)
    out_tMax = PoinCare.run()[0]
    ## IDENTIFY LAST-CLOSED FLUX SURFACE
    PoinCare.identifyLCFS(LCFStype='inner', t_maxs=out_tMax)

    ## END RUN ##
    simIO.log.info('## SIM FINISHED ##\n\n\n\n')

if __name__ == '__main__': main()