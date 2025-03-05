import os
import numpy as np

import classes.class_outputHandler as out
from classes.mesh import Mesh
from utility.anlys_funcs import identifyLCFS
from solver.poincare_gen import Gen_Poincare

def main():

    ## SET UP RUN DIRECTORY (*DATA AND PLOTS WILL BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!*)
    simIO = out.IOHandler("Iota1q3_Fit1_Hel1p00_89at72deg_600DUB_rtol2p49e12_atol2p49e9") 
    simIO.startLog()

    ## DEFINE MESH AND LOAD FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.err_mag = 3.168E-4 
    b_hidra.err_dir = 268.6 * np.pi/180
    #b_hidra.loadCartesianField('input_files/It486_Ih900_Iv000_0p943_0p955.npy', errField=True)
    b_hidra.loadCartesianField('input_files/It486_Ih900_Iv000_0p943_1p00.npy', errField=True)

    ## SET UP INITIAL POSITIONS IN RTP COORDS
    NLINES = 12 + 11 + 22 + 44
    IC_RAD = np.array(np.linspace( 0.130, 0.020, NLINES)) # 0.130, 0.010
    IC_THETA = np.pi 
    IC_PHI = 72 * np.pi/180.

    ##CREATE INITIAL CONDITIONS ARRAY IN (IN RTP)
    ICs_RTP = np.array([[R, IC_THETA, IC_PHI] for R in IC_RAD])

    ## SET UP MAX LENGTH OF FIELD LINE TO INTEGRATE (1 SPIN=2*pi*R0)
    SPINS = 600

    ## SOLVER PARAMETERS
    SOLVER = 'LSODA'
    RTOL = 2.49e-12 #1.49e-8
    ATOL = 2.49e-9
    THREADS = os.cpu_count() - 1
    DOUBLE_LINE = True

    ## RUN POINCARE MAP
    tMax, Poincare_output, wallPt_output = Gen_Poincare(ICs_RTP, SPINS, b_hidra, simIO, 'Poincare',  
                                                        SOLVER, RTOL, ATOL, THREADS, DOUBLE_LINE)
    ## IDENTIFY LAST-CLOSED FLUX SURFACE
    LCFS_index = identifyLCFS(LCFStype='inner', iconds=IC_RAD, t_maxs=tMax, outputHandler=simIO)
   
    ## END RUN ##
    simIO.log.info('## SIM FINISHED ##\n\n\n\n')


if __name__ == '__main__':
    main()