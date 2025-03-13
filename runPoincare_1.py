import os
import numpy as np

import classes.class_outputHandler as out
from classes.mesh import Mesh
from utility.anlys_funcs import identifyLCFS
from solver.poincare_gen import Gen_Poincare

def main():

    ## SET UP RUN DIRECTORY (*DATA AND PLOTS WILL BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!*)
    simIO = out.IOHandler("9deg_1q3_40p_40s_new_threads") 
    simIO.startLog()

    ## DEFINE MESH AND LOAD FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.err_mag = 3.168E-4 
    b_hidra.err_dir = 270 * np.pi/180
    #b_hidra.loadCartesianField('input_files/It486_Ih900_Iv000_0p943_0p955.npy', errField=True)
    b_hidra.loadCartesianField('input_files/i1q3_hires.npy', errField=True)

    ## SET UP INITIAL POSITIONS IN RTP COORDS
    NLINES = 40
    IC_RAD = np.array(np.linspace( 0.120, 0.020, NLINES)) # 0.130, 0.010
    IC_THETA = 0#np.pi 
    IC_PHI = 2*np.pi - (np.pi/5)#72 * np.pi/180.

    ##CREATE INITIAL CONDITIONS ARRAY IN (IN RTP)
    ICs_RTP = np.array([[R, IC_THETA, IC_PHI] for R in IC_RAD])

    ## SET UP MAX LENGTH OF FIELD LINE TO INTEGRATE (1 SPIN=2*pi*R0)
    SPINS = 40

    ## SOLVER PARAMETERS
    SOLVER = 'LSODA'
    RTOL = 1e-7#2.49e-12 #1.49e-8
    ATOL = 1e-32#2.49e-9
    THREADS = 40#os.cpu_count() - 1
    DOUBLE_LINE = False

    ## RUN POINCARE MAP
    tMax, Poincare_output, wallPt_output = Gen_Poincare(ICs_RTP, SPINS, b_hidra, simIO, 'Poincare',  
                                                        SOLVER, RTOL, ATOL, THREADS, DOUBLE_LINE)
    ## IDENTIFY LAST-CLOSED FLUX SURFACE
    LCFS_index = identifyLCFS(LCFStype='inner', iconds=IC_RAD, t_maxs=tMax, outputHandler=simIO)
   
    ## END RUN ##
    simIO.log.info('## SIM FINISHED ##\n\n\n\n')


if __name__ == '__main__':
    main()