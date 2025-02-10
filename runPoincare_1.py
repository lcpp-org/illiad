
import numpy as np

import classes.class_outputHandler as out
from classes.mesh import Mesh

from utility.anlys_funcs import identifyLCFS
from solver.poincare_gen import Gen_Poincare

def main():

    ## SET UP RUN DIRECTORY (*DATA AND PLOTS WILL BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!*)
    simIO = out.IOHandler("FitCheck_020925_hel-0p950_45l_300s_rtol6") 
    
    simIO.startLog()

    ## DEFINE MESH AND LOAD FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    #b_hidra.loadCartesianField('input_files/It486_Ih900_Iv000_0p955_ATTEN_hires.npy', errField=True)
    #b_hidra.loadCartesianField('input_files/It486_Ih900_Iv000_1p0_hires.npy', errField=True)
    #b_hidra.loadCartesianField('input_files/It486_Ih900_Iv000_0p955_hires.npy', errField=True)
    #b_hidra.loadCartesianField('input_files/It486_Ih900_Iv000_ALL0p9670_Hel0p955.npy', errField=True)
    #b_hidra.loadCartesianField('input_files/It000_Ih2000_Iv000_ALL1p0.npy', errField=True)
    #b_hidra.loadCartesianField('input_files/FITTED_02092025.npy', errField=True)

    b_hidra.loadCartesianField('input_files/FITTED_02092025_hel-0p950.npy', errField=True)
    b_hidra.err_mag = 0.0003167943268031759
    b_hidra.err_dir = 268.6154452940438 * np.pi/180

    ## SET UP INITIAL POSITIONS IN RTP COORDS
    NLINES = 12+11 + 22 #+44 #12+11+10+9 #13
    IC_RAD = np.array(np.linspace( 0.140, 0.020, NLINES)) # 0.130, 0.010
    IC_THETA = np.pi 
    IC_PHI = np.pi * 216./180

    ## SET UP MAX LENGTH OF FIELD LINE TO INTEGRATE (1 SPIN=2*pi*R0)
    SPINS = 300

    ##CREATE INITIAL CONDITIONS ARRAY IN (IN RTP)
    ICs_RTP = np.array([[R, IC_THETA, IC_PHI] for R in IC_RAD])

    ## SOLVER PARAMETERS
    SOLVER = 'LSODA'
    RTOL = 1e-6
    ATOL = 1e-16
    THREADS = 31
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