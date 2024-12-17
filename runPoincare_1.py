
import numpy as np

import classes.class_outputHandler as out
from classes.mesh import Mesh

from utility.anlys_funcs import identifyLCFS
from solver.poincare_gen import Gen_Poincare

def main():

    ## SET UP RUN DIRECTORY
    simOut = out.IOHandler("It486_Ih900_Iv000_0p955_rtol6_2x500spins") #DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
    simOut.startLog()

    ## DEFINE MESH AND LOAD FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField('input_files/It486_Ih900_Iv000_0p955_hires.npy', errField=True)


    ## SET UP INITIAL POSITIONS IN RTP COORDS
    NLINES = 13 + 12# + 24
    IC_Rad = np.array(np.linspace( 0.130, 0.01, NLINES))

    IC_THETA = 0. #np.pi
    IC_PHI = np.pi #2*np.pi - (np.pi/5.) # at 324deg., plasma lines up nicely with midplane

    ## SET UP MAX LENGTH OF FIELD LINE TO INTEGRATE (1 SPIN=2*pi*R0)
    SPINS = 500

    ##CREATE INITIAL CONDITIONS ARRAY IN (IN RTP)
    ICs_RTP = np.array([[R, IC_THETA, IC_PHI] for R in IC_Rad]) #THETA=pi, r increasing towards high-field side

    ## SOLVER PARAMETERS
    SOLVER = 'LSODA' #'RK45'
    RTOL = 1e-6
    ATOL = 1e-16
    THREADS = 32
    DOUBLE_LINE = True

    ## RUN POINCARE MAP
    tMax, Poincare_output, wallPt_output = Gen_Poincare(ICs_RTP, SPINS, b_hidra, simOut, 'Poincare',  
                                                        SOLVER, RTOL, ATOL, THREADS, DOUBLE_LINE)

    ## IDENTIFY LAST-CLOSED FLUX SURFACE
    LCFS_index = identifyLCFS(LCFStype='inner', iconds=IC_Rad, t_maxs=tMax, outputHandler=simOut)

    ## END RUN ##
    simOut.log.info('## SIM FINISHED ##\n\n\n\n')


if __name__ == '__main__':
    main()