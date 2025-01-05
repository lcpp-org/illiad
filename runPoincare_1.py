
import numpy as np

import classes.class_outputHandler as out
from classes.mesh import Mesh

from utility.anlys_funcs import identifyLCFS
from solver.poincare_gen import Gen_Poincare

def main():

    ## SET UP RUN DIRECTORY
    #* DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!! *#
    #simIO = out.IOHandler("It486_Ih900_Iv000_0p955_12lines_1500spins_rtol6") 
    simIO = out.IOHandler("It486_Ih900_Iv000_0p955_89lines_400spinsDOUBLED_rtol7") 
    simIO.startLog()

    ## DEFINE MESH AND LOAD FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField('input_files/It486_Ih900_Iv000_0p955_hires.npy', errField=True)
    simIO.log.info('Loaded field data from input_files/It486_Ih900_Iv000_0p955_hires.npy')

    ## SET UP INITIAL POSITIONS IN RTP COORDS
    NLINES = 12+11+22+44 #12+11+10+9 #13
    IC_RAD = np.array(np.linspace( 0.140, 0.020, NLINES)) # 0.130, 0.010
    IC_THETA = np.pi
    IC_PHI = 144. * np.pi/180.

    ## SET UP MAX LENGTH OF FIELD LINE TO INTEGRATE (1 SPIN=2*pi*R0)
    SPINS = 400

    ##CREATE INITIAL CONDITIONS ARRAY IN (IN RTP)
    ICs_RTP = np.array([[R, IC_THETA, IC_PHI] for R in IC_RAD])

    ## SOLVER PARAMETERS
    SOLVER = 'LSODA'
    RTOL = 1e-7
    ATOL = 1e-16
    THREADS = 31
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