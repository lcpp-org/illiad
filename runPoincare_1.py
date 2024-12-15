import numpy as np
import class_outputHandler as out
from mesh import *
from particle import *
from coordtrans import RTP_to_XYZ
from anlys_funcs import identifyLCFS
from poincare_gen import Gen_Poincare

def main():
    ## SET UP RUN DIRECTORY
    ## DEFINE MESH AND LOAD FIELD
    simOut = out.IOHandler("1q3_0p925_25lines_pp") #DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
    simOut.startLog()
    BX, BY, BZ = np.load('input_files/It486_Ih900_Iv000_0p925_hires.npy')


    mesh_prd = np.array([0, 1, 5], dtype=np.int32)
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField(BX, BY, BZ, mesh_prd, errField=True)


    ## SET UP INITIAL POSITIONS IN RTP COORDS
    NLINES = 13 + 12# + 24
    SPINS = 300

    IC_Rad = np.array(np.linspace( 0.130, 0.01, NLINES))
    IC_THETA = 0. #np.pi
    IC_PHI = 2*np.pi - (np.pi/5.) # at 324deg., plasma lines up nicely with midplane
    IC_PHI = 108. * np.pi / 180. # at 108deg., plasma lines up nicely with midplane (err-+)
    IC_PHI = np.pi # at 180deg., plasma lines up nicely with midplane (err++)
    #IC_PHI = np.pi * 6 / 5. # at 216deg., plasma lines up nicely with midplane(negative I_tor)

    ## SOLVER PARAMETERS
    SOLVER = 'LSODA' #'RK45'
    RTOL = 1e-6
    ATOL = 1e-16
    THREADS = 32

    ##CREATE INITIAL CONDITIONS ARRAY IN (IN RTP)
    ICs_RTP = np.array([[R, IC_THETA, IC_PHI] for R in IC_Rad]) #THETA=pi, r increasing towards high-field side
    ## CONVERT TO XYZ COORDS
    ICs_XYZ = np.zeros(shape=(NLINES, 3))
    for i in range(NLINES):
        ICs_XYZ[i] = RTP_to_XYZ(ICs_RTP[i], b_hidra.R0)

    # Print out a nicely-formatted boilerplate listing the parameters and their values as a table with a border
    simOut.log.info("+----------------+-------------------------+")
    simOut.log.info("| Parameter      | Value                   |")
    simOut.log.info("+----------------+-------------------------+")
    simOut.log.info(f"| SOLVER         | {SOLVER:<23} |")
    simOut.log.info(f"| RTOL           | {RTOL:<23} |")
    simOut.log.info(f"| ATOL           | {ATOL:<23} |")
    simOut.log.info(f"| THREADS        | {THREADS:<23} |")
    simOut.log.info("+----------------+-------------------------+")
    simOut.log.info(f"| NLINES         | {NLINES:<23} |")
    simOut.log.info(f"| SPINS          | {SPINS:<23} |")
    simOut.log.info("| Initial Conditions (RTP):                |")
    for ic in ICs_RTP:
        simOut.log.info(f"|     {str(ic):<23}   |")
    simOut.log.info("+----------------+-------------------------+")

    ## GENERATE POINCARE DATA
    length = (2*np.pi * b_hidra.R0) * SPINS
    fieldlines = [fieldLine(init_cond, length, direction = 1.0) for init_cond in ICs_XYZ]
    fieldlines += [fieldLine(init_cond, length, direction = -1.0) for init_cond in ICs_XYZ] #add fieldlines in opposite direction
    
    tMax, Poincare_output, wallPt_output = Gen_Poincare(b_hidra, fieldlines, simOut, 'Poincare', SOLVER, RTOL, ATOL, workers=THREADS)
    tMax = [tMax[i]+tMax[i+NLINES] for i in range(0,NLINES)]
    ## IDENTIFY LAST-CLOSED FLUX SURFACE
    LCFS_index = identifyLCFS(LCFStype='inner', iconds=IC_Rad, t_maxs=tMax, outputHandler=simOut)

    ## END RUN ##
    simOut.log.info('## SIM FINISHED ##\n\n\n\n')

if __name__ == '__main__':
    main()