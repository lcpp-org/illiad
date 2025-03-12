import numpy as np
import class_outputHandler as out
from mesh import *
from particle import *
from coordtrans import RTP_to_XYZ
from anlys_funcs import identifyLCFS
from poincare_gen import Gen_Poincare
import time


## SET UP RUN DIRECTORY
simOut = out.IOHandler("9deg_1q3_40p_40s_times") #DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simOut.startLog()

## DEFINE MESH AND LOAD FIELD
BX, BY, BZ = np.load('input_files/i1q3_hires.npy')
mesh_prd = np.array([0, 1, 5], dtype=np.int32)
b_hidra = Mesh(R0=0.72, a=0.19)
b_hidra.loadCartesianField(BX, BY, BZ, mesh_prd, errField=True)

## SET UP INITIAL POSITIONS IN RTP COORDS
Nlines = 40 #21
Angles = 1
spins = 40
length = (2*np.pi * b_hidra.R0) * spins
IC_Rad = np.array(np.linspace( 0.120, 0.020, Nlines))
IC_THETA = np.array(np.linspace(0, np.pi, Angles))
ICs_RTP = np.array([[R, T, 2*np.pi - (np.pi/5.)] for R in IC_Rad for T in IC_THETA])

## CONVERT TO XYZ COORDS
ICs_XYZ = np.zeros(shape=(Nlines*Angles, 3))
for i in range(Nlines*Angles):
    ICs_XYZ[i] = RTP_to_XYZ(ICs_RTP[i], b_hidra.R0)

## GENERATE POINCARE DATA
toc = time.time()

simOut.log.info('Initial Conditions (RTP):\n{}'.format(ICs_RTP))
fieldlines = [fieldLine(init_cond, length) for init_cond in ICs_XYZ]
tMax, Poincare_output, wallPt_output = Gen_Poincare(b_hidra, fieldlines, simOut, 'Poincare', 'LSODA', 1e-7, 1e-32)

## IDENTIFY LAST-CLOSED FLUX SURFACE
LCFS_index = identifyLCFS(LCFStype='inner', iconds=IC_Rad, t_maxs=tMax[::Angles], outputHandler=simOut)

tic = time.time()

## END RUN ##
simOut.log.info(f'Time for plotting = {tic-toc}')
simOut.log.info('## SIM FINISHED ##\n\n\n\n')