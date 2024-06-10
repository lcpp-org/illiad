import numpy as np
import class_outputHandler as out
from mesh import *
from coordtrans import RTP_to_XYZ
from anlys_funcs import identifyLCFS
from poincare_gen import Gen_Poincare


## SET UP RUN DIRECTORY
simOut = out.IOHandler("1q4ERR_run002") #DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simOut.startLog()

## DEFINE MESH AND LOAD FIELD
BX, BY, BZ = np.load('input_files/Bxyz_i-1q4_hires_5Period_IH-95p5pct.npy')
mesh_prd = np.array([0, 1, 5], dtype=np.int32)

b_hidra = Mesh(R0=0.72, a=0.19)
b_hidra.loadCartesianField(BX, BY, BZ, mesh_prd)

## SET UP POINCARE SEED POINTS IN RTP COORDS
Nlines = 41
spins = 1500
length = (2*np.pi * b_hidra.R0) * spins

IC_R0 = np.array(np.linspace( 0.120, 0.020, Nlines))
ICs_RTP = np.array([[R, 0., 2*np.pi - (np.pi/5.)] for R in IC_R0])
simOut.log.info('Initial Conditions (RTP):\n{}'.format(ICs_RTP))

## CONVERT POINTS TO XYZ COORDS
ICs_XYZ = np.zeros(shape=(Nlines, 3))
for ic_rtp, ic_xyz in zip(ICs_RTP, ICs_XYZ):
    ic_xyz[:] = RTP_to_XYZ(ic_rtp, b_hidra.R0)

## GENERATE POINCARE DATA
tMax, Poincare_output, wallPt_output = Gen_Poincare(b_hidra, ICs_XYZ, length, simOut, 'Poincare', 'LSODA', 1e-8, 1e-32)

## IDENTIFY LAST-CLOSED FLUX SURFACE
LCFS_index = identifyLCFS(LCFStype='inner', iconds=IC_R0, t_maxs=tMax, outputHandler=simOut)

## END RUN ##
simOut.log.info('## SIM FINISHED ##\n\n\n\n')