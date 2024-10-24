import numpy as np
import class_outputHandler as out
from mesh import *
from particle import *
from coordtrans import RTP_to_XYZ
from anlys_funcs import identifyLCFS
from poincare_gen import Gen_Poincare

## SET UP RUN DIRECTORY
simOut = out.IOHandler("HIDRA_1q3ERR_dphi1deg") #DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simOut.startLog()

## DEFINE MESH AND LOAD FIELD
BX, BY, BZ = np.load('input_files/HIDRA_i3ERR_hires.npy')
mesh_prd = np.array([0, 1, 5], dtype=np.int32)
b_hidra = Mesh(R0=0.72, a=0.19)
b_hidra.loadCartesianField(BX, BY, BZ, mesh_prd, errField=True)

## SET UP INITIAL POSITIONS IN RTP COORDS
Nlines = 16+15
IC_Rad = np.array(np.linspace( 0.170, 0.02, Nlines))
ICs_RTP = np.array([[R, np.pi, 2*np.pi - (np.pi/5.)] for R in IC_Rad]) #THETA=pi, r increasing towards high-field side

## CONVERT TO XYZ COORDS
ICs_XYZ = np.zeros(shape=(Nlines, 3))
for i in range(Nlines):
    ICs_XYZ[i] = RTP_to_XYZ(ICs_RTP[i], b_hidra.R0)

## GENERATE POINCARE DATA
simOut.log.info('Initial Conditions (RTP):\n{}'.format(ICs_RTP))
spins = 1000
length = (2*np.pi * b_hidra.R0) * spins
fieldlines = [fieldLine(init_cond, length) for init_cond in ICs_XYZ]
tMax, Poincare_output, wallPt_output = Gen_Poincare(b_hidra, fieldlines, simOut, 'Poincare', 'LSODA', 1e-7, 1e-32)

## IDENTIFY LAST-CLOSED FLUX SURFACE
LCFS_index = identifyLCFS(LCFStype='inner', iconds=IC_Rad, t_maxs=tMax, outputHandler=simOut)

## END RUN ##
simOut.log.info('## SIM FINISHED ##\n\n\n\n')