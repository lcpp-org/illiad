import numpy as np
import class_outputHandler as out
from mesh import *
from particle import *
from coordtrans import RTP_to_XYZ
from anlys_funcs import identifyLCFS
from poincare_gen import Gen_Poincare
#from point_generators import generateSeedShells


## SET UP RUN DIRECTORY
simOut = out.IOHandler("HIDRA-1q3-ERR_particles_5") #DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
simOut.startLog()

## DEFINE MESH AND LOAD FIELD
BX, BY, BZ = np.load('input_files/Bxyz_i-1q3_hires_5Period_IH-95p5pct.npy')
mesh_prd = np.array([0, 1, 5], dtype=np.int32)
b_hidra = Mesh(R0=0.72, a=0.19)
b_hidra.loadCartesianField(BX, BY, BZ, mesh_prd, errField=True)

## SET UP POINCARE SEED POINTS IN RTP COORDS
Nlines = 21
spins = 1000
length = (2*np.pi * b_hidra.R0) * spins

fl_R0     = np.array(np.linspace( 0.120, 0.020, Nlines))
fl_THETA0 = np.zeros(Nlines)
fl_PHI0   = np.ones(Nlines) * (2*np.pi - (np.pi/5.))

ICs_RTP = np.transpose(np.vstack([fl_R0, fl_THETA0, fl_PHI0]))
ICs_XYZ = np.zeros(shape=(Nlines, 3))
for i in range(Nlines):
    ICs_XYZ[i] = RTP_to_XYZ(ICs_RTP[i], b_hidra.R0)
fieldlines = [fieldLine(init_cond, length) for init_cond in ICs_XYZ]

## GENERATE POINCARE DATA
simOut.log.info('Initial Conditions (RTP):\n{}'.format(ICs_RTP))
tMax, Poincare_output, wallPt_output = Gen_Poincare(b_hidra, fieldlines, simOut, 'Poincare', 'LSODA', 1e-6, 1e-32)

## IDENTIFY LAST-CLOSED FLUX SURFACE
LCFS_index = identifyLCFS(LCFStype='inner', iconds=fl_R0, t_maxs=tMax, outputHandler=simOut)

## END RUN ##
simOut.log.info('## SIM FINISHED ##\n\n\n\n')