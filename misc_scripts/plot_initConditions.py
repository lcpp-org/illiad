"""
Runs the initialization pipeline for ion tracking in a magnetic and electric field.

This function performs the following steps:
  1. Sets up the output directory and logging.
  2. Calculates the number of emitters and particles.
  3. Initializes mesh objects for magnetic and electric fields and loads field data.
  4. Initializes ion properties and their initial positions/velocities.
  5. Saves the initial conditions to disk.
All configuration and physical parameters are expected to be defined in the global scope or imported modules.
"""

## IMPORTS
import os
import sys
# Allow running from any subdirectory: resolve the project root relative to this file
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)


import numpy as np
from time import perf_counter

from illiad.io import IOHandler
from illiad.mesh import TorchMesh as Mesh
from illiad.boris import Boris

from illiad.utilities.coordtrans import *
from illiad.utilities.point_generators import ionInitializer, generateSeedShells

## SOME PHYSICAL CONSTANTS
kg_per_amu = 1.660_539_068E-27
kboltz = 1.602_176_634E-19 # Joules/eV
Li_mass = 6.941 #amu
He_mass = 4.002602 #amu

## SET SIMULATION INPUTS:
# ANALYSIS DIRECTORY AND UNIQUE OUTPUT TAG
#OUTPUT_DIRECTORY_NAME = "AcceptedIota4_1500spins_atole-8_eng"
OUTPUT_DIRECTORY_NAME = "AcceptedIota3_1500spins_atole-9"
TAG = "IC_PLOT_IOTA3"
# TOROIDAL AND HELICAL MAGNETIC FIELDS
TOROIDAL_CURRENT = 0.486 #[kA]
HELICAL_CURRENT = 0.900 #[kA]
CONFIG_TOR = 'default_toroidal'
CONFIG_HEL = 'default_helical'

# ELECTRIC FIELD
#FIELD_FILE_ELECTRIC = 'input_files/Efield_AcceptedIota3_lcfs35.npy'
FIELD_FILE_ELECTRIC = 'input_files/Efield_acceptedSmoothed_linear_3.npy'
FIELD_SCALE_ELECTRIC = 60.0 # [Volts]

# INITIAL CONDITIONS
LCFS_INDEX = 35 # (from Poincare output (simIO.log))
DELTRS = [0.000] # [m]
NPHI = 180
NTHETA = 15



## RUN SIMULATION:
def main():
    ## SET UP RUN DIRECTORY AND LOGGING
    simIO = IOHandler(OUTPUT_DIRECTORY_NAME) 
    simIO.startLog()
    simIO.borisBoilerplate(globals())

    ## DEFINE STRING (FOR FILE NAME)
    delimiter = '-'
    dr_String = delimiter.join(str(int(dr*1000)) for dr in DELTRS)
    # cond_string = dr_String + 'mm_LCFS{}_{}eV_{}V_Z{}_'.format(int(LCFS_INDEX), int(ION_TEMP),
    #                                                            int(FIELD_SCALE_ELECTRIC), int(CHARGE_NUM))
    ## CALCULATE SOME CONSTANTS
    N_emitters = len(DELTRS) * NTHETA * NPHI
    # N_particles = NPARTICLES_PER_EMITTER * N_emitters

    # DEFINE MESH AND LOAD MAGNETIC AND ELECTRIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.setErrorField()
    b_hidra.loadCartesianField(coilCurrent=TOROIDAL_CURRENT, errField=True, att_mult=CONFIG_TOR)
    b_hidra.addFieldPerturbation(coilCurrent=HELICAL_CURRENT, att_mult=CONFIG_HEL)
    e_hidra = Mesh(R0=0.72, a=0.19)
    e_hidra.loadCartesianField(FIELD_FILE_ELECTRIC, period_=np.array([0, 1, 1]),
                                    att_mult=FIELD_SCALE_ELECTRIC)

    ## GENERATE INITIAL POSITIONS
    phiGen_arr = np.arange(360//NPHI, 361, 360//NPHI, dtype=int).tolist()
    generateSeedShells(DELTRS, NTHETA, phiGen_arr, LCFS_INDEX, 'IonSeedPts_{}mm'.format(dr_String),
                         b_hidra, Efield=e_hidra, genNormals=True, outputHandler=simIO)


    simIO.log.info('FINISHED PLOTTINGT IC DATA:')


if __name__ == "__main__":
    main()