## IMPORTS
import numpy as np
from time import perf_counter

from classes.iohandler import IOHandler
from classes.meshNew import Mesh
from classes.boris import Boris

#from plot_HistogramUnion import OUTPUT_DIRECTORY_NAME
from plot_funcs import plotFuncs
from runBorisEfield_i3Accepted import TAG
from utility.coordtrans import *
from utility.point_generators import ionInitializer, generateSeedShells

## SOME PHYSICAL CONSTANTS
kg_per_amu = 1.660_539_068E-27
kboltz = 1.602_176_634E-19 # Joules/eV
Li_mass = 6.941 #amu
He_mass = 4.002602 #amu

# INITIAL CONDITIONS
INPUT_DIR_NAME = "AcceptedIota3_1500spins_atole-9"

LCFS_INDEX = 37
DELTRS = [0.000] # [m]
NPHI = 180
NTHETA = 20
NPARTICLES_PER_EMITTER = 500


ION_TEMP = 2.0 #eV 
ION_MASS = Li_mass #amu
CHARGE_NUM = 1 # Z
FIELD_SCALE_ELECTRIC = 60.0 # [Volts]

## DEFINE STRING (FOR FILE NAME)
delimiter = '-'
dr_String = delimiter.join(str(int(dr*1000)) for dr in DELTRS)

# cond_string = dr_String + 'mm_LCFS{}_{}eV_{}V_Z{}_'.format(int(LCFS_INDEX), int(ION_TEMP),
#                                                            int(FIELD_SCALE_ELECTRIC), int(CHARGE_NUM))

cond_string = dr_String + 'mm_{}eV_LCFS{}_{}V_Li_Z{}_'.format(int(ION_TEMP), int(LCFS_INDEX),
                                                           int(FIELD_SCALE_ELECTRIC), int(CHARGE_NUM))
TAG_IN = 'SOFE25-2'
IC_filename = 'initVelPos_' + cond_string+TAG_IN


OUTPUT_DIRECTORY_NAME = "Test_Output"
TAGOUT = '_ICcheck'

def main():
    ## SET UP RUN DIRECTORY AND LOGGING
    simIO = IOHandler(OUTPUT_DIRECTORY_NAME) 
    simIO.startLog()
    sim_IN = IOHandler(INPUT_DIR_NAME)

    # ## PLOT INITIAL ENERGY DISTRIBUTION TO VALIDATE MAXWELLIAN PROFILE & ION TEMPERATURE
    plotFuncs.boris_plotInitEnergies(IC_filename+'.npy', ION_MASS, runString=IC_filename+TAGOUT, simIO=simIO, sim_in=sim_IN)



if __name__ == "__main__":
    main()