## IMPORTS
import numpy as np
from time import perf_counter

import classes.class_outputHandler as out
from classes.meshNew import Mesh
from classes.particle import Ion
from classes.boris import Boris

from utility.coordtrans import *
from utility.point_generators import generateSeedShells, generate_MB_velocities, ion_initializer

## SOME PHYSICAL CONSTANTS
kg_per_amu = 1.660_539_068E-27
kboltz = 1.602_176_634E-19 # Joules/eV
Li_mass = 6.941 #amu
He_mass = 4.002602 #amu


"""
This script sets up and runs a simulation of ion behavior in a magnetic field using the Boris-Buneman algorithm.
It initializes the simulation parameters, loads the magnetic and electric fields, generates initial ion positions and velocities,
and runs the BORIS solver to track the ions' motion. The results are then plotted and saved.
"""

############################
## SET SIMULATION INPUTS: ##
############################
# ANALYSIS DIRECTORY AND UNIQUE OUTPUT TAG
OUTPUT_DIRECTORY_NAME = "ChangeToIota3_1500spins_atole-9"
TAG = "0p4ms_borisCLASS4"

# TOROIDAL AND HELICAL MAGNETIC FIELDS
TOROIDAL_CURRENT = 0.486 #[kA]
HELICAL_CURRENT = 0.900 #[kA]
CONFIG_TOR = 'default_toroidal'
CONFIG_HEL = 'default_helical_rev'

# ELECTRIC FIELD
FIELD_FILE_ELECTRIC = 'input_files/Efield_ChangeToIota3.npy'
FIELD_SCALE_ELECTRIC = 90.0 # [Volts]

# ION PROPERTIES
ION_MASS = Li_mass # amu
ION_TEMP = 25.0 # eV 
CHARGE_NUM = 1 # Z

# INITIAL CONDITIONS
LCFS_INDEX = 31 #30 #29 #40 # from Poincare output (simIO.log)
NPHI = 60
NTHETA = 72 #90
DELTRS = [0.000]
NPARTICLES_PER_EMITTER = 100 #300

# SIMULATION PARAMETERS
DT = 1e-8
TMAX = 0.0004
NSTEPS = int(TMAX / DT)


#####################
## RUN SIMULATION: ##
#####################
def main():
    ## SET UP RUN DIRECTORY AND LOGGING
    simIO = out.IOHandler(OUTPUT_DIRECTORY_NAME) 
    simIO.startLog()
    simIO.borisBoilerplate(globals())

    ## DEFINE STRING (FOR FILE NAME)
    delimiter = '-'
    dr_String = delimiter.join(str(int(dr*1000)) for dr in DELTRS)
    cond_string = dr_String + 'mm_LCFS{}_{}eV_{}V_Z{}_'.format(int(LCFS_INDEX), int(ION_TEMP),
                                                               int(FIELD_SCALE_ELECTRIC), int(CHARGE_NUM))
    ## CALCULATE SOME CONSTANTS
    N_emitters = len(DELTRS) * NTHETA * NPHI
    N_particles = NPARTICLES_PER_EMITTER * N_emitters

    ## DEFINE MESH AND LOAD MAGNETIC AND ELECTRIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.set_nonPer_errField()
    b_hidra.loadCartesianField(coilCurrent=TOROIDAL_CURRENT, errField=True, att_mult=CONFIG_TOR)
    b_hidra.addFieldPerturbation(coilCurrent=HELICAL_CURRENT, att_mult=CONFIG_HEL)
    e_hidra = Mesh(R0=0.72, a=0.19)
    e_hidra.loadCartesianField(FIELD_FILE_ELECTRIC, period_=np.array([0, 1, 1]),
                                    att_mult=FIELD_SCALE_ELECTRIC)

    ## DEFINE LIST OF IONS AND THEIR INIT. POSITIONS/VELOCITIES
    init_conds = [LCFS_INDEX, NPHI, NTHETA, DELTRS, NPARTICLES_PER_EMITTER]
    ion_properties = [ION_MASS, CHARGE_NUM, ION_TEMP]
    ion_list, initVelPos = ion_initializer(init_conds, ion_properties, b_hidra, e_hidra, outputHandler=simIO)

    ## SAVE THE INITIAL VELOCITIES AND POSITIONS AS COMBINED ARRAY
    IC_filename = 'initVelPos_' + cond_string+TAG
    simIO.saveNumpyData(initVelPos, IC_filename)
    simIO.log.info('OUTPUT IC DATA: {}'.format(IC_filename))

    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
    ## RUN BORIS SOLVER FOR PARTICLES ##
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
    particle_tracker_list = [10,13,20,500,2346, 13130, 29777, 33333, 40266, 50000]
    ionTracer = Boris(simIO, OUTPUT_DIRECTORY_NAME, TAG)
    ionTracer.set_conditions(ion_list, cond_string, DT, TMAX)
    outputArray, energy_output, depo_angles_deg, ion_traces = ionTracer.run(b_hidra, e_hidra, particle_tracker_list)
    simIO.log.info('PYTORCH STATS:\n' + torch.cuda.memory_summary())

    ## PLOTTING
    # COORDINATE FLIPPING & CONVERSION
    phi_plot = (-1)*outputArray[2] + 2*np.pi # flip phi for the perspective outside the vacuum vessel
    theta_plot = outputArray[1]
    theta_plot[theta_plot>np.pi] -= 2*np.pi #shift so that (theta=0) is centered in the plot
    a_phi = -18. # degrees, phi_comp is 18 CW from south-side split
    phi_plot_deg = (phi_plot*(180/np.pi) + a_phi) % 360.
    theta_plot_deg = theta_plot*(180/np.pi)

    ionTracer.plotParticlesOverTime(outputArray[-1], N_particles, TMAX, DT, runString=cond_string+TAG, simIO=simIO)
    ionTracer.plotWallHist(outputArray[:3], cond_string+TAG, simIO=simIO)
    ionTracer.plotWallPoints3D(phi_plot_deg, theta_plot_deg, b_hidra, runString=cond_string+TAG, simIO=simIO)

    ## FINAL ENERGY AND DEPOSITION ANGLE PLOTS
    ionTracer.plotCombined(phi_plot_deg, theta_plot_deg, depo_angles_deg, colorRange=[0, 90], 
                                colorLabel='Ion Deposition Angle (deg. from normal)', myColormap='viridis',
                                runString=cond_string+TAG+'_AngleCombined', simIO=simIO)
    ionTracer.plotCombined(phi_plot_deg, theta_plot_deg, energy_output, 
                                colorLabel='Ion Deposition Energy (eV)', myColormap='magma',
                                runString=cond_string+TAG+'_EnergyCombined', simIO=simIO)

    ionTracer.plotTraces(ion_traces, b_hidra, runString=cond_string+TAG, simIO=simIO)
    ionTracer.plotWallPoints(phi_plot_deg, theta_plot_deg, runString=cond_string+TAG, simIO=simIO)
    ionTracer.plotInitEnergies(IC_filename+'.npy', ION_MASS, runString=cond_string+TAG, simIO=simIO)

    ## END RUN ##
    simIO.log.info('## SIM FINISHED! ##\n\n\n')


if __name__ == "__main__":
    main()