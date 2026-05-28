"""
Runs the full simulation pipeline for ion tracking in a magnetic and electric field environment.

This function performs the following steps:
  1. Sets up the output directory and logging.
  2. Calculates the number of emitters and particles.
  3. Initializes mesh objects for magnetic and electric fields and loads field data.
  4. Initializes ion properties and their initial positions/velocities.
  5. Saves the initial conditions to disk.
  6. Runs the Boris particle solver for the initialized ions.
  7. Processes and transforms output coordinates for plotting.
  8. Generates a series of plots for particle trajectories, wall hits, deposition angles, and energies.
  9. Logs the completion of the simulation.
All configuration and physical parameters are expected to be defined in the global scope or imported modules.
"""

## IMPORTS
import numpy as np
from time import perf_counter

from classes.iohandler import IOHandler
from classes.meshNew import Mesh
from classes.boris import Boris

from plot_funcs import plotFuncs
from utility.coordtrans import *
from utility.point_generators import ionInitializer

## SOME PHYSICAL CONSTANTS
kg_per_amu = 1.660_539_068E-27
kboltz = 1.602_176_634E-19 # Joules/eV
Li_mass = 6.941 #amu
He_mass = 4.002602 #amu


## RUN SIMULATION:
def boris_runner(input_params=None):
     ## LOAD INPUT PARAMETERS
    if input_params is not None:
        print(f'{input_params.keys()=}')
        for key, value in input_params.items():
            print(f'{key}: {value}')
            globals()[str(key)] = value

    ## SET UP RUN DIRECTORY AND LOGGING
    simIO = IOHandler(OUTPUT_DIRECTORY_NAME) 
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
    b_hidra.setErrorField()
    b_hidra.loadCartesianField(coilCurrent=TOROIDAL_CURRENT, errField=ENABLE_ERRFIELD, att_mult=CONFIG_TOR)
    b_hidra.addFieldPerturbation(coilCurrent=HELICAL_CURRENT, att_mult=CONFIG_HEL)
    e_hidra = Mesh(R0=0.72, a=0.19)
    e_hidra.loadCartesianField(FIELD_FILE_ELECTRIC, period_=np.array([0, 1, 1]),
                                    att_mult=FIELD_SCALE_ELECTRIC)

    ## DEFINE LIST OF IONS AND THEIR INIT. POSITIONS/VELOCITIES
    init_conds = [LCFS_INDEX, NPHI, NTHETA, DELTRS, NPARTICLES_PER_EMITTER]
    ion_properties = [ION_MASS, CHARGE_NUM, ION_TEMP]
    ion_list, initVelPos = ionInitializer(init_conds, ion_properties, b_hidra, e_hidra, outputHandler=simIO)

    ## SAVE THE INITIAL VELOCITIES AND POSITIONS AS COMBINED ARRAY
    IC_filename = 'initVelPos_' + cond_string+TAG
    simIO.saveNumpyData(initVelPos, IC_filename)
    simIO.log.info('OUTPUT IC DATA: {}'.format(IC_filename))

    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
    ## RUN BORIS SOLVER FOR PARTICLES ##
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
    ## Regularly-spaced tracker grid: adjust TRACK_NPHI and TRACK_NTHETA as needed.
    ## Selects all NPARTICLES_PER_EMITTER copies for each grid location.
    ## Particle layout: block p (0..NPARTICLES_PER_EMITTER-1) starts at p*N_emitters;
    ## within a block: phi_i * len(DELTRS) * NTHETA + dr_j * NTHETA + theta_k
    _track_phi_idx   = np.round(np.linspace(0, NPHI                  - 1, TRACK_NPHI                  )).astype(int)
    _track_theta_idx = np.round(np.linspace(0, NTHETA                - 1, TRACK_NTHETA                )).astype(int)
    _track_p_idx     = np.round(np.linspace(0, NPARTICLES_PER_EMITTER- 1, TRACK_NPARTICLES_PER_EMITTER)).astype(int)
    particle_tracker_list = [int(p) * N_emitters + int(pi * len(DELTRS) * NTHETA + theta_i)
                                for pi in _track_phi_idx
                                for theta_i in _track_theta_idx
                                for p in _track_p_idx]

    ion_tracer = Boris(simIO, OUTPUT_DIRECTORY_NAME, TAG)
    ion_tracer.setConditions(ion_list, cond_string, DT, TMAX)
    output_array, energy_output, depo_angles_deg, toroidal_angles_deg, ion_traces = ion_tracer.run(b_hidra, e_hidra, particle_tracker_list)
    simIO.log.info('PYTORCH STATS:\n' + torch.cuda.memory_summary())

    # COORDINATE FLIIPING & CONVERSION
    phi_plot = output_array[2]*(-1) + 2*np.pi # flip phi for the perspective outside the vacuum vessel
    a_phi = 18. #-36. # degrees, phi_comp is 18 CW from south-side split
    phi_plot_deg = (phi_plot*(180/np.pi) + a_phi) % 360.

    theta_plot = output_array[1]
    theta_plot[theta_plot>np.pi] -= 2*np.pi #shift so that (theta=0) is centered in the plot
    theta_plot_deg = theta_plot*(180/np.pi)

    ## PLOTTING
    ion_tracer.plotParticlesOverTime(output_array[-1], N_particles, TMAX, DT, runString=cond_string+TAG, simIO=simIO)
    ion_tracer.plotWallHist(output_array[:3], cond_string+TAG, simIO=simIO, cond_string=cond_string)
    ion_tracer.plotCombined(phi_plot_deg, theta_plot_deg, depo_angles_deg, colorRange=[0, 90], 
                                colorLabel='Ion Deposition Angle (deg. from normal)', myColormap='viridis',
                                runString=cond_string+TAG+'_AngleCombined', simIO=simIO, cond_string=cond_string)
    ion_tracer.plotCombined(phi_plot_deg, theta_plot_deg, energy_output,
                                colorLabel='Ion Deposition Energy (eV)', myColormap='magma',
                                runString=cond_string+TAG+'_EnergyCombined', simIO=simIO, cond_string=cond_string)
    ion_tracer.plotCombined(phi_plot_deg, theta_plot_deg, toroidal_angles_deg, colorRange=[0, 180], 
                                colorLabel='Ion Deposition Toroidal Angle (deg. from $\\hat{\\phi}$)', myColormap='coolwarm',
                                runString=cond_string+TAG+'_PHIAngleCombined', simIO=simIO, cond_string=cond_string)
    
    # ion_tracer.plotTraceAnim(ion_traces, b_hidra, runString=cond_string+TAG+'_28stride13_30f_480p2' , simIO=simIO,
    #                               interval=1000/120, stride=13, max_frames=200,
    #                               linewidth=0.5, linecolor=plotFuncs.UIUC['il_orange'], line_alpha=0.4, line_window=40,
    #                               trail_length=10, markersize=2, markercolor=plotFuncs.UIUC['il_blue'],
    #                               parallel=True, n_workers=18, resolution='720p')

    #ion_tracer.plotWallPoints3D(phi_plot_deg, theta_plot_deg, b_hidra, runString=cond_string+TAG, simIO=simIO)
    ion_tracer.plotTraces(ion_traces, b_hidra, runString=cond_string+TAG, simIO=simIO)
    #ion_tracer.plotWallPoints(phi_plot_deg, theta_plot_deg, runString=cond_string+TAG, simIO=simIO)
    #ion_tracer.plotInitEnergies(IC_filename+'.npy', ION_MASS, runString=cond_string+TAG, simIO=simIO)

    ## END RUN ##
    simIO.log.info('## SIM FINISHED! ##\n\n\n')

if __name__ == "__main__":

    ## SET SIMULATION INPUTS:
    # ANALYSIS DIRECTORY AND UNIQUE OUTPUT TAG
    #OUTPUT_DIRECTORY_NAME = "It-0486_Ih-0790_PHI180_1500spins_105Lines_LSODA1e9_newEvents"
    OUTPUT_DIRECTORY_NAME = "It-0486_Ih-0900_noErr_1500sp_LSODA1e8"
    TAG = "Lithium_FS86_1p0ms_TRACK4x4"
    # TOROIDAL AND HELICAL MAGNETIC FIELDS
    TOROIDAL_CURRENT = 0.486 #[kA]
    HELICAL_CURRENT = 0.900 #[kA]``
    CONFIG_TOR = 'default_toroidal'
    CONFIG_HEL = 'default_helical'
    ENABLE_ERRFIELD = False
    # ELECTRIC FIELD
    #FIELD_FILE_ELECTRIC = 'input_files/Efield_AcceptedIota3.npy'
    #FIELD_FILE_ELECTRIC = 'input_files/Efield_dftIota4_lcfs16.npy'
    FIELD_FILE_ELECTRIC = 'input_files/Efield_IdealIota3_lcfs30.npy'
    FIELD_SCALE_ELECTRIC = 60.0 # [Volts]
    # ION PROPERTIES
    ION_MASS = Li_mass # [amu]
    ION_TEMP = 2.0 # [eV]
    CHARGE_NUM = 1 # [Z]
    # INITIAL CONDITIONS
    LCFS_INDEX = 86 #30 #29 #40 (from Poincare output (simIO.log))
    DELTRS = [0.000] # [m]
    NPHI = 360
    NTHETA = 120
    NPARTICLES_PER_EMITTER = 50
    # SIMULATION PARAMETERS
    DT = 1e-8 # [s]
    TMAX = 0.0010 # [s]
    NSTEPS = int(TMAX / DT)
    TRACK_NPHI = 4#72     #60#18 #60  #18
    TRACK_NTHETA = 4#40   #40#12 #40  #12
    TRACK_NPARTICLES_PER_EMITTER = 1

    boris_runner()