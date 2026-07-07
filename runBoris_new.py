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
import argparse
import json
import numpy as np
import torch
from time import perf_counter
from pathlib import Path
from typing import Optional

from classes.iohandler import IOHandler
from classes.meshNew import Mesh
from classes.boris import Boris

from plot_funcs import plotFuncs
from utility.coordtrans import *
from utility.point_generators import ionInitializer
from utility.run_config import load_inputs_json, merge_input_params, normalize_phi_gens


## SOME PHYSICAL CONSTANTS
kg_per_amu = 1.660_539_068E-27
kboltz = 1.602_176_634E-19 # Joules/eV
Li_mass = 6.941 #amu
He_mass = 4.002602 #amu

# JSON-provided run inputs. These are assigned dynamically in boris_runner();
# the annotations keep static analyzers from reporting them as undefined.
CONFIG_TOR: str
CONFIG_HEL: str
ENABLE_ERRFIELD: bool
TOROIDAL_CURRENT: float
HELICAL_CURRENT: float

FIELD_FILE_DENSITY: str
FIELD_FILE_ELECTRIC: str
FIELD_SCALE_ELECTRIC: float
ION_NEUTRAL_COLLISIONS: Optional[str]
ION_ION_COLLISIONS: Optional[str]

FIELD_SCALE_ELECTRIC: float
BACKGROUND_GAS_SPECIES: str
NEUTRAL_GAS_TEMP_EV: float
NEUTRAL_GAS_DENSITY: float
BACKGROUND_ION_TEMP_EV: float
PLASMA_DENSITY: float

ION_MASS: float
ION_TEMP: float
CHARGE_NUM: int

LCFS_INDEX: int
DELTRS: list[float]
NPHI: int
NTHETA: int
NPARTICLES_PER_EMITTER: int

DT: float
TMAX: float
NSTEPS: int
TRACK_NPHI: int
TRACK_NTHETA: int
TRACK_NPARTICLES_PER_EMITTER: int
STRIDE: int
TRACE_STRIDE: int

OUTPUT_DIRECTORY_NAME: str
TAG: str


input_params = {
    "CONFIG_TOR": "default_toroidal",
    "CONFIG_HEL": "default_helical",
    "ENABLE_ERRFIELD": True,
    "TOROIDAL_CURRENT": 0.486,
    "HELICAL_CURRENT": 0.900,

    "FIELD_FILE_DENSITY": "output/AAAnewIO_iota3FWD_phi306_LSODA/data/LCFS20_360x180/big_grid_linear.npy",
    "FIELD_FILE_ELECTRIC": "output/AAAnewIO_iota3FWD_phi306_LSODA/data/LCFS20_360x180/Efield_testingOutput.npy",
    "ION_NEUTRAL_COLLISIONS": "langevin_in_hstep",
    "ION_ION_COLLISIONS": "fokker_planck_ii_hstep",

    "FIELD_SCALE_ELECTRIC": 60.0,
    "BACKGROUND_GAS_SPECIES": "He", 
    "NEUTRAL_GAS_TEMP_EV": 0.025,
    "NEUTRAL_GAS_DENSITY": 3e18,
    "BACKGROUND_ION_TEMP_EV": 2.0,
    "PLASMA_DENSITY": 5e18,

    "ION_MASS": 6.941,
    "ION_TEMP": 2.0,
    "CHARGE_NUM": 1,

    "LCFS_INDEX": 20,
    "DELTRS": [0.0],
    "NPHI": 180,
    "NTHETA": 120,
    "NPARTICLES_PER_EMITTER": 5,

    "DT": 1e-8,
    "TMAX": 0.001,

    "TRACK_NPHI": 180,
    "TRACK_NTHETA": 120,
    "TRACK_NPARTICLES_PER_EMITTER": 1,
    "STRIDE": 13,

    "OUTPUT_DIRECTORY_NAME": "AAAnewIO_iota3FWD_phi306_LSODA",
    "TAG": "pipelineTest"
    }


_CLI_INPUTS = object()

def parse_args():
    parser = argparse.ArgumentParser(description="Run Boris ion tracking from a JSON input file.")
    parser.add_argument(
        "--inputs-json",
        default=None,
        help="Path to Boris runner inputs JSON.",
    )
    return parser.parse_args()


## RUN SIMULATION:
def boris_runner(params):
     ## LOAD INPUT PARAMETERS
    if params is not None:
        print(f'{params.keys()=}')
        for key, value in params.items():
            print(f'{key}: {value}')
            globals()[str(key)] = value

    STRIDE = int(params.get("STRIDE", params.get("TRACE_STRIDE", 1)))
    if STRIDE < 1:
        raise ValueError("STRIDE must be a positive integer")
    params["TRACE_STRIDE"] = STRIDE
    params["STRIDE"] = STRIDE

    ## DEFINE STRING (FOR FILE NAME)
    delimiter = '-'
    dr_String = delimiter.join(str(int(dr*1000)) for dr in DELTRS)
    cond_string = dr_String + 'mm_LCFS{}_{}eV_{}V_Z{}_'.format(int(LCFS_INDEX), int(ION_TEMP),
                                                               int(FIELD_SCALE_ELECTRIC), int(CHARGE_NUM))
    boris_subdir = cond_string + TAG
    params["NSTEPS"] = int(TMAX / DT)

    ## SET UP RUN DIRECTORY AND LOGGING
    simIO = IOHandler(OUTPUT_DIRECTORY_NAME)
    simIO.setActiveSubDir(boris_subdir)
    simIO.startLog(log_name="boris.log", subdir=boris_subdir, logger_name="Boris")
    simIO.borisBoilerplate(params)

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
    if ION_ION_COLLISIONS:
      n_hidra = Mesh(R0=0.72, a=0.19)
      n_hidra.loadScalarField(FIELD_FILE_DENSITY, period_=np.array([0, 1, 1]),
                  att_mult=PLASMA_DENSITY)
    else:
      n_hidra = None

    ## DEFINE LIST OF IONS AND THEIR INIT. POSITIONS/VELOCITIES
    init_conds = [LCFS_INDEX, NPHI, NTHETA, DELTRS, NPARTICLES_PER_EMITTER]
    ion_properties = [ION_MASS, CHARGE_NUM, ION_TEMP]
    ion_list, initVelPos = ionInitializer(init_conds, ion_properties, b_hidra, e_hidra, outputHandler=simIO)

    ## SAVE THE INITIAL VELOCITIES AND POSITIONS AS COMBINED ARRAY
    IC_filename = 'initVelPos'
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
    ion_tracer.setConditions(ion_list, cond_string, DT, TMAX, NEUTRAL_GAS_TEMP_EV, BACKGROUND_ION_TEMP_EV,
                             n_gas=NEUTRAL_GAS_DENSITY, n_e=PLASMA_DENSITY, bg_gas_species=BACKGROUND_GAS_SPECIES)
    output_array, energy_out, depo_angles, toroidal_angles, traces = ion_tracer.run(Bfield=b_hidra,
                                                                                    Efield=e_hidra,
                                                                                    nfield=n_hidra,
                                                                                    ion_neutral_collisions=ION_NEUTRAL_COLLISIONS,
                                                                                    ion_ion_collisions=ION_ION_COLLISIONS,
                                                                                    trace_IDs=particle_tracker_list,
                                                                                    trace_stride=STRIDE)
    simIO.log.info('PYTORCH STATS:\n' + torch.cuda.memory_summary())

    # COORDINATE FLIIPING & CONVERSION
    phi_plot = output_array[2]*(-1) + 2*np.pi # flip phi for the perspective outside the vacuum vessel
    a_phi = 18. #-36. # degrees, phi_comp is 18 CW from south-side split
    phi_plot_deg = (phi_plot*(180/np.pi) + a_phi) % 360.

    theta_plot = output_array[1]
    theta_plot[theta_plot>np.pi] -= 2*np.pi #shift so that (theta=0) is centered in the plot
    theta_plot_deg = theta_plot*(180/np.pi)

    ## PLOTTING
    ion_tracer.plotParticlesOverTime(output_array[-1], N_particles, TMAX, DT, runString='RunningFraction', simIO=simIO)
    ion_tracer.plotWallHist(output_array[:3], 'WallHistogram', simIO=simIO, cond_string=cond_string)
    ion_tracer.plotCombined(phi_plot_deg, theta_plot_deg, depo_angles, colorRange=[0, 90], 
                                colorLabel='Ion Deposition Angle (deg. from normal)', myColormap='viridis',
                                runString='AngleCombined', simIO=simIO, cond_string=cond_string)
    ion_tracer.plotCombined(phi_plot_deg, theta_plot_deg, energy_out,
                                colorLabel='Ion Deposition Energy (eV)', myColormap='magma',
                                runString='EnergyCombined', simIO=simIO, cond_string=cond_string)
    ion_tracer.plotCombined(phi_plot_deg, theta_plot_deg, toroidal_angles, colorRange=[0, 180], 
                                colorLabel='Ion Deposition Toroidal Angle (deg. from $\\hat{\\phi}$)', myColormap='coolwarm',
                                runString='PHIAngleCombined', simIO=simIO, cond_string=cond_string)


    #ion_tracer.plotWallPoints3D(phi_plot_deg, theta_plot_deg, b_hidra, runString='WallPoints3D', simIO=simIO)
    ion_tracer.plotTraces(traces, b_hidra, runString='Traces', simIO=simIO)
    #ion_tracer.plotWallPoints(phi_plot_deg, theta_plot_deg, runString='WallPoints', simIO=simIO)
    #ion_tracer.plotInitEnergies(IC_filename+'.npy', ION_MASS, runString='InitEnergies', simIO=simIO)

    ## END RUN ##
    simIO.log.info('## SIM FINISHED! ##\n\n\n')


def main(input_params_override=_CLI_INPUTS):
    if input_params_override is _CLI_INPUTS:
            args = parse_args()
            input_params_override = load_inputs_json(args.inputs_json, "Boris inputs") if args.inputs_json else None
    params = merge_input_params(input_params, input_params_override)
    boris_runner(params)


if __name__ == "__main__":
    main()
