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
from illiad.utilities.coordtrans import *
from illiad.particle import *
#import plot_funcs.plotFuncs as plotFuncs
from illiad.plotting import *

## SOME PHYSICAL CONSTANTS
kg_per_amu = 1.660_539_068E-27
kboltz = 1.602_176_634E-19 # Joules/eV
Li_mass = 6.941 #amu
He_mass = 4.002602 #amu


## SET SIMULATION INPUTS: ##

INPUT_DIRECTORY_NAME = "It-0486_Ih-0900_noErr_1500sp_LSODA1e8"
# INITIAL CONDITIONS
LCFS_INDEX = 40 #30 #29 #40 (from Poincare output (simIO.log))
DELTRS = [0.000] # [m]
NPHI = 180
NTHETA = 120
NPARTICLES_PER_EMITTER = 50
# SIMULATION PARAMETERS
DT = 1e-8 # [s]
TMAX = 0.0010 # [s]
NSTEPS = int(TMAX / DT)
# ION PROPERTIES
ION_MASS = Li_mass # [amu]
CHARGE_NUM = 1 # [Z]


input_conditions = [
    {#IonsVtime_0mm_LCFS40_5eV_120V_Z1_Lithium_FS40_tauRes_1p0m
        "ION_TEMP": 5.0,  # [eV]
        "FIELD_SCALE_ELECTRIC": 120.0,  # [Volts]
        "TAG": "Lithium_FS40_tauRes_1p0ms",
        "N_PARTICLES": 360000
        },
    {#IonsVtime_0mm_LCFS30_1eV_60V_Z1_Lithium_FS40_2
        "ION_TEMP": 1.0,  # [eV]
        "FIELD_SCALE_ELECTRIC": 60.0,  # [Volts]
        "TAG": "Lithium_FS40_2",
        "N_PARTICLES": 720000
        },

    {#IonsVtime_0mm_LCFS40_5eV_40V_Z1_Lithium_FS40_tauRes_1p0ms
        "ION_TEMP": 5.0,  # [eV]
        "FIELD_SCALE_ELECTRIC": 40.0,  # [Volts]
        "TAG": "Lithium_FS40_tauRes_1p0ms",
        "N_PARTICLES": 360000
        },
    {#IonsVtime_0mm_LCFS40_2eV_40V_Z1_Lithium_FS40_1p0ms_PHANGLE2
        "ION_TEMP": 2.0,  # [eV]
        "FIELD_SCALE_ELECTRIC": 40.0,  # [Volts]
        "TAG": "Lithium_FS40_1p0ms_PHANGLE2",
        "N_PARTICLES": 1080000
        }
    ]


OUTPUT_DIRECTORY_NAME = 'SURVIVAL_TEST'

def calculate_tau(maxN_array, tot_particles, tmax, dt):
    """Plots the percent of particles running over time.
        maxN_array is an array of maximum timestep for each particle.
    """

    # Calculate the number of particles running over time (efficiently)
    time_steps = np.arange(0, tmax, dt)
    time_ms = time_steps * 1000
    sorted_maxTime = np.sort(maxN_array) * dt

    immortal_particles = tot_particles - len(maxN_array)  # Count particles that never hit the wall
    print(f"Immortal particles (never hit the wall): {immortal_particles} out of {tot_particles}")

    # Use searchsorted to find how many particles have maxTime > t for each t
    particles_running = len(sorted_maxTime) - np.searchsorted(sorted_maxTime, time_steps, side='right')


    frac_running = (particles_running + immortal_particles) / tot_particles  # include immortal particles in the fraction running

    # Estimate residence time using trapezoidal integration of the fraction running over time
    tau_res = np.trapezoid(frac_running, dx=dt)
    if frac_running[-1] > 0:
        slope = (np.log(frac_running[-1]) - np.log(frac_running[-101])) / 100 / dt # use wider range for slope to reduce noise
        slope = min(slope, -1e-1)  # prevent division by zero or very small slope
        tau_res_corr = -frac_running[-1] / slope
    else:
        tau_res_corr = 0.0

    tau_res_est = tau_res + tau_res_corr
    tau_res_est_clamped = np.clip(tau_res_est, time_steps[0], time_steps[-1])
    frac_at_tau_res = np.interp(tau_res_est_clamped, time_steps, frac_running)

    return (tau_res, tau_res_corr, frac_at_tau_res, time_ms, frac_running)


def plot_survivalFunction(out_conditions, in_conditions, runString='default', simIO=None):

    plt.figure(figsize=(8, 5))

    condition_colors = [
        UIUC['il_stormdark1'],
        UIUC['il_blue'],
        UIUC['il_orange'],
        UIUC['il_storm'],
    ]
    condition_markers = ['o', 's', 'D', '^']

    for idx, (condition_out, condition_in) in enumerate(zip(out_conditions, in_conditions)):

        cond_color = condition_colors[idx % len(condition_colors)] # %cycling


        tau_res, tau_res_corr, frac_at_tau_res, time_ms, frac_running = condition_out
        ion_temp = condition_in['ION_TEMP']
        field_scale_electric = condition_in['FIELD_SCALE_ELECTRIC']

        tau_res_est = tau_res + tau_res_corr
    
        plt.plot(time_ms, frac_running, color='k', linewidth=1.)#, label='Particles Running')
        plt.fill_between(time_ms, frac_running, color=UIUC['il_blue'], alpha=0.25)

        plt.scatter(tau_res_est*1000, frac_at_tau_res, marker=condition_markers[idx % len(condition_markers)], color=cond_color,
                     label=f'$\\mathrm{{ {ion_temp}eV, {field_scale_electric:.0f}V}}$', zorder=5)
        plt.vlines(tau_res_est*1000, 0.0, frac_at_tau_res,
                 color=cond_color, linestyle='--',
                   label=f'$\\tau={tau_res_est*1000:.3f}ms$', zorder=5)
    



    plt.xlabel('$t~[ms]$', fontsize=12)
    plt.ylabel('$\\dfrac{N_{active}}{N_{total}}$', fontsize=12, rotation=0, labelpad=22)
    # plt.title('Estimated Residence Time = {:.3f}$\\mathit{{(+{:.3f}ms~correction) }}$'
    #           .format(tau_res*1000, tau_res_corr*1000), fontsize=12)


    plt.xlim(0, time_ms[-1])    
    plt.ylim(0, 1.05)
    plt.xticks(np.arange(0, time_ms[-1]+0.1, 0.1), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=9.5, ncols=4)
    #plt.show()
    plotname = 'IonsVtime_' + runString + '.png'
    simIO.saveFig(plotname, dpi=300)
    simIO.log.info('OUTPUT PLOT: {}, residence time = {:.3f}ms, corr = {:.3f}ms,'
                   .format(plotname, tau_res*1000, tau_res_corr*1000))
    plt.close()


def main():
    ## SET UP RUN DIRECTORY AND LOGGING
    ## DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
    simOUT = IOHandler(OUTPUT_DIRECTORY_NAME) 
    simOUT.startLog()
    simIO = IOHandler(INPUT_DIRECTORY_NAME) 

    output_conditions = []
    ## LOAD DATA LOOP
    for condition in input_conditions:
        tag = condition['TAG']
        ion_temp = condition['ION_TEMP']
        field_scale_electric = condition['FIELD_SCALE_ELECTRIC']
        N_particles = condition['N_PARTICLES']
        simOUT.log.info('## RUNNING CONDITION: {} ##'.format(tag))

        ## DEFINE STRING (FOR FILE NAME)
        delimiter = '-'
        dr_String = delimiter.join(str(int(dr*1000)) for dr in DELTRS)
        cond_string = dr_String + 'mm_LCFS{}_{}eV_{}V_Z{}_'.format(
            int(LCFS_INDEX), int(ion_temp), int(field_scale_electric), int(CHARGE_NUM)
        )

        ## LOAD WALL POINTS
        filename = 'Wallpt_OUTPUT_' + cond_string + tag
        outputArray = simIO.loadNumpyData(filename+'.npy')
        max_timeStep = outputArray[6, :]  # max time step for each particle

        output = calculate_tau(max_timeStep, N_particles, TMAX, DT)

        print(f"Condition: {tag}, Estimated Residence Time = {output[0]*1000:.3f}ms, Correction = {output[1]*1000:.3f}ms, Fraction at Residence Time = {output[2]:.3f}" )

        output_conditions += [output]


    plot_survivalFunction(output_conditions, input_conditions, runString='SURVIVALTEST', simIO=simOUT)

    ## END RUN ##
    simOUT.log.info('## SIM FINISHED! ##\n\n\n')


if __name__ == "__main__":
    main()