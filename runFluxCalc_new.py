"""
#------------------------------------------------------#
# RUNNING FLUX CALCULATOR FOR HIDRA MAGNETIC SURFACES #
#------------------------------------------------------#
#        COIL CURRENTS NORMALLY RUN ON HIDRA           #
#------------------------------------------------------#
#  IOTA  |   I_T   |   I_H   |   I_V   |  PHI FWD/REV  #
#        |  [Amp]  |  [Amp]  |  [Amp]  |     [deg]     #
#  1/3   |   486   |   900   |    00   |    324/???    #
#  1/4   |   486   |   790   |    00   |    180/144    #
#  1/5   |   486   |   710   |    00   |    360/???    #
#  1/7   |   581   |   581   |    00   |    ???/???    #
#  MAX.  |  3500   |  7000   |    ??   |    ???/???    #
#------------------------------------------------------#
"""
import argparse
import numpy as np
from classes.iohandler import IOHandler
from classes.mesh import Mesh
from classes.calculate_flux import FluxCalculator
from utility.run_config import load_inputs_json, merge_input_params, normalize_phi_gens


#################
## USER INPUTS ##
#################

NPHI = 3 # Number of phi planes to evaluate
input_params = {
    'ANLYS_DIR': "iota3_entire_pipeline_test", # Existing Poincare input directory
    'ANLYS_SUBDIR': "test9", # Name of new output subdirectory inside ANLYS_DIR
    'FIELD_FILE_TOR': 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy',
    'FIELD_FILE_HEL': 'input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy',
    'CURRENT_TOR': 0.486, #[kA]
    'CURRENT_HEL': 0.900, #[kA]
    'CONFIG_TOR': 'default_toroidal',
    'CONFIG_HEL': 'default_helical',
    'ENABLE_ERRFIELD': True,
    'LCFS_INDEX': 1, # Surface index selected from Poincare log
    'NPHI': NPHI,
    'NTHETA': 180,
    'MAX_SUBSETS': 3, # Number of magnetic islands
    'SMOOTH_FCTR': 7.5e-5, #7.5e-6, 3e-5 #baseline 1e-6
    'INTEGRATE_EPSABS': 5e-2,
    'INTEGRATE_EPSREL': 5e-3,
    'ISLAND_ALGORITHM': 'histogram', # 'kmeans', 'spectral'
    'HIST_BINS': 90,
    'PLOT_ALL': True,
    'BIG_MESH': True,
}

_CLI_INPUTS = object()

def parse_args():
    parser = argparse.ArgumentParser(description="Run ILLIAD flux-surface integration.")
    parser.add_argument(
        "--inputs-json",
        default=None,
        help="Optional path to a JSON object overriding runFluxCalc.py defaults.",
    )
    return parser.parse_args()


def main(input_params_override=_CLI_INPUTS):
    if input_params_override is _CLI_INPUTS:
        args = parse_args()
        input_params_override = load_inputs_json(args.inputs_json, "Flux calculator inputs") if args.inputs_json else None
    params = merge_input_params(input_params, input_params_override)
    normalize_phi_gens(params)
    ## SET UP RUN DIRECTORY (*DATA AND PLOTS WILL BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!*)
    ANLYS_DIR = params['ANLYS_DIR']
    ANLYS_SUBDIR = params['ANLYS_SUBDIR']
    simIO = IOHandler(ANLYS_DIR)
    simIO.setActiveSubDir(ANLYS_SUBDIR)
    simIO.startLog(log_name="fluxCalc.log", subdir=ANLYS_SUBDIR, logger_name="FluxCalculator")
    simIO.inputsBoilerplate(
        "FLUX CALCULATOR INPUTS",
        params,
        [
            "ANLYS_DIR",
            "ANLYS_SUBDIR",
            "FIELD_FILE_TOR",
            "FIELD_FILE_HEL",
            "CURRENT_TOR",
            "CURRENT_HEL",
            "CONFIG_TOR",
            "CONFIG_HEL",
            "ENABLE_ERRFIELD",
            "LCFS_INDEX",
            "NPHI",
            "NTHETA",
            "PHI_GENs",
            "MAX_SUBSETS",
            "SMOOTH_FCTR",
            "INTEGRATE_EPSABS",
            "INTEGRATE_EPSREL",
            "ISLAND_ALGORITHM",
            "HIST_BINS",
            "PLOT_ALL",
            "BIG_MESH",
        ],
    )

    ## DEFINE MESH AND LOAD MAGNETIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.setErrorField()
    b_hidra.loadCartesianField(coilCurrent=params['CURRENT_TOR'],
                               errField=params['ENABLE_ERRFIELD'],
                               att_mult=params['CONFIG_TOR'])
    b_hidra.addFieldPerturbation(coilCurrent=params['CURRENT_HEL'],
                                 att_mult=params['CONFIG_HEL'])
    ## RUN ANALYSIS
    flux_calc = FluxCalculator(simIO, b_hidra, params)
    flux_calc.run()


if __name__ == '__main__':
    main()