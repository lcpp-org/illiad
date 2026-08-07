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
from illiad.mesh import Mesh
from illiad.io import IOHandler
from illiad.flux import FluxCalculator
from illiad.utilities.run_config import load_inputs_json, merge_input_params, normalize_phi_gens

DEFAULT_INPUTS = {
    "ANLYS_DIR": "It-0486_Ih-0790_PHI180_1500spins_105Lines_LSODA1e9_newEvents",
    "ANLYS_SUBDIR": "LCFS30_20x720_furtherTESTING",
    "FIELD_FILE_TOR": "input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy",
    "FIELD_FILE_HEL": "input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy",

    "CURRENT_TOR": 0.486, # [kA]
    "CURRENT_HEL": 0.790, # [kA]
    "CONFIG_TOR": "default_toroidal",
    "CONFIG_HEL": "default_helical",
    "ENABLE_ERRFIELD": True,

    "LCFS_INDEX": 30,
    "NPHI": 20,
    "NTHETA": 720,
    "MAX_SUBSETS": 4,
    "SMOOTH_FCTR": 1e-6,
    "INTEGRATE_EPSABS": 1e-2,
    "INTEGRATE_EPSREL": 1e-2,
    "ISLAND_ALGORITHM": "histogram",
    "HIST_BINS": 90,
    "PLOT_ALL": True,
    "BIG_MESH": True,
}

_CLI_INPUTS = object()


def parse_args():
    parser = argparse.ArgumentParser(description="Run ILLIAD flux-surface integration.")
    parser.add_argument(
        "inputs_path",
        nargs="?",
        metavar="INPUTS",
        help="Optional positional path to the workflow JSON input.",
    )
    inputs_group = parser.add_mutually_exclusive_group()
    inputs_group.add_argument(
        "--inputs",
        dest="inputs",
        default=None,
        help="Optional path to a JSON object overriding built-in workflow defaults.",
    )
    inputs_group.add_argument(
        "--inputs-json",
        dest="inputs",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.inputs_path is not None and args.inputs is not None:
        parser.error("provide INPUTS or --inputs, not both")
    args.inputs = args.inputs if args.inputs is not None else args.inputs_path
    return args


def main(input_overrides=_CLI_INPUTS):
    if input_overrides is _CLI_INPUTS:
        args = parse_args()
        input_overrides = load_inputs_json(args.inputs, "Flux calculator inputs") if args.inputs else None
    params = merge_input_params(DEFAULT_INPUTS, input_overrides)
    normalize_phi_gens(params)
    ## SET UP RUN DIRECTORY (*DATA AND PLOTS WILL BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!*)
    anlys_dir = params["ANLYS_DIR"]
    anlys_subdir = params["ANLYS_SUBDIR"]
    simIO = IOHandler(anlys_dir)
    simIO.setActiveSubDir(anlys_subdir)
    simIO.startLog(log_name="fluxCalc.log", subdir=anlys_subdir, logger_name="FluxCalculator")
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
    b_hidra.loadCartesianField(coilCurrent=params["CURRENT_TOR"],
                               errField=params["ENABLE_ERRFIELD"],
                               att_mult=params["CONFIG_TOR"])
    b_hidra.addFieldPerturbation(coilCurrent=params["CURRENT_HEL"],
                                 att_mult=params["CONFIG_HEL"])
    ## RUN ANALYSIS
    flux_calc = FluxCalculator(simIO, b_hidra, params)
    flux_calc.run()


if __name__ == '__main__':
    main()
