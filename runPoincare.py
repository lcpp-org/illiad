"""
#------------------------------------------------------#
# GENERATING POINCARE PLOTS FOR HIDRA'S MAGNETIC FIELD #
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
##  NTHREADS:
#    > 0: use N threads
#    = 0: use all available threads
#    < 0: use all but the last N threads
## DOUBLE_LINE:
#    True: run each fieldline in both directions from the init pos *!ONLY USE WHEN NTHREADS > NLINES!*
#    False: run each fieldline in +B direction from the init pos
"""
import argparse
import numpy as np
from classes.mesh import Mesh
from classes.poincare import Poincare
from classes.iohandler import IOHandler
from utility.run_config import load_inputs_json, merge_input_params

DEFAULT_INPUTS = {
    "CURRENT_TOR": 0.486,  # [kA]
    "CURRENT_HEL": 0.900,  # [kA]
    "CONFIG_TOR": "default_toroidal",
    "CONFIG_HEL": "default_helical",
    "ENABLE_ERRFIELD": True,

    "IC_PHI_DEG": 306.0,  # [deg]
    "IC_THETA_DEG": 180.0,  # [deg]
    "START_RADIUS": 0.150,  # [m]
    "END_RADIUS": 0.020,  # [m]
    "NLINES": 53,
    "SPINS": 600,
    "NPLANES": 360,
    "SOLVER": "LSODA",

    "RTOL": 2.49e-12,
    "ATOL": 1e-8,
    "NTHREADS": -1,
    "DOUBLE_LINE": False,

    "OUTPUT_DIR": "AAAnewIO_iota3FWD_phi306_LSODA",
}

_CLI_INPUTS = object()


def parse_args():
    parser = argparse.ArgumentParser(description="Run ILLIAD Poincare field-line tracing.")
    parser.add_argument(
        "--inputs-json",
        default=None,
        help="Optional path to a JSON object overriding runPoincare.py defaults.",
    )
    return parser.parse_args()


def main(input_overrides=_CLI_INPUTS):
    """
    Main function to set up the mesh, load magnetic field data, and generate Poincare plots.
    """
    if input_overrides is _CLI_INPUTS:
        args = parse_args()
        input_overrides = load_inputs_json(args.inputs_json, "Poincare inputs") if args.inputs_json else None
    params = merge_input_params(DEFAULT_INPUTS, input_overrides)
   
    ## SET UP RUN DIRECTORY (*DATA AND PLOTS WILL BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!*)
    simIO = IOHandler(params["OUTPUT_DIR"]) 
    simIO.startLog(log_name="poincare.log", subdir="Poincare", logger_name="Poincare")
    simIO.inputsBoilerplate(
        "POINCARE INPUTS",
        params,
        [
            "CURRENT_TOR",
            "CURRENT_HEL",
            "CONFIG_TOR",
            "CONFIG_HEL",
            "ENABLE_ERRFIELD",
            "IC_PHI_DEG",
            "IC_THETA_DEG",
            "START_RADIUS",
            "END_RADIUS",
            "NLINES",
            "SPINS",
            "NPLANES",
            "SOLVER",
            "RTOL",
            "ATOL",
            "NTHREADS",
            "DOUBLE_LINE",
            "OUTPUT_DIR",
        ],
    )

    ## DEFINE MESH AND LOAD MAGNETIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField(coilCurrent=params["CURRENT_TOR"], 
                               errField=params["ENABLE_ERRFIELD"], 
                               att_mult=params["CONFIG_TOR"])
    b_hidra.set_nonPer_errField()
    b_hidra.addFieldPerturbation(coilCurrent=params["CURRENT_HEL"], 
                                 att_mult=params["CONFIG_HEL"])

    ## SET UP INITIAL CONDITIONS
    ic_radii = np.array(np.linspace(params["START_RADIUS"], params["END_RADIUS"], params["NLINES"]))
    ic_theta = params["IC_THETA_DEG"] * np.pi/180.
    ic_phi = params["IC_PHI_DEG"] * np.pi/180.
    init_conds_rtp = np.array([[ic_r, ic_theta, ic_phi] for ic_r in ic_radii])

    ## GENERATE POINCARE PLOTS
    solver_args = [params["SOLVER"], params["RTOL"], params["ATOL"], params["NTHREADS"], params["DOUBLE_LINE"]]
    poincare = Poincare(simIO, *solver_args)
    poincare.set_conditions(init_conds_rtp, params["SPINS"], b_hidra, nplanes=params["NPLANES"])
    out_tMax = poincare.run()[0]

    ## IDENTIFY LAST-CLOSED FLUX SURFACE
    poincare.identifyLCFS(LCFStype='inner', t_maxs=out_tMax)

    ## END RUN ##
    simIO.log.info('## SIM FINISHED ##\n\n\n\n')

if __name__ == '__main__':
    main()
