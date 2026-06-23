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
import matplotlib.pyplot as plt
from classes.iohandler import IOHandler
from classes.mesh import Mesh
from classes.poincare import Poincare
import utility.phi_events as phi_event_defs
from utility.run_config import load_inputs_json, merge_input_params

#################
## USER INPUTS ##
#################

# DEFINE FIELDS #
CURRENT_TOR = 0.486 #[kA]
CURRENT_HEL = 0.790 #[kA]
CONFIG_TOR = "default_toroidal"
CONFIG_HEL = "default_helical"
ENABLE_ERRFIELD = True

# DEFINE INITIAL CONDITIONS #
IC_PHI_DEG = 306. #[deg]
IC_THETA_DEG = 180. #[deg]
START_RADIUS = 0.150 #[m]
END_RADIUS = 0.020 #[m]
NLINES = 14 + 13 + 26 #+ 52 + 104
SPINS = 1000 #1500 # max length, SPIN = 2pi*R0 [meters]
NPLANES = 360

# DEFINE SOLVER PARAMETERS #
SOLVER = "LSODA"#"RK45"#
RTOL = 2.49e-12
ATOL = 1e-8

##  NTHREADS:
#   n > 0: use n threads
#   n = 0: use all available threads
#   n < 0: use all but the last n threads
NTHREADS = -1

# DOUBLE_LINE:
#   True  : trace each field line in both +B and -B directions from the initial position
#           Only use when NTHREADS > NLINES.
#   False : trace each field line only in the +B direction
DOUBLE_LINE = False

# DEFINE PLOT PARAMETERS #
PLOT_TITLE_ON = False
PLOT_DPI = 100

# DEFINE OUTPUT DIRECTORY #
#OUTPUT_DIR = f"Iota4FWD_{SPINS}spins_{NLINES}Lines_RK45_1e8_newEvents"
OUTPUT_DIR = f"Iiota4FWD_1000spins_54LinesLSODA_100DPI"


DEFAULT_INPUTS = {
    "CURRENT_TOR": CURRENT_TOR,
    "CURRENT_HEL": CURRENT_HEL,
    "CONFIG_TOR": CONFIG_TOR,
    "CONFIG_HEL": CONFIG_HEL,
    "ENABLE_ERRFIELD": ENABLE_ERRFIELD,
    "IC_PHI_DEG": IC_PHI_DEG,
    "IC_THETA_DEG": IC_THETA_DEG,
    "START_RADIUS": START_RADIUS,
    "END_RADIUS": END_RADIUS,
    "NLINES": NLINES,
    "SPINS": SPINS,
    "NPLANES": NPLANES,
    "SOLVER": SOLVER,
    "RTOL": RTOL,
    "ATOL": ATOL,
    "NTHREADS": NTHREADS,
    "DOUBLE_LINE": DOUBLE_LINE,
    "OUTPUT_DIR": OUTPUT_DIR,
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


def main(input_params=_CLI_INPUTS):
    
    """
    Main function to set up the mesh, load magnetic field data, and generate Poincare plots.
    """
    if input_params is _CLI_INPUTS:
        args = parse_args()
        input_params = load_inputs_json(args.inputs_json, "Poincare inputs") if args.inputs_json else None
    if input_params is not None:
        globals().update(merge_input_params(DEFAULT_INPUTS, input_params))

    ## SET UP RUN DIRECTORY (*DATA AND PLOTS WILL BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!*)
    simIO = IOHandler(OUTPUT_DIR) 
    simIO.startLog(log_name="poincare.log", subdir="Poincare", logger_name="Poincare")
    simIO.inputsBoilerplate(
        "POINCARE INPUTS",
        globals(),
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
    b_hidra.loadCartesianField(coilCurrent=CURRENT_TOR, errField=ENABLE_ERRFIELD, att_mult=CONFIG_TOR)
    b_hidra.set_nonPer_errField()
    b_hidra.addFieldPerturbation(coilCurrent=CURRENT_HEL, att_mult=CONFIG_HEL)

    ## SET UP INITIAL CONDITIONS
    ic_radii = np.linspace(START_RADIUS, END_RADIUS, NLINES)
    ic_theta = IC_THETA_DEG * np.pi/180.
    ic_phi = IC_PHI_DEG * np.pi/180.
    init_conds_rtp = np.array([[ic_r, ic_theta, ic_phi] for ic_r in ic_radii])

    ## GENERATE POINCARE PLOTS
    solver_args = [SOLVER, RTOL, ATOL, NTHREADS, DOUBLE_LINE]
    PoinCare = Poincare(simIO, *solver_args)
    PoinCare.set_conditions(init_conds_rtp, SPINS, b_hidra, nplanes=NPLANES)
    out_tMax = PoinCare.run(plot_args={'title_on': PLOT_TITLE_ON, 'dpi': PLOT_DPI})[0]

    ## IDENTIFY LAST-CLOSED FLUX SURFACE
    PoinCare.identifyLCFS(LCFStype='inner', t_maxs=out_tMax)

    ## END RUN ##
    simIO.log.info('## SIM FINISHED ##\n\n\n\n')

if __name__ == '__main__':
    main()
