import argparse
import numpy as np
from classes.mesh import Mesh
import classes.calculate_flux as fc
import classes.flux_gradientor as fg
import classes.interpolate_flux as fi
from classes.iohandler import IOHandler
from utility.run_config import load_inputs_json, merge_input_params, normalize_phi_gens

N_phi = 10
input_params = {
    # 'ANLYS_DIR': "AcceptedIota3_1500spins_atole-9",
    # 'ANLYS_SUBDIR': "LCFS19_360x180_ARTICLE_smooth3e-5",
    #'ANLYS_SUBDIR': "LCFS14_360x180_ARTICLE",
    #'ANLYS_DIR': "AAAnewIO_iota3FWD_phi306_LSODA",
    'ANLYS_DIR': "iota3_1200spins_53Lines_LSODA_165deg",
    #'ANLYS_SUBDIR': "LCFS21_20x180",
    #'ANLYS_SUBDIR': "LCFS40_20x360_fixd2_abserr1e-2_relerr1e-1_meshErrTest2",
    'ANLYS_SUBDIR': "Aiota3_gradtest3_165deg",

    'FIELD_FILE_TOR': 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy',
    'FIELD_FILE_HEL': 'input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy',
    'CURRENT_TOR': 0.486, #[kA]
    'CURRENT_HEL': 0.900, #[kA]
    'CONFIG_TOR': 'default_toroidal',
    'CONFIG_HEL': 'default_helical',
    'ENABLE_ERRFIELD': True,
    'LCFS_INDEX': 10,
    'NPHI': N_phi,
    'NTHETA': 180,
    'MAX_SUBSETS': 3,
    'SMALLEST_ISLAND_INDEX': 53, #61, #53 #39
    # 'SMOOTH_FCTR': 3e-5, #7.5e-6 #baseline 1e-6
    # 'INTEGRATE_EPSABS': 5e-1,
    # 'INTEGRATE_EPSREL': 5e-2,
    # 'ISLAND_ALGORITHM': 'histogram', # 'kmeans', 'spectral'
    # 'HIST_BINS': 90,
    # 'PLOT_ALL': True,
    # 'BIG_MESH': False

    }

_CLI_INPUTS = object()

def parse_args():
    parser = argparse.ArgumentParser(description="Run ILLIAD flux interpolation and gradient generation.")
    parser.add_argument(
        "--inputs-json",
        default=None,
        help="Optional path to a JSON object overriding runFluxGrad.py defaults.",
    )
    return parser.parse_args()


def setup_IO(params, log_name, logger_name):
    ANLYS_DIR = params['ANLYS_DIR']
    ANLYS_SUBDIR = params['ANLYS_SUBDIR']
    simIO = IOHandler(ANLYS_DIR)
    simIO.setActiveSubDir(ANLYS_SUBDIR)
    simIO.startLog(log_name=log_name, subdir=ANLYS_SUBDIR, logger_name=logger_name)

    return simIO


def log_inputs(io_handler, title, params, input_keys):
    io_handler.inputsBoilerplate(title, params, input_keys)


def setupField(params):
    ## DEFINE MESH AND LOAD MAGNETIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.setErrorField()
    b_hidra.loadCartesianField(coilCurrent=params['CURRENT_TOR'],
                               errField=params['ENABLE_ERRFIELD'],
                               att_mult=params['CONFIG_TOR'])
    b_hidra.addFieldPerturbation(coilCurrent=params['CURRENT_HEL'],
                                 att_mult=params['CONFIG_HEL'])
    
    return b_hidra


def main(input_params_override=_CLI_INPUTS):
    if input_params_override is _CLI_INPUTS:
        args = parse_args()
        input_params_override = load_inputs_json(args.inputs_json, "Flux gradient inputs") if args.inputs_json else None
    params = merge_input_params(input_params, input_params_override)

    b_hidra = setupField(params)

    ## RUN ANALYSIS
    params.setdefault('ALPHA', 1.0)
    params.setdefault('DEBUG', True)
    params.setdefault('LCFS_INDEX', 10)
    params.setdefault('INV_SURF_INDICES', [20])
    params.setdefault('GUESS_PHI_INDEX', -3)
    params.setdefault('OUTPUT_FILE_NAME', "to_delete_test1")
    normalize_phi_gens(params)

    interpIO = setup_IO(params, log_name="fluxInterpolator.log", logger_name="FluxInterpolator")
    log_inputs(interpIO, "FLUX INTERPOLATOR INPUTS", params,
        [
            "ANLYS_DIR",
            "ANLYS_SUBDIR",
            "CURRENT_TOR",
            "CURRENT_HEL",
            "CONFIG_TOR",
            "CONFIG_HEL",
            "ENABLE_ERRFIELD",
            "LCFS_INDEX",
            "SMALLEST_ISLAND_INDEX",
            "PHI_GENs",
            "MAX_SUBSETS",
            "ALPHA",
            "INV_SURF_INDICES",
            "GUESS_PHI_INDEX",
            "OUTPUT_FILE_NAME",
            "RBF_KERNEL",
            "RBF_NEIGHBORS",
            "RBF_SMOOTHING",
            "RBF_EPSILON",
        ],)
    flux_grad = fi.FluxInterpolator(interpIO, b_hidra, params)
    flux_grad.run()

    gradIO = setup_IO(params, log_name="fluxGradientor.log", logger_name="FluxGradientor")
    log_inputs(gradIO, "FLUX GRADIENTOR INPUTS", params, 
        [
            "ANLYS_DIR",
            "ANLYS_SUBDIR",
            "CURRENT_TOR",
            "CURRENT_HEL",
            "CONFIG_TOR",
            "CONFIG_HEL",
            "ENABLE_ERRFIELD",
            "LCFS_INDEX",
            "PHI_GENs",
            "OUTPUT_FILE_NAME",
        ],)
    flux_grad = fg.FluxGradientor(gradIO, b_hidra, params)
    flux_grad.run()


if __name__ == "__main__":
    main()