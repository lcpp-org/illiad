"""Interpolate normalized flux and generate the electric field."""

import argparse
from illiad.flux import FluxGradientor, FluxInterpolator
from illiad.mesh import Mesh
from illiad.io import IOHandler
from illiad.utilities.run_config import load_inputs_json, merge_input_params, normalize_phi_gens

DEFAULT_INPUTS = {
    "ANLYS_DIR": "It-0486_Ih-0790_PHI180_1500spins_105Lines_LSODA1e9_newEvents",
    "ANLYS_SUBDIR": "LCFS30_20x720_furtherTESTING",
    "FIELD_FILE_TOR": "input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy",
    "FIELD_FILE_HEL": "input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy",

    "CURRENT_TOR": 0.486,
    "CURRENT_HEL": 0.790,
    "CONFIG_TOR": "default_toroidal",
    "CONFIG_HEL": "default_helical",
    "ENABLE_ERRFIELD": True,

    "LCFS_INDEX": 30,
    "NPHI": 360,
    "NTHETA": 720,
    "MAX_SUBSETS": 4,
    "SMALLEST_ISLAND_INDEX": 61,

    "ALPHA": 1.0,
    "DEBUG": True,
    "INV_SURF_INDICES": [],
    "GUESS_PHI_INDEX": -3,
    "OUTPUT_FILE_NAME": "LCFS30alpha1p0",

    "RBF_KERNEL": "multiquadric",
    "RBF_NEIGHBORS": 128,
    "RBF_SMOOTHING": 1.0,
    "RBF_EPSILON": 1000.0,
    "FLUX_INTERPOLATION_MODE": "3d",
    "RBF_PHI_HALF_WINDOW": 2,
    "RBF_PHI_SCALE": 0.72,
    "RBF_POINTS_PER_SURFACE_PER_PHI": 72,

    "LEGACY_FILTER_GRADIENTS_OUTSIDE_LCFS": False,
    "GRADIENT_FILTER_BUFFER": 0.01,
}
_CLI_INPUTS = object()


def parse_args():
    parser = argparse.ArgumentParser(description="Run ILLIAD flux interpolation and gradient generation.")
    parser.add_argument(
        "--inputs-json",
        default=None,
        help="Optional path to a JSON object overriding built-in workflow defaults.",
    )
    return parser.parse_args()


def setup_IO(params, log_name, logger_name):
    anlys_dir = params["ANLYS_DIR"]
    anlys_subdir = params["ANLYS_SUBDIR"]
    simIO = IOHandler(anlys_dir)
    simIO.setActiveSubDir(anlys_subdir)
    simIO.startLog(log_name=log_name, subdir=anlys_subdir, logger_name=logger_name)

    return simIO


def log_inputs(io_handler, title, params, input_keys):
    io_handler.inputsBoilerplate(title, params, input_keys)


def setup_field(params):
    ## DEFINE MESH AND LOAD MAGNETIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.setErrorField()
    b_hidra.loadCartesianField(coilCurrent=params["CURRENT_TOR"],
                               errField=params["ENABLE_ERRFIELD"],
                               att_mult=params["CONFIG_TOR"])
    b_hidra.addFieldPerturbation(coilCurrent=params["CURRENT_HEL"],
                                 att_mult=params["CONFIG_HEL"])

    return b_hidra


def main(input_overrides=_CLI_INPUTS):
    if input_overrides is _CLI_INPUTS:
        args = parse_args()
        input_overrides = load_inputs_json(args.inputs_json, "Flux gradient inputs") if args.inputs_json else None
    params = merge_input_params(DEFAULT_INPUTS, input_overrides)

    b_hidra = setup_field(params)

    ## RUN ANALYSIS
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
            "FLUX_INTERPOLATION_MODE",
            "RBF_PHI_HALF_WINDOW",
            "RBF_PHI_SCALE",
            "RBF_POINTS_PER_SURFACE_PER_PHI",
        ],)
    flux_interpolator = FluxInterpolator(interpIO, b_hidra, params)
    flux_interpolator.run()

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
            "LEGACY_FILTER_GRADIENTS_OUTSIDE_LCFS",
            "GRADIENT_FILTER_BUFFER",
        ],)
    flux_gradientor = FluxGradientor(gradIO, b_hidra, params)
    flux_gradientor.run()


if __name__ == '__main__':
    main()
