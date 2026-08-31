"""Command-line adapter for SOL connection-length regularization."""

import argparse

from illiad.io import IOHandler
from illiad.sol import SOLRegularizer
from illiad.utilities.run_config import load_inputs_json, merge_input_params


DEFAULT_INPUTS = {
    "ANLYS_DIR": "DEFAULT",
    "ANLYS_SUBDIR": "SOLTrace_RegularGrid",
    "TRACE_SUBDIR": "SOLTrace",
    "LCFS_INDEX": 19,

    "N_RHO": 191,
    "N_THETA": 180,
    "RHO_MIN": 0.0,
    "RHO_MAX": 0.19,

    "INTERPOLATION_SPACE": "log",
    "FILL_METHOD": "idw",
    "IDW_NEIGHBORS": 8,
    "IDW_POWER": 2.0,
    "TREE_WORKERS": -1,
    "RAW_CHUNK_SIZE": 250000,
    "OUTPUT_FIELD_FILENAME": "connection_length_field_m.npy",

    "GENERATE_PLOTS": True,
    "SHOW_PROGRESS": True,
    "COLOR_SCALE": "log",
    "COLORMAP": "afmhot",
    "N_LEVELS": 50,
    "VMIN": None,
    "VMAX": None,
    "CONTOUR_EXTEND": "both",
    "DPI": 250,
    "VESSEL_RADIUS_M": 0.19,
    "PHYSICAL_PHI_OFFSET_DEG": 198.0,
}

_CLI_INPUTS = object()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Regularize saved SOL crossings onto an ILLIAD "
            "(phi, theta, rho) field mesh."
        )
    )
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
        help="Optional path to a JSON object overriding built-in defaults.",
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
        input_overrides = (
            load_inputs_json(args.inputs, "SOL regularizer inputs")
            if args.inputs
            else None
        )
    params = merge_input_params(DEFAULT_INPUTS, input_overrides)

    sim_io = IOHandler(params["ANLYS_DIR"])
    sim_io.startLog(
        log_name="solRegularizer.log",
        subdir=params["ANLYS_SUBDIR"],
        logger_name="SOLRegularizer",
    )
    analysis = SOLRegularizer(sim_io, params)
    analysis.run()

    print(f"Saved regular connection-length field: {analysis.output_path}")
    if params["GENERATE_PLOTS"]:
        print(
            "Saved regular-grid plots: "
            f"{sim_io.plot_dir}/{params['ANLYS_SUBDIR']}"
        )


if __name__ == "__main__":
    main()
