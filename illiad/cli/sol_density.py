"""Command-line adapter for SOL plasma-density construction."""

import argparse

from illiad.io import IOHandler
from illiad.sol import SOLDensity
from illiad.utilities.run_config import load_inputs_json, merge_input_params


DEFAULT_INPUTS = {
    "ANLYS_DIR": "IOTA3_1000sp_atol1e-9",
    "ANLYS_SUBDIR": (
        "ConLenVolume_REDO_250spins_rk1mm_RegularGrid_Density_v2"
    ),
    "SOL_SUBDIR": "ConLenVolume_REDO_250spins_rk1mm_RegularGrid",
    "SOL_FIELD_FILENAME": "connection_length_field_m.npy",
    "NFIELD_SUBDIR": "LCFS40",
    "NFIELD_FILENAME": "nField_LCFS40_linear.npy",
    "LCFS_INDEX": None,

    "N_AXIS": 1.0,
    "N_LCFS": 0.3,
    "N_WALL": 1e-4,
    "ALPHA": 0.85,
    "SOL_BETA": 0.5,
    "L_PARALLEL_0_M": None,

    "GENERATE_PLOTS": True,
    "SHOW_LCFS": False,
    "COLOR_SCALE": "log",
    "SHOW_PROGRESS": True,
}

_CLI_INPUTS = object()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct an ILLIAD piecewise core/SOL plasma-density field."
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
            load_inputs_json(args.inputs, "SOL density inputs")
            if args.inputs else None
        )
    params = merge_input_params(DEFAULT_INPUTS, input_overrides)

    sim_io = IOHandler(params["ANLYS_DIR"])
    sim_io.startLog(
        log_name="solDensity.log",
        subdir=params["ANLYS_SUBDIR"],
        logger_name="SOLDensity",
    )
    analysis = SOLDensity(sim_io, params)
    analysis.run()

    print(f"Saved density field: {analysis.output_path}")
    print(f"Saved density metadata: {analysis.metadata_path}")
    if params["GENERATE_PLOTS"]:
        print(f"Saved density plots: {sim_io.plot_dir}/{params['ANLYS_SUBDIR']}")


if __name__ == "__main__":
    main()
