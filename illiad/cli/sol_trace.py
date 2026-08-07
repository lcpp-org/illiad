"""Command-line adapter for scrape-off-layer field-line tracing."""

import argparse

from illiad.sol import (
    SOLTracer,
    build_torch_magnetic_field,
    resolve_device,
)
from illiad.io import IOHandler
from illiad.utilities.run_config import load_inputs_json, merge_input_params
from illiad.sol.tracer import validate_runtime_settings


DEFAULT_INPUTS = {
    "ANLYS_DIR": "DEFAULT",
    "ANLYS_SUBDIR": "SOLTrace",

    "CURRENT_TOR": 0.486,
    "CURRENT_HEL": 0.900,
    "CONFIG_TOR": "default_toroidal",
    "CONFIG_HEL": "default_helical",
    "ENABLE_ERRFIELD": True,
    "MAJOR_RADIUS_M": 0.72,
    "VESSEL_RADIUS_M": 0.19,

    "LCFS_INDEX": 19,
    "N_PLANES": 360,
    "N_SEED_PLANES": 1,
    "SEED_PHI_DEG": 324.0,
    "SPINS": 250,
    "N_RHO": 40,
    "N_THETA": 45,
    "RHO_MIN": 0.002,
    "RHO_MAX": 0.188,
    "LCFS_CLEARANCE_M": 0.0,
    "LCFS_SPLINE_SMOOTHING": 1e-5,
    "LCFS_BOUNDARY_POINTS": 1000,

    "DEVICE": "auto",
    "INTEGRATOR": "midpoint",
    "STEP_SIZE_M": 0.002,
    "BATCH_SIZE": 524288,
    "CROSSING_BUFFER_SIZE": 1000000,
    "STEP_CHUNK_SIZE": 8,
    "COMPILE_STEP_CHUNKS": True,
    "WALL_BISECTION_STEPS": 24,
    "MIN_FIELD_MAGNITUDE": 1e-14,
    "PROGRESS_INTERVAL_STEPS": 5000,
    "PROGRESS_REFRESH_STEPS": 100,
    "SHOW_PROGRESS": True,

    "GENERATE_PLOTS": True,
    "COLOR_SCALE": "log",
    "COLORMAP": "viridis",
    "N_LEVELS": 50,
    "VMIN": None,
    "VMAX": None,
    "DPI": 300,
    "PLOT_MAX_SAMPLES": 150000,
    "PLOT_SAMPLE_SEED": 0,
    "PHYSICAL_PHI_OFFSET_DEG": 198.0,
}

_CLI_INPUTS = object()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trace open SOL field lines with ILLIAD's PyTorch solver."
    )
    parser.add_argument(
        "--inputs-json",
        default=None,
        help="Optional path to a JSON object overriding built-in defaults.",
    )
    return parser.parse_args()


def main(input_overrides=_CLI_INPUTS):
    if input_overrides is _CLI_INPUTS:
        args = parse_args()
        input_overrides = (
            load_inputs_json(args.inputs_json, "SOL trace inputs")
            if args.inputs_json
            else None
        )
    params = merge_input_params(DEFAULT_INPUTS, input_overrides)
    validate_runtime_settings(params)

    simIO = IOHandler(params["ANLYS_DIR"])
    simIO.startLog(
        log_name="solTrace.log",
        subdir=params["ANLYS_SUBDIR"],
        logger_name="SOLTracer",
    )

    device = resolve_device(params["DEVICE"])
    magnetic_field = build_torch_magnetic_field(params, device)
    analysis = SOLTracer(simIO, magnetic_field, params)
    analysis.run()

    print(f"Saved raw data: {analysis.data_dir}")
    if params["GENERATE_PLOTS"]:
        print(
            "Saved contour plots: "
            f"{simIO.plot_dir}/{params['ANLYS_SUBDIR']}"
        )


if __name__ == "__main__":
    main()
