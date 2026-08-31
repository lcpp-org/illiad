"""Unified CLI for SOL tracing and connection-length regularization."""

import argparse
from copy import deepcopy

from illiad.cli.sol_regularize import DEFAULT_INPUTS as REGULAR_DEFAULTS
from illiad.cli.sol_trace import DEFAULT_INPUTS as TRACE_DEFAULTS
from illiad.io import IOHandler
from illiad.sol import (
    SOLRegularizer,
    SOLTraceRegularizer,
    SOLTracer,
    build_torch_magnetic_field,
    resolve_device,
)
from illiad.utilities.run_config import load_inputs_json


DEFAULT_INPUTS = {
    "TRACE": TRACE_DEFAULTS,
    "REGULARIZE": REGULAR_DEFAULTS,
}
DEFAULT_INPUTS["TRACE"] = {
    **DEFAULT_INPUTS["TRACE"],
    "BATCH_SIZE": 16384,
    "CROSSING_BUFFER_SIZE": 250000,
}
MODES = ("trace", "regularize", "trace_regularize")
_CLI_INPUTS = object()


def merge_workflow_inputs(overrides=None):
    """Deep-merge the two independently named stage configurations."""
    params = deepcopy(DEFAULT_INPUTS)
    if not overrides:
        return params
    unknown = set(overrides) - set(params)
    if unknown:
        raise ValueError(
            "Unknown top-level connection-length inputs: "
            + ", ".join(sorted(unknown))
        )
    for section_name, section_overrides in overrides.items():
        if not isinstance(section_overrides, dict):
            raise ValueError(f"{section_name} must be a JSON object.")
        params[section_name].update(section_overrides)
    return params


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run SOL tracing, saved-crossing regularization, or direct "
            "bounded-memory trace regularization."
        )
    )
    parser.add_argument(
        "mode",
        choices=MODES,
        help=(
            "trace retains raw shards; regularize reads existing raw data; "
            "trace_regularize writes the regular field without retaining raw "
            "crossings"
        ),
    )
    parser.add_argument(
        "--inputs",
        help="Optional path to nested TRACE/REGULARIZE JSON overrides.",
    )
    return parser.parse_args()


def main(input_overrides=_CLI_INPUTS, mode=None):
    if input_overrides is _CLI_INPUTS:
        args = parse_args()
        mode = args.mode
        input_overrides = (
            load_inputs_json(args.inputs, "SOL connection-length inputs")
            if args.inputs
            else None
        )
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}.")
    params = merge_workflow_inputs(input_overrides)
    trace_params = params["TRACE"]
    regular_params = params["REGULARIZE"]

    if mode == "regularize":
        sim_io = IOHandler(regular_params["ANLYS_DIR"])
        sim_io.startLog(
            log_name="solRegularizer.log",
            subdir=regular_params["ANLYS_SUBDIR"],
            logger_name="SOLRegularizer",
        )
        analysis = SOLRegularizer(sim_io, regular_params)
        analysis.run()
        print(f"Saved regular connection-length field: {analysis.output_path}")
        return analysis

    device = resolve_device(trace_params["DEVICE"])
    magnetic_field = build_torch_magnetic_field(trace_params, device)
    if mode == "trace":
        sim_io = IOHandler(trace_params["ANLYS_DIR"])
        sim_io.startLog(
            log_name="solTrace.log",
            subdir=trace_params["ANLYS_SUBDIR"],
            logger_name="SOLTracer",
        )
        analysis = SOLTracer(sim_io, magnetic_field, trace_params)
        analysis.run()
        print(f"Saved sharded raw data: {analysis.data_dir}")
        return analysis

    sim_io = IOHandler(regular_params["ANLYS_DIR"])
    sim_io.startLog(
        log_name="solTraceRegularizer.log",
        subdir=regular_params["ANLYS_SUBDIR"],
        logger_name="SOLTraceRegularizer",
    )
    analysis = SOLTraceRegularizer(
        sim_io,
        magnetic_field,
        trace_params,
        regular_params,
    )
    analysis.run()
    print(f"Saved direct regular connection-length field: {analysis.output_path}")
    return analysis


if __name__ == "__main__":
    main()
