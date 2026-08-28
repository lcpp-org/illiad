"""High-level SOL trace and regularization workflows."""

from pathlib import Path

import numpy as np
import torch

from .regularizer import (
    GridIndexSpool,
    PHI_FILENAME,
    RHO_FILENAME,
    THETA_FILENAME,
    RegularGridAccumulator,
    make_regular_grid,
    plot_field,
    regularize_accumulator,
    validate_regularizer_settings,
)
from .tracer import (
    SOLTracer,
    save_trace_metadata,
    trace_connection_length_volume_paired,
    validate_runtime_settings,
)


class _TraceValueSource:
    """Minimal value source used to scale direct regular-grid plots."""

    def __init__(self, plane_phi_deg, connection_length_m):
        self.plane_phi_deg = np.asarray(plane_phi_deg)
        self.plane_count = int(self.plane_phi_deg.size)
        self._values = np.asarray(connection_length_m)

    def iter_value_chunks(self, chunk_size):
        for start in range(0, self._values.size, chunk_size):
            yield self._values[start : start + chunk_size]


class SOLTraceRegularizer:
    """Trace paired field lines directly into bounded regular statistics."""

    def __init__(self, io_handler, magnetic_field, trace_params, regular_params):
        self.simIO = io_handler
        self.field_model = magnetic_field
        self.trace_params = dict(trace_params)
        self.regular_params = dict(regular_params)
        if self.trace_params["ANLYS_DIR"] != self.regular_params["ANLYS_DIR"]:
            raise ValueError(
                "TRACE.ANLYS_DIR and REGULARIZE.ANLYS_DIR must match in "
                "trace_regularize mode."
            )
        if self.trace_params["LCFS_INDEX"] != self.regular_params["LCFS_INDEX"]:
            raise ValueError(
                "TRACE.LCFS_INDEX and REGULARIZE.LCFS_INDEX must match in "
                "trace_regularize mode."
            )
        if not np.isclose(
            self.trace_params["VESSEL_RADIUS_M"],
            self.regular_params["VESSEL_RADIUS_M"],
        ):
            raise ValueError(
                "TRACE.VESSEL_RADIUS_M and REGULARIZE.VESSEL_RADIUS_M must "
                "match in trace_regularize mode."
            )
        validate_runtime_settings(self.trace_params)
        validate_regularizer_settings(self.regular_params)
        if self.trace_params["BATCH_SIZE"] < 2:
            raise ValueError(
                "TRACE.BATCH_SIZE must be at least 2 in trace_regularize "
                "mode so both directions fit in one batch."
            )
        self.trace_data = None
        self.field = None
        self.output_path = None
        self.data_dir = None

    def run(self):
        trace_params = dict(self.trace_params)
        regular_params = self.regular_params
        output_subdir = regular_params["ANLYS_SUBDIR"]
        trace_params["ANLYS_SUBDIR"] = output_subdir
        tracer = SOLTracer(self.simIO, self.field_model, trace_params)
        if tracer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(tracer.device)
        tracer.build_initial_conditions()
        tracer.log_inputs()

        rho, theta, _, _, grid_x, grid_z = make_regular_grid(
            regular_params["N_RHO"],
            regular_params["N_THETA"],
            regular_params["RHO_MIN"],
            regular_params["RHO_MAX"],
        )
        accumulator = RegularGridAccumulator(
            trace_params["N_PLANES"],
            regular_params,
        )
        self.data_dir = Path(self.simIO.data_dir) / output_subdir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir = self.data_dir / ".trace_regularize_scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        def sink_factory(batch_number):
            return GridIndexSpool(
                scratch_dir / f"batch_{batch_number:06d}.bin",
                trace_params["N_PLANES"],
                self.field_model.R0,
                regular_params,
            )

        self.trace_data = trace_connection_length_volume_paired(
            tracer.seed_data,
            trace_params,
            self.field_model,
            trace_params["STEP_SIZE_M"],
            trace_params["BATCH_SIZE"],
            trace_params["CROSSING_BUFFER_SIZE"],
            sink_factory,
            accumulator,
            regular_params["RAW_CHUNK_SIZE"],
            self.simIO,
            trace_params["SHOW_PROGRESS"],
            trace_params["STEP_CHUNK_SIZE"],
            tracer.compile_chunks,
            trace_params["INTEGRATOR"],
            trace_params["MIN_FIELD_MAGNITUDE"],
            trace_params["WALL_BISECTION_STEPS"],
            trace_params["PROGRESS_REFRESH_STEPS"],
            trace_params["PROGRESS_INTERVAL_STEPS"],
        )
        try:
            scratch_dir.rmdir()
        except OSError:
            self.simIO.log.warning(
                "Trace scratch directory was not empty after processing: %s",
                scratch_dir,
            )

        save_trace_metadata(
            self.simIO,
            tracer.seed_data,
            self.trace_data,
            self.field_model.R0,
            output_subdir,
        )
        self.simIO.saveNumpyData(
            rho,
            RHO_FILENAME.removesuffix(".npy"),
            subdir=output_subdir,
        )
        self.simIO.saveNumpyData(
            theta,
            THETA_FILENAME.removesuffix(".npy"),
            subdir=output_subdir,
        )
        self.simIO.saveNumpyData(
            self.trace_data["plane_phi_deg"],
            PHI_FILENAME.removesuffix(".npy"),
            subdir=output_subdir,
        )
        self.output_path = self.data_dir / regular_params[
            "OUTPUT_FIELD_FILENAME"
        ]
        self.simIO.inputsBoilerplate(
            "DIRECT TRACE-REGULARIZE INPUTS",
            {
                "MODE": "trace_regularize",
                "TRACE": trace_params,
                "REGULARIZE": regular_params,
                "FIELD_SHAPE": (
                    trace_params["N_PLANES"],
                    theta.size,
                    rho.size,
                ),
                "RAW_CROSSINGS_RETAINED": False,
                "ACCUMULATOR_BYTES": (
                    accumulator.value_sum.nbytes
                    + accumulator.sample_count.nbytes
                ),
            },
        )
        self.field = regularize_accumulator(
            trace_params["ANLYS_DIR"],
            accumulator,
            self.trace_data["plane_phi_deg"],
            trace_params["LCFS_INDEX"],
            rho,
            theta,
            grid_x,
            grid_z,
            self.output_path,
            self.simIO,
            regular_params,
        )
        self.simIO.log.info(
            "Saved direct regular connection-length field: %s",
            self.output_path,
        )

        if regular_params["GENERATE_PLOTS"]:
            value_source = _TraceValueSource(
                self.trace_data["plane_phi_deg"],
                self.trace_data["connection_length"],
            )
            plot_field(
                trace_params["ANLYS_DIR"],
                self.field,
                rho,
                theta,
                value_source,
                trace_params["LCFS_INDEX"],
                self.simIO,
                regular_params,
            )

        hit_count = np.count_nonzero(self.trace_data["hit_wall"])
        self.simIO.log.info(
            "Wall intersections: %d of %d directional traces.",
            hit_count,
            self.trace_data["hit_wall"].size,
        )
        if tracer.device.type == "cuda":
            peak_gib = torch.cuda.max_memory_allocated(tracer.device) / (
                1024.0**3
            )
            self.simIO.log.info("PEAK CUDA MEMORY ALLOCATED: %.3f GiB", peak_gib)
        self.simIO.log.info("## DIRECT TRACE-REGULARIZE FINISHED ##")
        return self.field
