"""Standalone Warp SOL tracer; run this file with --help for JSON usage.

TorchMesh is used only to load/combine the existing magnetic-field inputs.
Integration, wall refinement and crossing detection run in float64 Warp kernels.
The production SOL command and its Torch backend are unchanged.
"""

import argparse
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
try:
    import warp as wp
except ImportError as exc:
    raise ImportError("Warp SOL requires the optional dependency: pip install 'illiad-fieldlines[warp]'") from exc
from tqdm import tqdm

from illiad.boris_warp import VectorGrid, make_grid, sample_vector
from illiad.cli.sol_trace import DEFAULT_INPUTS
from illiad.io import IOHandler
from illiad.sol.tracer import (
    SOLTracer, build_torch_magnetic_field, resolve_device,
    validate_runtime_settings, save_torch_outputs,
)
from illiad.sol.crossings import PlaneShardWriter
from illiad.utilities.coordtrans import RTP_to_XYZ_many, XYZ_to_RTP_many
from illiad.utilities.run_config import load_inputs_json, merge_input_params


# Warp 1.17's default loop unrolling triggers a release-mode CUDA illegal
# access in this fused tracer on an RTX 4080 SUPER. Keeping the small nested
# interpolation loops as loops fixes the reproduced first-chunk failure while
# retaining release optimization. Scope this to the prototype, not Boris.
wp.set_module_options({"max_unroll": 0}, module=__name__)


@wp.func
def sol_radius(x: wp.vec3d, R0: wp.float64):
    # Use the SOL wall expression (avoids cancellation near the magnetic axis).
    radial = wp.sqrt(x[0]*x[0] + x[1]*x[1]) - R0
    return wp.sqrt(radial*radial + x[2]*x[2])


@wp.func
def field_direction(g: VectorGrid, x: wp.vec3d, sign: wp.float64,
                    minimum: wp.float64):
    b = sample_vector(g, x)
    magnitude = wp.length(b)
    valid = wp.isfinite(b[0]) and wp.isfinite(b[1]) and wp.isfinite(b[2])
    valid = valid and wp.isfinite(magnitude) and magnitude > minimum
    derivative = wp.vec3d(wp.float64(0.0))
    if valid:
        derivative = b * (sign / magnitude)
    return derivative, valid


@wp.func
def sol_step(g: VectorGrid, x: wp.vec3d, sign: wp.float64,
             h: wp.float64, rk4: int, minimum: wp.float64):
    k1, v1 = field_direction(g, x, sign, minimum)
    k2, v2 = field_direction(g, x + wp.float64(0.5)*h*k1, sign, minimum)
    candidate = x + h*k2
    valid = v1 and v2
    if rk4 == 1:
        k3, v3 = field_direction(g, x + wp.float64(0.5)*h*k2, sign, minimum)
        k4, v4 = field_direction(g, x + h*k3, sign, minimum)
        candidate = x + (h/wp.float64(6.0))*(k1 + wp.float64(2.0)*k2 + wp.float64(2.0)*k3 + k4)
        valid = valid and v3 and v4
    valid = valid and wp.isfinite(candidate[0]) and wp.isfinite(candidate[1]) and wp.isfinite(candidate[2])
    return candidate, valid


@wp.kernel
def trace_chunk(
    g: VectorGrid,
    positions: wp.array(dtype=wp.vec3d),
    phi: wp.array(dtype=wp.float64),
    signs: wp.array(dtype=wp.float64),
    lengths: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    walls: wp.array(dtype=wp.vec3d),
    first_step: int, steps: int, max_steps: int,
    step_size: wp.float64, max_length: wp.float64,
    rk4: int, minimum: wp.float64, bisections: int, n_planes: int,
    crossing_xyz: wp.array(dtype=wp.vec3d),
    crossing_plane: wp.array(dtype=wp.int32),
    crossing_row: wp.array(dtype=wp.int32),
    counters: wp.array(dtype=wp.int32),
):
    row = wp.tid()
    if status[row] != 0:
        return
    x = positions[row]
    unwrapped = phi[row]
    distance = lengths[row]
    state = int(0)  # 0 active, 1 wall, 2 length limit, 3 invalid field
    tau = wp.float64(6.283185307179586)
    spacing = tau / wp.float64(n_planes)
    epsilon = wp.float64(1.0e-10)
    for local_step in range(steps):
        h = wp.min(step_size, max_length - wp.float64(first_step + local_step)*step_size)
        candidate, valid = sol_step(g, x, signs[row], h, rk4, minimum)
        if not valid:
            state = 3
            break
        fraction = wp.float64(1.0)
        if sol_radius(candidate, g.R0) >= g.a:
            low = wp.float64(0.0)
            high = wp.float64(1.0)
            for iteration in range(bisections):
                middle = wp.float64(0.5)*(low + high)
                if sol_radius(x + middle*(candidate-x), g.R0) < g.a:
                    low = middle
                else:
                    high = middle
            fraction = high
            candidate = x + fraction*(candidate-x)
            walls[row] = candidate
            state = 1

        angle_change = -wp.atan2(candidate[1], candidate[0]) + wp.atan2(x[1], x[0])
        delta = wp.atan2(wp.sin(angle_change), wp.cos(angle_change))
        endpoint_phi = unwrapped + delta
        plane_id = int(0)
        crosses = bool(False)
        if delta > epsilon:
            plane_id = int(wp.floor((unwrapped + epsilon)/spacing)) + 1
            crosses = endpoint_phi + epsilon >= wp.float64(plane_id)*spacing
        elif delta < -epsilon:
            plane_id = int(wp.ceil((unwrapped - epsilon)/spacing)) - 1
            crosses = endpoint_phi - epsilon <= wp.float64(plane_id)*spacing
        if crosses:
            alpha = wp.clamp((wp.float64(plane_id)*spacing - unwrapped)/delta,
                             wp.float64(0.0), wp.float64(1.0))
            slot = wp.atomic_add(counters, 0, 1)
            if slot >= 0 and slot < crossing_xyz.shape[0]:
                crossing_xyz[slot] = x + alpha*(candidate-x)
                crossing_plane[slot] = ((plane_id - 1) % n_planes + n_planes) % n_planes
                crossing_row[slot] = row
            else:
                wp.atomic_add(counters, 2, 1)
        distance = distance + h*fraction
        x = candidate
        unwrapped = endpoint_phi
        if state == 1:
            break
        if first_step + local_step + 1 == max_steps:
            state = 2
            break
    positions[row] = x
    phi[row] = unwrapped
    lengths[row] = distance
    status[row] = state
    if state == 0:
        wp.atomic_add(counters, 1, 1)


def integrate_warp_batch(initial_xyz, initial_phi, fieldline_id, direction_sign,
                         grid, max_length, settings, sink, log=None):
    """Trace one directional batch, streaming crossings to an append_xyz sink.

    At most one crossing is possible per step under the shared step-size guard.
    Batch/chunk sizes are bounded so every possible event fits before flushing.
    State rows always retain their original field-line IDs; atomic event order
    is unspecified on CUDA.
    """
    validate_runtime_settings(settings)
    xyz = np.asarray(initial_xyz, dtype=np.float64)
    initial_phi = np.asarray(initial_phi, dtype=np.float64)
    ids = np.asarray(fieldline_id)
    signs = np.asarray(direction_sign, dtype=np.float64)
    count = len(xyz)
    capacity = settings["CROSSING_BUFFER_SIZE"]
    if xyz.shape != (count, 3) or any(a.shape != (count,) for a in (initial_phi, ids, signs)):
        raise ValueError("Batch arrays must have matching rows and XYZ shape (n, 3).")
    if count > capacity:
        raise ValueError("Directional batch must fit CROSSING_BUFFER_SIZE.")
    if not np.isfinite(xyz).all() or not np.isfinite(initial_phi).all() or not np.isin(signs, [-1, 1]).all():
        raise ValueError("Seeds must be finite and direction signs must be +/-1.")
    if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids > np.iinfo(np.int32).max):
        raise ValueError("Field-line IDs must be nonnegative int32-compatible integers.")
    if not np.isfinite(max_length) or max_length <= 0:
        raise ValueError("max_length must be positive and finite.")
    if not np.allclose([grid.R0, grid.a], [settings["MAJOR_RADIUS_M"], settings["VESSEL_RADIUS_M"]], rtol=0, atol=1e-14):
        raise ValueError("Grid geometry must match SOL settings.")
    radii = np.hypot(np.hypot(xyz[:, 0], xyz[:, 1]) - grid.R0, xyz[:, 2])
    if np.any(radii > grid.a + 1e-14):
        raise ValueError("SOL seeds must lie inside or on the vessel.")
    step_size = settings["STEP_SIZE_M"]
    max_steps = int(np.ceil(max_length / step_size))
    if max_steps > np.iinfo(np.int32).max or capacity > np.iinfo(np.int32).max:
        raise ValueError("Step count and crossing capacity must fit int32.")
    device = grid.values.device
    chunk_size = min(settings["STEP_CHUNK_SIZE"], max_steps, capacity // max(count, 1))
    event_capacity = max(1, count*chunk_size)
    positions = wp.array(xyz, dtype=wp.vec3d, device=device)
    phi = wp.array(initial_phi, dtype=wp.float64, device=device)
    directions = wp.array(signs, dtype=wp.float64, device=device)
    lengths = wp.zeros(count, dtype=wp.float64, device=device)
    status = wp.zeros(count, dtype=wp.int32, device=device)
    walls = wp.array(np.full((count, 3), np.nan), dtype=wp.vec3d, device=device)
    events = wp.empty(event_capacity, dtype=wp.vec3d, device=device)
    planes = wp.empty(event_capacity, dtype=wp.int32, device=device)
    rows = wp.empty(event_capacity, dtype=wp.int32, device=device)
    counters = wp.zeros(3, dtype=wp.int32, device=device)
    if log:
        log.info("Warp: %d directions, %d steps/chunk, %d event slots.", count, chunk_size, event_capacity)
    with tqdm(total=max_steps, desc="Warp SOL", unit="step", disable=not settings["SHOW_PROGRESS"], mininterval=10) as progress:
        for first in range(0, max_steps, chunk_size):
            if count == 0:
                break
            steps = min(chunk_size, max_steps-first)
            counters.zero_()
            wp.launch(trace_chunk, dim=count, inputs=[
                grid, positions, phi, directions, lengths, status, walls,
                first, steps, max_steps, step_size, max_length,
                int(settings["INTEGRATOR"] == "rk4"), settings["MIN_FIELD_MAGNITUDE"],
                settings["WALL_BISECTION_STEPS"], settings["N_PLANES"],
                events, planes, rows, counters,
            ], device=device)
            captured, active, overflow = counters.numpy()
            if overflow or captured < 0 or captured > event_capacity:
                raise RuntimeError(f"Warp SOL crossing buffer overflow at step {first}: "
                                   f"{captured} events for {event_capacity} slots.")
            if captured:
                points = events[:int(captured)].numpy()
                plane = planes[:int(captured)].numpy()
                row = rows[:int(captured)].numpy()
                order = np.argsort(plane, kind="stable")
                unique, starts, sizes = np.unique(plane[order], return_index=True, return_counts=True)
                for index, start, size in zip(unique, starts, sizes):
                    selection = order[start:start+size]
                    sink.append_xyz(int(index), points[selection], ids[row[selection]], signs[row[selection]].astype(np.int8))
            completed = first + steps
            progress.update(steps)
            if log and (completed % settings["PROGRESS_INTERVAL_STEPS"] < steps or active == 0):
                log.info("Warp step %d/%d: %d active.", completed, max_steps, active)
            if active == 0:
                break
    states = status.numpy()
    return dict(path_length=lengths.numpy(), wall_xyz=walls.numpy(),
                hit_wall=states == 1, reached_limit=states == 2,
                valid_trace=(states == 1) | (states == 2))


def trace_warp_volume(seed_data, settings, grid, sink, log=None):
    """Trace both directions and assemble the production SOL metadata layout."""
    rtp = seed_data["initial_conditions_rtp"]
    count = len(rtp)
    if count > np.iinfo(np.int32).max:
        raise ValueError("Field-line count exceeds the int32 raw-ID range.")
    max_length = 2*np.pi*grid.R0*settings["SPINS"]
    result = dict(direction_length=np.full((count, 2), np.nan),
                  wall_xyz=np.full((count, 2, 3), np.nan),
                  hit_wall=np.zeros((count, 2), dtype=bool),
                  reached_limit=np.zeros((count, 2), dtype=bool),
                  valid_trace=np.zeros((count, 2), dtype=bool))
    batch_size = min(settings["BATCH_SIZE"], settings["CROSSING_BUFFER_SIZE"])
    for start in range(0, 2*count, batch_size):
        directional_ids = np.arange(start, min(start+batch_size, 2*count))
        ids = directional_ids % count
        columns = directional_ids // count
        signs = np.where(columns == 0, 1.0, -1.0)
        batch = integrate_warp_batch(RTP_to_XYZ_many(rtp[ids], grid.R0), rtp[ids, 2],
                                     ids, signs, grid, max_length, settings, sink, log)
        for key in result:
            result[key][ids, columns] = batch["path_length" if key == "direction_length" else key]
    connection_length = result["direction_length"].sum(axis=1)
    connection_length[~result["valid_trace"].all(axis=1)] = np.nan
    wall_rtp = np.full_like(result["wall_xyz"], np.nan)
    finite = np.isfinite(result["wall_xyz"]).all(axis=-1)
    wall_rtp[finite] = XYZ_to_RTP_many(result["wall_xyz"][finite], grid.R0)
    result.update(connection_length=connection_length, wall_rtp=wall_rtp,
                  trace_length_limit_m=float(max_length), trace_spins=int(settings["SPINS"]),
                  trace_lcfs_index=int(settings["LCFS_INDEX"]),
                  trace_vessel_radius_m=float(grid.a),
                  plane_phi_deg=np.linspace(360/settings["N_PLANES"], 360, settings["N_PLANES"]))
    return result


class WarpSOLTracer(SOLTracer):
    """Reuse SOL seeding, sharded output and plotting with a Warp integration."""

    def __init__(self, io_handler, magnetic_field, input_params, device="cpu"):
        super().__init__(io_handler, magnetic_field, input_params)
        wp.init()
        periods = np.asarray(magnetic_field.periodicity)
        if periods.shape != (3,) or tuple(periods[:2]) != (0, 1):
            raise ValueError("Warp SOL requires field periodicity [0, 1, P].")
        addend = np.zeros(3)
        if magnetic_field._sol_error_enabled:
            addend = np.asarray(magnetic_field.err_adder.cpu(), dtype=np.float64)
        self.grid = make_grid(magnetic_field.B.detach().cpu().numpy(),
                              R0=magnetic_field.R0, a=magnetic_field.a,
                              periods=periods[2], device=device, addend=addend)
        self.warp_device = self.grid.values.device

    def log_inputs(self):
        self.simIO.inputsBoilerplate("WARP SOL PROTOTYPE INPUTS", {
            **self.input_params, "SOL_BACKEND": "warp_prototype", "WARP_VERSION": wp.__version__,
            "DEVICE_RESOLVED": str(self.warp_device), "DTYPE": "float64",
            "COMPILE_STEP_CHUNKS_RESOLVED": "Warp JIT (always)",
            "SEED_PHI_DEG_RESOLVED": self.seed_data["seed_phi_deg"].tolist(),
            "SEED_COUNTS": self.seed_data["seed_counts"].tolist(),
            "SEED_POINCARE_FILES": [str(p) for p in self.seed_data["poincare_paths"]],
            "TRACED_FIELD_LINES": len(self.seed_data["initial_conditions_rtp"]),
        })

    def trace(self):
        self.data_dir = Path(self.simIO.data_dir) / self.analysis_subdir
        planes = np.linspace(360/self.input_params["N_PLANES"], 360, self.input_params["N_PLANES"])
        writer = PlaneShardWriter(self.data_dir, planes, self.grid.R0)
        try:
            self.trace_data = trace_warp_volume(self.seed_data, self.input_params, self.grid, writer, self.simIO.log)
            self.data_dir = save_torch_outputs(self.simIO, self.seed_data, self.trace_data,
                                               writer, self.grid.R0, self.analysis_subdir)
        except Exception:
            writer.abort()
            raise
        return self.trace_data

    def run(self):
        self.build_initial_conditions()
        self.log_inputs()
        start = perf_counter()
        self.trace()
        self.simIO.log.info("Warp tracing and output finished in %.3f s (includes first-use JIT).", perf_counter()-start)
        if self.input_params["GENERATE_PLOTS"]:
            self.plot()
        self.simIO.log.info("Wall intersections: %d/%d; saved %s", np.count_nonzero(self.trace_data["hit_wall"]),
                            self.trace_data["hit_wall"].size, self.data_dir)
        return self.trace_data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs_path", nargs="?", help="SOL JSON input (defaults to input_files/sol_trace_inputs.json).")
    parser.add_argument("--inputs", help="Alternative to positional INPUTS.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Override JSON DEVICE.")
    parser.add_argument("--output-subdir", help="Override output subdirectory; default: JSON ANLYS_SUBDIR + '_Warp'.")
    parser.add_argument("--no-plots", action="store_true", help="Skip per-plane contour plots.")
    args = parser.parse_args(argv)
    if args.inputs and args.inputs_path:
        parser.error("provide INPUTS or --inputs, not both")
    path = args.inputs or args.inputs_path or ROOT / "input_files/sol_trace_inputs.json"
    params = merge_input_params(DEFAULT_INPUTS, load_inputs_json(path, "Warp SOL inputs"))
    params["ANLYS_SUBDIR"] = args.output_subdir or params["ANLYS_SUBDIR"] + "_Warp"
    if args.device:
        params["DEVICE"] = args.device
    if args.no_plots:
        params["GENERATE_PLOTS"] = False
    validate_runtime_settings(params)
    wp.init()
    requested = params["DEVICE"]
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError('DEVICE must be "auto", "cpu", or "cuda".')
    selected = ("cuda" if wp.is_cuda_available() else "cpu") if requested == "auto" else requested
    device = wp.get_device(selected)
    # Existing loaders use Torch for setup only; Warp may run CUDA even with CPU Torch.
    field = build_torch_magnetic_field(params, resolve_device("cpu"))
    sim_io = IOHandler(params["ANLYS_DIR"])
    sim_io.startLog(log_name="warpSolTrace.log", subdir=params["ANLYS_SUBDIR"], logger_name="WarpSOLTracer")
    analysis = WarpSOLTracer(sim_io, field, params, device=device)
    analysis.run()
    print(f"Saved Warp SOL data: {analysis.data_dir}")


if __name__ == "__main__":
    main()
