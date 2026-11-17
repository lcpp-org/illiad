"""GPU LCFS distance tracer with a refillable queue spanning launch planes.

Owns its CLI, scan workflow and Warp kernels. Companion utilities handle
geometry/output and plotting; integration steps reuse the Warp SOL helpers.
Default: float64 RK4 on CUDA. Use --device cpu only for testing.
See docs/lcfs_distance_warp.md for numerical differences and queue controls.
"""

import argparse
from copy import copy
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import warp as wp
except ImportError as exc:
    raise ImportError("This tracer requires the optional Warp dependency; use warpenv.") from exc

from illiad.sol.tracer import load_poincare_settings
from misc_scripts import lcfs_distance_utils as utils
from misc_scripts.lcfs_distance_geometry import SurfaceDistance
from misc_scripts.warp_sol_prototype import sol_radius, sol_step
from illiad.boris_warp import VectorGrid, make_grid

# Editable defaults; command-line arguments override these.
RUN_DIR = PROJECT_ROOT / "output/IOTA3_1000sp_atol1e-9"
SURFACE_INDEX = 23  # Exclusion surface; None uses the distance-reference LCFS.
PHI_DEG = [306]
NPOINTS_RHO = None  # None reads the selected magnetic field mesh (currently 191).
NPOINTS_THETA = None  # None reads the selected magnetic field mesh (currently 180).
OFFSET_MM = 0.005  # Scalar or list, e.g. [0, 5, 10, 20]; positive expands the excluded LCFS interior.
LENGTH_M = 20.0  # Per direction, unless the vessel wall is reached first.
MAJOR_RADIUS_M = 0.72
VESSEL_RADIUS_M = 0.19
COLOR_SCALE = "log"  # "linear" or "log"; applies to new plots and --plot-only.


# The exclusion surface defaults to this distance reference.
DISTANCE_SURFACE_INDEX = 23  # None uses the run's logged LCFS.
DISTANCE_POLOIDAL_POINTS = 256  # Per saved toroidal slice; increase to check convergence.
DISTANCE_STATISTIC = "min"  # Closest sampled approach across both directions, including launch.
DELETE_PATHS = False  # Remove seed path files after summaries and any requested plots succeed.
ALL_PHI = False  # Pool all saved launch planes; PHI_BATCHES selects sequential folders.
PATH_PLOTS = True  # Disable for dense batches; all per-point color plots still run.

# GPU execution controls; sample spacing and step size set numerical resolution.
DEVICE = "cuda"
INTEGRATOR = "rk4"
STEP_SIZE_M = 0.005
ACTIVE_SEEDS = 1024 * 2048  # Each slot has two directional lanes; refill after both finish.
SAMPLES_PER_CHUNK = 32 * 6
WALL_BISECTIONS = 32
MIN_FIELD_MAGNITUDE = 1e-14
PHI_BATCHES = False

# Same module-local release-mode loop-unrolling workaround as Warp SOL.
wp.set_module_options({"max_unroll": 0}, module=__name__)

# Each seed owns adjacent forward/backward lanes. Only ACTIVE lanes advance;
# a slot can be reused after both lanes reach WALL or LENGTH_CAP.
UNUSED = wp.constant(-1)
ACTIVE = wp.constant(0)
WALL = wp.constant(1)
LENGTH_CAP = wp.constant(2)
INVALID_FIELD = wp.constant(3)


@wp.struct
class DistanceTree:
    lower: wp.array(dtype=wp.vec3d)
    upper: wp.array(dtype=wp.vec3d)
    escape: wp.array(dtype=wp.int32)
    first: wp.array(dtype=wp.int32)
    size: wp.array(dtype=wp.int32)
    triangles: wp.array2d(dtype=wp.vec3d)


def make_distance_tree(vertices, faces, device):
    """Flatten the existing exact float64 AABB tree in preorder, with skip links."""
    reference = SurfaceDistance(vertices, faces)
    lower, upper, escape, first, size, triangles = [], [], [], [], [], []

    def visit(node):
        lo, hi, left, right = node
        index = len(lower)
        lower.append(lo)
        upper.append(hi)
        escape.append(0)
        first.append(len(triangles))
        size.append(0)
        if right is None:
            size[index] = len(left)
            triangles.extend(reference.triangles[left])
        else:
            visit(left)
            visit(right)
        escape[index] = len(lower)

    visit(reference.root)
    tree = DistanceTree()
    for name, values, dtype in (("lower", lower, wp.vec3d), ("upper", upper, wp.vec3d),
                                ("escape", escape, wp.int32), ("first", first, wp.int32),
                                ("size", size, wp.int32), ("triangles", triangles, wp.vec3d)):
        setattr(tree, name, wp.array(np.asarray(values), dtype=dtype, device=device))
    return tree


@wp.func
def segment_distance2(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d):
    edge = b - a
    denominator = wp.dot(edge, edge)
    t = wp.float64(0.0)
    if denominator > wp.float64(0.0):
        t = wp.clamp(wp.dot(p-a, edge)/denominator, wp.float64(0.0), wp.float64(1.0))
    delta = p-a-t*edge
    return wp.dot(delta, delta)


@wp.func
def triangle_distance2(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d):
    ab = b-a
    ac = c-a
    ap = p-a
    normal = wp.cross(ab, ac)
    n2 = wp.dot(normal, normal)
    best = wp.min(segment_distance2(p, a, b),
                  wp.min(segment_distance2(p, b, c), segment_distance2(p, c, a)))
    if n2 > wp.float64(0.0):
        v = wp.dot(wp.cross(ap, ac), normal)/n2
        w = wp.dot(wp.cross(ab, ap), normal)/n2
        if v >= wp.float64(0.0) and w >= wp.float64(0.0) and v+w <= wp.float64(1.0):
            height = wp.dot(ap, normal)
            best = wp.min(best, height*height/n2)
    return best


@wp.kernel
def query_distances(tree: DistanceTree, points: wp.array(dtype=wp.vec3d),
                    counts: wp.array(dtype=wp.int32), stride: int,
                    distances: wp.array(dtype=wp.float64)):
    tid = wp.tid()
    if tid % stride >= counts[tid // stride]:
        return
    p = points[tid]
    delta = p-tree.triangles[0, 0]
    best = wp.dot(delta, delta)
    node = int(0)
    while node < tree.lower.shape[0]:
        lo = tree.lower[node]
        hi = tree.upper[node]
        d = wp.vec3d(wp.float64(0.0))
        for axis in range(3):
            d[axis] = wp.max(wp.max(lo[axis]-p[axis], p[axis]-hi[axis]), wp.float64(0.0))
        if wp.dot(d, d) > best + wp.float64(1e-24):
            node = tree.escape[node]
        elif tree.size[node] > 0:
            for j in range(tree.first[node], tree.first[node]+tree.size[node]):
                best = wp.min(best, triangle_distance2(p, tree.triangles[j, 0],
                                                       tree.triangles[j, 1], tree.triangles[j, 2]))
            node = tree.escape[node]
        else:
            node = node+1
    distances[tid] = wp.sqrt(best)


@wp.kernel
def trace_samples(g: VectorGrid, positions: wp.array(dtype=wp.vec3d),
                  lengths: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32),
                  sample_index: wp.array(dtype=wp.int32), chunk: int,
                  step: wp.float64, spacing: wp.float64, limit: wp.float64,
                  rk4: int, minimum: wp.float64, bisections: int,
                  points: wp.array(dtype=wp.vec3d), arcs: wp.array(dtype=wp.float64),
                  counts: wp.array(dtype=wp.int32)):
    row = wp.tid()
    counts[row] = 0
    if status[row] != ACTIVE:
        return
    sign = wp.float64(1.0)
    if row % 2 == 1:
        sign = wp.float64(-1.0)
    x = positions[row]
    s = lengths[row]
    index = sample_index[row]
    state = int(ACTIVE)
    for j in range(chunk):
        target = wp.min(wp.float64(index)*spacing, limit)
        # Index zero emits the launch point, preserving exact tie semantics.
        while s < target:
            h = wp.min(step, target-s)
            candidate, valid = sol_step(g, x, sign, h, rk4, minimum)
            if not valid:
                state = INVALID_FIELD
                break
            fraction = wp.float64(1.0)
            if sol_radius(candidate, g.R0) >= g.a:
                low = wp.float64(0.0)
                high = wp.float64(1.0)
                for iteration in range(bisections):
                    middle = wp.float64(0.5)*(low+high)
                    if sol_radius(x+middle*(candidate-x), g.R0) < g.a:
                        low = middle
                    else:
                        high = middle
                fraction = high
                candidate = x+fraction*(candidate-x)
                state = WALL
            x = candidate
            if state == WALL:
                s = s+h*fraction
                break
            # Assign the exact sample target on the last substep to avoid tiny steps.
            if h >= target-s:
                s = target
            else:
                s = s+h
        if state == INVALID_FIELD:
            break
        points[row*chunk+j] = x
        arcs[row*chunk+j] = s
        counts[row] = counts[row]+1
        index = index+1
        if state == WALL:
            break
        if s >= limit:
            state = LENGTH_CAP
            break
    positions[row] = x
    lengths[row] = s
    status[row] = state
    sample_index[row] = index


def validate_options(args):
    for name in ("step_size_m", "sample_step_m", "length_m", "min_field_magnitude"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    for name in ("active_seeds", "samples_per_chunk", "wall_bisections"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive.")
    if args.wall_bisections > 60:
        raise ValueError("wall_bisections must not exceed 60 (float64 refinement limit).")
    if args.length_m/args.sample_step_m + 2 >= np.iinfo(np.int32).max:
        raise ValueError("Saved sample count must fit int32.")
    if 2*args.active_seeds*args.samples_per_chunk >= np.iinfo(np.int32).max:
        raise ValueError("Chunk sample capacity must fit int32.")
    if min(args.step_size_m, args.sample_step_m) <= np.spacing(args.length_m):
        raise ValueError("Step/sample spacing is too small for the requested float64 length.")


def integrate_queue(seeds, grid, tree, args):
    """Yield (seed ID, paths, wall flags) as slots finish, irrespective of phi.

    State and path histories are bounded by active_seeds, not total scan size.
    Refills occur between chunks. A lane that finishes early waits for its paired
    direction; other seed slots continue and refill independently.
    """
    validate_options(args)
    seeds = np.asarray(seeds, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 3 or not np.isfinite(seeds).all():
        raise ValueError("Seeds must be finite with shape (N, 3).")
    radii = np.hypot(np.hypot(seeds[:, 0], seeds[:, 1])-grid.R0, seeds[:, 2])
    if np.any(radii >= grid.a):
        raise ValueError("Seeds must be strictly inside the vessel.")
    if len(seeds) == 0:
        return
    slots = min(args.active_seeds, len(seeds))
    rows, chunk = 2*slots, args.samples_per_chunk
    device = grid.values.device
    host_x = np.zeros((rows, 3))
    host_s = np.zeros(rows)
    host_status = np.full(rows, UNUSED, dtype=np.int32)
    host_index = np.zeros(rows, dtype=np.int32)
    ids = np.full(slots, -1, dtype=int)
    history = [None]*slots
    positions = wp.empty(rows, dtype=wp.vec3d, device=device)
    lengths = wp.empty(rows, dtype=wp.float64, device=device)
    status = wp.empty(rows, dtype=wp.int32, device=device)
    indices = wp.empty(rows, dtype=wp.int32, device=device)
    points = wp.empty(rows*chunk, dtype=wp.vec3d, device=device)
    arcs = wp.empty(rows*chunk, dtype=wp.float64, device=device)
    distances = wp.empty(rows*chunk, dtype=wp.float64, device=device)
    counts = wp.empty(rows, dtype=wp.int32, device=device)
    next_seed = 0
    last_report = perf_counter()
    completed = 0
    while next_seed < len(seeds) or np.any(ids >= 0):
        # Refill free slots, resetting both directional lanes and their histories.
        for slot in np.flatnonzero(ids < 0):
            if next_seed == len(seeds):
                break
            ids[slot] = next_seed
            selection = slice(2*slot, 2*slot+2)
            host_x[selection] = seeds[next_seed]
            host_s[selection] = 0
            host_status[selection] = ACTIVE
            host_index[selection] = 0
            history[slot] = [[], []]
            next_seed += 1
        # Advance at most one chunk of saved samples per active lane.
        for buffer, values in ((positions, host_x), (lengths, host_s),
                               (status, host_status), (indices, host_index)):
            buffer.assign(values)
        wp.launch(trace_samples, dim=rows, inputs=[
            grid, positions, lengths, status, indices, chunk,
            args.step_size_m, args.sample_step_m, args.length_m,
            int(args.integrator == "rk4"), args.min_field_magnitude, args.wall_bisections,
            points, arcs, counts], device=device)
        host_status = status.numpy()
        if np.any(host_status == INVALID_FIELD):
            bad = np.flatnonzero(host_status == INVALID_FIELD)[0]
            raise RuntimeError(f"Invalid/nonfinite magnetic field for seed {ids[bad//2]}, "
                               f"direction {'forward' if bad % 2 == 0 else 'backward'}; no summary published.")
        wp.launch(query_distances, dim=rows*chunk,
                  inputs=[tree, points, counts, chunk, distances], device=device)
        host_x, host_s, host_index = positions.numpy(), lengths.numpy(), indices.numpy()
        sizes = counts.numpy()
        xyz = points.numpy().reshape(rows, chunk, 3)
        ss = arcs.numpy().reshape(rows, chunk)
        dd = distances.numpy().reshape(rows, chunk)
        # Keep valid samples only. Emit a seed once both directions finish;
        # the stable seed ID preserves output order even when slots finish early.
        for slot in np.flatnonzero(ids >= 0):
            for direction in range(2):
                row = 2*slot+direction
                count = sizes[row]
                if count:
                    # Copy only valid samples, releasing the full downloaded chunk.
                    history[slot][direction].append((ss[row, :count].copy(),
                                                    xyz[row, :count].copy(), dd[row, :count].copy()))
            if np.all(host_status[2*slot:2*slot+2] > 0):
                paths = {}
                for direction, name in enumerate(("forward", "backward")):
                    parts = history[slot][direction]
                    for column, suffix in enumerate(("s_m", "xyz_m", "lcfs_distance_m")):
                        paths[f"{name}_{suffix}"] = np.concatenate([p[column] for p in parts])
                yield int(ids[slot]), paths, host_status[2*slot:2*slot+2] == WALL
                ids[slot] = -1
                host_status[2*slot:2*slot+2] = UNUSED
                history[slot] = None
                completed += 1
        if perf_counter()-last_report >= 10:
            print(f"Warp queue: {completed}/{len(seeds)} complete; {next_seed} launched; "
                  f"{np.count_nonzero(ids >= 0)} occupied slots", flush=True)
            last_report = perf_counter()


class WarpBackend:
    """Load field/tree once per scan and turn queue results into saved seed records."""

    def __init__(self, args):
        self.args = args
        self.grid = None
        self.tree = None

    def write_paths(self):
        """Retain temporary paths only when requested output needs them."""
        return not self.args.delete_paths or (not self.args.no_plots and self.args.path_plots)

    def metadata(self):
        # Keep legacy settings keys for saved-output readers; Warp has no LSODA tolerances.
        return dict(solver=f"Warp {self.args.integrator.upper()}", backend="warp",
                    paths_written=self.write_paths(),
                    workers=1, max_step_m=self.args.step_size_m,
                    warp_version=wp.__version__, dtype="float64",
                    queue_mode="per_phi" if self.args.phi_batches else "pooled_phi",
                    queue_refill="completed seed slots after each sample chunk",
                    path_sampling="uniform arc length plus exact launch and terminal samples",
                    wall_refinement="SOL chord bisection", rtol=None, atol=None,
                    distance_backend="Warp float64 nearest triangle with AABB pruning")

    def trace(self, tasks, config, vertices, faces):
        if self.grid is None:
            wp.init()
            device = wp.get_device(self.args.device)
            # Load the CPU Poincare field once, including its configured error addend.
            field = utils.load_field(config)
            if tuple(field.periodicity[:2]) != (0, 1):
                raise ValueError("Warp requires field periodicity [0, 1, P].")
            self.grid = make_grid(np.stack((field.Bx, field.By, field.Bz), axis=-1),
                                  R0=field.R0, a=field.a, periods=field.periodicity[2],
                                  device=device, addend=field.err_adder if field.errField else (0, 0, 0))
            self.tree = make_distance_tree(vertices, faces, device)
            print(f"Warp tracing on {device}; {self.args.active_seeds} seed slots, "
                  f"{self.args.samples_per_chunk} samples/chunk", flush=True)
        for local_id, paths, hits in integrate_queue([xyz for _, xyz in tasks], self.grid, self.tree, self.args):
            seed_id = tasks[local_id][0]
            metrics = utils.path_metrics(paths, hits)  # Also validates all samples before saving.
            lengths, integrals, maxima, minima = [], [], [], []
            for name in ("forward", "backward"):
                arc, distance = paths[f"{name}_s_m"], paths[f"{name}_lcfs_distance_m"]
                lengths.append(float(arc[-1]))
                integrals.append(float(np.sum(np.diff(arc)*(distance[:-1]+distance[1:])/2)))
                maxima.append(float(distance.max()))
                minima.append(float(distance.min()))
            if self.write_paths():
                np.savez_compressed(Path(config["output"])/"paths"/f"seed_{seed_id:05d}.npz", **paths)
            yield seed_id, lengths, hits, integrals, maxima, minima, metrics


def build_parser():
    """Expose geometry, output and Warp controls in one place."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-json", type=Path, help="JSON settings file; explicit CLI arguments override it")
    parser.add_argument("run_dir", nargs="?", type=Path, default=RUN_DIR)
    planes = parser.add_mutually_exclusive_group()
    planes.add_argument("--phi-deg", nargs="+", type=int, default=PHI_DEG, help="Computational launch planes in degrees")
    planes.add_argument("--all-phi", action="store_true", default=ALL_PHI,
                        help="Queue all saved launch planes together; --phi-batches restores phiDDD batches")
    parser.add_argument("--delete-paths", action=argparse.BooleanOptionalAction, default=DELETE_PATHS,
                        help="Skip writing paths unless full-path plots need them, then delete after saving metrics/plots; supports --plot-only")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip all figures, including combined plots; still save every summary and CSV")
    parser.add_argument("--path-plots", action=argparse.BooleanOptionalAction, default=PATH_PLOTS,
                        help="Render full 3D/projected paths; --no-path-plots keeps only per-point color plots")
    parser.add_argument("--nrho", "--npoints-rho", type=int, default=NPOINTS_RHO,
                        help="Radial grid count before filtering; default: field mesh")
    parser.add_argument("--ntheta", "--npoints-theta", type=int, default=NPOINTS_THETA,
                        help="Poloidal grid count before filtering; default: field mesh")
    surface = parser.add_mutually_exclusive_group()
    surface.add_argument("--surface-index", type=int, default=SURFACE_INDEX,
                         help="Zero-based exclusion surface; default: distance-reference LCFS")
    surface.add_argument("--lcfs-index", dest="surface_index", type=int, default=argparse.SUPPRESS,
                         help="Backward-compatible alias for --surface-index")
    parser.add_argument("--offset-mm", "--offsets-mm", nargs="+", type=float, default=OFFSET_MM,
                        help="One or more signed poloidal exclusion margins in mm; positive expands LCFS")
    parser.add_argument("--length-m", type=float, default=LENGTH_M, help="Maximum arc length PER DIRECTION")
    parser.add_argument("--sample-step-m", type=float, default=0.01, help="Maximum saved path spacing (default: 0.01 m)")
    parser.add_argument("--smoothing", type=float, default=1e-5, help="Selected surface spline smoothing (same as SOL loader)")
    parser.add_argument("--output", type=Path, help="Default: RUN_DIR/lcfs_distance_warp")
    parser.add_argument("--plot-only", action="store_true", help="Replot saved paths and settings without retracing")
    parser.add_argument("--combined-only", action="store_true",
                        help="Replot only combined initial points from offset scan summaries; no tracing")
    parser.add_argument("--color-scale", choices=("linear", "log"), default=COLOR_SCALE,
                        help="Shared path/seed color scale; default: COLOR_SCALE in script")
    parser.add_argument("--toroidal-field", type=Path, default=PROJECT_ROOT / "input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy")
    parser.add_argument("--helical-field", type=Path, default=PROJECT_ROOT / "input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy")
    parser.add_argument("--distance-surface-index", type=int, default=DISTANCE_SURFACE_INDEX,
                        help="Distance reference; default: DISTANCE_SURFACE_INDEX (None uses logged LCFS)")
    parser.add_argument("--distance-poloidal-points", type=int, default=DISTANCE_POLOIDAL_POINTS,
                        help="Mesh samples per saved toroidal plane (default: 256)")
    parser.add_argument("--distance-statistic", choices=("min", "mean", "max"), default=DISTANCE_STATISTIC,
                        help="Initial-point colors: closest sampled approach (default), weighted mean, or maximum")
    parser.add_argument("--device", default=DEVICE, help="Warp device (default: cuda; cpu for testing)")
    parser.add_argument("--integrator", choices=("rk4", "midpoint"), default=INTEGRATOR)
    parser.add_argument("--step-size-m", type=float, default=STEP_SIZE_M,
                        help="Maximum fixed integration step; shortened at saved samples and length cap")
    parser.add_argument("--active-seeds", type=int, default=ACTIVE_SEEDS,
                        help="Number of refillable seed slots, each tracing both directions")
    parser.add_argument("--samples-per-chunk", type=int, default=SAMPLES_PER_CHUNK)
    parser.add_argument("--wall-bisections", type=int, default=WALL_BISECTIONS)
    parser.add_argument("--min-field-magnitude", type=float, default=MIN_FIELD_MAGNITUDE)
    parser.add_argument("--phi-batches", action="store_true", default=PHI_BATCHES,
                        help="With --all-phi, finish each phiDDD folder sequentially (default: pool all planes)")
    parser.set_defaults(major_radius_m=MAJOR_RADIUS_M, vessel_radius_m=VESSEL_RADIUS_M,
                        pooled_all_phi=False)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--input-json", type=Path)
    config_path = probe.parse_known_args(argv)[0].input_json
    if config_path is not None:
        try:
            settings = json.loads(config_path.read_text())
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot read input JSON: {exc}")
        if not isinstance(settings, dict):
            parser.error("Input JSON must be an object of argument names and values.")
        actions = {a.dest: a for a in parser._actions if a.dest not in ("help", "input_json")}
        unknown = settings.keys() - actions.keys()
        if unknown:
            parser.error(f"Unknown JSON settings: {sorted(unknown)}")
        for key, value in settings.items():
            action = actions[key]
            if value is None:
                if key not in ("output", "nrho", "ntheta", "surface_index", "distance_surface_index"):
                    parser.error(f"{key} cannot be null")
                continue
            if isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction)):
                if not isinstance(value, bool):
                    parser.error(f"{key} must be a JSON boolean")
                continue
            values = value if action.nargs == "+" and isinstance(value, list) else [value]
            if not values:
                parser.error(f"{key} must not be empty")
            converted = []
            for item in values:
                try:
                    if action.type in (int, float) and (isinstance(item, bool) or not isinstance(item, (int, float))):
                        raise ValueError("expected a number")
                    if action.type is int and int(item) != item:
                        raise ValueError("expected an integer")
                    result = action.type(item) if action.type else item
                    if action.choices and result not in action.choices:
                        raise ValueError(f"choose from {action.choices}")
                except (TypeError, ValueError) as exc:
                    parser.error(f"Invalid {key}: {exc}")
                converted.append(result)
            settings[key] = converted if action.nargs == "+" else converted[0]
        parser.set_defaults(**settings)
    args = parser.parse_args(argv)
    if args.no_plots and (args.plot_only or args.combined_only):
        parser.error("--no-plots cannot be combined with --plot-only or --combined-only")
    if args.combined_only and args.delete_paths:
        parser.error("Use --plot-only --delete-paths before cleanup")
    if args.phi_batches and not args.all_phi and not (args.plot_only or args.combined_only):
        parser.error("--phi-batches requires --all-phi")
    args.output = args.output or args.run_dir.resolve()/"lcfs_distance_warp"
    validate_options(args)
    return args


def run_offset(args, backend, *, validate_only=False):
    """Validate, trace and save one offset using the shared Warp backend."""
    run = args.run_dir.resolve()
    output = args.output.resolve()
    if args.plot_only:
        from misc_scripts.lcfs_distance_plots import plot_results
        plot_results(output, color_scale=args.color_scale, statistic=args.distance_statistic,
                     include_paths=args.path_plots)
        if args.delete_paths:
            utils.delete_saved_paths(output)
        print(f"Replotted saved results: {output / 'plots'}")
        return
    if (output / "settings.json").exists():
        raise ValueError(f"Trace output already exists: {output}. Choose --output or use --plot-only.")
    settings = load_poincare_settings(str(run))
    required = ("CURRENT_TOR", "CURRENT_HEL", "CONFIG_TOR", "CONFIG_HEL", "ENABLE_ERRFIELD")
    missing = [key for key in required if key not in settings]
    if missing:
        raise ValueError(f"Missing field settings in Poincare log: {missing}")
    reference_index = args.distance_surface_index
    if reference_index is None:
        reference_index = settings.get("LCFS_INDEX")
    if reference_index is None or reference_index < 0:
        raise ValueError("No valid LCFS reference; provide --distance-surface-index.")
    index = args.surface_index if args.surface_index is not None else reference_index
    if index is None or index < 0:
        raise ValueError("Surface index must be nonnegative; provide --surface-index if no LCFS is logged.")
    validate_options(args)
    if not np.isfinite(args.offset_mm) or not np.isfinite(args.smoothing) or args.smoothing < 0:
        raise ValueError("Offset must be finite and smoothing finite and nonnegative.")
    planes = [phi % 360 for phi in args.phi_deg]
    if len(set(planes)) != len(planes):
        raise ValueError("Launch planes must be distinct modulo 360 degrees.")
    for path in (args.toroidal_field, args.helical_field):
        if not path.is_file():
            raise FileNotFoundError(path)
    nrho, ntheta, field_shape = utils.resolve_grid_counts(args)
    grid = utils.make_grid_seeds(run, index, planes, nrho, ntheta, args.offset_mm, args.smoothing,
                                major_radius_m=args.major_radius_m, vessel_radius_m=args.vessel_radius_m)
    xyz, plane_id = grid["seed_xyz_m"], grid["plane_id"]
    config = {key: settings[key] for key in required}
    config.update({key: str(value.resolve()) if isinstance(value, Path) else value
                   for key, value in vars(args).items()})
    # Keep the old lcfs_index key as an alias for consumers of previous outputs.
    config.update(output=str(output), surface_index=int(index), lcfs_index=int(index),
                  logged_lcfs_index=settings.get("LCFS_INDEX"), phi_deg=planes,
                  retained_seeds_per_plane=np.bincount(plane_id, minlength=len(planes)).tolist(),
                  major_radius_m=args.major_radius_m, vessel_radius_m=args.vessel_radius_m,
                  seed_mode="regular_rho_theta_grid", nrho=nrho, ntheta=ntheta,
                  field_mesh_shape=field_shape, requested_seeds_per_plane=nrho * ntheta,
                  grid_order="plane, rho, theta (theta fastest); axis represented once",
                  exclusion_method="signed poloidal distance > offset; boundary excluded",
                  grid_rho_range_m=[0.0, args.vessel_radius_m], grid_theta_endpoint=False)
    vertices, faces, mesh_planes = utils.get_distance_mesh(args, int(reference_index))
    rings = vertices.reshape(len(mesh_planes), args.distance_poloidal_points, 3)
    reference_rings = rings[[int(np.flatnonzero(mesh_planes == phi)[0]) for phi in planes]]
    reference_boundaries = np.stack((np.hypot(reference_rings[..., 0], reference_rings[..., 1])
                                    - args.major_radius_m, reference_rings[..., 2]), axis=-1)
    config.update(distance_surface_index=int(reference_index),
                  distance_method="unsigned nearest point on closed triangle mesh",
                  distance_phi_deg=mesh_planes.tolist(),
                  distance_mean_method="trapezoidal arc-length integral over both traced directions",
                  distance_max_method="maximum over saved path samples",
                  distance_min_method="minimum over both directions at saved samples, including launch",
                  distance_mesh_file="lcfs_distance_mesh.npz",
                  lcfs_min_s_method="arc length from launch; global sampled minimum; exact ties shortest s then forward",
                  wall_connection_method="minimum forward/backward arc length from launch; lower bound if selected direction is capped",
                  paths_deleted=False)
    config.update(backend.metadata())
    if validate_only:
        return
    (output / "paths").mkdir(parents=True, exist_ok=True)
    (output / "settings.json").write_text(json.dumps(config, indent=2) + "\n")
    np.savez_compressed(output / "lcfs_distance_mesh.npz", vertices_xyz_m=vertices,
                        faces=faces, phi_deg=mesh_planes, surface_index=int(reference_index))
    integrals = np.zeros((len(xyz), 2))
    maxima = np.zeros((len(xyz), 2))
    minima = np.zeros((len(xyz), 2))
    length = np.zeros((len(xyz), 2))
    hit = np.zeros((len(xyz), 2), dtype=bool)
    records = [None] * len(xyz)
    tasks = list(enumerate(xyz))
    print(f"Grid {nrho} x {ntheta}; exclusion surface {index}; {len(xyz)} seeds; offset {args.offset_mm:g} mm; "
          f"maximum {args.length_m:g} m per direction", flush=True)

    if not tasks:
        print("No grid seeds survive LCFS/vessel filtering; saving empty results without tracing.", flush=True)
    else:
        for seed_id, lengths, hits, distance_integrals, distance_maxima, distance_minima, metrics in backend.trace(
                tasks, config, vertices, faces):
            length[seed_id], hit[seed_id] = lengths, hits
            integrals[seed_id], maxima[seed_id] = distance_integrals, distance_maxima
            minima[seed_id] = distance_minima
            records[seed_id] = metrics
    means = np.divide(integrals, length, out=np.zeros_like(integrals), where=length > 0)
    total_length = length.sum(axis=1)
    total_mean = np.divide(integrals.sum(axis=1), total_length,
                           out=np.zeros(len(xyz)), where=total_length > 0)
    total_max = maxima.max(axis=1)
    total_min = minima.min(axis=1)
    utils.save_summary(output, **utils.stack_path_metrics(records),
                        lcfs_reference_boundaries_xz_m=reference_boundaries,
                        lcfs_distance_min_m=total_min, lcfs_distance_direction_min_m=minima,
                        lcfs_distance_integral_m2=integrals,
                        lcfs_distance_direction_mean_m=means,
                        lcfs_distance_direction_max_m=maxima,
                        lcfs_distance_mean_m=total_mean, lcfs_distance_max_m=total_max, **grid,
                        length_m=length, hit_wall=hit, total_length_m=length.sum(axis=1),
                        connection_complete=hit.all(axis=1))
    with (output / "connection_lengths.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("seed_id", "phi_comp_deg", "x_m", "y_m", "z_m", "L_forward_m",
                         "L_backward_m", "L_total_m", "hit_forward", "hit_backward", "is_lower_bound"))
        for i in range(len(xyz)):
            writer.writerow((i, planes[plane_id[i]], *xyz[i], *length[i], length[i].sum(),
                             *hit[i], not hit[i].all()))
    with (output / "lcfs_distances.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("seed_id", "phi_comp_deg", "mean_forward_m", "mean_backward_m",
                         "mean_both_m", "max_forward_m", "max_backward_m", "max_both_m",
                         "min_forward_m", "min_backward_m", "min_both_m",
                         "L_total_m", "connection_complete"))
        for i in range(len(xyz)):
            writer.writerow((i, planes[plane_id[i]], *means[i], total_mean[i],
                             *maxima[i], total_max[i], *minima[i], total_min[i], total_length[i], hit[i].all()))
    utils.write_point_metrics(output, utils.load_summary(output), config)
    if not args.no_plots:
        from misc_scripts.lcfs_distance_plots import plot_results
        plot_results(output, color_scale=args.color_scale, statistic=args.distance_statistic,
                     include_paths=args.path_plots)
    if args.delete_paths:
        utils.delete_saved_paths(output, paths_written=config.get("paths_written", True))
    print(f"Saved per-point metrics{' (plots skipped)' if args.no_plots else ' and plots'}: {output}")


def run_scan(args, backend):
    """Dispatch saved-result plotting, plane batches and offset scans."""
    if not args.no_plots:
        from misc_scripts.lcfs_distance_plots import (
            clear_plot_cache, plot_combined_offsets, plot_results,
        )
    output = args.output.resolve()
    phi_manifest = output / "phi_scan.json"
    if phi_manifest.is_file():
        if not (args.plot_only or args.combined_only):
            raise ValueError(f"Phi scan already exists: {output}. Choose --output or use --plot-only.")
        for entry in json.loads(phi_manifest.read_text())["planes"]:
            child = copy(args)
            child.all_phi = False
            child.output = output / entry["directory"]
            if args.combined_only and not (child.output / "offset_scan.json").is_file():
                plot_results(child.output, args.color_scale, args.distance_statistic, include_paths=False)
            else:
                run_scan(child, backend=backend)
            if not args.no_plots:
                clear_plot_cache()
        return
    if args.all_phi and args.phi_batches and not (args.plot_only or args.combined_only):
        if (output / "settings.json").exists() or (output / "offset_scan.json").exists():
            raise ValueError(f"Trace output already exists: {output}. Choose a new --output for all planes.")
        planes = utils.saved_launch_planes(args.run_dir)
        entries = [dict(phi_deg=phi, directory=f"phi{phi:03d}") for phi in sorted(planes)]
        for entry in entries:
            folder = output / entry["directory"]
            if folder.exists() and any(folder.iterdir()):
                raise ValueError(f"Phi output already contains files: {folder}. Choose a new --output.")
        output.mkdir(parents=True, exist_ok=True)
        phi_manifest.write_text(json.dumps({"planes": entries}, indent=2) + "\n")
        for number, entry in enumerate(entries, 1):
            print(f"Launch plane {number}/{len(entries)}: {entry['phi_deg']} degrees", flush=True)
            child = copy(args)
            child.all_phi = False
            child.phi_deg = [entry["phi_deg"]]
            child.output = output / entry["directory"]
            run_scan(child, backend=backend)
            if not args.no_plots:
                clear_plot_cache()
        return
    manifest = output / "offset_scan.json"
    if args.plot_only or args.combined_only:
        if manifest.is_file():
            if not args.combined_only:
                for entry in json.loads(manifest.read_text())["offsets"]:
                    child = copy(args)
                    child.output = output / entry["directory"]
                    run_offset(child, backend=backend)
            plot_combined_offsets(output, color_scale=args.color_scale, statistic=args.distance_statistic)
        elif args.combined_only:
            raise ValueError("--combined-only requires an offset scan directory containing offset_scan.json.")
        else:
            run_offset(args, backend=backend)
        return

    offsets = np.atleast_1d(np.asarray(args.offset_mm, dtype=float))
    if offsets.ndim != 1 or not offsets.size or not np.isfinite(offsets).all():
        raise ValueError("Offsets must be a nonempty scalar or list of finite numbers.")
    if len(np.unique(offsets)) != len(offsets):
        raise ValueError("Offsets must be distinct; remove duplicate values.")
    if manifest.exists():
        raise ValueError(f"Offset scan already exists: {output}. Choose --output or use --plot-only.")
    if len(offsets) == 1:
        args.offset_mm = float(offsets[0])
        run_offset(args, backend=backend)
        return

    if (output / "settings.json").exists():
        raise ValueError(f"Trace output already exists: {output}. Choose a new --output for the scan.")
    entries, jobs = [], []
    for offset in offsets:
        value = float(offset)
        directory = f"offset_{str(value).removesuffix('.0')}mm"
        child = copy(args)
        child.offset_mm = value
        child.output = output / directory
        entries.append({"offset_mm": value, "directory": directory})
        jobs.append(child)
    # Check every offset before starting a potentially long scan. This catches
    # invalid geometry and existing outputs even at the end of the list.
    for child in jobs:
        run_offset(child, validate_only=True, backend=backend)
    output.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"offsets": entries}, indent=2) + "\n")
    for number, child in enumerate(jobs, start=1):
        print(f"Offset {number}/{len(jobs)}: {child.offset_mm:g} mm", flush=True)
        run_offset(child, backend=backend)
    if not args.no_plots:
        plot_combined_offsets(output, color_scale=args.color_scale, statistic=args.distance_statistic)


def main(argv=None):
    args = parse_args(argv)
    if args.all_phi and not args.phi_batches and not (args.plot_only or args.combined_only):
        args.phi_deg = utils.saved_launch_planes(args.run_dir)
        args.pooled_all_phi = True
        args.all_phi = False  # Resolved planes now follow the ordinary multi-plane workflow.
    run_scan(args, backend=WarpBackend(args))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, IndexError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
