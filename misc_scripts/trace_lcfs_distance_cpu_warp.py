"""GPU LCFS distance tracer with a refillable queue spanning launch planes.

Reuses v2's geometry, metrics, archives and plots, and the Warp SOL integrator.
Default: float64 RK4 on CUDA. Use --device cpu only for testing.
See docs/lcfs_distance_warp.md for numerical differences and queue controls.
"""

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

from misc_scripts import trace_lcfs_distance_cpu_v2 as cpu
from misc_scripts.lcfs_distance_geometry import SurfaceDistance
from misc_scripts.warp_sol_prototype import sol_radius, sol_step
from illiad.boris_warp import VectorGrid, make_grid

# Editable GPU defaults. Geometry/plot defaults are shared with the v2 script.
DEVICE = "cuda"
INTEGRATOR = "rk4"
STEP_SIZE_M = 0.005
ACTIVE_SEEDS = 512*1024  # Each slot has two directional lanes; refill after both finish.
SAMPLES_PER_CHUNK = 32*8
WALL_BISECTIONS = 32
MIN_FIELD_MAGNITUDE = 1e-14
PHI_BATCHES = False

# Same module-local release-mode loop-unrolling workaround as Warp SOL.
wp.set_module_options({"max_unroll": 0}, module=__name__)


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
    if status[row] != 0:
        return
    sign = wp.float64(1.0)
    if row % 2 == 1:
        sign = wp.float64(-1.0)
    x = positions[row]
    s = lengths[row]
    index = sample_index[row]
    state = int(0)  # 1 wall, 2 length cap, 3 invalid field; -1 unused slot
    for j in range(chunk):
        target = wp.min(wp.float64(index)*spacing, limit)
        # Index zero emits the launch point, preserving exact tie semantics.
        while s < target:
            h = wp.min(step, target-s)
            candidate, valid = sol_step(g, x, sign, h, rk4, minimum)
            if not valid:
                state = 3
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
                state = 1
            x = candidate
            if state == 1:
                s = s+h*fraction
                break
            # Assign the exact sample target on the last substep to avoid tiny steps.
            if h >= target-s:
                s = target
            else:
                s = s+h
        if state == 3:
            break
        points[row*chunk+j] = x
        arcs[row*chunk+j] = s
        counts[row] = counts[row]+1
        index = index+1
        if state == 1:
            break
        if s >= limit:
            state = 2
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
    host_status = np.full(rows, -1, dtype=np.int32)
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
        for slot in np.flatnonzero(ids < 0):
            if next_seed == len(seeds):
                break
            ids[slot] = next_seed
            selection = slice(2*slot, 2*slot+2)
            host_x[selection] = seeds[next_seed]
            host_s[selection] = 0
            host_status[selection] = 0
            host_index[selection] = 0
            history[slot] = [[], []]
            next_seed += 1
        for buffer, values in ((positions, host_x), (lengths, host_s),
                               (status, host_status), (indices, host_index)):
            buffer.assign(values)
        wp.launch(trace_samples, dim=rows, inputs=[
            grid, positions, lengths, status, indices, chunk,
            args.step_size_m, args.sample_step_m, args.length_m,
            int(args.integrator == "rk4"), args.min_field_magnitude, args.wall_bisections,
            points, arcs, counts], device=device)
        host_status = status.numpy()
        if np.any(host_status == 3):
            bad = np.flatnonzero(host_status == 3)[0]
            raise RuntimeError(f"Invalid/nonfinite magnetic field for seed {ids[bad//2]}, "
                               f"direction {'forward' if bad % 2 == 0 else 'backward'}; no summary published.")
        wp.launch(query_distances, dim=rows*chunk,
                  inputs=[tree, points, counts, chunk, distances], device=device)
        host_x, host_s, host_index = positions.numpy(), lengths.numpy(), indices.numpy()
        sizes = counts.numpy()
        xyz = points.numpy().reshape(rows, chunk, 3)
        ss = arcs.numpy().reshape(rows, chunk)
        dd = distances.numpy().reshape(rows, chunk)
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
                yield int(ids[slot]), paths, host_status[2*slot:2*slot+2] == 1
                ids[slot] = -1
                host_status[2*slot:2*slot+2] = -1
                history[slot] = None
                completed += 1
        if perf_counter()-last_report >= 10:
            print(f"Warp queue: {completed}/{len(seeds)} complete; {next_seed} launched; "
                  f"{np.count_nonzero(ids >= 0)} occupied slots", flush=True)
            last_report = perf_counter()


class WarpBackend:
    def __init__(self, args):
        self.args = args
        self.grid = None
        self.tree = None

    def metadata(self, config):
        return dict(solver=f"Warp {self.args.integrator.upper()}", backend="warp",
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
            field = cpu.load_field(config)
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
            metrics = cpu.path_metrics(paths, hits)  # Also validates all samples before saving.
            lengths, integrals, maxima, minima = [], [], [], []
            for name in ("forward", "backward"):
                arc, distance = paths[f"{name}_s_m"], paths[f"{name}_lcfs_distance_m"]
                lengths.append(float(arc[-1]))
                integrals.append(float(np.sum(np.diff(arc)*(distance[:-1]+distance[1:])/2)))
                maxima.append(float(distance.max()))
                minima.append(float(distance.min()))
            np.savez_compressed(Path(config["output"])/"paths"/f"seed_{seed_id:05d}.npz", **paths)
            yield seed_id, lengths, hits, integrals, maxima, minima, metrics


def parse_args(argv=None):
    parser = cpu.build_parser(cpu_options=False)
    parser.description = __doc__
    parser.set_defaults(workers=1, rtol=None, atol=None, max_step_m=STEP_SIZE_M,
                        pooled_all_phi=False)
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
    # The inherited output and all-phi help is specialized without duplicating the CLI.
    for action in parser._actions:
        if action.dest == "output":
            action.default = None
            action.help = "Default: RUN_DIR/lcfs_distance_warp"
        elif action.dest == "all_phi":
            action.help = "Queue all saved launch planes together; --phi-batches restores phiDDD batches"
    args = parser.parse_args(argv)
    if args.no_plots and (args.plot_only or args.combined_only):
        parser.error("--no-plots cannot be combined with --plot-only or --combined-only")
    if args.combined_only and args.delete_paths:
        parser.error("Use --plot-only --delete-paths before cleanup")
    if args.phi_batches and not args.all_phi and not (args.plot_only or args.combined_only):
        parser.error("--phi-batches requires --all-phi")
    args.output = args.output or args.run_dir.resolve()/"lcfs_distance_warp"
    args.max_step_m = args.step_size_m
    validate_options(args)
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.all_phi and not args.phi_batches and not (args.plot_only or args.combined_only):
        planes = set()
        for path in (args.run_dir/"data/Poincare").glob("Poincare_*.npy"):
            try:
                planes.add(int(path.stem.split("_")[-1]) % 360)
            except ValueError:
                continue
        if not planes:
            raise ValueError("--all-phi requires saved Poincare planes.")
        args.phi_deg = sorted(planes)
        args.pooled_all_phi = True
        args.all_phi = False  # Reuse the v2 multi-plane output/offset workflow.
    cpu.run_args(args, backend=WarpBackend(args))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, IndexError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
