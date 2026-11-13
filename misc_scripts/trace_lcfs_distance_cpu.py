"""Trace field lines and their unsigned Euclidean distance to a closed 3D LCFS.

Example (length is the maximum in EACH direction)::

    python misc_scripts/trace_lcfs_distance_cpu.py output/IOTA3_1000sp_atol1e-9 \
        --surface-index 40 --phi-deg 306 --npoints 32 --offset-mm 5 --length-m 100 --workers 4

See docs/lcfs_distance_cpu_trace.md for outputs, offset geometry, and length limits.
"""

import argparse
from copy import copy
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import sys
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from illiad.sol.tracer import load_poincare_settings
from misc_scripts import trace_lcfs_cpu as base
from misc_scripts.trace_lcfs_cpu import make_seeds, load_field, trace_direction, save_figure
from misc_scripts.lcfs_distance_geometry import build_lcfs_mesh, SurfaceDistance

# Editable defaults; command-line arguments override these.
RUN_DIR = PROJECT_ROOT / "output/IOTA3_1000sp_atol1e-9"
SURFACE_INDEX = 21  # Zero-based saved surface index; None uses the logged LCFS.
PHI_DEG = [306]
NPOINTS = 64  # Per launch plane, equally spaced along the selected surface perimeter.
OFFSET_MM = 0.0  # Scalar or list, e.g. [0, 5, 10, 20]; positive = outward normal.
LENGTH_M = 100.0  # Per direction, unless the vessel wall is reached first.
WORKERS = 24  # Processes, not threads (LSODA is not reentrant).
MAJOR_RADIUS_M = 0.72
VESSEL_RADIUS_M = 0.19
COLORMAP = "Blues"  # Dark long-path colors stay visible on a white background.
COLOR_SCALE = "log"  # "linear" or "log"; applies to new plots and --plot-only.


# The distance reference is independent of the seed-launch surface.
DISTANCE_SURFACE_INDEX = 21  # None uses the run's logged LCFS.
DISTANCE_POLOIDAL_POINTS = 256  # Per saved toroidal slice; increase to check convergence.
DISTANCE_STATISTIC = "min"  # Closest sampled approach across both directions, including launch.


def initialize_worker(settings):
    global _FIELD, _SETTINGS, _DISTANCE
    _FIELD = load_field(settings)
    _SETTINGS = settings
    with np.load(Path(settings["output"]) / "lcfs_distance_mesh.npz") as mesh:
        _DISTANCE = SurfaceDistance(mesh["vertices_xyz_m"], mesh["faces"])


def trace_seed(task):
    seed_id, xyz = task
    s = _SETTINGS
    output = {}
    lengths, hits, integrals, maxima, minima = [], [], [], [], []
    for name, direction in (("forward", 1), ("backward", -1)):
        distance, path, length, hit = trace_direction(
            _FIELD, xyz, direction, s["length_m"], s["rtol"], s["atol"],
            s["max_step_m"], s["sample_step_m"])
        output[f"{name}_s_m"] = distance
        output[f"{name}_xyz_m"] = path
        lcfs_distance = _DISTANCE(path)
        output[f"{name}_lcfs_distance_m"] = lcfs_distance
        # LSODA sample spacing is adaptive: a plain sample average is biased.
        integrals.append(float(np.sum(np.diff(distance) *
                                      (lcfs_distance[:-1] + lcfs_distance[1:]) / 2)))
        maxima.append(float(lcfs_distance.max()))
        minima.append(float(lcfs_distance.min()))
        lengths.append(length)
        hits.append(hit)
    np.savez_compressed(Path(s["output"]) / "paths" / f"seed_{seed_id:05d}.npz", **output)
    return seed_id, lengths, hits, integrals, maxima, minima


_MESH_CACHE = {}


def get_distance_mesh(args, index):
    """Reuse one reference mesh for all offsets in this invocation."""
    key = (str(args.run_dir.resolve()), index, args.distance_poloidal_points, args.smoothing)
    if key not in _MESH_CACHE:
        print(f"Building 3D distance mesh for surface {index}...", flush=True)
        _MESH_CACHE[key] = build_lcfs_mesh(
            args.run_dir, index, args.distance_poloidal_points, args.smoothing, MAJOR_RADIUS_M)
    return _MESH_CACHE[key]


def distance_color_scale(values, color_scale, statistic):
    values = np.asarray(values)
    if not np.all(np.isfinite(values) & (values >= 0)):
        raise ValueError("LCFS distances must be finite and nonnegative.")
    maximum = max(float(values.max()) if values.size else 0.0, 1e-9)
    if color_scale == "linear":
        norm = Normalize(0, maximum)
    elif color_scale == "log":
        positive = values[values > 0]
        minimum = float(positive.min()) if positive.size else maximum / 10
        if minimum == maximum:
            minimum /= np.sqrt(10)
            maximum *= np.sqrt(10)
        # clip=True maps exact zeros to the low color instead of hiding seeds.
        norm = LogNorm(minimum, maximum, clip=True)
    else:
        raise ValueError("Color scale must be linear or log.")
    label = {"min": "Closest sampled approach (minimum)",
             "mean": "Arc-length-weighted mean", "max": "Sampled maximum"}[statistic]
    label += " distance to 3D LCFS [mm]"
    if color_scale == "log":
        label += " (log; zeros use lowest color)"
    return norm, plt.get_cmap(COLORMAP), label


def plot_results(output, color_scale=None, statistic=DISTANCE_STATISTIC):
    base.plot_results(output, color_scale=color_scale)
    settings = json.loads((output / "settings.json").read_text())
    with np.load(output / "summary.npz") as saved:
        data = dict(saved)
    norm, cmap, label = distance_color_scale(data[f"lcfs_distance_{statistic}_m"] * 1000,
                                            color_scale or COLOR_SCALE, statistic)
    title = (f"Launch surface {settings['surface_index']}, offset {settings['offset_mm']:g} mm; "
             f"distance to LCFS {settings['distance_surface_index']}\n"
             "Distance statistic covers the traced portions only")
    plot_distance_initial_points(output / "plots", settings, [data], norm, cmap, label,
                                 title, statistic=statistic)


def plot_distance_initial_points(plots, settings, datasets, norm, cmap, label, title, suffix="", statistic=DISTANCE_STATISTIC):
    """Overlay seed datasets in list order, retaining wall/capped marker meanings."""
    surface_index = settings.get("surface_index", settings.get("lcfs_index"))
    has_seeds = any(len(data["seed_xyz_m"]) for data in datasets)
    for plane_id, phi in enumerate(settings["phi_deg"]):
        boundary = datasets[0]["boundaries_xz_m"][plane_id]
        fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
        closed = np.vstack((boundary, boundary[0]))
        ax.plot(*closed.T, color="black", linewidth=1, label=f"Fitted surface {surface_index}")
        reference = datasets[0]["lcfs_reference_boundaries_xz_m"][plane_id]
        ax.plot(*np.vstack((reference, reference[0])).T, color="tab:orange", linestyle="--",
                linewidth=1, label=f"Distance LCFS {settings['distance_surface_index']}")
        angle = np.linspace(0.0, 2 * np.pi, 361)
        ax.plot(settings["vessel_radius_m"] * np.cos(angle), settings["vessel_radius_m"] * np.sin(angle),
                color="gray", linewidth=1, label="Vessel wall")
        seen = set()
        for data in datasets:
            mask = data["plane_id"] == plane_id
            xyz = data["seed_xyz_m"][mask]
            points = np.column_stack((np.hypot(xyz[:, 0], xyz[:, 1]) - settings["major_radius_m"], xyz[:, 2]))
            total = data[f"lcfs_distance_{statistic}_m"][mask] * 1000
            complete = data["hit_wall"][mask].all(axis=1)
            for is_complete, marker, description in ((True, "o", "Both ends hit wall"),
                                                      (False, "^", "Length limit reached")):
                selected = complete == is_complete
                if np.any(selected):
                    ax.scatter(*points[selected].T, c=total[selected], norm=norm, cmap=cmap,
                               marker=marker, s=35, edgecolors="black", linewidths=0.5,
                               label=description if description not in seen else "_nolegend_")
                    seen.add(description)
        if not seen:
            ax.text(0.5, 0.5, "No seeds remain inside the vessel", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
        ax.set(xlabel="$R-R_0$ [m]", ylabel="Z [m]", aspect="equal",
               title=title + f"\nLaunch plane $\\phi_{{comp}}={phi:g}^\\circ$")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        if has_seeds:
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=label)
        save_figure(fig, plots, f"initial_points_lcfs_distance_{statistic}{suffix}_phi{phi:03d}")


def plot_combined_offsets(output, color_scale=None, statistic=DISTANCE_STATISTIC):
    """Plot all offsets using saved summaries only, with a global color range."""
    base.plot_combined_offsets(output, color_scale=color_scale)
    color_scale = COLOR_SCALE if color_scale is None else color_scale
    entries = json.loads((output / "offset_scan.json").read_text())["offsets"]
    if not entries:
        raise ValueError("Offset scan contains no offsets.")
    datasets, reference = [], None
    for entry in entries:
        folder = output / entry["directory"]
        settings = json.loads((folder / "settings.json").read_text())
        if reference is None:
            reference = settings
        for key in ("phi_deg", "major_radius_m", "vessel_radius_m", "distance_surface_index", "distance_poloidal_points", "smoothing"):
            if settings[key] != reference[key]:
                raise ValueError(f"Cannot combine offset plots with different {key}: {folder}")
        if settings.get("surface_index", settings.get("lcfs_index")) != reference.get("surface_index", reference.get("lcfs_index")):
            raise ValueError(f"Cannot combine different flux surfaces: {folder}")
        with np.load(folder / "summary.npz") as saved:
            datasets.append(dict(saved))
    total = np.concatenate([data[f"lcfs_distance_{statistic}_m"] * 1000 for data in datasets])
    norm, cmap, label = distance_color_scale(total, color_scale, statistic)
    index = reference.get("surface_index", reference.get("lcfs_index"))
    offsets_text = ", ".join(f"{entry['offset_mm']:g}" for entry in entries)
    title = f"Launch surface {index}; distance to LCFS {reference['distance_surface_index']}; LSODA\n" + fill(f"Offsets [mm]: {offsets_text}", width=60)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    plot_distance_initial_points(plots, reference, datasets, norm, cmap, label, title,
                                 suffix="_all_offsets", statistic=statistic)
    print(f"Saved combined initial-point plots: {plots}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="?", type=Path, default=RUN_DIR)
    parser.add_argument("--phi-deg", nargs="+", type=int, default=PHI_DEG, help="Computational launch planes in degrees")
    parser.add_argument("--npoints", type=int, default=NPOINTS, help="Seeds per plane")
    surface = parser.add_mutually_exclusive_group()
    surface.add_argument("--surface-index", type=int, default=SURFACE_INDEX,
                         help="Zero-based saved flux surface; default: SURFACE_INDEX or logged LCFS")
    surface.add_argument("--lcfs-index", dest="surface_index", type=int, default=argparse.SUPPRESS,
                         help="Backward-compatible alias for --surface-index")
    parser.add_argument("--offset-mm", "--offsets-mm", nargs="+", type=float, default=OFFSET_MM,
                        help="One or more signed poloidal normal offsets in mm")
    parser.add_argument("--length-m", type=float, default=LENGTH_M, help="Maximum arc length PER DIRECTION")
    parser.add_argument("--workers", type=int, default=WORKERS, help="CPU processes; each loads its own field")
    parser.add_argument("--rtol", type=float, help="Default: saved Poincare RTOL")
    parser.add_argument("--atol", type=float, help="Default: saved Poincare ATOL")
    parser.add_argument("--max-step-m", type=float, default=0.02, help="Maximum LSODA step (default: 0.02 m)")
    parser.add_argument("--sample-step-m", type=float, default=0.01, help="Maximum saved path spacing (default: 0.01 m)")
    parser.add_argument("--smoothing", type=float, default=1e-5, help="Selected surface spline smoothing (same as SOL loader)")
    parser.add_argument("--output", type=Path, help="Default: RUN_DIR/lcfs_distance_cpu_trace; use distinct folders for scans")
    parser.add_argument("--plot-only", action="store_true", help="Replot saved paths and settings without retracing")
    parser.add_argument("--combined-only", action="store_true",
                        help="Replot only combined initial points from offset scan summaries; no tracing")
    parser.add_argument("--color-scale", choices=("linear", "log"), default=COLOR_SCALE,
                        help="Shared path/seed color scale; default: COLOR_SCALE in script")
    parser.add_argument("--toroidal-field", type=Path, default=PROJECT_ROOT / "input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy")
    parser.add_argument("--helical-field", type=Path, default=PROJECT_ROOT / "input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy")
    parser.add_argument("--distance-surface-index", type=int, default=DISTANCE_SURFACE_INDEX,
                        help="Distance reference surface; default: logged LCFS (independent of launch surface)")
    parser.add_argument("--distance-poloidal-points", type=int, default=DISTANCE_POLOIDAL_POINTS,
                        help="Mesh samples per saved toroidal plane (default: 256)")
    parser.add_argument("--distance-statistic", choices=("min", "mean", "max"), default=DISTANCE_STATISTIC,
                        help="Initial-point colors: closest sampled approach (default), weighted mean, or maximum")
    return parser.parse_args(argv)


def run_offset(args, validate_only=False):
    """Trace one offset, or validate its inputs/geometry without writing output."""
    run = args.run_dir.resolve()
    output = (args.output or run / "lcfs_distance_cpu_trace").resolve()
    if args.plot_only:
        plot_results(output, color_scale=args.color_scale, statistic=args.distance_statistic)
        print(f"Replotted saved paths: {output / 'plots'}")
        return
    if (output / "settings.json").exists():
        raise ValueError(f"Trace output already exists: {output}. Choose --output or use --plot-only.")
    settings = load_poincare_settings(str(run))
    required = ("CURRENT_TOR", "CURRENT_HEL", "CONFIG_TOR", "CONFIG_HEL", "ENABLE_ERRFIELD")
    missing = [key for key in required if key not in settings]
    if missing:
        raise ValueError(f"Missing field settings in Poincare log: {missing}")
    index = args.surface_index if args.surface_index is not None else settings.get("LCFS_INDEX")
    if index is None or index < 0:
        raise ValueError("Surface index must be nonnegative; provide --surface-index if no LCFS is logged.")
    args.rtol = args.rtol if args.rtol is not None else settings.get("RTOL", 2.49e-12)
    args.atol = args.atol if args.atol is not None else settings.get("ATOL", 1e-9)
    for key in ("length_m", "max_step_m", "sample_step_m", "rtol", "atol"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if args.npoints < 1 or args.workers < 1:
        raise ValueError("npoints and workers must be positive integers.")
    if not np.isfinite(args.offset_mm) or not np.isfinite(args.smoothing) or args.smoothing < 0:
        raise ValueError("Offset must be finite and smoothing finite and nonnegative.")
    planes = [phi % 360 for phi in args.phi_deg]
    if len(set(planes)) != len(planes):
        raise ValueError("Launch planes must be distinct modulo 360 degrees.")
    for path in (args.toroidal_field, args.helical_field):
        if not path.is_file():
            raise FileNotFoundError(path)
    boundaries, bases, xyz, plane_id = make_seeds(
        run, index, planes, args.npoints, args.offset_mm, args.smoothing, return_plane_ids=True)
    config = {key: settings[key] for key in required}
    config.update({key: str(value.resolve()) if isinstance(value, Path) else value
                   for key, value in vars(args).items()})
    # Keep the old lcfs_index key as an alias for consumers of previous outputs.
    config.update(output=str(output), surface_index=int(index), lcfs_index=int(index),
                  logged_lcfs_index=settings.get("LCFS_INDEX"), phi_deg=planes,
                  retained_seeds_per_plane=np.bincount(plane_id, minlength=len(planes)).tolist(),
                  major_radius_m=MAJOR_RADIUS_M, vessel_radius_m=VESSEL_RADIUS_M, solver="LSODA")
    reference_index = args.distance_surface_index
    if reference_index is None:
        reference_index = settings.get("LCFS_INDEX")
    if reference_index is None or reference_index < 0:
        raise ValueError("No valid logged LCFS; provide --distance-surface-index.")
    vertices, faces, mesh_planes = get_distance_mesh(args, int(reference_index))
    rings = vertices.reshape(len(mesh_planes), args.distance_poloidal_points, 3)
    reference_rings = rings[[int(np.flatnonzero(mesh_planes == phi)[0]) for phi in planes]]
    reference_boundaries = np.stack((np.hypot(reference_rings[..., 0], reference_rings[..., 1])
                                    - MAJOR_RADIUS_M, reference_rings[..., 2]), axis=-1)
    config.update(distance_surface_index=int(reference_index),
                  distance_method="unsigned nearest point on closed triangle mesh",
                  distance_phi_deg=mesh_planes.tolist(),
                  distance_mean_method="trapezoidal arc-length integral over both traced directions",
                  distance_max_method="maximum over saved path samples",
                  distance_min_method="minimum over both directions at saved samples, including launch",
                  distance_mesh_file="lcfs_distance_mesh.npz")
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
    tasks = list(enumerate(xyz))
    print(f"Surface index {index}; {len(xyz)} seeds; offset {args.offset_mm:g} mm; "
          f"maximum {args.length_m:g} m per direction", flush=True)

    def collect(results):
        for seed_id, lengths, hits, distance_integrals, distance_maxima, distance_minima in results:
            length[seed_id], hit[seed_id] = lengths, hits
            integrals[seed_id], maxima[seed_id] = distance_integrals, distance_maxima
            minima[seed_id] = distance_minima
            print(f"Seed {seed_id + 1}/{len(xyz)}: L+={lengths[0]:.6g} m, "
                  f"L-={lengths[1]:.6g} m; "
                  f"{'wall-to-wall' if all(hits) else 'length capped'}", flush=True)

    if not tasks:
        print("No seeds remain inside the vessel; saving empty results without tracing.", flush=True)
    elif args.workers == 1:
        initialize_worker(config)
        collect(map(trace_seed, tasks))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)),
                                 initializer=initialize_worker, initargs=(config,)) as pool:
            collect(pool.map(trace_seed, tasks))
    means = np.divide(integrals, length, out=np.zeros_like(integrals), where=length > 0)
    total_length = length.sum(axis=1)
    total_mean = np.divide(integrals.sum(axis=1), total_length,
                           out=np.zeros(len(xyz)), where=total_length > 0)
    total_max = maxima.max(axis=1)
    total_min = minima.min(axis=1)
    np.savez_compressed(output / "summary.npz",
                        lcfs_reference_boundaries_xz_m=reference_boundaries,
                        lcfs_distance_min_m=total_min, lcfs_distance_direction_min_m=minima,
                        lcfs_distance_integral_m2=integrals,
                        lcfs_distance_direction_mean_m=means,
                        lcfs_distance_direction_max_m=maxima,
                        lcfs_distance_mean_m=total_mean, lcfs_distance_max_m=total_max, seed_xyz_m=xyz, surface_base_xz_m=bases,
                        lcfs_base_xz_m=bases,
                        boundaries_xz_m=boundaries, plane_id=plane_id,
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
    plot_results(output, color_scale=args.color_scale, statistic=args.distance_statistic)
    print(f"Saved paths, lengths and plots: {output}")


def main(argv=None):
    args = parse_args(argv)
    output = (args.output or args.run_dir.resolve() / "lcfs_distance_cpu_trace").resolve()
    manifest = output / "offset_scan.json"
    if args.plot_only or args.combined_only:
        if manifest.is_file():
            if not args.combined_only:
                for entry in json.loads(manifest.read_text())["offsets"]:
                    child = copy(args)
                    child.output = output / entry["directory"]
                    run_offset(child)
            plot_combined_offsets(output, color_scale=args.color_scale, statistic=args.distance_statistic)
        elif args.combined_only:
            raise ValueError("--combined-only requires an offset scan directory containing offset_scan.json.")
        else:
            run_offset(args)
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
        run_offset(args)
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
        run_offset(child, validate_only=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"offsets": entries}, indent=2) + "\n")
    for number, child in enumerate(jobs, start=1):
        print(f"Offset {number}/{len(jobs)}: {child.offset_mm:g} mm", flush=True)
        run_offset(child)
    plot_combined_offsets(output, color_scale=args.color_scale, statistic=args.distance_statistic)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, IndexError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
