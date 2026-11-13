"""Trace a regular rho-theta launch grid outside an offset LCFS, tracking 3D distance.

Example (length is the maximum in EACH direction)::

    python misc_scripts/trace_lcfs_distance_cpu_v2.py output/IOTA3_1000sp_atol1e-9 \
        --phi-deg 306 --nrho 32 --ntheta 48 --offset-mm 5 --length-m 100 --workers 4

See docs/lcfs_distance_cpu_trace_v2.md for outputs, offset geometry, and length limits.
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

from illiad.sol.tracer import load_poincare_settings, load_lcfs_boundary, minimum_boundary_distance
from illiad.utilities.coordtrans import RTP_to_XYZ_many
from matplotlib.path import Path as PlotPath
import warnings
from misc_scripts import trace_lcfs_cpu as base
from misc_scripts.trace_lcfs_cpu import load_field, trace_direction, save_figure
from misc_scripts.lcfs_distance_geometry import build_lcfs_mesh, SurfaceDistance

# Editable defaults; command-line arguments override these.
RUN_DIR = PROJECT_ROOT / "output/IOTA3_1000sp_atol1e-9"
SURFACE_INDEX = None  # Exclusion surface; None uses the distance-reference LCFS.
PHI_DEG = [306]
NPOINTS_RHO = None  # None reads the selected magnetic field mesh (currently 191).
NPOINTS_THETA = None  # None reads the selected magnetic field mesh (currently 180).
OFFSET_MM = 0.0  # Scalar or list, e.g. [0, 5, 10, 20]; positive expands the excluded LCFS interior.
LENGTH_M = 100.0  # Per direction, unless the vessel wall is reached first.
WORKERS = 24  # Processes, not threads (LSODA is not reentrant).
MAJOR_RADIUS_M = 0.72
VESSEL_RADIUS_M = 0.19
COLORMAP = "Blues"  # Dark long-path colors stay visible on a white background.
COLOR_SCALE = "log"  # "linear" or "log"; applies to new plots and --plot-only.


# The exclusion surface defaults to this distance reference.
DISTANCE_SURFACE_INDEX = 21  # None uses the run's logged LCFS.
DISTANCE_POLOIDAL_POINTS = 256  # Per saved toroidal slice; increase to check convergence.
DISTANCE_STATISTIC = "min"  # Closest sampled approach across both directions, including launch.
DELETE_PATHS = False  # Remove this run's seed path files only after summaries/plots succeed.
ALL_PHI = False  # Run all saved toroidal launch planes sequentially in phiDDD folders.
PATH_PLOTS = True  # Disable for dense batches; all per-point color plots still run.

EXTRA_METRIC_SHAPES = {
    "lcfs_min_s_direction_m": (2,), "lcfs_min_sample_direction_index": (2,),
    "lcfs_min_s_m": (), "lcfs_min_direction": (), "lcfs_min_signed_s_m": (),
    "lcfs_min_xyz_m": (3,), "wall_connection_length_m": (),
    "wall_connection_is_lower_bound": (), "wall_connection_direction": (),
}


def wall_connection_metrics(lengths, hits):
    """Shortest one-sided connection, retaining censoring when the nearer end is capped."""
    lengths, hits = np.asarray(lengths), np.asarray(hits, dtype=bool)
    # A wall hit wins an exact length tie; otherwise shortest traced length wins.
    column = min(range(2), key=lambda i: (lengths[i], not hits[i], i))
    return dict(wall_connection_length_m=float(lengths[column]),
                wall_connection_is_lower_bound=not bool(hits[column]),
                wall_connection_direction=1 if column == 0 else -1)


def path_metrics(paths, hits):
    """Summarize arc lengths to closest sampled LCFS approach and to the nearest wall."""
    minima, arc_lengths, indices, positions, lengths = [], [], [], [], []
    for column, name in enumerate(("forward", "backward")):
        distance = paths[f"{name}_lcfs_distance_m"]
        arc = paths[f"{name}_s_m"]
        xyz = paths[f"{name}_xyz_m"]
        if not (len(distance) == len(arc) == len(xyz)) or not len(arc):
            raise ValueError("Path distances, arc lengths and positions must be nonempty and aligned.")
        if not all(np.isfinite(values).all() for values in (distance, arc, xyz)):
            raise ValueError("Path metrics require finite samples.")
        index = int(np.argmin(distance))  # First/shortest-s occurrence within one direction.
        minima.append(float(distance[index]))
        arc_lengths.append(float(arc[index]))
        indices.append(index)
        positions.append(xyz[index])
        lengths.append(float(arc[-1]))
    selected = min(range(2), key=lambda i: (minima[i], arc_lengths[i], i))
    direction = 1 if selected == 0 else -1
    return dict(lcfs_min_s_direction_m=np.asarray(arc_lengths),
                lcfs_min_sample_direction_index=np.asarray(indices, dtype=int),
                lcfs_min_s_m=arc_lengths[selected], lcfs_min_direction=direction,
                lcfs_min_signed_s_m=direction * arc_lengths[selected],
                lcfs_min_xyz_m=np.asarray(positions[selected]),
                **wall_connection_metrics(lengths, hits))


def stack_path_metrics(records):
    return {key: np.asarray([record[key] for record in records]).reshape((len(records),) + shape)
            .astype(bool if key == "wall_connection_is_lower_bound" else
                    int if key in ("lcfs_min_direction", "lcfs_min_sample_direction_index",
                                   "wall_connection_direction") else float)
            for key, shape in EXTRA_METRIC_SHAPES.items()}


def write_point_metrics(output, data, settings):
    """One compact CSV row per retained grid point; independent of path retention."""
    with (output / "point_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("seed_id", "phi_comp_deg", "rho_index", "theta_index", "rho_m", "theta_rad",
                         "x_m", "y_m", "z_m", "L_forward_m", "L_backward_m", "L_total_m",
                         "hit_forward", "hit_backward", "is_lower_bound", "lcfs_min_m", "lcfs_mean_m",
                         "lcfs_max_m", "s_to_lcfs_min_m", "lcfs_min_direction", "signed_s_to_lcfs_min_m",
                         "s_to_forward_lcfs_min_m", "s_to_backward_lcfs_min_m",
                         "lcfs_min_x_m", "lcfs_min_y_m", "lcfs_min_z_m",
                         "wall_connection_length_m", "wall_connection_is_lower_bound", "wall_connection_direction"))
        for i, xyz in enumerate(data["seed_xyz_m"]):
            writer.writerow((i, settings["phi_deg"][data["plane_id"][i]],
                             data["seed_rho_index"][i], data["seed_theta_index"][i],
                             data["seed_rho_m"][i], data["seed_theta_rad"][i], *xyz,
                             *data["length_m"][i], data["total_length_m"][i], *data["hit_wall"][i],
                             not data["hit_wall"][i].all(), data["lcfs_distance_min_m"][i],
                             data["lcfs_distance_mean_m"][i], data["lcfs_distance_max_m"][i],
                             data["lcfs_min_s_m"][i], data["lcfs_min_direction"][i],
                             data["lcfs_min_signed_s_m"][i], *data["lcfs_min_s_direction_m"][i],
                             *data["lcfs_min_xyz_m"][i], data["wall_connection_length_m"][i],
                             data["wall_connection_is_lower_bound"][i], data["wall_connection_direction"][i]))


def ensure_path_metrics(output, data, settings):
    """Upgrade older V2 summaries from existing paths before any optional deletion."""
    if any(key not in data for key in EXTRA_METRIC_SHAPES):
        if any(key not in data for key in EXTRA_METRIC_SHAPES if key.startswith("lcfs_")):
            records = []
            for i in range(len(data["seed_xyz_m"])):
                with np.load(output / "paths" / f"seed_{i:05d}.npz") as paths:
                    records.append(path_metrics(paths, data["hit_wall"][i]))
            data.update(stack_path_metrics(records))
        else:
            # Wall arc lengths require only saved lengths/flags, even after cleanup.
            records = [wall_connection_metrics(length, hit)
                       for length, hit in zip(data["length_m"], data["hit_wall"])]
            for key in EXTRA_METRIC_SHAPES:
                if key.startswith("wall_connection_"):
                    dtype = bool if key.endswith("lower_bound") else int if key.endswith("direction") else float
                    data[key] = np.asarray([record[key] for record in records], dtype=dtype)
        temporary = output / "summary.updated.npz"
        np.savez_compressed(temporary, **data)
        temporary.replace(output / "summary.npz")
    write_point_metrics(output, data, settings)


def delete_saved_paths(output):
    """Delete only enumerated seed files after durable summaries and plotting succeed."""
    with np.load(output / "summary.npz") as saved:
        data = dict(saved)
    if any(key not in data for key in EXTRA_METRIC_SHAPES) or not (output / "point_metrics.csv").is_file():
        raise ValueError("Cannot delete paths before all per-point metrics have been saved.")
    removed = 0
    for i in range(len(data["seed_xyz_m"])):
        path = output / "paths" / f"seed_{i:05d}.npz"
        if path.is_file():
            path.unlink()
            removed += 1
    settings = json.loads((output / "settings.json").read_text())
    settings.update(paths_deleted=True)
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Deleted {removed} seed path files; retained per-point summaries and plots: {output}")


def resolve_grid_counts(args):
    """Read raw field dimensions without loading/scaling the vector arrays."""
    field = np.load(args.toroidal_field, mmap_mode="r", allow_pickle=False)
    helical = np.load(args.helical_field, mmap_mode="r", allow_pickle=False)
    if field.ndim != 4 or field.shape[0] != 3 or min(field.shape[1:]) < 2:
        raise ValueError("Magnetic field must have shape (3, nrho, ntheta, nphi), each grid count >= 2.")
    if helical.shape != field.shape:
        raise ValueError("Toroidal and helical field mesh shapes must match.")
    nrho = field.shape[1] if args.nrho is None else args.nrho
    ntheta = field.shape[2] if args.ntheta is None else args.ntheta
    if nrho < 2 or ntheta < 3:
        raise ValueError("Grid counts require nrho >= 2 and ntheta >= 3.")
    return int(nrho), int(ntheta), list(field.shape)


def signed_boundary_distance(points, boundary):
    """Poloidal Euclidean distance, negative inside the fitted closed contour."""
    distance = minimum_boundary_distance(points, boundary)
    inside = PlotPath(np.vstack((boundary, boundary[0]))).contains_points(points)
    return np.where(inside, -distance, distance)


_PLOT_DISTANCE_CACHE = {}


def exclusion_plot_grid(boundary, vessel_radius):
    """Resolve the displayed buffer independently of a deliberately coarse launch grid."""
    key = (boundary.tobytes(), vessel_radius)
    if key not in _PLOT_DISTANCE_CACHE:
        axis = np.linspace(-vessel_radius, vessel_radius, 201)
        x, z = np.meshgrid(axis, axis)
        distance = signed_boundary_distance(np.column_stack((x.ravel(), z.ravel())), boundary)
        _PLOT_DISTANCE_CACHE[key] = x, z, distance.reshape(x.shape)
    return _PLOT_DISTANCE_CACHE[key]


def make_grid_seeds(run_dir, index, planes, nrho, ntheta, offset_mm, smoothing):
    """Filter a fixed regular grid without moving any retained launch points."""
    if nrho < 2 or ntheta < 3:
        raise ValueError("Grid counts require nrho >= 2 and ntheta >= 3.")
    if not np.isfinite(offset_mm):
        raise ValueError("Offset must be finite.")
    rho = np.linspace(0.0, VESSEL_RADIUS_M, nrho)
    theta = np.linspace(0.0, 2 * np.pi, ntheta, endpoint=False)
    rr, tt = np.meshgrid(rho, theta, indexing="ij")
    xz = np.column_stack((rr.ravel() * np.cos(tt.ravel()), rr.ravel() * np.sin(tt.ravel())))
    boundaries, masks, signed_grids, xyz, plane_ids, r_ids, t_ids = [], [], [], [], [], [], []
    for plane_id, phi in enumerate(planes):
        plane = 360 if phi == 0 and not (run_dir / "data/Poincare/Poincare_000.npy").is_file() else phi
        boundary, _ = load_lcfs_boundary(str(run_dir), plane, index,
                                        spline_smoothing=smoothing, boundary_points=2048)
        signed = signed_boundary_distance(xz, boundary).reshape(nrho, ntheta)
        if not np.isfinite(signed).all():
            raise ValueError(f"Plane {phi}: LCFS distances contain non-finite values.")
        # Exclude the offset boundary itself, with a tiny metric roundoff margin.
        excluded = signed <= offset_mm / 1000.0 + 1e-12
        in_vessel = rr < VESSEL_RADIUS_M
        keep = ~excluded & in_vessel
        duplicate_axis = np.count_nonzero(keep[0, 1:])
        keep[0, 1:] = False  # All theta values at rho=0 represent one physical point.
        ir, it = np.nonzero(keep)
        rtp = np.column_stack((rho[ir], theta[it], np.full(len(ir), np.deg2rad(phi))))
        boundaries.append(boundary)
        masks.append(keep)
        signed_grids.append(signed)
        xyz.append(RTP_to_XYZ_many(rtp, MAJOR_RADIUS_M))
        plane_ids.append(np.full(len(ir), plane_id, dtype=int))
        r_ids.append(ir)
        t_ids.append(it)
        warnings.warn(f"Plane {phi}, offset {offset_mm:g} mm: {len(ir)}/{nrho * ntheta} grid seeds remain; "
                      f"removed {np.count_nonzero(~in_vessel)} at the vessel wall, "
                      f"{np.count_nonzero(excluded & in_vessel)} inside/on the offset LCFS, "
                      f"and {duplicate_axis} duplicate axis points.", stacklevel=2)
    return dict(boundaries_xz_m=np.stack(boundaries), seed_xyz_m=np.concatenate(xyz),
                plane_id=np.concatenate(plane_ids), grid_rho_m=rho, grid_theta_rad=theta,
                grid_keep_mask=np.stack(masks), grid_signed_lcfs_distance_m=np.stack(signed_grids),
                seed_rho_index=np.concatenate(r_ids), seed_theta_index=np.concatenate(t_ids),
                seed_rho_m=rho[np.concatenate(r_ids)], seed_theta_rad=theta[np.concatenate(t_ids)])


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
    metrics = path_metrics(output, hits)
    return seed_id, lengths, hits, integrals, maxima, minima, metrics


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
    label = {"min": "Minimum sampled",
             "mean": "Arc-length-weighted mean", "max": "Sampled maximum"}[statistic]
    label += " distance\nto 3D LCFS [mm]"
    if color_scale == "log":
        label += "\n(log; zeros use lowest color)"
    return norm, plt.get_cmap(COLORMAP), label


def plot_results(output, color_scale=None, statistic=DISTANCE_STATISTIC, include_paths=True):
    settings = json.loads((output / "settings.json").read_text())
    with np.load(output / "summary.npz") as saved:
        data = dict(saved)
    ensure_path_metrics(output, data, settings)
    if include_paths and all((output / "paths" / f"seed_{i:05d}.npz").is_file() for i in range(len(data["seed_xyz_m"]))):
        base.plot_results(output, color_scale=color_scale)
    elif include_paths:
        print(f"Full paths unavailable; replotting saved per-point metrics: {output}")
    (output / "plots").mkdir(exist_ok=True)
    norm, cmap, label = distance_color_scale(data[f"lcfs_distance_{statistic}_m"] * 1000,
                                            color_scale or COLOR_SCALE, statistic)
    title = (f"Grid {settings['nrho']} x {settings['ntheta']}; exclusion offset {settings['offset_mm']:g} mm; "
             f"distance to LCFS {settings['distance_surface_index']}\n"
             "Distance statistic covers the traced portions only")
    lengths = data["length_m"].sum(axis=1)
    length_norm, length_cmap, length_label = base.connection_color_scale(lengths, color_scale or COLOR_SCALE)
    plot_distance_initial_points(output / "plots", settings, [data], length_norm, length_cmap,
                                 length_label, title, statistic="length")
    plot_distance_initial_points(output / "plots", settings, [data], norm, cmap, label,
                                 title, statistic=statistic)
    plot_extra_metrics(output / "plots", settings, [data], title, color_scale or COLOR_SCALE)


def metric_values(data, statistic):
    if statistic == "length":
        return data["length_m"].sum(axis=1)
    if statistic == "lcfs_min_s":
        return data["lcfs_min_s_m"]
    if statistic == "wall_connection":
        return data["wall_connection_length_m"]
    return data[f"lcfs_distance_{statistic}_m"] * 1000


def plot_extra_metrics(plots, settings, datasets, title, color_scale, suffix=""):
    for metric, label in (("lcfs_min_s", "Arc length from launch to\nclosest sampled LCFS approach [m]"),
                          ("wall_connection", "Shortest arc length to wall [m]\n(lower bound if capped)")):
        values = np.concatenate([metric_values(data, metric) for data in datasets])
        norm, cmap, _ = distance_color_scale(values, color_scale, "min")
        if color_scale == "log":
            label += "\n(log; zeros use lowest color)"
        plot_distance_initial_points(plots, settings, datasets, norm, cmap, label, title,
                                     suffix=suffix, statistic=metric)


def plot_distance_initial_points(plots, settings, datasets, norm, cmap, label, title, suffix="", statistic=DISTANCE_STATISTIC):
    """Overlay seed datasets in list order, retaining wall/capped marker meanings."""
    surface_index = settings.get("surface_index", settings.get("lcfs_index"))
    has_seeds = any(len(data["seed_xyz_m"]) for data in datasets)
    for plane_id, phi in enumerate(settings["phi_deg"]):
        boundary = datasets[0]["boundaries_xz_m"][plane_id]
        fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
        closed = np.vstack((boundary, boundary[0]))
        ax.plot(*closed.T, color="black", linewidth=1, label=f"Exclusion LCFS {surface_index}")
        reference = datasets[0]["lcfs_reference_boundaries_xz_m"][plane_id]
        ax.plot(*np.vstack((reference, reference[0])).T, color="tab:orange", linestyle="--",
                linewidth=1, label=f"Distance LCFS {settings['distance_surface_index']}")
        angle = np.linspace(0.0, 2 * np.pi, 361)
        ax.plot(settings["vessel_radius_m"] * np.cos(angle), settings["vessel_radius_m"] * np.sin(angle),
                color="gray", linewidth=1, label="Vessel wall")
        seen = set()
        for data in datasets:
            offset = data.get("offset_mm", settings["offset_mm"])
            if offset != 0:
                x, z, values = exclusion_plot_grid(boundary, settings["vessel_radius_m"])
                if values.min() < offset / 1000.0 < values.max():
                    ax.contour(x, z, values, levels=[offset / 1000.0],
                               colors=["tab:green"], linestyles=":", linewidths=1)
                    description = f"Exclusion margin {offset:g} mm"
                    if description not in seen:
                        ax.plot([], [], color="tab:green", linestyle=":", label=description)
                        seen.add(description)
            mask = data["plane_id"] == plane_id
            xyz = data["seed_xyz_m"][mask]
            points = np.column_stack((np.hypot(xyz[:, 0], xyz[:, 1]) - settings["major_radius_m"], xyz[:, 2]))
            total = metric_values(data, statistic)[mask]
            complete = (~data["wall_connection_is_lower_bound"][mask] if statistic == "wall_connection"
                        else data["hit_wall"][mask].all(axis=1))
            #marker_size = max(3, min(55, 30000 / max(len(xyz), 1)))
            for is_complete, marker, description in ((True, "o", "Nearest wall reached" if statistic == "wall_connection" else "Both ends hit wall"),
                                                      (False, "^", "Length limit reached")):
                selected = complete == is_complete
                if np.any(selected):
                    ax.scatter(*points[selected].T, c=total[selected], norm=norm, cmap=cmap,
                               marker=marker, s=50, edgecolors="black",
                               linewidths=0.5 if len(xyz) < 500 else 0.1,
                               label=description if description not in seen else "_nolegend_")
                    seen.add(description)
        if not any(np.any(data["plane_id"] == plane_id) for data in datasets):
            ax.text(0.5, 0.5, "No grid seeds survive LCFS/vessel filtering", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
        ax.set(xlabel="$R-R_0$ [m]", ylabel="Z [m]", aspect="equal",
               xlim=(-1.1 * settings["vessel_radius_m"], 1.1 * settings["vessel_radius_m"]),
               ylim=(-1.1 * settings["vessel_radius_m"], 1.1 * settings["vessel_radius_m"]),
               title=title + f"\nLaunch plane $\\phi_{{comp}}={phi:g}^\\circ$")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        if has_seeds:
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=label)
        prefix = {"length": "initial_points", "lcfs_min_s": "initial_points_lcfs_min_s",
                  "wall_connection": "initial_points_wall_connection_length"}.get(statistic, f"initial_points_lcfs_distance_{statistic}")
        save_figure(fig, plots, f"{prefix}{suffix}_phi{phi:03d}")


def plot_combined_offsets(output, color_scale=None, statistic=DISTANCE_STATISTIC):
    """Plot all offsets using saved summaries only, with a global color range."""
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
        for key in ("phi_deg", "major_radius_m", "vessel_radius_m", "distance_surface_index", "distance_poloidal_points", "smoothing", "nrho", "ntheta"):
            if settings[key] != reference[key]:
                raise ValueError(f"Cannot combine offset plots with different {key}: {folder}")
        if settings.get("surface_index", settings.get("lcfs_index")) != reference.get("surface_index", reference.get("lcfs_index")):
            raise ValueError(f"Cannot combine different flux surfaces: {folder}")
        with np.load(folder / "summary.npz") as saved:
            data = dict(saved)
        ensure_path_metrics(folder, data, settings)
        datasets.append(dict(data, offset_mm=settings["offset_mm"]))
    total = np.concatenate([data[f"lcfs_distance_{statistic}_m"] * 1000 for data in datasets])
    norm, cmap, label = distance_color_scale(total, color_scale, statistic)
    index = reference.get("surface_index", reference.get("lcfs_index"))
    offsets_text = ", ".join(f"{entry['offset_mm']:g}" for entry in entries)
    title = f"Grid {reference['nrho']} x {reference['ntheta']}; exclusion LCFS {index}; distance to LCFS {reference['distance_surface_index']}; LSODA\n" + fill(f"Offsets [mm]: {offsets_text}", width=60)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    lengths = np.concatenate([data["length_m"].sum(axis=1) for data in datasets])
    ln, lc, ll = base.connection_color_scale(lengths, color_scale)
    plot_distance_initial_points(plots, reference, datasets, ln, lc, ll, title,
                                 suffix="_all_offsets", statistic="length")
    plot_distance_initial_points(plots, reference, datasets, norm, cmap, label, title,
                                 suffix="_all_offsets", statistic=statistic)
    plot_extra_metrics(plots, reference, datasets, title, color_scale, suffix="_all_offsets")
    print(f"Saved combined initial-point plots: {plots}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="?", type=Path, default=RUN_DIR)
    planes = parser.add_mutually_exclusive_group()
    planes.add_argument("--phi-deg", nargs="+", type=int, default=PHI_DEG, help="Computational launch planes in degrees")
    planes.add_argument("--all-phi", action="store_true", default=ALL_PHI,
                        help="Batch all saved toroidal planes sequentially in phiDDD folders")
    parser.add_argument("--delete-paths", action=argparse.BooleanOptionalAction, default=DELETE_PATHS,
                        help="Delete seed path files after summaries and plots succeed; also supports --plot-only")
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
    parser.add_argument("--workers", type=int, default=WORKERS, help="CPU processes; each loads its own field")
    parser.add_argument("--rtol", type=float, help="Default: saved Poincare RTOL")
    parser.add_argument("--atol", type=float, help="Default: saved Poincare ATOL")
    parser.add_argument("--max-step-m", type=float, default=0.02, help="Maximum LSODA step (default: 0.02 m)")
    parser.add_argument("--sample-step-m", type=float, default=0.01, help="Maximum saved path spacing (default: 0.01 m)")
    parser.add_argument("--smoothing", type=float, default=1e-5, help="Selected surface spline smoothing (same as SOL loader)")
    parser.add_argument("--output", type=Path, help="Default: RUN_DIR/lcfs_distance_cpu_trace_v2; use distinct folders for scans")
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
    return parser.parse_args(argv)


def run_offset(args, validate_only=False):
    """Trace one offset, or validate its inputs/geometry without writing output."""
    run = args.run_dir.resolve()
    output = (args.output or run / "lcfs_distance_cpu_trace_v2").resolve()
    if args.plot_only:
        plot_results(output, color_scale=args.color_scale, statistic=args.distance_statistic,
                     include_paths=args.path_plots)
        if args.delete_paths:
            delete_saved_paths(output)
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
    args.rtol = args.rtol if args.rtol is not None else settings.get("RTOL", 2.49e-12)
    args.atol = args.atol if args.atol is not None else settings.get("ATOL", 1e-9)
    for key in ("length_m", "max_step_m", "sample_step_m", "rtol", "atol"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if args.workers < 1:
        raise ValueError("workers must be a positive integer.")
    if not np.isfinite(args.offset_mm) or not np.isfinite(args.smoothing) or args.smoothing < 0:
        raise ValueError("Offset must be finite and smoothing finite and nonnegative.")
    planes = [phi % 360 for phi in args.phi_deg]
    if len(set(planes)) != len(planes):
        raise ValueError("Launch planes must be distinct modulo 360 degrees.")
    for path in (args.toroidal_field, args.helical_field):
        if not path.is_file():
            raise FileNotFoundError(path)
    nrho, ntheta, field_shape = resolve_grid_counts(args)
    grid = make_grid_seeds(run, index, planes, nrho, ntheta, args.offset_mm, args.smoothing)
    xyz, plane_id = grid["seed_xyz_m"], grid["plane_id"]
    config = {key: settings[key] for key in required}
    config.update({key: str(value.resolve()) if isinstance(value, Path) else value
                   for key, value in vars(args).items()})
    # Keep the old lcfs_index key as an alias for consumers of previous outputs.
    config.update(output=str(output), surface_index=int(index), lcfs_index=int(index),
                  logged_lcfs_index=settings.get("LCFS_INDEX"), phi_deg=planes,
                  retained_seeds_per_plane=np.bincount(plane_id, minlength=len(planes)).tolist(),
                  major_radius_m=MAJOR_RADIUS_M, vessel_radius_m=VESSEL_RADIUS_M, solver="LSODA",
                  seed_mode="regular_rho_theta_grid", nrho=nrho, ntheta=ntheta,
                  field_mesh_shape=field_shape, requested_seeds_per_plane=nrho * ntheta,
                  grid_order="plane, rho, theta (theta fastest); axis represented once",
                  exclusion_method="signed poloidal distance > offset; boundary excluded",
                  grid_rho_range_m=[0.0, VESSEL_RADIUS_M], grid_theta_endpoint=False)
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
                  distance_mesh_file="lcfs_distance_mesh.npz",
                  lcfs_min_s_method="arc length from launch; global sampled minimum; exact ties shortest s then forward",
                  wall_connection_method="minimum forward/backward arc length from launch; lower bound if selected direction is capped",
                  paths_deleted=False)
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

    def collect(results):
        for seed_id, lengths, hits, distance_integrals, distance_maxima, distance_minima, metrics in results:
            length[seed_id], hit[seed_id] = lengths, hits
            integrals[seed_id], maxima[seed_id] = distance_integrals, distance_maxima
            minima[seed_id] = distance_minima
            records[seed_id] = metrics
            print(f"Seed {seed_id + 1}/{len(xyz)}: L+={lengths[0]:.6g} m, "
                  f"L-={lengths[1]:.6g} m; "
                  f"{'wall-to-wall' if all(hits) else 'length capped'}", flush=True)

    if not tasks:
        print("No grid seeds survive LCFS/vessel filtering; saving empty results without tracing.", flush=True)
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
    np.savez_compressed(output / "summary.npz", **stack_path_metrics(records),
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
    with np.load(output / "summary.npz") as saved:
        write_point_metrics(output, saved, config)
    plot_results(output, color_scale=args.color_scale, statistic=args.distance_statistic,
                 include_paths=args.path_plots)
    if args.delete_paths:
        delete_saved_paths(output)
    print(f"Saved per-point metrics and plots: {output}")


def main(argv=None):
    args = parse_args(argv)
    if args.combined_only and args.delete_paths:
        raise ValueError("Use --plot-only --delete-paths to save all metrics/plots before cleanup.")
    run_args(args)


def run_args(args):
    output = (args.output or args.run_dir.resolve() / "lcfs_distance_cpu_trace_v2").resolve()
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
                run_args(child)
            _PLOT_DISTANCE_CACHE.clear()
        return
    if args.all_phi and not (args.plot_only or args.combined_only):
        if (output / "settings.json").exists() or (output / "offset_scan.json").exists():
            raise ValueError(f"Trace output already exists: {output}. Choose a new --output for all planes.")
        planes = set()
        for path in (args.run_dir / "data/Poincare").glob("Poincare_*.npy"):
            try:
                planes.add(int(path.stem.split("_")[-1]) % 360)
            except ValueError:
                continue
        if not planes:
            raise ValueError("--all-phi requires saved Poincare planes.")
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
            run_args(child)
            _PLOT_DISTANCE_CACHE.clear()
        return
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
