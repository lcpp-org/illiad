"""Trace bidirectional paths from a saved flux surface with CPU Poincare LSODA.

Example (length is the maximum in EACH direction)::

    python misc_scripts/trace_lcfs_cpu.py output/IOTA3_1000sp_atol1e-9 \
        --surface-index 40 --phi-deg 306 --npoints 32 --offset-mm 5 --length-m 100 --workers 4

See docs/lcfs_cpu_trace.md for outputs, offset geometry, and length limits.
"""

import argparse
from copy import copy
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import sys
from textwrap import fill
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.path import Path as PlotPath
import numpy as np
from scipy.integrate import solve_ivp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from illiad.mesh import Mesh
from illiad.particle import FieldLine
from illiad.sol.tracer import load_lcfs_boundary, load_poincare_settings
from illiad.sol.stitching import outward_normals, resample_closed_curve
from illiad.utilities.coordtrans import RTP_to_XYZ_many
from illiad.utilities.phi_events import inVV

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
COLORMAP = "Blues_r"  # Dark long-path colors stay visible on a white background.
COLOR_SCALE = "linear"  # "linear" or "log"; applies to new plots and --plot-only.


def offset_seeds(boundary, npoints, offset_m):
    """Interpolate dense contour normals at equally spaced perimeter locations."""
    # Compute normals on a dense curve even when only one seed is requested.
    dense = resample_closed_curve(boundary, max(2048, npoints * 8))
    normals = outward_normals(dense)
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(
        np.diff(np.vstack((dense, dense[0])), axis=0), axis=1))]
    target = np.linspace(0.0, distance[-1], npoints, endpoint=False)
    base = np.column_stack([np.interp(target, distance, np.r_[dense[:, j], dense[0, j]])
                            for j in range(2)])
    normal = np.column_stack([np.interp(target, distance, np.r_[normals[:, j], normals[0, j]])
                              for j in range(2)])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    shifted = base + offset_m * normal
    if offset_m != 0:
        inside = PlotPath(np.vstack((boundary, boundary[0]))).contains_points(shifted)
        if np.any(inside if offset_m > 0 else ~inside):
            raise ValueError("Offset crosses the selected surface again; reduce --offset-mm or inspect the contour.")
    return base, shifted


def make_seeds(run_dir, lcfs_index, planes, npoints, offset_mm, smoothing, *, return_plane_ids=False):
    boundaries, bases, positions, plane_ids = [], [], [], []
    for plane_id, phi in enumerate(planes):
        # Absolute run paths are supported by the existing LCFS loader.
        plane = 360 if phi == 0 and not (run_dir / "data/Poincare/Poincare_000.npy").is_file() else phi
        boundary, _ = load_lcfs_boundary(str(run_dir), plane, lcfs_index,
                                        spline_smoothing=smoothing, boundary_points=2048)
        base, shifted = offset_seeds(boundary, npoints, offset_mm / 1000.0)
        if not np.isfinite(shifted).all():
            raise ValueError(f"Plane {phi}: offset seeds contain non-finite coordinates.")
        rho = np.linalg.norm(shifted, axis=1)
        keep = rho < VESSEL_RADIUS_M
        removed = int(np.count_nonzero(~keep))
        if removed:
            warnings.warn(
                f"Plane {phi}, offset {offset_mm:g} mm: removed {removed}/{len(rho)} seeds "
                f"at or outside the vessel (minor radius >= {VESSEL_RADIUS_M:g} m); "
                f"{np.count_nonzero(keep)} remain.", UserWarning, stacklevel=2)
        base, shifted, rho = base[keep], shifted[keep], rho[keep]
        rtp = np.column_stack((rho, np.arctan2(shifted[:, 1], shifted[:, 0]),
                               np.full(len(rho), np.deg2rad(phi))))
        boundaries.append(boundary)
        bases.append(base)
        positions.append(RTP_to_XYZ_many(rtp, MAJOR_RADIUS_M))
        plane_ids.append(np.full(len(rho), plane_id, dtype=int))
    result = np.stack(boundaries), np.concatenate(bases), np.concatenate(positions)
    return (*result, np.concatenate(plane_ids)) if return_plane_ids else result


def load_field(settings):
    """Use exactly the field setup in illiad.cli.poincare, with absolute files."""
    field = Mesh(R0=MAJOR_RADIUS_M, a=VESSEL_RADIUS_M)
    field.loadCartesianField(file_path=settings["toroidal_field"],
                             coilCurrent=settings["CURRENT_TOR"],
                             errField=settings["ENABLE_ERRFIELD"],
                             att_mult=settings["CONFIG_TOR"])
    field.set_nonPer_errField()
    field.addFieldPerturbation(file_path=settings["helical_field"],
                               coilCurrent=settings["CURRENT_HEL"],
                               att_mult=settings["CONFIG_HEL"])
    return field


def trace_direction(field, xyz, direction, length_m, rtol, atol, max_step_m, sample_step_m):
    """Retain the CPU Poincare RHS/wall event, adding full-path dense sampling."""
    if inVV(0.0, xyz, field) <= 0:
        raise ValueError("Initial position must be inside the vessel.")
    line = FieldLine(np.asarray(xyz, dtype=float), length_m, direction=direction)
    result = solve_ivp(line.pushXYZ, (0.0, length_m), line.pos0_XYZ,
                       args=(field,), method="LSODA", events=inVV, dense_output=True,
                       rtol=rtol, atol=atol, max_step=max_step_m)
    # Fail rather than labeling a solver failure as a physical connection length.
    if not result.success or not np.isfinite(result.y).all():
        raise RuntimeError(f"LSODA failed in direction {direction:+d}: {result.message}")
    length = float(result.t[-1])
    # Retain every accepted solver point and refine long steps for plotting.
    samples = np.unique(np.r_[result.t, np.arange(0.0, length, sample_step_m), length])
    path = result.sol(samples).T
    if not np.isfinite(path).all():
        raise RuntimeError("LSODA dense output contains non-finite coordinates.")
    return samples, path, length, bool(len(result.t_events[0]))


def initialize_worker(settings):
    global _FIELD, _SETTINGS
    _FIELD = load_field(settings)
    _SETTINGS = settings


def trace_seed(task):
    seed_id, xyz = task
    s = _SETTINGS
    output = {}
    lengths, hits = [], []
    for name, direction in (("forward", 1), ("backward", -1)):
        distance, path, length, hit = trace_direction(
            _FIELD, xyz, direction, s["length_m"], s["rtol"], s["atol"],
            s["max_step_m"], s["sample_step_m"])
        output[f"{name}_s_m"] = distance
        output[f"{name}_xyz_m"] = path
        lengths.append(length)
        hits.append(hit)
    np.savez_compressed(Path(s["output"]) / "paths" / f"seed_{seed_id:05d}.npz", **output)
    return seed_id, lengths, hits


def save_figure(fig, directory, name):
    #for extension in ("png", "pdf"):
    #    fig.savefig(directory / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def connection_color_scale(total, color_scale):
    """Use the same length normalization for individual and combined plots."""
    # Empty traces still produce a geometry-only initial-point plot. The dummy
    # normalization is never displayed when the entire dataset is empty.
    if total.size == 0:
        total = np.array([1.0])
    if color_scale == "log":
        if not np.all(np.isfinite(total) & (total > 0)):
            raise ValueError("Log color scale requires finite, positive total connection lengths.")
        vmin, vmax = float(total.min()), float(total.max())
        if vmin == vmax:
            # Keep a nonzero logarithmic range when all lines reach the same cap.
            vmin /= np.sqrt(10.0)
            vmax *= np.sqrt(10.0)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif color_scale == "linear":
        norm = Normalize(0.0, max(float(total.max()), 1e-12))
    else:
        raise ValueError("Color scale must be 'linear' or 'log'.")
    cmap = plt.get_cmap(COLORMAP)
    label = "Traced length $L_+ + L_-$ [m]; lower bound if capped"
    if color_scale == "log":
        label += " (log scale)"
    return norm, cmap, label


def plot_results(output, color_scale=None):
    color_scale = COLOR_SCALE if color_scale is None else color_scale
    settings = json.loads((output / "settings.json").read_text())
    major_radius = settings["major_radius_m"]
    vessel_radius = settings["vessel_radius_m"]
    with np.load(output / "summary.npz") as saved:
        data = dict(saved)
    xyz = data["seed_xyz_m"]
    total = data["length_m"].sum(axis=1)
    norm, cmap, label = connection_color_scale(total, color_scale)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    surface_index = settings.get("surface_index", settings.get("lcfs_index"))
    title = f"Surface {surface_index}, offset {settings['offset_mm']:g} mm; LSODA"
    if not len(xyz):
        plot_initial_points(plots, settings, [data], norm, cmap, label, title)
        return

    fig3 = plt.figure(figsize=(10, 6.5), layout="constrained")
    ax3 = fig3.add_subplot(projection="3d")
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for i in range(len(xyz)):
        with np.load(output / "paths" / f"seed_{i:05d}.npz") as paths:
            for name, style in (("forward", "-"), ("backward", "--")):
                p = paths[f"{name}_xyz_m"]
                color = cmap(norm(total[i]))
                ax3.plot(*p.T, style, color=color, linewidth=0.8, alpha=0.8)
                axes[0].plot(p[:, 0], p[:, 1], style, color=color, linewidth=0.5, alpha=0.7)
                axes[1].plot(np.hypot(p[:, 0], p[:, 1]) - major_radius, p[:, 2],
                             style, color=color, linewidth=0.5, alpha=0.7)
    ax3.scatter(*xyz.T, c=total, norm=norm, cmap=cmap, s=18, edgecolors="black")
    # Fixed vessel extents provide equal physical scaling across all runs.
    extent = major_radius + vessel_radius
    ax3.set(xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]",
            xlim=(-extent, extent), ylim=(-extent, extent),
            zlim=(-vessel_radius, vessel_radius),
            title=title + "\nSolid: +B; dashed: -B (computational coordinates)")
    ax3.set_box_aspect((2 * extent, 2 * extent, 2 * vessel_radius), zoom=0.85)
    ax3.tick_params(labelsize=8)
    fig3.colorbar(mappable, ax=ax3, shrink=0.65, pad=0.12, label=label)
    save_figure(fig3, plots, "paths_3d")
    axes[0].set(xlabel="X [m]", ylabel="Y [m]", title="Top view")
    axes[1].set(xlabel="$R-R_0$ [m]", ylabel="Z [m]", title="Poloidal projection (all toroidal angles)")
    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    fig2.suptitle(title + "\nSolid: +B; dashed: -B")
    fig2.colorbar(mappable, ax=axes, label=label, shrink=0.8)
    save_figure(fig2, plots, "paths_projections")

    plot_initial_points(plots, settings, [data], norm, cmap, label, title)


def plot_initial_points(plots, settings, datasets, norm, cmap, label, title, suffix=""):
    """Overlay seed datasets in list order, retaining wall/capped marker meanings."""
    surface_index = settings.get("surface_index", settings.get("lcfs_index"))
    has_seeds = any(len(data["seed_xyz_m"]) for data in datasets)
    for plane_id, phi in enumerate(settings["phi_deg"]):
        boundary = datasets[0]["boundaries_xz_m"][plane_id]
        fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
        closed = np.vstack((boundary, boundary[0]))
        ax.plot(*closed.T, color="black", linewidth=1, label=f"Fitted surface {surface_index}")
        angle = np.linspace(0.0, 2 * np.pi, 361)
        ax.plot(settings["vessel_radius_m"] * np.cos(angle), settings["vessel_radius_m"] * np.sin(angle),
                color="gray", linewidth=1, label="Vessel wall")
        seen = set()
        for data in datasets:
            mask = data["plane_id"] == plane_id
            xyz = data["seed_xyz_m"][mask]
            points = np.column_stack((np.hypot(xyz[:, 0], xyz[:, 1]) - settings["major_radius_m"], xyz[:, 2]))
            total = data["length_m"][mask].sum(axis=1)
            complete = data["hit_wall"][mask].all(axis=1)
            for is_complete, marker, description in ((True, "o", "Both ends hit wall"),
                                                      (False, "^", "Length limit reached (lower bound)")):
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
        save_figure(fig, plots, f"initial_points{suffix}_phi{phi:03d}")


def plot_combined_offsets(output, color_scale=None):
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
        for key in ("phi_deg", "major_radius_m", "vessel_radius_m"):
            if settings[key] != reference[key]:
                raise ValueError(f"Cannot combine offset plots with different {key}: {folder}")
        if settings.get("surface_index", settings.get("lcfs_index")) != reference.get("surface_index", reference.get("lcfs_index")):
            raise ValueError(f"Cannot combine different flux surfaces: {folder}")
        with np.load(folder / "summary.npz") as saved:
            datasets.append(dict(saved))
    total = np.concatenate([data["length_m"].sum(axis=1) for data in datasets])
    norm, cmap, label = connection_color_scale(total, color_scale)
    index = reference.get("surface_index", reference.get("lcfs_index"))
    offsets_text = ", ".join(f"{entry['offset_mm']:g}" for entry in entries)
    title = f"Surface {index}; all offsets; LSODA\n" + fill(f"Offsets [mm]: {offsets_text}", width=60)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    plot_initial_points(plots, reference, datasets, norm, cmap, label, title, suffix="_all_offsets")
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
    parser.add_argument("--output", type=Path, help="Default: RUN_DIR/lcfs_cpu_trace; use distinct folders for scans")
    parser.add_argument("--plot-only", action="store_true", help="Replot saved paths and settings without retracing")
    parser.add_argument("--combined-only", action="store_true",
                        help="Replot only combined initial points from offset scan summaries; no tracing")
    parser.add_argument("--color-scale", choices=("linear", "log"), default=COLOR_SCALE,
                        help="Shared path/seed color scale; default: COLOR_SCALE in script")
    parser.add_argument("--toroidal-field", type=Path, default=PROJECT_ROOT / "input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy")
    parser.add_argument("--helical-field", type=Path, default=PROJECT_ROOT / "input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy")
    return parser.parse_args(argv)


def run_offset(args, validate_only=False):
    """Trace one offset, or validate its inputs/geometry without writing output."""
    run = args.run_dir.resolve()
    output = (args.output or run / "lcfs_cpu_trace").resolve()
    if args.plot_only:
        plot_results(output, color_scale=args.color_scale)
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
    if validate_only:
        return
    (output / "paths").mkdir(parents=True, exist_ok=True)
    (output / "settings.json").write_text(json.dumps(config, indent=2) + "\n")
    length = np.zeros((len(xyz), 2))
    hit = np.zeros((len(xyz), 2), dtype=bool)
    tasks = list(enumerate(xyz))
    print(f"Surface index {index}; {len(xyz)} seeds; offset {args.offset_mm:g} mm; "
          f"maximum {args.length_m:g} m per direction", flush=True)

    def collect(results):
        for seed_id, lengths, hits in results:
            length[seed_id], hit[seed_id] = lengths, hits
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
    np.savez_compressed(output / "summary.npz", seed_xyz_m=xyz, surface_base_xz_m=bases,
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
    plot_results(output, color_scale=args.color_scale)
    print(f"Saved paths, lengths and plots: {output}")


def main(argv=None):
    args = parse_args(argv)
    output = (args.output or args.run_dir.resolve() / "lcfs_cpu_trace").resolve()
    manifest = output / "offset_scan.json"
    if args.plot_only or args.combined_only:
        if manifest.is_file():
            if not args.combined_only:
                for entry in json.loads(manifest.read_text())["offsets"]:
                    child = copy(args)
                    child.output = output / entry["directory"]
                    run_offset(child)
            plot_combined_offsets(output, color_scale=args.color_scale)
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
    plot_combined_offsets(output, color_scale=args.color_scale)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, IndexError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
