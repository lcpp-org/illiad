"""Geometry, field loading, path metrics and durable output for the LCFS tracer.

This module has no tracing entry point or dependency on a runnable trace script.
Plotting lives in lcfs_distance_plots; Warp execution lives in
trace_lcfs_distance_cpu_warp. Geometry dimensions are supplied by the caller.
"""

import csv
import json
import os
from pathlib import Path
import tempfile
import warnings
import zipfile

from matplotlib.path import Path as PlotPath
import numpy as np

from illiad.mesh import Mesh
from illiad.sol.tracer import load_lcfs_boundary, minimum_boundary_distance
from illiad.utilities.coordtrans import RTP_to_XYZ_many
from misc_scripts.lcfs_distance_geometry import build_lcfs_mesh

EXTRA_METRIC_SHAPES = {
    "lcfs_min_s_direction_m": (2,), "lcfs_min_sample_direction_index": (2,),
    "lcfs_min_s_m": (), "lcfs_min_direction": (), "lcfs_min_signed_s_m": (),
    "lcfs_min_xyz_m": (3,), "wall_connection_length_m": (),
    "wall_connection_is_lower_bound": (), "wall_connection_direction": (),
}


def load_field(settings):
    """Use exactly the field setup in illiad.cli.poincare, with absolute files."""
    field = Mesh(R0=settings["major_radius_m"], a=settings["vessel_radius_m"])
    field.loadCartesianField(file_path=settings["toroidal_field"],
                             coilCurrent=settings["CURRENT_TOR"],
                             errField=settings["ENABLE_ERRFIELD"],
                             att_mult=settings["CONFIG_TOR"])
    field.set_nonPer_errField()
    field.addFieldPerturbation(file_path=settings["helical_field"],
                               coilCurrent=settings["CURRENT_HEL"],
                               att_mult=settings["CONFIG_HEL"])
    return field


def load_summary(output):
    """Read every array so truncated archives and corrupt members fail before plotting."""
    path = Path(output) / "summary.npz"
    try:
        with path.open("rb") as stream:
            with np.load(stream, allow_pickle=False) as saved:
                return dict(saved)
    except (zipfile.BadZipFile, EOFError, ValueError) as exc:
        raise ValueError(
            f"Cannot read summary archive {path} ({path.stat().st_size:,} bytes): {exc}. "
            "The file may be truncated or corrupt. Copy the complete summary.npz from the "
            "original run and verify its size/checksum; also check the CSV and mesh files. "
            "--plot-only cannot reconstruct missing data. The existing files were not changed."
        ) from exc


def save_summary(output, **data):
    """Publish only a fully written, readable archive; preserve any previous summary."""
    destination = Path(output) / "summary.npz"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", prefix=".summary-", suffix=".npz",
                                         dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **data)
            stream.flush()
            os.fsync(stream.fileno())
        # Loading all members checks the archive and CRCs before publication/cleanup.
        with temporary.open("rb") as stream:
            with np.load(stream, allow_pickle=False) as saved:
                for key in saved.files:
                    saved[key]
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
        save_summary(output, **data)
    write_point_metrics(output, data, settings)


def delete_saved_paths(output, *, paths_written=True):
    """Delete enumerated seed files after durable summaries and any requested plots succeed."""
    data = load_summary(output)
    if any(key not in data for key in EXTRA_METRIC_SHAPES) or not (output / "point_metrics.csv").is_file():
        raise ValueError("Cannot delete paths before all per-point metrics have been saved.")
    removed = 0
    for i in range(len(data["seed_xyz_m"]) if paths_written else 0):
        path = output / "paths" / f"seed_{i:05d}.npz"
        if path.is_file():
            path.unlink()
            removed += 1
    settings = json.loads((output / "settings.json").read_text())
    settings.update(paths_deleted=True)
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Deleted {removed} seed path files; retained per-point summaries: {output}")


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


def make_grid_seeds(run_dir, index, planes, nrho, ntheta, offset_mm, smoothing, *, major_radius_m, vessel_radius_m):
    """Filter a fixed regular grid without moving any retained launch points."""
    if nrho < 2 or ntheta < 3:
        raise ValueError("Grid counts require nrho >= 2 and ntheta >= 3.")
    if not np.isfinite(offset_mm):
        raise ValueError("Offset must be finite.")
    rho = np.linspace(0.0, vessel_radius_m, nrho)
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
        in_vessel = rr < vessel_radius_m
        keep = ~excluded & in_vessel
        duplicate_axis = np.count_nonzero(keep[0, 1:])
        keep[0, 1:] = False  # All theta values at rho=0 represent one physical point.
        ir, it = np.nonzero(keep)
        rtp = np.column_stack((rho[ir], theta[it], np.full(len(ir), np.deg2rad(phi))))
        boundaries.append(boundary)
        masks.append(keep)
        signed_grids.append(signed)
        xyz.append(RTP_to_XYZ_many(rtp, major_radius_m))
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


_MESH_CACHE = {}


def get_distance_mesh(args, index):
    """Reuse one reference mesh for all offsets in this invocation."""
    key = (str(args.run_dir.resolve()), index, args.distance_poloidal_points, args.smoothing, args.major_radius_m)
    if key not in _MESH_CACHE:
        print(f"Building 3D distance mesh for surface {index}...", flush=True)
        _MESH_CACHE[key] = build_lcfs_mesh(
            args.run_dir, index, args.distance_poloidal_points, args.smoothing, args.major_radius_m)
    return _MESH_CACHE[key]


def saved_launch_planes(run_dir):
    """Return distinct saved computational planes, normalizing 360 degrees to zero."""
    planes = set()
    for path in (Path(run_dir) / "data/Poincare").glob("Poincare_*.npy"):
        try:
            planes.add(int(path.stem.split("_")[-1]) % 360)
        except ValueError:
            continue
    if not planes:
        raise ValueError("--all-phi requires saved Poincare planes.")
    return sorted(planes)
