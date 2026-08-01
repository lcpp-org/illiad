"""Trace connection lengths into an unstructured three-dimensional sample set.

Sparse, LCFS-masked grids are launched from one or more equally spaced
toroidal planes. Each field line is traced once in both directions, and every
crossing of the requested toroidal planes is assigned that field line's total
wall-to-wall connection length.

This initial workflow saves the raw samples and produces triangulated filled
contours at every toroidal plane. It intentionally does not interpolate the
samples onto a regular three-dimensional mesh.
"""

import argparse
import gc
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.path import Path as MplPath
import matplotlib.tri as mtri
import numpy as np


# Allow this script to be run from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from illiad.io import IOHandler
from illiad.poincare import Poincare
from illiad.utilities.coordtrans import RTP_to_XYZ_many, XYZ_to_RTP_many
from misc_scripts.connection_lengths_outside_lcfs import (
    build_magnetic_field,
    load_lcfs_boundary,
    load_poincare_settings,
    minimum_boundary_distance,
    resolve_workers,
)


ANALYSIS_SUBDIR = "ConnectionLengthVolume"

# Sampling and tracing settings
N_PLANES = 360
N_SEED_PLANES = 1
MAX_SPINS = 111  # Set to None to inherit SPINS from the Poincare log
N_RHO = 40
N_THETA = 45
RHO_MIN = 0.002  # [m]
RHO_MAX = 0.188  # [m], 2 mm inside the r=0.19 m vessel wall
LCFS_CLEARANCE = 0.0  # [m]
VESSEL_RADIUS = 0.19  # [m]

# Plot settings
COLOR_SCALE = "log"  # "log" or "linear"
COLORMAP = "viridis"
N_LEVELS = 50
VMIN = None
VMAX = None
DPI = 300
PLOT_MAX_SAMPLES = 150_000  # None disables deterministic plot-only thinning
PLOT_SAMPLE_SEED = 0
COLOR_RANGE_CHUNK_SIZE = 1_000_000


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Trace LCFS-masked seed grids into raw three-dimensional "
            "connection-length samples."
        )
    )
    parser.add_argument("analysis_dir", help="Existing directory under output/.")
    parser.add_argument(
        "--lcfs-index",
        type=int,
        default=None,
        help="LCFS surface index (default: read from the Poincare log).",
    )
    parser.add_argument(
        "--spins",
        type=int,
        default=None,
        help=(
            "Maximum toroidal spins in each direction "
            "(default: MAX_SPINS in this script)."
        ),
    )
    parser.add_argument(
        "--seed-planes",
        type=int,
        default=None,
        help=(
            "Number of equally spaced seed planes "
            f"(default: {N_SEED_PLANES}; must divide {N_PLANES})."
        ),
    )
    parser.add_argument(
        "--seed-phi-deg",
        type=float,
        default=None,
        help=(
            "First seed plane in computational degrees "
            "(default: IC_PHI_DEG from the Poincare log)."
        ),
    )
    return parser.parse_args()


def seed_plane_degrees(
    n_seed_planes,
    seed_phi_deg=360.0,
    n_planes=N_PLANES,
):
    """Return equally spaced seed planes selected from the target planes."""
    if (
        isinstance(n_seed_planes, bool)
        or not isinstance(n_seed_planes, int)
        or n_seed_planes <= 0
    ):
        raise ValueError("N_SEED_PLANES must be a positive integer.")
    if n_seed_planes > n_planes or n_planes % n_seed_planes:
        raise ValueError(
            f"N_SEED_PLANES must be a positive divisor of N_PLANES={n_planes}."
        )

    plane_step_deg = 360.0 / n_planes
    normalized_seed_phi = seed_phi_deg % 360.0
    if np.isclose(normalized_seed_phi, 0.0):
        normalized_seed_phi = 360.0
    first_plane_number = int(np.rint(normalized_seed_phi / plane_step_deg))
    first_plane_phi = first_plane_number * plane_step_deg
    if not np.isclose(normalized_seed_phi, first_plane_phi):
        raise ValueError(
            f"SEED_PHI_DEG must lie on the {plane_step_deg:g}-degree "
            "target-plane grid."
        )

    plane_spacing = n_planes // n_seed_planes
    first_plane_index = (first_plane_number - 1) % n_planes
    plane_indices = (
        first_plane_index
        + np.arange(n_seed_planes, dtype=np.int32) * plane_spacing
    ) % n_planes
    phi_degrees = (plane_indices + 1) * (360.0 / n_planes)
    return plane_indices, phi_degrees


def make_seed_initial_conditions(
    analysis_dir,
    lcfs_index,
    n_seed_planes,
    seed_phi_deg=360.0,
    n_rho=N_RHO,
    n_theta=N_THETA,
    rho_min=RHO_MIN,
    rho_max=RHO_MAX,
    lcfs_clearance=LCFS_CLEARANCE,
    n_planes=N_PLANES,
):
    """Build sparse polar grids outside the LCFS on each seed plane."""
    if n_rho < 2 or n_theta < 3:
        raise ValueError("The sparse grid requires N_RHO >= 2 and N_THETA >= 3.")
    if not 0.0 <= rho_min < rho_max <= VESSEL_RADIUS:
        raise ValueError(
            "Require 0 <= RHO_MIN < RHO_MAX <= VESSEL_RADIUS."
        )
    if lcfs_clearance < 0.0:
        raise ValueError("LCFS_CLEARANCE must be non-negative.")

    seed_plane_indices, seed_phi_degrees = seed_plane_degrees(
        n_seed_planes,
        seed_phi_deg=seed_phi_deg,
        n_planes=n_planes,
    )
    rho_values = np.linspace(rho_min, rho_max, n_rho)
    theta_values = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    theta_grid, rho_grid = np.meshgrid(theta_values, rho_values)
    grid_xz = np.column_stack(
        (
            rho_grid.ravel() * np.cos(theta_grid.ravel()),
            rho_grid.ravel() * np.sin(theta_grid.ravel()),
        )
    )

    condition_blocks = []
    seed_id_blocks = []
    seed_counts = []
    poincare_paths = []
    for seed_id, phi_deg in enumerate(seed_phi_degrees):
        boundary, poincare_path = load_lcfs_boundary(
            analysis_dir,
            phi_deg,
            lcfs_index,
        )
        closed_boundary = np.vstack((boundary, boundary[0]))
        inside_lcfs = MplPath(closed_boundary).contains_points(grid_xz)
        if lcfs_clearance > 0.0:
            near_lcfs = (
                minimum_boundary_distance(grid_xz, boundary)
                <= lcfs_clearance
            )
        else:
            near_lcfs = np.zeros(grid_xz.shape[0], dtype=bool)
        trace_mask = ~(inside_lcfs | near_lcfs)

        count = int(np.count_nonzero(trace_mask))
        seed_counts.append(count)
        poincare_paths.append(poincare_path)
        if count == 0:
            continue

        condition_blocks.append(
            np.column_stack(
                (
                    rho_grid.ravel()[trace_mask],
                    theta_grid.ravel()[trace_mask],
                    np.full(count, np.deg2rad(phi_deg)),
                )
            )
        )
        seed_id_blocks.append(np.full(count, seed_id, dtype=np.int16))

    if not condition_blocks:
        raise ValueError("The LCFS masks removed every sparse-grid seed point.")

    return {
        "initial_conditions_rtp": np.vstack(condition_blocks),
        "seed_id": np.concatenate(seed_id_blocks),
        "seed_plane_index": seed_plane_indices,
        "seed_phi_deg": seed_phi_degrees,
        "seed_counts": np.asarray(seed_counts, dtype=np.int64),
        "poincare_paths": poincare_paths,
    }


def _parse_directional_results(
    solver_output,
    n_fieldlines,
    max_length,
    magnetic_field,
):
    if len(solver_output) != 2 * n_fieldlines:
        raise RuntimeError(
            f"Expected {2 * n_fieldlines} directional traces; "
            f"received {len(solver_output)}."
        )

    direction_length = np.full((n_fieldlines, 2), np.nan)
    wall_xyz = np.full((n_fieldlines, 2, 3), np.nan)
    hit_wall = np.zeros((n_fieldlines, 2), dtype=bool)
    for output_index, (length, _plane_output, wall_output) in enumerate(
        solver_output
    ):
        direction_index = 0 if output_index < n_fieldlines else 1
        fieldline_index = output_index % n_fieldlines
        direction_length[fieldline_index, direction_index] = length
        if isinstance(wall_output, np.ndarray) and wall_output.size:
            wall_xyz[fieldline_index, direction_index] = wall_output[0]
            hit_wall[fieldline_index, direction_index] = True

    reached_limit = np.isclose(
        direction_length,
        max_length,
        rtol=1e-7,
        atol=1e-9,
    )
    valid_trace = hit_wall | reached_limit
    connection_length = np.sum(direction_length, axis=1)
    connection_length[~np.all(valid_trace, axis=1)] = np.nan

    wall_rtp = np.full_like(wall_xyz, np.nan)
    finite_wall = np.all(np.isfinite(wall_xyz), axis=-1)
    wall_rtp[finite_wall] = XYZ_to_RTP_many(
        wall_xyz[finite_wall],
        magnetic_field.R0,
    )
    return {
        "direction_length": direction_length,
        "connection_length": connection_length,
        "wall_xyz": wall_xyz,
        "wall_rtp": wall_rtp,
        "hit_wall": hit_wall,
        "reached_limit": reached_limit,
        "valid_trace": valid_trace,
    }


def assemble_raw_samples(
    solver_output,
    initial_conditions_rtp,
    seed_id,
    seed_plane_indices,
    plot_angles,
    magnetic_field,
    max_length,
):
    """Attach each total connection length to all reconstructed crossings."""
    n_fieldlines = initial_conditions_rtp.shape[0]
    trace_data = _parse_directional_results(
        solver_output,
        n_fieldlines,
        max_length,
        magnetic_field,
    )
    n_planes = len(plot_angles)
    xyz_blocks = [[] for _ in range(n_planes)]
    rtp_blocks = [[] for _ in range(n_planes)]
    value_blocks = [[] for _ in range(n_planes)]
    fieldline_id_blocks = [[] for _ in range(n_planes)]
    direction_blocks = [[] for _ in range(n_planes)]

    for fieldline_index in range(n_fieldlines):
        connection_length = trace_data["connection_length"][fieldline_index]
        for direction_index, direction_value in enumerate((1, -1)):
            output_index = (
                fieldline_index
                if direction_index == 0
                else n_fieldlines + fieldline_index
            )
            plane_output = solver_output[output_index][1]
            if len(plane_output) != n_planes:
                raise RuntimeError(
                    f"Field line {fieldline_index} returned "
                    f"{len(plane_output)} planes; expected {n_planes}."
                )

            for plane_index, points_xyz in enumerate(plane_output):
                if not isinstance(points_xyz, np.ndarray) or not points_xyz.size:
                    continue
                points_xyz = np.asarray(points_xyz, dtype=np.float64)
                points_rtp = XYZ_to_RTP_many(
                    points_xyz,
                    magnetic_field.R0,
                )
                points_rtp[:, 2] = plot_angles[plane_index]
                point_count = points_xyz.shape[0]
                xyz_blocks[plane_index].append(points_xyz)
                rtp_blocks[plane_index].append(points_rtp)
                value_blocks[plane_index].append(
                    np.full(point_count, connection_length)
                )
                fieldline_id_blocks[plane_index].append(
                    np.full(point_count, fieldline_index, dtype=np.int64)
                )
                direction_blocks[plane_index].append(
                    np.full(point_count, direction_value, dtype=np.int8)
                )

        # A plane crossing at integration time zero is not reconstructed, so
        # retain every seed itself as a sample on its launch plane.
        seed_plane_index = seed_plane_indices[seed_id[fieldline_index]]
        seed_rtp = initial_conditions_rtp[fieldline_index : fieldline_index + 1]
        seed_xyz = RTP_to_XYZ_many(seed_rtp, magnetic_field.R0)
        xyz_blocks[seed_plane_index].append(seed_xyz)
        rtp_blocks[seed_plane_index].append(seed_rtp.copy())
        value_blocks[seed_plane_index].append(
            np.array([connection_length])
        )
        fieldline_id_blocks[seed_plane_index].append(
            np.array([fieldline_index], dtype=np.int64)
        )
        direction_blocks[seed_plane_index].append(
            np.array([0], dtype=np.int8)
        )

    raw_xyz = []
    raw_rtp = []
    raw_values = []
    raw_fieldline_id = []
    raw_direction = []
    raw_plane_index = []
    plane_offsets = [0]
    for plane_index in range(n_planes):
        if xyz_blocks[plane_index]:
            plane_xyz = np.vstack(xyz_blocks[plane_index])
            plane_rtp = np.vstack(rtp_blocks[plane_index])
            plane_values = np.concatenate(value_blocks[plane_index])
            plane_fieldline_id = np.concatenate(
                fieldline_id_blocks[plane_index]
            )
            plane_direction = np.concatenate(direction_blocks[plane_index])
        else:
            plane_xyz = np.empty((0, 3), dtype=np.float64)
            plane_rtp = np.empty((0, 3), dtype=np.float64)
            plane_values = np.empty(0, dtype=np.float64)
            plane_fieldline_id = np.empty(0, dtype=np.int64)
            plane_direction = np.empty(0, dtype=np.int8)

        raw_xyz.append(plane_xyz)
        raw_rtp.append(plane_rtp)
        raw_values.append(plane_values)
        raw_fieldline_id.append(plane_fieldline_id)
        raw_direction.append(plane_direction)
        raw_plane_index.append(
            np.full(plane_values.size, plane_index, dtype=np.int16)
        )
        plane_offsets.append(plane_offsets[-1] + plane_values.size)

    trace_data.update(
        {
            "raw_points_xyz": np.vstack(raw_xyz),
            "raw_points_rtp": np.vstack(raw_rtp),
            "raw_connection_length": np.concatenate(raw_values),
            "raw_fieldline_id": np.concatenate(raw_fieldline_id),
            "raw_source_direction": np.concatenate(raw_direction),
            "raw_plane_index": np.concatenate(raw_plane_index),
            "plane_offsets": np.asarray(plane_offsets, dtype=np.int64),
            "plane_phi_deg": np.rad2deg(np.asarray(plot_angles)),
        }
    )
    return trace_data


def trace_connection_length_volume(
    initial_conditions_rtp,
    seed_id,
    seed_plane_indices,
    settings,
    magnetic_field,
    sim_io,
    n_planes=N_PLANES,
):
    """Trace each sparse seed once and reconstruct all toroidal crossings."""
    tracer = Poincare(
        sim_io,
        solvr=settings["SOLVER"],
        r_tol=settings["RTOL"],
        a_tol=settings["ATOL"],
        workers=resolve_workers(settings["NTHREADS"]),
        double_line=True,
        anlys_name=ANALYSIS_SUBDIR,
    )
    tracer.set_conditions(
        initial_conditions_rtp,
        spins=settings["SPINS"],
        field=magnetic_field,
        nplanes=n_planes,
    )
    solver_output = list(tracer.parallel_solver())
    max_length = 2.0 * np.pi * magnetic_field.R0 * settings["SPINS"]
    return assemble_raw_samples(
        solver_output,
        initial_conditions_rtp,
        seed_id,
        seed_plane_indices,
        tracer.plot_angles,
        magnetic_field,
        max_length,
    )


def save_raw_outputs(sim_io, seed_data, trace_data):
    """Save raw crossings and field-line metadata as NumPy artifacts."""
    outputs = {
        "raw_points_xyz": trace_data["raw_points_xyz"],
        "raw_points_rtp": trace_data["raw_points_rtp"],
        "raw_connection_length_m": trace_data["raw_connection_length"],
        "raw_plane_index": trace_data["raw_plane_index"],
        "raw_fieldline_id": trace_data["raw_fieldline_id"],
        "raw_source_direction": trace_data["raw_source_direction"],
        "plane_offsets": trace_data["plane_offsets"],
        "plane_phi_deg": trace_data["plane_phi_deg"],
        "seed_initial_conditions_rtp": seed_data["initial_conditions_rtp"],
        "seed_id": seed_data["seed_id"],
        "seed_plane_index": seed_data["seed_plane_index"],
        "seed_phi_deg": seed_data["seed_phi_deg"],
        "seed_counts": seed_data["seed_counts"],
        "fieldline_connection_length_m": trace_data["connection_length"],
        "direction_connection_length_m": trace_data["direction_length"],
        "wall_intersection_xyz": trace_data["wall_xyz"],
        "wall_intersection_rtp": trace_data["wall_rtp"],
        "hit_wall": trace_data["hit_wall"],
        "reached_max_length": trace_data["reached_limit"],
        "valid_trace": trace_data["valid_trace"],
    }
    for name, values in outputs.items():
        sim_io.saveNumpyData(values, name, subdir=ANALYSIS_SUBDIR)

    data_dir = Path(sim_io.data_dir) / ANALYSIS_SUBDIR
    sim_io.log.info(
        "Saved %d raw samples across %d toroidal planes: %s",
        trace_data["raw_points_rtp"].shape[0],
        trace_data["plane_phi_deg"].size,
        data_dir,
    )
    return data_dir


def make_color_scale(values):
    data_min = np.inf
    data_max = -np.inf
    for start in range(0, values.size, COLOR_RANGE_CHUNK_SIZE):
        chunk = np.asarray(values[start : start + COLOR_RANGE_CHUNK_SIZE])
        positive = chunk[np.isfinite(chunk) & (chunk > 0.0)]
        if positive.size:
            data_min = min(data_min, float(positive.min()))
            data_max = max(data_max, float(positive.max()))
    if not np.isfinite(data_min):
        raise ValueError("No positive finite connection lengths are available.")

    value_min = data_min if VMIN is None else VMIN
    value_max = data_max if VMAX is None else VMAX
    if np.isclose(value_min, value_max):
        delta = max(0.01 * value_min, np.finfo(float).eps)
        value_min -= delta
        value_max += delta
    if value_min >= value_max:
        raise ValueError("The resolved color limits require VMIN < VMAX.")

    if COLOR_SCALE == "log":
        levels = np.geomspace(value_min, value_max, N_LEVELS)
        norm = LogNorm(vmin=value_min, vmax=value_max)
    elif COLOR_SCALE == "linear":
        levels = np.linspace(value_min, value_max, N_LEVELS)
        norm = Normalize(vmin=value_min, vmax=value_max)
    else:
        raise ValueError('COLOR_SCALE must be either "log" or "linear".')

    clipped_below = data_min < value_min
    clipped_above = data_max > value_max
    if clipped_below and clipped_above:
        extend = "both"
    elif clipped_below:
        extend = "min"
    elif clipped_above:
        extend = "max"
    else:
        extend = "neither"
    return levels, norm, extend


def _unique_plane_samples(points_rtp, values):
    finite = (
        np.all(np.isfinite(points_rtp), axis=1)
        & np.isfinite(values)
        & (values > 0.0)
    )
    points_rtp = points_rtp[finite]
    values = values[finite]
    if not points_rtp.size:
        return points_rtp, values

    if PLOT_MAX_SAMPLES is not None and points_rtp.shape[0] > PLOT_MAX_SAMPLES:
        sample_indices = np.sort(
            np.random.default_rng(PLOT_SAMPLE_SEED).choice(
                points_rtp.shape[0],
                PLOT_MAX_SAMPLES,
                replace=False,
            )
        )
        points_rtp = points_rtp[sample_indices]
        values = values[sample_indices]

    x = points_rtp[:, 0] * np.cos(points_rtp[:, 1])
    z = points_rtp[:, 0] * np.sin(points_rtp[:, 1])
    _, unique_indices = np.unique(
        np.column_stack((x, z)),
        axis=0,
        return_index=True,
    )
    unique_indices.sort()
    return points_rtp[unique_indices], values[unique_indices]


def plot_plane_samples(
    points_rtp,
    values,
    boundary,
    phi_deg,
    levels,
    norm,
    extend,
    sim_io,
    analysis_subdir=ANALYSIS_SUBDIR,
):
    """Plot one unstructured toroidal slice with LCFS-interior triangles masked."""
    points_rtp, values = _unique_plane_samples(points_rtp, values)
    fig, ax = plt.subplots(figsize=(7, 6))
    color_artist = None

    if points_rtp.shape[0] >= 3:
        x = points_rtp[:, 0] * np.cos(points_rtp[:, 1])
        z = points_rtp[:, 0] * np.sin(points_rtp[:, 1])
        try:
            triangulation = mtri.Triangulation(x, z)
            triangle_x = x[triangulation.triangles]
            triangle_z = z[triangulation.triangles]
            triangle_probes = np.stack(
                (
                    np.column_stack(
                        (triangle_x.mean(axis=1), triangle_z.mean(axis=1))
                    ),
                    np.column_stack(
                        (
                            0.5 * (triangle_x[:, 0] + triangle_x[:, 1]),
                            0.5 * (triangle_z[:, 0] + triangle_z[:, 1]),
                        )
                    ),
                    np.column_stack(
                        (
                            0.5 * (triangle_x[:, 1] + triangle_x[:, 2]),
                            0.5 * (triangle_z[:, 1] + triangle_z[:, 2]),
                        )
                    ),
                    np.column_stack(
                        (
                            0.5 * (triangle_x[:, 2] + triangle_x[:, 0]),
                            0.5 * (triangle_z[:, 2] + triangle_z[:, 0]),
                        )
                    ),
                ),
                axis=1,
            )
            closed_boundary = np.vstack((boundary, boundary[0]))
            inside_lcfs = MplPath(closed_boundary).contains_points(
                triangle_probes.reshape(-1, 2)
            ).reshape(-1, triangle_probes.shape[1]).any(axis=1)
            triangulation.set_mask(inside_lcfs)
            if np.all(inside_lcfs):
                raise ValueError("Every triangulation element is inside the LCFS.")

            color_artist = ax.tricontourf(
                triangulation,
                values,
                levels=levels,
                norm=norm,
                cmap=COLORMAP,
                extend=extend,
            )
        except (RuntimeError, ValueError) as exc:
            sim_io.log.warning(
                "Falling back to scatter at phi=%g deg: %s",
                phi_deg,
                exc,
            )

    if color_artist is None and points_rtp.size:
        x = points_rtp[:, 0] * np.cos(points_rtp[:, 1])
        z = points_rtp[:, 0] * np.sin(points_rtp[:, 1])
        color_artist = ax.scatter(
            x,
            z,
            c=values,
            s=4.0,
            linewidths=0.0,
            norm=norm,
            cmap=COLORMAP,
        )
    elif color_artist is None:
        ax.text(
            0.5,
            0.5,
            "No connection-length samples",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    closed_boundary = np.vstack((boundary, boundary[0]))
    ax.plot(
        closed_boundary[:, 0],
        closed_boundary[:, 1],
        color="black",
        linewidth=1.0,
        label="LCFS",
    )
    vessel_angle = np.linspace(0.0, 2.0 * np.pi, 720)
    ax.plot(
        VESSEL_RADIUS * np.cos(vessel_angle),
        VESSEL_RADIUS * np.sin(vessel_angle),
        color="0.35",
        linewidth=1.0,
        label="Vessel wall",
    )

    phi_phys_deg = (phi_deg + 198.0) % 360.0
    ax.set_title(
        "Connection length samples\n"
        f"$\\phi_{{phy}}={phi_phys_deg:03.0f}^\\circ$ CW from North split, "
        f"$\\phi_c={phi_deg:03.0f}^\\circ$"
    )
    ax.set_xlabel(r"$x=\rho\cos\theta$ [m]")
    ax.set_ylabel(r"$z=\rho\sin\theta$ [m]")
    ax.set_xlim(-VESSEL_RADIUS, VESSEL_RADIUS)
    ax.set_ylim(-VESSEL_RADIUS, VESSEL_RADIUS)
    ax.set_aspect("equal")
    ax.grid(linewidth=0.4, color="0.75")
    ax.legend(loc="upper right")
    if color_artist is not None:
        colorbar = fig.colorbar(color_artist, ax=ax, pad=0.03)
        colorbar.set_label("Connection length [m]")

    plot_name = f"connection_length_{phi_deg:03.0f}.png"
    sim_io.saveFig(plot_name, dpi=DPI, subdir=analysis_subdir)
    sim_io.log.info(
        "Saved figure with %d samples: %s/%s",
        points_rtp.shape[0],
        analysis_subdir,
        plot_name,
    )
    plt.close(fig)
    gc.collect()


def plot_all_planes(
    analysis_dir,
    lcfs_index,
    trace_data,
    sim_io,
    analysis_subdir=ANALYSIS_SUBDIR,
):
    """Produce one unstructured filled-contour plot for every toroidal plane."""
    fieldline_values = trace_data.get("fieldline_connection_length")
    raw_values = trace_data.get("raw_connection_length")
    raw_fieldline_id = trace_data.get("raw_fieldline_id")
    if raw_values is None and (
        fieldline_values is None or raw_fieldline_id is None
    ):
        raise ValueError(
            "Trace data must provide either expanded raw connection lengths or "
            "fieldline lengths plus raw fieldline IDs."
        )

    color_values = raw_values if raw_values is not None else fieldline_values
    levels, norm, extend = make_color_scale(color_values)
    raw_points_rtp = trace_data["raw_points_rtp"]
    plane_offsets = trace_data["plane_offsets"]
    for plane_index, phi_deg in enumerate(trace_data["plane_phi_deg"]):
        start = int(plane_offsets[plane_index])
        stop = int(plane_offsets[plane_index + 1])
        if raw_values is None:
            plane_values = fieldline_values[raw_fieldline_id[start:stop]]
        else:
            plane_values = raw_values[start:stop]
        boundary, _ = load_lcfs_boundary(
            analysis_dir,
            phi_deg,
            lcfs_index,
        )
        plot_plane_samples(
            raw_points_rtp[start:stop],
            plane_values,
            boundary,
            phi_deg,
            levels,
            norm,
            extend,
            sim_io,
            analysis_subdir=analysis_subdir,
        )


def main():
    args = parse_args()
    settings = load_poincare_settings(args.analysis_dir)

    lcfs_index = (
        settings.get("LCFS_INDEX")
        if args.lcfs_index is None
        else args.lcfs_index
    )
    if lcfs_index is None:
        raise ValueError("No LCFS index was found; provide --lcfs-index.")

    if MAX_SPINS is not None:
        settings["SPINS"] = MAX_SPINS
    if args.spins is not None:
        settings["SPINS"] = args.spins
    if (
        isinstance(settings["SPINS"], bool)
        or not isinstance(settings["SPINS"], int)
        or settings["SPINS"] <= 0
    ):
        raise ValueError("SPINS must be a positive integer.")

    n_seed_planes = (
        N_SEED_PLANES if args.seed_planes is None else args.seed_planes
    )
    seed_phi_deg = (
        settings["IC_PHI_DEG"]
        if args.seed_phi_deg is None
        else args.seed_phi_deg
    )
    seed_data = make_seed_initial_conditions(
        args.analysis_dir,
        lcfs_index,
        n_seed_planes,
        seed_phi_deg=seed_phi_deg,
    )

    sim_io = IOHandler(args.analysis_dir)
    sim_io.startLog(
        log_name="connection_length_volume.log",
        subdir=ANALYSIS_SUBDIR,
        logger_name=ANALYSIS_SUBDIR,
    )
    run_settings = {
        **settings,
        "LCFS_INDEX": lcfs_index,
        "LCFS_CLEARANCE_M": LCFS_CLEARANCE,
        "N_PLANES": N_PLANES,
        "N_SEED_PLANES": n_seed_planes,
        "SEED_PHI_OFFSET_DEG": seed_phi_deg,
        "SEED_PHI_DEG": seed_data["seed_phi_deg"].tolist(),
        "SEED_COUNTS": seed_data["seed_counts"].tolist(),
        "SEED_POINCARE_FILES": [
            str(path) for path in seed_data["poincare_paths"]
        ],
        "N_RHO": N_RHO,
        "N_THETA": N_THETA,
        "RHO_MIN": RHO_MIN,
        "RHO_MAX": RHO_MAX,
        "TRACED_FIELD_LINES": seed_data["initial_conditions_rtp"].shape[0],
        "DIRECTIONAL_SOLVES": (
            2 * seed_data["initial_conditions_rtp"].shape[0]
        ),
        "DOUBLE_LINE": True,
        "COLOR_SCALE": COLOR_SCALE,
        "COLORMAP": COLORMAP,
        "N_LEVELS": N_LEVELS,
        "VMIN": VMIN,
        "VMAX": VMAX,
        "DPI": DPI,
    }
    sim_io.inputsBoilerplate(
        "CONNECTION-LENGTH VOLUME INPUTS",
        run_settings,
        [
            "CURRENT_TOR",
            "CURRENT_HEL",
            "CONFIG_TOR",
            "CONFIG_HEL",
            "ENABLE_ERRFIELD",
            "LCFS_INDEX",
            "LCFS_CLEARANCE_M",
            "N_PLANES",
            "N_SEED_PLANES",
            "SEED_PHI_OFFSET_DEG",
            "SEED_PHI_DEG",
            "SEED_COUNTS",
            "SEED_POINCARE_FILES",
            "N_RHO",
            "N_THETA",
            "RHO_MIN",
            "RHO_MAX",
            "TRACED_FIELD_LINES",
            "DIRECTIONAL_SOLVES",
            "SPINS",
            "SOLVER",
            "RTOL",
            "ATOL",
            "NTHREADS",
            "DOUBLE_LINE",
            "COLOR_SCALE",
            "COLORMAP",
            "N_LEVELS",
            "VMIN",
            "VMAX",
            "DPI",
        ],
    )

    magnetic_field = build_magnetic_field(settings)
    trace_data = trace_connection_length_volume(
        seed_data["initial_conditions_rtp"],
        seed_data["seed_id"],
        seed_data["seed_plane_index"],
        settings,
        magnetic_field,
        sim_io,
    )
    data_dir = save_raw_outputs(sim_io, seed_data, trace_data)
    plot_all_planes(args.analysis_dir, lcfs_index, trace_data, sim_io)

    hit_count = np.count_nonzero(trace_data["hit_wall"])
    total_directions = trace_data["hit_wall"].size
    sim_io.log.info(
        "Wall intersections: %d of %d directional traces.",
        hit_count,
        total_directions,
    )
    sim_io.log.info("Saved raw connection-length data: %s", data_dir)
    sim_io.log.info(
        "Saved %d contour plots: %s",
        N_PLANES,
        Path(sim_io.plot_dir) / ANALYSIS_SUBDIR,
    )
    sim_io.log.info("## CONNECTION-LENGTH VOLUME ANALYSIS FINISHED ##")
    print(f"Saved raw data: {data_dir}")
    print(
        "Saved contour plots: "
        f"{Path(sim_io.plot_dir) / ANALYSIS_SUBDIR}"
    )


if __name__ == "__main__":
    main()
