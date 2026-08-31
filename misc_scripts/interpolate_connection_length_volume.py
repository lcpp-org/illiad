"""Interpolate raw connection-length crossings onto a regular field mesh.

The saved field follows the normalized-flux scalar-field convention:
``(phi, theta, rho)``. Raw samples are accumulated into their nearest regular
grid cells, and missing cells outside the LCFS are filled from neighboring
occupied cells in the seam-free poloidal ``(x, z)`` plane. Cells inside the
LCFS remain NaN because the connection-length analysis does not sample them.
"""

import argparse
from contextlib import nullcontext
import gc
import os
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from illiad.io import IOHandler
from illiad.sol import (
    load_lcfs_boundary,
    load_poincare_settings,
)
from misc_scripts.plot_connection_length_volume import load_raw_samples


# Analysis settings
ANALYSIS_DIR = "IOTA3_1000sp_atol1e-9"
#DATA_SUBDIR = "ConLenVolumeTorch_30spins_MID2mm"
DATA_SUBDIR = "SOLTrace3c"
OUTPUT_SUBDIR = "SOLTrace3c_RegularGrid_log"  # None uses f"{DATA_SUBDIR}_RegularGrid"
LCFS_INDEX = 40     # None reads LCFS_INDEX from the Poincare log

# Regular grid. These defaults match the field/surface-parameter mesh.
N_RHO = 191
N_THETA = 180
RHO_MIN = 0.0
RHO_MAX = 0.19

# Interpolation settings
INTERPOLATION_SPACE = "log"  # "linear" or "log"
FILL_METHOD = "idw"             # "idw", "nearest", or "none"
IDW_NEIGHBORS = 8
IDW_POWER = 2.0
TREE_WORKERS = -1
RAW_CHUNK_SIZE = 250_000

# Plot settings
GENERATE_PLOTS = True
SHOW_PROGRESS = True
COLOR_SCALE = "log"             # "log" or "linear"
COLORMAP = "afmhot"
N_LEVELS = 50
VMIN = None
VMAX = None
CONTOUR_EXTEND = "both"         # "auto", "neither", "both", "min", "max"
DPI = 250
VESSEL_RADIUS = 0.19

FIELD_FILENAME = "connection_length_field_m.npy"
RHO_FILENAME = "rho_grid_m.npy"
THETA_FILENAME = "theta_grid_rad.npy"
PHI_FILENAME = "phi_grid_deg.npy"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate raw connection-length crossings onto a regular "
            "(phi, theta, rho) field mesh."
        )
    )
    parser.add_argument(
        "analysis_dir",
        nargs="?",
        default=ANALYSIS_DIR,
        help=f"Existing directory under output/ (default: {ANALYSIS_DIR}).",
    )
    parser.add_argument(
        "--data-subdir",
        default=DATA_SUBDIR,
        help=f"Raw connection-length data subdirectory (default: {DATA_SUBDIR}).",
    )
    parser.add_argument(
        "--output-subdir",
        default=OUTPUT_SUBDIR,
        help="Output data/plot/log subdirectory (default: <data-subdir>_RegularGrid).",
    )
    parser.add_argument(
        "--lcfs-index",
        type=int,
        default=LCFS_INDEX,
        help="LCFS surface index (default: read from the Poincare log).",
    )
    parser.add_argument(
        "--rho-count",
        type=int,
        default=N_RHO,
        help=f"Regular radial grid count (default: {N_RHO}).",
    )
    parser.add_argument(
        "--theta-count",
        type=int,
        default=N_THETA,
        help=f"Regular poloidal grid count (default: {N_THETA}).",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=GENERATE_PLOTS,
        help="Generate one regular-grid contour plot per toroidal plane.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=SHOW_PROGRESS,
        help="Show interpolation and plotting progress bars.",
    )
    return parser.parse_args()


def validate_settings(n_rho, n_theta):
    if n_rho < 2:
        raise ValueError("N_RHO must be at least 2.")
    if n_theta < 3:
        raise ValueError("N_THETA must be at least 3.")
    if not 0.0 <= RHO_MIN < RHO_MAX:
        raise ValueError("Require 0 <= RHO_MIN < RHO_MAX.")
    if RAW_CHUNK_SIZE <= 0:
        raise ValueError("RAW_CHUNK_SIZE must be positive.")
    if INTERPOLATION_SPACE not in {"linear", "log"}:
        raise ValueError('INTERPOLATION_SPACE must be "linear" or "log".')
    if FILL_METHOD not in {"idw", "nearest", "none"}:
        raise ValueError('FILL_METHOD must be "idw", "nearest", or "none".')
    if IDW_NEIGHBORS < 1:
        raise ValueError("IDW_NEIGHBORS must be positive.")
    if not np.isfinite(IDW_POWER) or IDW_POWER <= 0.0:
        raise ValueError("IDW_POWER must be positive and finite.")


def make_regular_grid(n_rho, n_theta):
    """Return flux-compatible radial and poloidal node arrays."""
    rho = np.linspace(RHO_MIN, RHO_MAX, n_rho, dtype=np.float64)
    theta = np.linspace(2.0 * np.pi / n_theta, 2.0 * np.pi, n_theta, dtype=np.float64)
    grid_theta, grid_rho = np.meshgrid(theta, rho, indexing="ij")
    grid_x = grid_rho * np.cos(grid_theta)
    grid_z = grid_rho * np.sin(grid_theta)

    return rho, theta, grid_rho, grid_theta, grid_x, grid_z


def values_for_slice(value_source, start, stop):
    """Materialize connection lengths for one slice of raw point indices."""
    if value_source["expanded"] is not None:
        return np.asarray(value_source["expanded"][start:stop])
    fieldline_id = np.asarray(value_source["fieldline_id"][start:stop])
    return np.asarray(value_source["fieldline"][fieldline_id])


def source_value_array(value_source):
    """Select the smallest array containing the global value population."""
    if value_source["fieldline"] is not None:
        return value_source["fieldline"]
    return value_source["expanded"]


def positive_data_range(values):
    data_min = np.inf
    data_max = -np.inf
    for start in range(0, values.size, RAW_CHUNK_SIZE):
        chunk = np.asarray(values[start : start + RAW_CHUNK_SIZE])
        finite = chunk[np.isfinite(chunk) & (chunk > 0.0)]
        if finite.size:
            data_min = min(data_min, float(finite.min()))
            data_max = max(data_max, float(finite.max()))
    if not np.isfinite(data_min):
        raise ValueError("No positive finite connection lengths are available.")
    return data_min, data_max


def make_color_scale(values):
    data_min, data_max = positive_data_range(values)
    value_min = data_min if VMIN is None else VMIN
    value_max = data_max if VMAX is None else VMAX
    if np.isclose(value_min, value_max):
        delta = max(0.01 * value_min, np.finfo(float).eps)
        value_min -= delta
        value_max += delta
    if not value_min < value_max:
        raise ValueError("Resolved color limits require VMIN < VMAX.")

    if COLOR_SCALE == "log":
        if value_min <= 0.0:
            raise ValueError("Logarithmic plots require VMIN > 0.")
        levels = np.geomspace(value_min, value_max, N_LEVELS)
        norm = LogNorm(vmin=value_min, vmax=value_max)
    elif COLOR_SCALE == "linear":
        levels = np.linspace(value_min, value_max, N_LEVELS)
        norm = Normalize(vmin=value_min, vmax=value_max)
    else:
        raise ValueError('COLOR_SCALE must be "linear" or "log".')

    if CONTOUR_EXTEND == "auto":
        below = data_min < value_min
        above = data_max > value_max
        if below and above:
            extend = "both"
        elif below:
            extend = "min"
        elif above:
            extend = "max"
        else:
            extend = "neither"
    elif CONTOUR_EXTEND in {"neither", "both", "min", "max"}:
        extend = CONTOUR_EXTEND
    else:
        raise ValueError("Invalid CONTOUR_EXTEND setting.")
    return levels, norm, extend, value_min, value_max


def grid_indices(points_rtp, n_rho, n_theta):
    """Map RTP points to their nearest periodic regular-grid nodes."""
    radial_spacing = (RHO_MAX - RHO_MIN) / (n_rho - 1)
    rho_index = np.rint( (points_rtp[:, 0] - RHO_MIN) / radial_spacing ).astype(np.int64)
    theta_spacing = 2.0 * np.pi / n_theta
    theta_index = ( np.rint(np.remainder(points_rtp[:, 1], 2.0 * np.pi) / theta_spacing).astype(np.int64) - 1) % n_theta
    return rho_index, theta_index


def accumulate_plane(points, value_source, start, stop, n_rho, n_theta,):
    """Aggregate every finite raw plane sample into its nearest grid cell."""
    cell_count = n_rho * n_theta
    value_sum = np.zeros(cell_count, dtype=np.float64)
    sample_count = np.zeros(cell_count, dtype=np.int64)
    used_samples = 0

    for chunk_start in range(start, stop, RAW_CHUNK_SIZE):
        chunk_stop = min(chunk_start + RAW_CHUNK_SIZE, stop)
        point_chunk = np.asarray(points[chunk_start:chunk_stop])
        value_chunk = values_for_slice(value_source, chunk_start, chunk_stop)
        finite = (
            np.all(np.isfinite(point_chunk[:, :2]), axis=1)
            & np.isfinite(value_chunk)
            & (value_chunk > 0.0)
            & (point_chunk[:, 0] >= RHO_MIN)
            & (point_chunk[:, 0] <= RHO_MAX)
        )
        if not np.any(finite):
            continue

        point_chunk = point_chunk[finite]
        value_chunk = value_chunk[finite]
        if INTERPOLATION_SPACE == "log":
            value_chunk = np.log(value_chunk)
        rho_index, theta_index = grid_indices(point_chunk, n_rho, n_theta)

        valid_index = (rho_index >= 0) & (rho_index < n_rho)
        flat_index = theta_index[valid_index] * n_rho + rho_index[valid_index]
        value_sum += np.bincount(flat_index, weights=value_chunk[valid_index], minlength=cell_count)
        sample_count += np.bincount(flat_index, minlength=cell_count)
        used_samples += int(np.count_nonzero(valid_index))

    occupied = sample_count > 0
    gridded = np.full(cell_count, np.nan, dtype=np.float64)
    gridded[occupied] = value_sum[occupied] / sample_count[occupied]
    if INTERPOLATION_SPACE == "log":
        gridded[occupied] = np.exp(gridded[occupied])

    return (gridded.reshape(n_theta, n_rho), sample_count.reshape(n_theta, n_rho), used_samples)


def exterior_mask(boundary, grid_x, grid_z):
    closed_boundary = np.vstack((boundary, boundary[0]))
    inside = MplPath(closed_boundary).contains_points( np.column_stack((grid_x.ravel(), grid_z.ravel())))
    return ~inside.reshape(grid_x.shape)


def fill_missing_cells(field, exterior, grid_x, grid_z):
    """Fill unsampled exterior cells from occupied grid nodes in x-z space."""
    field = field.copy()
    field[~exterior] = np.nan
    occupied = exterior & np.isfinite(field) & (field > 0.0)
    missing = exterior & ~occupied
    if not np.any(missing) or FILL_METHOD == "none":
        return field, int(np.count_nonzero(occupied)), 0
    if not np.any(occupied):
        raise ValueError("No occupied regular-grid cells remain outside the LCFS.")

    source_points = np.column_stack((grid_x[occupied], grid_z[occupied]))
    target_points = np.column_stack((grid_x[missing], grid_z[missing]))
    source_values = field[occupied]
    if INTERPOLATION_SPACE == "log":
        source_values = np.log(source_values)

    tree = cKDTree(source_points)
    neighbor_count = 1 if FILL_METHOD == "nearest" else min(IDW_NEIGHBORS, source_points.shape[0])
    distances, indices = tree.query(target_points, k=neighbor_count, workers=TREE_WORKERS,)
    if neighbor_count == 1:
        filled_values = source_values[indices]
    else:
        distances = np.maximum(distances, np.finfo(np.float64).eps)
        weights = distances ** (-IDW_POWER)
        filled_values = np.sum(weights * source_values[indices], axis=1) / np.sum(weights, axis=1)
    if INTERPOLATION_SPACE == "log":
        filled_values = np.exp(filled_values)
    field[missing] = filled_values
    return (field, int(np.count_nonzero(occupied)), int(np.count_nonzero(missing)),)


def interpolate_field(analysis_dir, points, value_source, offsets,
                      phi_deg, lcfs_index, rho, theta, grid_x, grid_z,
                      output_path, sim_io, show_progress):

    """Build and save the complete regular connection-length field."""
    field = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float64, shape=(phi_deg.size, theta.size, rho.size))

    start_time = perf_counter()
    progress = tqdm(range(phi_deg.size), desc="Regularizing connection length", unit="plane", dynamic_ncols=True, disable=not show_progress)

    log_context = (logging_redirect_tqdm(loggers=[sim_io.log]) if show_progress else nullcontext())
    with log_context:
        for plane_index in progress:
            plane_start = int(offsets[plane_index])
            plane_stop = int(offsets[plane_index + 1])
            boundary, _ = load_lcfs_boundary( analysis_dir, float(phi_deg[plane_index]), lcfs_index)
            exterior = exterior_mask(boundary, grid_x, grid_z)
            binned, counts, used_samples = accumulate_plane(points, value_source, plane_start, plane_stop, rho.size, theta.size)
            regular, occupied_count, filled_count = fill_missing_cells(binned, exterior, grid_x, grid_z)
            field[plane_index] = regular
            sim_io.log.info("Regularized phi=%03.0f deg: %d/%d raw samples used, %d directly occupied exterior cells, %d filled cells, %d samples in the busiest cell.",
                            phi_deg[plane_index], used_samples, plane_stop - plane_start, occupied_count, filled_count, int(counts.max(initial=0)))

            if (plane_index + 1) % 10 == 0:
                field.flush()
                gc.collect()
    field.flush()
    sim_io.log.info("REGULAR CONNECTION-LENGTH FIELD FINISHED IN %.3f seconds.", perf_counter() - start_time)
    return np.load(output_path, mmap_mode="r")


def plot_plane(plane, rho, theta, phi_deg,
               boundary, levels, norm, extend,
               sim_io, output_subdir):

    """Plot one regular field plane in the original Cartesian cross-section."""
    plot_theta = np.concatenate(([0.0], theta))
    plot_data = np.vstack((plane[-1], plane))
    plot_theta_grid, plot_rho_grid = np.meshgrid(plot_theta, rho, indexing="ij")
    plot_x = plot_rho_grid * np.cos(plot_theta_grid)
    plot_z = plot_rho_grid * np.sin(plot_theta_grid)

    fig, ax = plt.subplots(figsize=(7, 6))
    color_artist = ax.contourf(plot_x, plot_z, plot_data, levels=levels, norm=norm, cmap=COLORMAP, extend=extend)
    closed_boundary = np.vstack((boundary, boundary[0]))
    ax.plot(closed_boundary[:, 0], closed_boundary[:, 1], color="black", linewidth=1.0, label="LCFS")
    vessel_angle = np.linspace(0.0, 2.0 * np.pi, 720)
    ax.plot(VESSEL_RADIUS * np.cos(vessel_angle), VESSEL_RADIUS * np.sin(vessel_angle), color="0.35", linewidth=1.0, label="Vessel wall")
    physical_phi = (phi_deg + 198.0) % 360.0
    ax.set_title(f"Regular-grid connection length\n$\\phi_{{phy}}={physical_phi:03.0f}^\\circ$ CW from North split, $\\phi_c={phi_deg:03.0f}^\\circ$")
    ax.set_xlabel(r"$x=\rho\cos\theta$ [m]")
    ax.set_ylabel(r"$z=\rho\sin\theta$ [m]")
    ax.set_xlim(-VESSEL_RADIUS, VESSEL_RADIUS)
    ax.set_ylim(-VESSEL_RADIUS, VESSEL_RADIUS)
    ax.set_aspect("equal")
    ax.grid(linewidth=0.4, color="0.75")
    ax.legend(loc="upper right")
    colorbar = fig.colorbar(color_artist, ax=ax, pad=0.03)
    colorbar.set_label("Connection length [m]")

    plot_name = f"connection_length_field_{phi_deg:03.0f}.png"
    sim_io.saveFig(plot_name, dpi=DPI, subdir=output_subdir)
    sim_io.log.info("Saved regular-grid contour: %s/%s", output_subdir, plot_name)
    plt.close(fig)
    gc.collect()


def plot_field(analysis_dir, field,
               rho, theta, phi_deg, lcfs_index, value_source,
               sim_io, output_subdir, show_progress):

    levels, norm, extend, value_min, value_max = make_color_scale(source_value_array(value_source))
    sim_io.log.info("Regular-grid plot color range: %g to %g m (%s).", value_min, value_max, COLOR_SCALE)
    progress = tqdm( range(phi_deg.size), desc="Plotting regular field", unit="plane", dynamic_ncols=True, disable=not show_progress)
    log_context = (logging_redirect_tqdm(loggers=[sim_io.log]) if show_progress else nullcontext())
    with log_context:
        for plane_index in progress:
            boundary, _ = load_lcfs_boundary(analysis_dir, float(phi_deg[plane_index]), lcfs_index)
            plot_plane(field[plane_index], rho, theta, float(phi_deg[plane_index]),
                       boundary, levels, norm, extend,
                       sim_io, output_subdir)


def main():
    args = parse_args()
    validate_settings(args.rho_count, args.theta_count)
    output_subdir = (
        f"{args.data_subdir}_RegularGrid"
        if args.output_subdir is None
        else args.output_subdir
    )
    settings = load_poincare_settings(args.analysis_dir)
    lcfs_index = (
        settings.get("LCFS_INDEX")
        if args.lcfs_index is None
        else args.lcfs_index
    )
    if lcfs_index is None:
        raise ValueError("No LCFS index was found; provide --lcfs-index.")

    raw_data_dir = (
        PROJECT_ROOT
        / "output"
        / args.analysis_dir
        / "data"
        / args.data_subdir
    )
    points, value_source, offsets, phi_deg = load_raw_samples(raw_data_dir)
    rho, theta, _, _, grid_x, grid_z = make_regular_grid(args.rho_count, args.theta_count)

    sim_io = IOHandler(args.analysis_dir)
    sim_io.startLog(log_name="interpolate_connection_length_volume.log", subdir=output_subdir, logger_name=output_subdir,)
    output_data_dir = Path(sim_io.data_dir) / output_subdir
    output_data_dir.mkdir(parents=True, exist_ok=True)
    field_path = output_data_dir / FIELD_FILENAME
    input_format = (
        "expanded"
        if value_source["expanded"] is not None
        else "compact_fieldline_indexed"
    )
    run_settings = {
        "ANALYSIS_DIR": args.analysis_dir,
        "DATA_SUBDIR": args.data_subdir,
        "OUTPUT_SUBDIR": output_subdir,
        "RAW_DATA_DIR": str(raw_data_dir),
        "INPUT_FORMAT": input_format,
        "RAW_SAMPLE_COUNT": points.shape[0],
        "LCFS_INDEX": lcfs_index,
        "N_PHI": phi_deg.size,
        "N_THETA": theta.size,
        "N_RHO": rho.size,
        "FIELD_SHAPE": (phi_deg.size, theta.size, rho.size),
        "RHO_MIN": RHO_MIN,
        "RHO_MAX": RHO_MAX,
        "INTERPOLATION_SPACE": INTERPOLATION_SPACE,
        "FILL_METHOD": FILL_METHOD,
        "IDW_NEIGHBORS": IDW_NEIGHBORS,
        "IDW_POWER": IDW_POWER,
        "TREE_WORKERS": TREE_WORKERS,
        "RAW_CHUNK_SIZE": RAW_CHUNK_SIZE,
        "FIELD_FILENAME": FIELD_FILENAME,
        "GENERATE_PLOTS": args.plots,
        "SHOW_PROGRESS": args.progress,
        "COLOR_SCALE": COLOR_SCALE,
        "COLORMAP": COLORMAP,
        "N_LEVELS": N_LEVELS,
        "VMIN": VMIN,
        "VMAX": VMAX,
        "CONTOUR_EXTEND": CONTOUR_EXTEND,
        "DPI": DPI,
    }

    input_keys = list(run_settings)
    sim_io.inputsBoilerplate( "CONNECTION-LENGTH REGULAR-FIELD INPUTS", run_settings, input_keys)

    sim_io.saveNumpyData(rho, RHO_FILENAME.removesuffix(".npy"), subdir=output_subdir)
    sim_io.saveNumpyData(theta, THETA_FILENAME.removesuffix(".npy"), subdir=output_subdir)
    sim_io.saveNumpyData(phi_deg, PHI_FILENAME.removesuffix(".npy"), subdir=output_subdir)

    print(f"Reading raw connection-length data: {raw_data_dir}")
    print(f"Regular field shape (phi, theta, rho): {run_settings['FIELD_SHAPE']}")
    field = interpolate_field(args.analysis_dir, points, value_source, offsets,
                              phi_deg, lcfs_index, rho, theta, grid_x, grid_z,
                              field_path, sim_io, args.progress)

    sim_io.log.info("Saved regular connection-length field: %s", field_path)

    if args.plots:
        plot_field(args.analysis_dir, field,
                   rho, theta, phi_deg, lcfs_index, value_source,
                   sim_io, output_subdir, args.progress)
        sim_io.log.info("Saved %d regular-grid contour plots: %s", phi_deg.size, Path(sim_io.plot_dir) / output_subdir)


    sim_io.log.info("## CONNECTION-LENGTH REGULAR-FIELD ANALYSIS FINISHED ##")


if __name__ == "__main__":
    main()
