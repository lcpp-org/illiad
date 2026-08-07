"""Replot an already-generated stitched nField / connection-length field.

This helper reads the regular ``(phi, theta, rho)`` field, coordinate arrays,
and saved stitch-zone boundaries. It performs no interpolation, stitching, or
field-line tracing, so the plot settings below can be changed independently of
the original analysis.
"""

import argparse
from contextlib import nullcontext
import gc
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from illiad.io import IOHandler
from illiad.sol import (
    load_lcfs_boundary,
    load_poincare_settings,
)


# DATA AND OUTPUT SETTINGS
ANALYSIS_DIR = "IOTA3_1000sp_atol1e-9"
DATA_SUBDIR = "ConLenVolume_REDO_250spins_rk1mm_RegularGrid_Stitched_v2"
OUTPUT_SUBDIR = None  # None writes replots to the input data subdirectory
LCFS_INDEX = None     # None reads LCFS_INDEX from the Poincare log

# None replots every saved plane. A scalar or iterable selects computational
# toroidal angles in degrees; for example, 18 or [18, 90, 180]. The same
# selection can be supplied with ``--phi`` without editing this file.
PHI_DEG = None

FIELD_FILENAME = "stitched_nfield_connection_length.npy"
RHO_FILENAME = "rho_grid_m.npy"
THETA_FILENAME = "theta_grid_rad.npy"
PHI_FILENAME = "phi_grid_deg.npy"
OUTER_BOUNDARY_FILENAME = "stitch_outer_boundary_xz_m.npy"
BOUNDARY_METADATA_FILENAME = "stitch_boundary_metadata.npz"
OUTPUT_FILENAME = "stitched_field_{phi_deg:03.0f}_replot.png"

# PLOT SETTINGS
FIGSIZE = (7, 6)
DPI = 300

COLOR_SCALE = "log"       # "log" or "linear"
COLORMAP = "afmhot"       # Any Matplotlib colormap
N_LEVELS = 10
VMIN = 4e-5                # None uses the minimum finite saved value
VMAX = 1.0                 # None uses the maximum finite saved value
CONTOUR_EXTEND = "both"   # "auto", "neither", "both", "min", or "max"
ANTIALIASED = False

SHOW_LCFS = False
LCFS_COLOR = "black"
LCFS_LINESTYLE = "-"
LCFS_LINEWIDTH = 1.2
LCFS_LABEL = "LCFS"

SHOW_STITCH_BOUNDARY = False
STITCH_BOUNDARY_COLOR = "cyan"
STITCH_BOUNDARY_LINESTYLE = "--"
STITCH_BOUNDARY_LINEWIDTH = 1.0
STITCH_BOUNDARY_LABEL = None  # None uses the saved contour level

SHOW_VESSEL = True
VESSEL_RADIUS = 0.19
VESSEL_COLOR = "0.35"
VESSEL_LINESTYLE = "-"
VESSEL_LINEWIDTH = 1.0
VESSEL_LABEL = "Vessel wall"

# None creates the standard title. A custom title may use the fields
# {phi_deg}, {phi_phys_deg}, and {boundary_level}.
TITLE = None
TITLE_SIZE = 12
TITLE_PAD = 8
PHYSICAL_PHI_OFFSET_DEG = 198.0

X_LIMITS = None            # None uses (-VESSEL_RADIUS, VESSEL_RADIUS)
Y_LIMITS = None
ASPECT = "equal"
X_LABEL = r"$x=\rho\cos\theta$ [m]"
Y_LABEL = r"$z=\rho\sin\theta$ [m]"

SHOW_GRID = True
GRID_COLOR = "0.75"
GRID_LINEWIDTH = 0.4
GRID_ALPHA = 1.0

SHOW_LEGEND = False
LEGEND_LOCATION = "upper right"

COLORBAR_LABEL = "Stitched scalar field"
COLORBAR_PAD = 0.03
COLORBAR_TICKS = None

SHOW_PLOT = False
SHOW_PROGRESS = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replot an existing stitched nField / connection-length field."
    )
    parser.add_argument(
        "analysis_dir",
        nargs="?",
        default=ANALYSIS_DIR,
        help=f"Directory under output/ (default: {ANALYSIS_DIR}).",
    )
    parser.add_argument(
        "data_subdir",
        nargs="?",
        default=DATA_SUBDIR,
        help=f"Stitched-field data subdirectory (default: {DATA_SUBDIR}).",
    )
    parser.add_argument(
        "--output-subdir",
        default=OUTPUT_SUBDIR,
        help="Plot/log output subdirectory (default: input data subdirectory).",
    )
    parser.add_argument(
        "--lcfs-index",
        type=int,
        default=LCFS_INDEX,
        help="LCFS Poincare surface index (default: read from the Poincare log).",
    )
    parser.add_argument(
        "--phi",
        nargs="+",
        type=float,
        default=None,
        help="Only replot these computational toroidal angles in degrees.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=SHOW_PROGRESS,
        help="Show a plotting progress bar.",
    )
    return parser.parse_args()


def require_file(path, description):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def load_stitched_data(data_dir):
    field_path = require_file(data_dir / FIELD_FILENAME, "stitched field")
    rho_path = require_file(data_dir / RHO_FILENAME, "rho grid")
    theta_path = require_file(data_dir / THETA_FILENAME, "theta grid")
    phi_path = require_file(data_dir / PHI_FILENAME, "phi grid")
    boundary_path = require_file(
        data_dir / OUTER_BOUNDARY_FILENAME,
        "stitch outer-boundary data",
    )

    field = np.load(field_path, mmap_mode="r")
    rho = np.load(rho_path)
    theta = np.load(theta_path)
    phi_deg = np.load(phi_path)
    boundaries = np.load(boundary_path, mmap_mode="r")

    expected_shape = (phi_deg.size, theta.size, rho.size)
    if field.shape != expected_shape:
        raise ValueError(
            f"Stitched field shape {field.shape} does not match coordinate "
            f"shape {expected_shape}."
        )
    if boundaries.ndim != 3 or boundaries.shape[0] != phi_deg.size:
        raise ValueError(
            "stitch_outer_boundary_xz_m.npy must have shape "
            f"(phi, boundary_point, 2); found {boundaries.shape}."
        )
    if boundaries.shape[2] != 2 or boundaries.shape[1] < 3:
        raise ValueError(
            "Every saved stitch boundary must contain at least three x-z points."
        )
    if not np.all(np.isfinite(boundaries)):
        raise ValueError("Saved stitch boundaries contain non-finite values.")
    if np.any(np.diff(rho) <= 0.0):
        raise ValueError("rho_grid_m.npy must be strictly increasing.")
    if np.any(np.diff(theta) <= 0.0):
        raise ValueError("theta_grid_rad.npy must be strictly increasing.")
    if np.any(np.diff(phi_deg) <= 0.0):
        raise ValueError("phi_grid_deg.npy must be strictly increasing.")

    metadata_path = data_dir / BOUNDARY_METADATA_FILENAME
    if metadata_path.is_file():
        with np.load(metadata_path) as metadata:
            if "contour_level" in metadata:
                boundary_levels = np.asarray(
                    metadata["contour_level"],
                    dtype=np.float64,
                )
            else:
                boundary_levels = np.full(phi_deg.size, np.nan)
    else:
        boundary_levels = np.full(phi_deg.size, np.nan)
    if boundary_levels.shape != (phi_deg.size,):
        raise ValueError(
            "Saved contour_level metadata must contain one value per phi plane."
        )

    return field, rho, theta, phi_deg, boundaries, boundary_levels, field_path


def data_range(field):
    data_min = np.inf
    data_max = -np.inf
    for plane_index in range(field.shape[0]):
        plane = np.asarray(field[plane_index])
        finite = plane[np.isfinite(plane)]
        if COLOR_SCALE == "log":
            finite = finite[finite > 0.0]
        if finite.size:
            data_min = min(data_min, float(finite.min()))
            data_max = max(data_max, float(finite.max()))
    if not np.isfinite(data_min):
        raise ValueError("The stitched field has no finite plottable values.")
    return data_min, data_max


def make_color_scale(field):
    data_min, data_max = data_range(field)
    value_min = data_min if VMIN is None else float(VMIN)
    value_max = data_max if VMAX is None else float(VMAX)
    if not value_min < value_max:
        raise ValueError("Resolved color limits require VMIN < VMAX.")

    if COLOR_SCALE == "log":
        if value_min <= 0.0:
            raise ValueError("A logarithmic color scale requires VMIN > 0.")
        levels = np.geomspace(value_min, value_max, N_LEVELS)
        norm = LogNorm(vmin=value_min, vmax=value_max)
    elif COLOR_SCALE == "linear":
        levels = np.linspace(value_min, value_max, N_LEVELS)
        norm = Normalize(vmin=value_min, vmax=value_max)
    else:
        raise ValueError('COLOR_SCALE must be either "log" or "linear".')

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
    return levels, norm, extend, data_min, data_max, value_min, value_max


def resolve_plane_indices(saved_phi_deg, cli_phi):
    selection = PHI_DEG if cli_phi is None else cli_phi
    if selection is None:
        return np.arange(saved_phi_deg.size, dtype=np.int64)
    requested = (
        [float(selection)]
        if np.isscalar(selection)
        else [float(phi) for phi in selection]
    )
    indices = []
    for requested_phi in requested:
        normalized_phi = requested_phi % 360.0
        if np.isclose(normalized_phi, 0.0):
            normalized_phi = 360.0
        distance = np.abs(
            (np.asarray(saved_phi_deg) - normalized_phi + 180.0) % 360.0
            - 180.0
        )
        plane_index = int(np.argmin(distance))
        if not np.isclose(distance[plane_index], 0.0, atol=1.0e-8):
            raise ValueError(
                f"No saved plane exists at phi={requested_phi:g} degrees."
            )
        if plane_index not in indices:
            indices.append(plane_index)
    return np.asarray(indices, dtype=np.int64)


def plot_plane(
    plane,
    rho,
    theta,
    phi_deg,
    lcfs_boundary,
    outer_boundary,
    boundary_level,
    levels,
    norm,
    extend,
    sim_io,
    output_subdir,
):
    plot_theta = np.concatenate(([0.0], theta))
    plot_data = np.vstack((plane[-1], plane))
    plot_theta_grid, plot_rho_grid = np.meshgrid(
        plot_theta,
        rho,
        indexing="ij",
    )
    plot_x = plot_rho_grid * np.cos(plot_theta_grid)
    plot_z = plot_rho_grid * np.sin(plot_theta_grid)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    color_artist = ax.contourf(
        plot_x,
        plot_z,
        plot_data,
        levels=levels,
        norm=norm,
        cmap=COLORMAP,
        extend=extend,
        antialiased=ANTIALIASED,
    )

    if SHOW_LCFS:
        closed_lcfs = np.vstack((lcfs_boundary, lcfs_boundary[0]))
        ax.plot(
            closed_lcfs[:, 0],
            closed_lcfs[:, 1],
            color=LCFS_COLOR,
            linestyle=LCFS_LINESTYLE,
            linewidth=LCFS_LINEWIDTH,
            label=LCFS_LABEL,
        )
    if SHOW_STITCH_BOUNDARY:
        closed_outer = np.vstack((outer_boundary, outer_boundary[0]))
        if STITCH_BOUNDARY_LABEL is not None:
            boundary_label = STITCH_BOUNDARY_LABEL
        elif np.isfinite(boundary_level):
            boundary_label = (
                f"$L_c/L_{{c,max}}={boundary_level:g}$ boundary"
            )
        else:
            boundary_label = "Stitch-zone boundary"
        ax.plot(
            closed_outer[:, 0],
            closed_outer[:, 1],
            color=STITCH_BOUNDARY_COLOR,
            linestyle=STITCH_BOUNDARY_LINESTYLE,
            linewidth=STITCH_BOUNDARY_LINEWIDTH,
            label=boundary_label,
        )
    if SHOW_VESSEL:
        vessel_angle = np.linspace(0.0, 2.0 * np.pi, 720)
        ax.plot(
            VESSEL_RADIUS * np.cos(vessel_angle),
            VESSEL_RADIUS * np.sin(vessel_angle),
            color=VESSEL_COLOR,
            linestyle=VESSEL_LINESTYLE,
            linewidth=VESSEL_LINEWIDTH,
            label=VESSEL_LABEL,
        )

    phi_phys_deg = (phi_deg + PHYSICAL_PHI_OFFSET_DEG) % 360.0
    if TITLE is None:
        title = (
            "Stitched nField and normalized connection length\n"
            f"$\\phi_{{phy}}={phi_phys_deg:03.0f}^\\circ$, "
            f"$\\phi_c={phi_deg:03.0f}^\\circ$"
        )
    else:
        title = TITLE.format(
            phi_deg=phi_deg,
            phi_phys_deg=phi_phys_deg,
            boundary_level=boundary_level,
        )
    ax.set_title(title, fontsize=TITLE_SIZE, pad=TITLE_PAD)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_xlim(
        (-VESSEL_RADIUS, VESSEL_RADIUS) if X_LIMITS is None else X_LIMITS
    )
    ax.set_ylim(
        (-VESSEL_RADIUS, VESSEL_RADIUS) if Y_LIMITS is None else Y_LIMITS
    )
    ax.set_aspect(ASPECT)
    ax.grid(
        SHOW_GRID,
        color=GRID_COLOR,
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA,
    )
    if SHOW_LEGEND and (SHOW_LCFS or SHOW_STITCH_BOUNDARY or SHOW_VESSEL):
        ax.legend(loc=LEGEND_LOCATION)

    colorbar_ticks = COLORBAR_TICKS
    if colorbar_ticks is None and COLOR_SCALE == "log":
        first_decade = int(np.ceil(np.log10(norm.vmin)))
        last_decade = int(np.floor(np.log10(norm.vmax)))
        colorbar_ticks = 10.0 ** np.arange(first_decade, last_decade + 1)
    colorbar = fig.colorbar(
        color_artist,
        ax=ax,
        pad=COLORBAR_PAD,
        ticks=colorbar_ticks,
    )
    colorbar.set_label(COLORBAR_LABEL)

    output_name = OUTPUT_FILENAME.format(phi_deg=phi_deg)
    sim_io.saveFig(output_name, dpi=DPI, subdir=output_subdir)
    sim_io.log.info("Saved stitched-field replot: %s/%s", output_subdir, output_name)
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)
        gc.collect()


def main():
    args = parse_args()
    output_subdir = (
        args.data_subdir
        if args.output_subdir is None
        else args.output_subdir
    )
    data_dir = (
        PROJECT_ROOT
        / "output"
        / args.analysis_dir
        / "data"
        / args.data_subdir
    )
    (
        field,
        rho,
        theta,
        saved_phi_deg,
        boundaries,
        boundary_levels,
        field_path,
    ) = load_stitched_data(data_dir)
    plane_indices = resolve_plane_indices(saved_phi_deg, args.phi)
    (
        levels,
        norm,
        extend,
        data_min,
        data_max,
        value_min,
        value_max,
    ) = make_color_scale(field)

    settings = load_poincare_settings(args.analysis_dir)
    lcfs_index = (
        settings.get("LCFS_INDEX")
        if args.lcfs_index is None
        else args.lcfs_index
    )
    if lcfs_index is None:
        raise ValueError("No LCFS index was found; provide --lcfs-index.")

    sim_io = IOHandler(args.analysis_dir)
    sim_io.startLog(
        log_name="plot_stitched_connection_length_nfield.log",
        subdir=output_subdir,
        logger_name=f"{output_subdir}_stitched_replot",
    )
    run_settings = {
        "ANALYSIS_DIR": args.analysis_dir,
        "DATA_SUBDIR": args.data_subdir,
        "OUTPUT_SUBDIR": output_subdir,
        "FIELD_PATH": str(field_path),
        "FIELD_SHAPE": field.shape,
        "SELECTED_PLANES": plane_indices.size,
        "LCFS_INDEX": lcfs_index,
        "COLOR_SCALE": COLOR_SCALE,
        "COLORMAP": COLORMAP,
        "N_LEVELS": N_LEVELS,
        "DATA_MIN": data_min,
        "DATA_MAX": data_max,
        "VMIN": value_min,
        "VMAX": value_max,
        "CONTOUR_EXTEND": extend,
        "SHOW_LCFS": SHOW_LCFS,
        "SHOW_STITCH_BOUNDARY": SHOW_STITCH_BOUNDARY,
        "SHOW_VESSEL": SHOW_VESSEL,
        "DPI": DPI,
        "SHOW_PROGRESS": args.progress,
    }
    sim_io.inputsBoilerplate(
        "STITCHED-FIELD REPLOT INPUTS",
        run_settings,
        list(run_settings),
    )

    print(f"Reading stitched field: {field_path}")
    print(f"Planes selected: {plane_indices.size} of {saved_phi_deg.size}")
    print(
        f"Color range: {value_min:g} to {value_max:g} "
        f"({COLOR_SCALE}; data {data_min:g} to {data_max:g})"
    )
    progress = tqdm(
        plane_indices,
        desc="Replotting stitched field",
        unit="plane",
        dynamic_ncols=True,
        disable=not args.progress,
    )
    log_context = (
        logging_redirect_tqdm(loggers=[sim_io.log])
        if args.progress
        else nullcontext()
    )
    with log_context:
        for plane_index in progress:
            phi_deg = float(saved_phi_deg[plane_index])
            lcfs_boundary, _ = load_lcfs_boundary(
                args.analysis_dir,
                phi_deg,
                lcfs_index,
            )
            plot_plane(
                field[plane_index],
                rho,
                theta,
                phi_deg,
                lcfs_boundary,
                np.asarray(boundaries[plane_index]),
                float(boundary_levels[plane_index]),
                levels,
                norm,
                extend,
                sim_io,
                output_subdir,
            )

    plot_dir = Path(sim_io.plot_dir) / output_subdir
    sim_io.log.info(
        "## STITCHED-FIELD REPLOT FINISHED: %d plots saved to %s ##",
        plane_indices.size,
        plot_dir,
    )
    print(f"Saved {plane_indices.size} replot(s): {plot_dir}")


if __name__ == "__main__":
    main()
