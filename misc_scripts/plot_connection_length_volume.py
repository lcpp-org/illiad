"""Replot connection-length volume slices from an already-completed analysis.

This helper reads raw, plane-sorted connection-length samples. It accepts the
current plane-sharded output as well as historical monolithic NumPy outputs.
It does not construct a magnetic field or trace field lines.
"""

import argparse
import gc
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.path import Path as MplPath
import matplotlib.tri as mtri
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from illiad.sol import (
    load_lcfs_boundary,
    load_poincare_settings,
    open_plane_crossing_source,
)


# DATA AND OUTPUT SETTINGS
ANALYSIS_DIR = "IOTA4_1000sp_atol1e-9"
DATA_SUBDIR = "SOLtrace_500"
OUTPUT_SUBDIR = DATA_SUBDIR

# None replots every saved plane. A single number or an iterable of numbers
# selects computational toroidal angles in degrees; for example, 18 or
# [18, 90, 180].
PHI_DEG = None
LCFS_INDEX = 30  # None reads LCFS_INDEX from the Poincare log

OUTPUT_FILENAME = "connection_length_{phi_deg:03.0f}_replot_3.png"

# PLOT SETTINGS
FIGSIZE = (7, 6)
DPI = 250
COLOR_SCALE = "log"       # "log" or "linear"
COLORMAP = "afmhot"      # Any Matplotlib colormap
N_LEVELS = 50
VMIN = 0.05               # None uses the minimum positive saved value
VMAX = 6000               # None uses the maximum positive saved value
CONTOUR_EXTEND = "auto"   # "auto", "neither", "both", "min", or "max"
MASK_COLOR = "white"
ANTIALIASED = False
PLOT_MAX_SAMPLES = 150_000*9  # None disables deterministic plot-only thinning
PLOT_SAMPLE_SEED = 0
PLOT_EXTEND_TO_WALL = True  # Plot-only nearest-sample extrapolation to the wall

# Triangles whose centroid or any edge midpoint lies inside the LCFS are
# removed from the filled contour. The raw samples themselves are unchanged.
MASK_LCFS_INTERIOR = True

SHOW_RAW_POINTS = False
RAW_POINT_COLOR = "black"
RAW_POINT_SIZE = 0.1
RAW_POINT_ALPHA = 0.7
RAW_POINT_ZORDER = 5

SHOW_LCFS = True
LCFS_COLOR = "black"
LCFS_LINEWIDTH = 1.0
LCFS_LABEL = "LCFS"

SHOW_VESSEL = True
VESSEL_RADIUS = 0.19
VESSEL_COLOR = "0.35"
VESSEL_LINEWIDTH = 1.0
VESSEL_LABEL = "Vessel wall"

# None creates the standard title. A custom title may use the format fields
# {phi_deg} and {phi_phys_deg}.
TITLE = None
TITLE_SIZE = 12
TITLE_PAD = 8
PHYSICAL_PHI_OFFSET_DEG = 198.0

X_LIMITS = None           # None uses (-VESSEL_RADIUS, VESSEL_RADIUS)
Y_LIMITS = None
ASPECT = "equal"
X_LABEL = r"$x=\rho\cos\theta$ [m]"
Y_LABEL = r"$z=\rho\sin\theta$ [m]"

SHOW_GRID = True
GRID_COLOR = "0.75"
GRID_LINEWIDTH = 0.4
GRID_ALPHA = 0.8

SHOW_LEGEND = False
LEGEND_LOCATION = "upper right"

COLORBAR_LABEL = "Connection length [m]"
COLORBAR_PAD = 0.03
COLORBAR_TICKS = None

SCATTER_FALLBACK_SIZE = 4.0
SAVE_BBOX = "tight"
SHOW_PLOT = False

# Number of samples read at once. Plane sampling happens while these chunks
# are streamed, so the complete raw plane is never loaded when a plot limit is
# configured.
PLOT_READ_CHUNK_SIZE = 1_000_000
COLOR_RANGE_CHUNK_SIZE = 1_000_000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replot an existing connection-length volume dataset."
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
        help=f"Subdirectory under data/ (default: {DATA_SUBDIR}).",
    )
    parser.add_argument(
        "--phi",
        nargs="+",
        type=float,
        default=None,
        help="Only replot these computational toroidal angles in degrees.",
    )
    return parser.parse_args()


def data_range(source):
    """Find the positive finite range from the compact value population."""
    data_min = np.inf
    data_max = -np.inf
    for chunk in source.iter_value_chunks(COLOR_RANGE_CHUNK_SIZE):
        positive = chunk[np.isfinite(chunk) & (chunk > 0.0)]
        if positive.size:
            data_min = min(data_min, float(positive.min()))
            data_max = max(data_max, float(positive.max()))

    if not np.isfinite(data_min):
        raise ValueError("No positive finite connection lengths were found.")
    return data_min, data_max


def make_color_scale(source):
    data_min, data_max = data_range(source)
    value_min = data_min if VMIN is None else VMIN
    value_max = data_max if VMAX is None else VMAX

    if np.isclose(value_min, value_max):
        value_min *= 0.99
        value_max *= 1.01
    if not value_min < value_max:
        raise ValueError("Plot limits require VMIN < VMAX.")

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

    valid_extend = {"auto", "neither", "both", "min", "max"}
    if CONTOUR_EXTEND not in valid_extend:
        raise ValueError(
            "CONTOUR_EXTEND must be one of "
            '"auto", "neither", "both", "min", or "max".'
        )
    if CONTOUR_EXTEND == "auto":
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
    else:
        extend = CONTOUR_EXTEND

    return levels, norm, extend, value_min, value_max


def resolve_plane_indices(saved_phi_deg, command_line_phi=None):
    """Map requested computational angles to saved plane indices."""
    requested_phi = PHI_DEG if command_line_phi is None else command_line_phi
    if requested_phi is None:
        return np.arange(saved_phi_deg.size, dtype=np.int64)

    requested = (
        [float(requested_phi)]
        if np.isscalar(requested_phi)
        else [float(phi) for phi in requested_phi]
    )
    plane_indices = []
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
        if plane_index not in plane_indices:
            plane_indices.append(plane_index)
    return np.asarray(plane_indices, dtype=np.int64)


def stream_plane_samples(source, plane_index):
    """Read one plane and retain a bounded deterministic uniform sample."""
    if (
        PLOT_MAX_SAMPLES is not None
        and (
            isinstance(PLOT_MAX_SAMPLES, bool)
            or not isinstance(PLOT_MAX_SAMPLES, int)
            or PLOT_MAX_SAMPLES <= 0
        )
    ):
        raise ValueError("PLOT_MAX_SAMPLES must be None or a positive integer.")

    rng = np.random.default_rng(PLOT_SAMPLE_SEED)
    sampled_points = np.empty((0, 3), dtype=np.float64)
    sampled_values = np.empty(0, dtype=np.float64)
    sampled_keys = np.empty(0, dtype=np.float64)
    valid_count = 0

    for chunk in source.iter_plane_chunks(plane_index, PLOT_READ_CHUNK_SIZE):
        finite = (
            np.all(np.isfinite(chunk.points_rtp), axis=1)
            & np.isfinite(chunk.connection_length_m)
            & (chunk.connection_length_m > 0.0)
        )
        points = np.asarray(chunk.points_rtp[finite])
        values = np.asarray(chunk.connection_length_m[finite])
        valid_count += points.shape[0]
        if not points.size:
            continue

        if PLOT_MAX_SAMPLES is None:
            sampled_points = np.concatenate((sampled_points, points))
            sampled_values = np.concatenate((sampled_values, values))
            continue

        keys = rng.random(points.shape[0])
        sampled_points = np.concatenate((sampled_points, points))
        sampled_values = np.concatenate((sampled_values, values))
        sampled_keys = np.concatenate((sampled_keys, keys))
        if sampled_keys.size > PLOT_MAX_SAMPLES:
            keep = np.argpartition(
                sampled_keys,
                PLOT_MAX_SAMPLES - 1,
            )[:PLOT_MAX_SAMPLES]
            sampled_points = sampled_points[keep]
            sampled_values = sampled_values[keep]
            sampled_keys = sampled_keys[keep]

    print(
        f"Plane {plane_index}: retained {sampled_points.shape[0]} of "
        f"{valid_count} positive finite samples."
    )
    return sampled_points, sampled_values


def unique_plane_samples(points_rtp, values):
    """Remove invalid and exactly duplicated x-z samples for triangulation."""
    finite = (
        np.all(np.isfinite(points_rtp), axis=1)
        & np.isfinite(values)
        & (values > 0.0)
    )
    points_rtp = np.asarray(points_rtp[finite])
    values = np.asarray(values[finite])
    if not points_rtp.size:
        return points_rtp, values

    x = points_rtp[:, 0] * np.cos(points_rtp[:, 1])
    z = points_rtp[:, 0] * np.sin(points_rtp[:, 1])
    _, unique_indices = np.unique(
        np.column_stack((x, z)),
        axis=0,
        return_index=True,
    )
    unique_indices.sort()
    return points_rtp[unique_indices], values[unique_indices]


def mask_lcfs_triangles(triangulation, x, z, boundary):
    """Mask triangles probing inside the LCFS at their center or edge centers."""
    triangle_x = x[triangulation.triangles]
    triangle_z = z[triangulation.triangles]
    probes = np.stack(
        (
            np.column_stack((triangle_x.mean(axis=1), triangle_z.mean(axis=1))),
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
        probes.reshape(-1, 2)
    ).reshape(-1, probes.shape[1]).any(axis=1)
    triangulation.set_mask(inside_lcfs)
    if np.all(inside_lcfs):
        raise ValueError("Every triangulation element is inside the LCFS.")


def plot_plane(
    points_rtp,
    values,
    boundary,
    phi_deg,
    levels,
    norm,
    extend,
    output_path,
):
    """Render one saved unstructured toroidal slice."""
    points_rtp, values = unique_plane_samples(points_rtp, values)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    color_artist = None

    if points_rtp.shape[0] >= 3:
        x = points_rtp[:, 0] * np.cos(points_rtp[:, 1])
        z = points_rtp[:, 0] * np.sin(points_rtp[:, 1])
        try:
            from illiad.sol.tracer import extend_plot_samples_to_wall

            plot_values = values
            if PLOT_EXTEND_TO_WALL:
                x, z, plot_values = extend_plot_samples_to_wall(
                    x, z, values, VESSEL_RADIUS
                )
            triangulation = mtri.Triangulation(x, z)
            if MASK_LCFS_INTERIOR:
                mask_lcfs_triangles(triangulation, x, z, boundary)
            cmap = plt.get_cmap(COLORMAP).copy()
            cmap.set_bad(MASK_COLOR)
            color_artist = ax.tricontourf(
                triangulation,
                plot_values,
                levels=levels,
                norm=norm,
                cmap=cmap,
                extend=extend,
                antialiased=ANTIALIASED,
            )
        except (RuntimeError, ValueError) as exc:
            print(
                f"Warning: falling back to scatter at phi={phi_deg:g} deg: "
                f"{exc}"
            )

    if color_artist is None and points_rtp.size:
        x = points_rtp[:, 0] * np.cos(points_rtp[:, 1])
        z = points_rtp[:, 0] * np.sin(points_rtp[:, 1])
        color_artist = ax.scatter(
            x,
            z,
            c=values,
            s=SCATTER_FALLBACK_SIZE,
            linewidths=0.0,
            norm=norm,
            cmap=COLORMAP,
        )
    elif color_artist is None:
        x = np.empty(0)
        z = np.empty(0)
        ax.text(
            0.5,
            0.5,
            "No connection-length samples",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    if SHOW_RAW_POINTS and points_rtp.size:
        ax.scatter(
            x,
            z,
            color=RAW_POINT_COLOR,
            s=RAW_POINT_SIZE,
            alpha=RAW_POINT_ALPHA,
            linewidths=0.0,
            zorder=RAW_POINT_ZORDER,
        )

    if SHOW_LCFS:
        closed_boundary = np.vstack((boundary, boundary[0]))
        ax.plot(
            closed_boundary[:, 0],
            closed_boundary[:, 1],
            color=LCFS_COLOR,
            linewidth=LCFS_LINEWIDTH,
            label=LCFS_LABEL,
        )
    if SHOW_VESSEL:
        vessel_angle = np.linspace(0.0, 2.0 * np.pi, 720)
        ax.plot(
            VESSEL_RADIUS * np.cos(vessel_angle),
            VESSEL_RADIUS * np.sin(vessel_angle),
            color=VESSEL_COLOR,
            linewidth=VESSEL_LINEWIDTH,
            label=VESSEL_LABEL,
        )

    phi_phys_deg = (phi_deg + PHYSICAL_PHI_OFFSET_DEG) % 360.0
    if TITLE is None:
        title = (
            "Connection length samples\n"
            f"$\\phi_{{phy}}={phi_phys_deg:03.0f}^\\circ$ CW from North split, "
            f"$\\phi_c={phi_deg:03.0f}^\\circ$"
        )
    else:
        title = TITLE.format(phi_deg=phi_deg, phi_phys_deg=phi_phys_deg)
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
    if SHOW_GRID:
        ax.grid(
            True,
            color=GRID_COLOR,
            linewidth=GRID_LINEWIDTH,
            alpha=GRID_ALPHA,
        )
    else:
        ax.grid(False)
    if SHOW_LEGEND and (SHOW_LCFS or SHOW_VESSEL):
        ax.legend(loc=LEGEND_LOCATION)

    if color_artist is not None:
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches=SAVE_BBOX)
    print(f"Saved plot with {points_rtp.shape[0]} samples: {output_path}")
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)
        gc.collect()


def main():
    args = parse_args()
    analysis_dir = args.analysis_dir
    data_subdir = args.data_subdir
    output_subdir = (
        data_subdir if OUTPUT_SUBDIR == DATA_SUBDIR else OUTPUT_SUBDIR
    )
    data_dir = PROJECT_ROOT / "output" / analysis_dir / "data" / data_subdir
    output_dir = PROJECT_ROOT / "output" / analysis_dir / "plots" / output_subdir
    source = open_plane_crossing_source(data_dir)
    saved_phi_deg = np.asarray(source.plane_phi_deg)
    plane_indices = resolve_plane_indices(saved_phi_deg, args.phi)
    levels, norm, extend, value_min, value_max = make_color_scale(source)

    settings = load_poincare_settings(analysis_dir)
    lcfs_index = settings.get("LCFS_INDEX") if LCFS_INDEX is None else LCFS_INDEX
    if lcfs_index is None:
        raise ValueError("No LCFS index was found; set LCFS_INDEX in this script.")

    print(f"Reading raw data: {data_dir}")
    print(f"Input format: {source.input_format}")
    print(f"Raw samples: {source.sample_count}")
    print(f"Planes selected: {plane_indices.size} of {saved_phi_deg.size}")
    print(f"Color range: {value_min:g} to {value_max:g} m ({COLOR_SCALE})")

    for plane_index in plane_indices:
        phi_deg = float(saved_phi_deg[plane_index])
        points, values = stream_plane_samples(source, int(plane_index))
        boundary, _ = load_lcfs_boundary(
            analysis_dir,
            phi_deg,
            lcfs_index,
        )
        output_name = OUTPUT_FILENAME.format(phi_deg=phi_deg)
        plot_plane(
            points,
            values,
            boundary,
            phi_deg,
            levels,
            norm,
            extend,
            output_dir / output_name,
        )

    print(f"Saved {plane_indices.size} replot(s): {output_dir}")


if __name__ == "__main__":
    main()
