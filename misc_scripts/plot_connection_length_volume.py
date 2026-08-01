"""Replot connection-length volume slices from an already-completed analysis.

This helper reads the raw, plane-sorted samples saved by either connection-
length volume tracer. It accepts both the original expanded value array and
the Torch tracer's compact fieldline-ID representation. It does not construct
a magnetic field or trace any field lines.
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

from misc_scripts.connection_lengths_outside_lcfs import (
    load_lcfs_boundary,
    load_poincare_settings,
)


# DATA AND OUTPUT SETTINGS
ANALYSIS_DIR = "IOTA3_1000sp_atol1e-9"
DATA_SUBDIR = "ConLenVolume_REDO_250spins_rk2mm"
OUTPUT_SUBDIR = DATA_SUBDIR

# None replots every saved plane. A single number or an iterable of numbers
# selects computational toroidal angles in degrees; for example, 18 or
# [18, 90, 180].
PHI_DEG = None
LCFS_INDEX = None  # None reads LCFS_INDEX from the Poincare log

OUTPUT_FILENAME = "connection_length_{phi_deg:03.0f}_replot.png"

# PLOT SETTINGS
FIGSIZE = (7, 6)
DPI = 300

COLOR_SCALE = "log"       # "log" or "linear"
COLORMAP = "afmhot"      # Any Matplotlib colormap
N_LEVELS = 50
VMIN = None               # None uses the minimum positive saved value
VMAX = None               # None uses the maximum positive saved value
CONTOUR_EXTEND = "both"   # "auto", "neither", "both", "min", or "max"
MASK_COLOR = "white"
ANTIALIASED = False
PLOT_MAX_SAMPLES = 150_000  # None disables deterministic plot-only thinning
PLOT_SAMPLE_SEED = 0

# Triangles whose centroid or any edge midpoint lies inside the LCFS are
# removed from the filled contour. The raw samples themselves are unchanged.
MASK_LCFS_INTERIOR = True

SHOW_RAW_POINTS = False
RAW_POINT_COLOR = "black"
RAW_POINT_SIZE = 0.1
RAW_POINT_ALPHA = 0.4
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
GRID_ALPHA = 1.0

SHOW_LEGEND = True
LEGEND_LOCATION = "upper right"

COLORBAR_LABEL = "Connection length [m]"
COLORBAR_PAD = 0.03
COLORBAR_TICKS = None

SCATTER_FALLBACK_SIZE = 4.0
SAVE_BBOX = "tight"
SHOW_PLOT = False

# Number of values examined at once while finding a global color range. The
# raw array remains memory-mapped throughout the replot.
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
    return parser.parse_args()


def load_raw_samples(data_dir):
    """Memory-map the raw sample arrays and validate their shared indexing."""
    required_paths = {
        "points": data_dir / "raw_points_rtp.npy",
        "offsets": data_dir / "plane_offsets.npy",
        "phi": data_dir / "plane_phi_deg.npy",
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing connection-length volume data:\n" + "\n".join(missing)
        )

    points = np.load(required_paths["points"], mmap_mode="r")
    offsets = np.load(required_paths["offsets"], mmap_mode="r")
    phi_deg = np.load(required_paths["phi"], mmap_mode="r")

    expanded_values_path = data_dir / "raw_connection_length_m.npy"
    fieldline_id_path = data_dir / "raw_fieldline_id.npy"
    fieldline_values_path = data_dir / "fieldline_connection_length_m.npy"
    if fieldline_id_path.is_file() and fieldline_values_path.is_file():
        value_source = {
            "expanded": None,
            "fieldline_id": np.load(fieldline_id_path, mmap_mode="r"),
            "fieldline": np.load(fieldline_values_path, mmap_mode="r"),
        }
    elif expanded_values_path.is_file():
        value_source = {
            "expanded": np.load(expanded_values_path, mmap_mode="r"),
            "fieldline_id": None,
            "fieldline": None,
        }
    else:
        raise FileNotFoundError(
            "Missing connection-length values. Expected either "
            "raw_connection_length_m.npy or both raw_fieldline_id.npy and "
            "fieldline_connection_length_m.npy."
        )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "raw_points_rtp.npy must have shape (sample, 3); "
            f"found {points.shape}."
        )
    if value_source["expanded"] is not None:
        values = value_source["expanded"]
        if values.ndim != 1 or values.shape[0] != points.shape[0]:
            raise ValueError(
                "raw_connection_length_m.npy must contain one value per point."
            )
    else:
        fieldline_id = value_source["fieldline_id"]
        fieldline_values = value_source["fieldline"]
        if fieldline_id.ndim != 1 or fieldline_id.shape[0] != points.shape[0]:
            raise ValueError(
                "raw_fieldline_id.npy must contain one ID per raw point."
            )
        if fieldline_values.ndim != 1:
            raise ValueError(
                "fieldline_connection_length_m.npy must be one-dimensional."
            )
    if phi_deg.ndim != 1:
        raise ValueError("plane_phi_deg.npy must be one-dimensional.")
    if offsets.ndim != 1 or offsets.size != phi_deg.size + 1:
        raise ValueError(
            "plane_offsets.npy must have one more entry than plane_phi_deg.npy."
        )
    if offsets[0] != 0 or offsets[-1] != points.shape[0]:
        raise ValueError(
            "plane_offsets.npy does not span the complete raw sample array."
        )
    if np.any(np.diff(offsets) < 0):
        raise ValueError("plane_offsets.npy must be monotonically increasing.")

    return points, value_source, offsets, phi_deg


def data_range(value_source):
    """Find the positive finite range without copying the full mapped array."""
    values = value_source["expanded"]
    if values is None:
        values = value_source["fieldline"]
    data_min = np.inf
    data_max = -np.inf
    for start in range(0, values.size, COLOR_RANGE_CHUNK_SIZE):
        chunk = np.asarray(values[start : start + COLOR_RANGE_CHUNK_SIZE])
        positive = chunk[np.isfinite(chunk) & (chunk > 0.0)]
        if positive.size:
            data_min = min(data_min, float(positive.min()))
            data_max = max(data_max, float(positive.max()))

    if not np.isfinite(data_min):
        raise ValueError("No positive finite connection lengths were found.")
    return data_min, data_max


def make_color_scale(value_source):
    data_min, data_max = data_range(value_source)
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


def resolve_plane_indices(saved_phi_deg):
    """Map the editable PHI_DEG setting to saved plane indices."""
    if PHI_DEG is None:
        return np.arange(saved_phi_deg.size, dtype=np.int64)

    requested = (
        [float(PHI_DEG)]
        if np.isscalar(PHI_DEG)
        else [float(phi) for phi in PHI_DEG]
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
            triangulation = mtri.Triangulation(x, z)
            if MASK_LCFS_INTERIOR:
                mask_lcfs_triangles(triangulation, x, z, boundary)
            cmap = plt.get_cmap(COLORMAP).copy()
            cmap.set_bad(MASK_COLOR)
            color_artist = ax.tricontourf(
                triangulation,
                values,
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
    ax.grid(
        SHOW_GRID,
        color=GRID_COLOR,
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA,
    )
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
    points, value_source, offsets, saved_phi_deg = load_raw_samples(data_dir)
    plane_indices = resolve_plane_indices(saved_phi_deg)
    levels, norm, extend, value_min, value_max = make_color_scale(value_source)

    settings = load_poincare_settings(analysis_dir)
    lcfs_index = settings.get("LCFS_INDEX") if LCFS_INDEX is None else LCFS_INDEX
    if lcfs_index is None:
        raise ValueError("No LCFS index was found; set LCFS_INDEX in this script.")

    print(f"Reading raw data: {data_dir}")
    print(f"Raw samples: {points.shape[0]}")
    print(f"Planes selected: {plane_indices.size} of {saved_phi_deg.size}")
    print(f"Color range: {value_min:g} to {value_max:g} m ({COLOR_SCALE})")

    for plane_index in plane_indices:
        phi_deg = float(saved_phi_deg[plane_index])
        start = int(offsets[plane_index])
        stop = int(offsets[plane_index + 1])
        if value_source["expanded"] is None:
            values = value_source["fieldline"][
                value_source["fieldline_id"][start:stop]
            ]
        else:
            values = value_source["expanded"][start:stop]
        boundary, _ = load_lcfs_boundary(
            analysis_dir,
            phi_deg,
            lcfs_index,
        )
        output_name = OUTPUT_FILENAME.format(phi_deg=phi_deg)
        plot_plane(
            points[start:stop],
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
