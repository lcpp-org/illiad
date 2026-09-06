"""Replot a completed regular connection-length field without retracing.

The input field must follow the ILLIAD ``(phi, theta, rho)`` convention used
by ``illiad-sol-connection-length trace_regularize``.
"""

import argparse
import gc
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from illiad.sol import load_lcfs_boundary, load_poincare_settings


# DEFAULT DATA SETTINGS
ANALYSIS_DIR = "IOTA5_1000sp_atol1e-9"
DATA_SUBDIR = "SOLtrace_500c_RegularGrid"
OUTPUT_SUBDIR = None  # None writes under plots/<DATA_SUBDIR>/
LCFS_INDEX = None     # None reads LCFS_INDEX from the Poincare log

# None plots every plane. A scalar or iterable selects computational angles;
# for example, 18 or [18, 90, 180]. The --phi option overrides this setting.
PHI_DEG = None

FIELD_FILENAME = "connection_length_field_m.npy"
RHO_FILENAME = "rho_grid_m.npy"
THETA_FILENAME = "theta_grid_rad.npy"
PHI_FILENAME = "phi_grid_deg.npy"
OUTPUT_FILENAME = "connection_length_field_{phi_deg:03.0f}_replot.png"

# PLOT SETTINGS
FIGSIZE = (7, 6)
DPI = 250

COLOR_SCALE = "log"       # "log" or "linear"
COLORMAP = "afmhot"       # Any Matplotlib colormap
N_LEVELS = 50
VMIN = 0.05                # None uses the minimum plottable field value
VMAX = 10000.0                # None uses the maximum plottable field value
CONTOUR_EXTEND = "auto"   # "auto", "neither", "both", "min", or "max"
ANTIALIASED = False
MASK_COLOR = "white"

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

X_LIMITS = None            # None uses (-VESSEL_RADIUS, VESSEL_RADIUS)
Y_LIMITS = None
ASPECT = "equal"
X_LABEL = r"$x=\rho\cos\theta$ [m]"
Y_LABEL = r"$z=\rho\sin\theta$ [m]"

SHOW_GRID = True
GRID_COLOR = "0.75"
GRID_LINEWIDTH = 0.4
GRID_ALPHA = 0.9

SHOW_LEGEND = False
LEGEND_LOCATION = "upper right"

COLORBAR_LABEL = "Connection length [m]"
COLORBAR_PAD = 0.03
COLORBAR_TICKS = None

SAVE_BBOX = "tight"
SHOW_PLOT = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replot a completed regular connection-length field."
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
        help=f"Regular-field data subdirectory (default: {DATA_SUBDIR}).",
    )
    parser.add_argument(
        "--output-subdir",
        default=OUTPUT_SUBDIR,
        help="Plot output subdirectory (default: input data subdirectory).",
    )
    parser.add_argument(
        "--lcfs-index",
        type=int,
        default=LCFS_INDEX,
        help="LCFS surface index (default: read from the Poincare log).",
    )
    parser.add_argument(
        "--phi",
        nargs="+",
        type=float,
        default=None,
        help="Only replot these computational toroidal angles in degrees.",
    )
    return parser.parse_args()


def require_file(path, description):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def load_regular_field(data_dir):
    field = np.load(
        require_file(data_dir / FIELD_FILENAME, "connection-length field"),
        mmap_mode="r",
    )
    rho = np.load(require_file(data_dir / RHO_FILENAME, "rho grid"))
    theta = np.load(require_file(data_dir / THETA_FILENAME, "theta grid"))
    phi_deg = np.load(require_file(data_dir / PHI_FILENAME, "phi grid"))

    if field.ndim != 3:
        raise ValueError(
            f"{FIELD_FILENAME} must have shape (phi, theta, rho); "
            f"found {field.shape}."
        )
    expected_shape = (phi_deg.size, theta.size, rho.size)
    if field.shape != expected_shape:
        raise ValueError(
            f"Field shape {field.shape} does not match coordinate shape "
            f"{expected_shape}."
        )
    for name, coordinate in (
        (RHO_FILENAME, rho),
        (THETA_FILENAME, theta),
        (PHI_FILENAME, phi_deg),
    ):
        if coordinate.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        if not np.all(np.isfinite(coordinate)):
            raise ValueError(f"{name} contains non-finite values.")
        if np.any(np.diff(coordinate) <= 0.0):
            raise ValueError(f"{name} must be strictly increasing.")
    return field, rho, theta, phi_deg


def resolve_plane_indices(saved_phi_deg, command_line_phi):
    requested_phi = PHI_DEG if command_line_phi is None else command_line_phi
    if requested_phi is None:
        return np.arange(saved_phi_deg.size, dtype=np.int64)
    if np.isscalar(requested_phi):
        requested_phi = [float(requested_phi)]

    indices = []
    for requested in requested_phi:
        normalized = float(requested) % 360.0
        if np.isclose(normalized, 0.0):
            normalized = 360.0
        distance = np.abs(
            (np.asarray(saved_phi_deg) - normalized + 180.0) % 360.0
            - 180.0
        )
        index = int(np.argmin(distance))
        if not np.isclose(distance[index], 0.0, atol=1.0e-8):
            raise ValueError(
                f"No saved plane exists at phi={float(requested):g} degrees."
            )
        if index not in indices:
            indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def data_range(field, plane_indices):
    data_min = np.inf
    data_max = -np.inf
    for plane_index in plane_indices:
        plane = np.asarray(field[plane_index])
        finite = plane[np.isfinite(plane)]
        if COLOR_SCALE == "log":
            finite = finite[finite > 0.0]
        if finite.size:
            data_min = min(data_min, float(finite.min()))
            data_max = max(data_max, float(finite.max()))
    if not np.isfinite(data_min):
        raise ValueError("The selected planes contain no plottable values.")
    return data_min, data_max


def make_color_scale(field, plane_indices):
    data_min, data_max = data_range(field, plane_indices)
    value_min = data_min if VMIN is None else float(VMIN)
    value_max = data_max if VMAX is None else float(VMAX)
    if np.isclose(value_min, value_max):
        delta = max(abs(value_min) * 0.01, np.finfo(float).eps)
        value_min -= delta
        value_max += delta
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


def plot_plane(plane, rho, theta, phi_deg, boundary, levels, norm, extend,
               output_path):
    plot_theta = np.concatenate(([0.0], theta))
    plot_data = np.ma.masked_invalid(np.vstack((plane[-1], plane)))
    if COLOR_SCALE == "log":
        plot_data = np.ma.masked_less_equal(plot_data, 0.0)
    theta_grid, rho_grid = np.meshgrid(plot_theta, rho, indexing="ij")
    plot_x = rho_grid * np.cos(theta_grid)
    plot_z = rho_grid * np.sin(theta_grid)

    cmap = plt.get_cmap(COLORMAP).copy()
    cmap.set_bad(MASK_COLOR)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    contours = ax.contourf(
        plot_x,
        plot_z,
        plot_data,
        levels=levels,
        norm=norm,
        cmap=cmap,
        extend=extend,
        antialiased=ANTIALIASED,
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
    title = (
        "Regular-grid connection length\n"
        f"$\\phi_{{phy}}={phi_phys_deg:03.0f}^\\circ$ CW from North split, "
        f"$\\phi_c={phi_deg:03.0f}^\\circ$"
        if TITLE is None
        else TITLE.format(phi_deg=phi_deg, phi_phys_deg=phi_phys_deg)
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
    if SHOW_LEGEND and (SHOW_LCFS or SHOW_VESSEL):
        ax.legend(loc=LEGEND_LOCATION)

    colorbar = fig.colorbar(
        contours,
        ax=ax,
        pad=COLORBAR_PAD,
        ticks=COLORBAR_TICKS,
    )
    colorbar.set_label(COLORBAR_LABEL)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches=SAVE_BBOX)
    print(f"Saved plot: {output_path}")
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)
        gc.collect()


def main():
    args = parse_args()
    data_dir = (
        PROJECT_ROOT / "output" / args.analysis_dir / "data" / args.data_subdir
    )
    output_subdir = (
        args.data_subdir if args.output_subdir is None else args.output_subdir
    )
    output_dir = (
        PROJECT_ROOT / "output" / args.analysis_dir / "plots" / output_subdir
    )

    field, rho, theta, saved_phi_deg = load_regular_field(data_dir)
    plane_indices = resolve_plane_indices(saved_phi_deg, args.phi)
    levels, norm, extend, value_min, value_max = make_color_scale(
        field,
        plane_indices,
    )

    lcfs_index = args.lcfs_index
    if SHOW_LCFS and lcfs_index is None:
        lcfs_index = load_poincare_settings(args.analysis_dir).get("LCFS_INDEX")
        if lcfs_index is None:
            raise ValueError(
                "No LCFS index was found; pass --lcfs-index or set LCFS_INDEX."
            )

    print(f"Reading regular field: {data_dir / FIELD_FILENAME}")
    print(f"Field shape (phi, theta, rho): {field.shape}")
    print(f"Planes selected: {plane_indices.size} of {saved_phi_deg.size}")
    print(f"Color range: {value_min:g} to {value_max:g} m ({COLOR_SCALE})")

    for plane_index in plane_indices:
        phi_deg = float(saved_phi_deg[plane_index])
        boundary = None
        if SHOW_LCFS:
            boundary, _ = load_lcfs_boundary(
                args.analysis_dir,
                phi_deg,
                lcfs_index,
            )
        output_name = OUTPUT_FILENAME.format(phi_deg=phi_deg)
        plot_plane(
            np.asarray(field[plane_index]),
            rho,
            theta,
            phi_deg,
            boundary,
            levels,
            norm,
            extend,
            output_dir / output_name,
        )

    print(f"Saved {plane_indices.size} replot(s): {output_dir}")


if __name__ == "__main__":
    main()
