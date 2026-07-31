"""Replot connection lengths from an already-completed analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DATA AND OUTPUT SETTINGS
ANALYSIS_DIR = "IOTA3_1000sp_atol1e-9"
ANALYSIS_SUBDIR = "ConnectionLengths_double_50spins_5mm"
PHI_DEG = 18

INPUT_FILENAME = f"connection_lengths_phi_{PHI_DEG:03.0f}_double_line.npz"
OUTPUT_FILENAME = f"connection_lengths_phi_{PHI_DEG:03.0f}_replot.png"

# PLOT SETTINGS
FIGSIZE = (9, 7)
DPI = 300

COLOR_SCALE = "log"       # "log" or "linear"
COLORMAP = "afmhot"      # Any Matplotlib colormap
N_LEVELS = 100
VMIN = 2e-1              # None uses the minimum finite connection length
VMAX = 2e3             # None uses the maximum finite connection length
CONTOUR_EXTEND = "both"  # "neither", "both", "min", or "max"
MASK_COLOR = "white"

TITLE = None              # None creates a title from the saved phi value
TITLE_SIZE = 16
TITLE_PAD = 30

SHOW_LCFS = False
LCFS_COLOR = "black"
LCFS_LINEWIDTH = 1.2
LCFS_LABEL = "LCFS"

SHOW_POINCARE = True
POINCARE_SUBDIR = "Poincare"
POINCARE_FILENAME = f"Poincare_{PHI_DEG:03.0f}.npy"
POINCARE_COLOR = "black"
POINCARE_MARKER_SIZE = 0.5
POINCARE_LABEL = "Poincare surfaces"
POINCARE_ZORDER = 5

SHOW_LEGEND = False
LEGEND_LOCATION = "upper right"

SHOW_GRID = False
GRID_COLOR = "0.65"
GRID_LINEWIDTH = 0.8
GRID_ALPHA = 0.8

RHO_MAX = None            # None uses the largest saved rho
R_LABEL_POSITION_DEG = 22.5
R_LABEL = r"$\rho$ [m]"
R_LABEL_PAD = 28

THETA_ZERO_LOCATION = "E"
THETA_DIRECTION = 1       # 1 = counterclockwise, -1 = clockwise

COLORBAR_LABEL = "Connection length [m]"
COLORBAR_PAD = 0.10
COLORBAR_TICKS = None      # Example: [1e-2, 1e0, 1e2, 1e4]

SAVE_BBOX = "tight"
SHOW_PLOT = False


def make_color_scale(connection_length):
    finite = connection_length[
        np.isfinite(connection_length) & (connection_length > 0.0)
    ]
    if finite.size == 0:
        raise ValueError("No positive finite connection lengths were found.")

    value_min = finite.min() if VMIN is None else VMIN
    value_max = finite.max() if VMAX is None else VMAX
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

    return levels, norm, value_min, value_max


def load_poincare_points(poincare_path):
    if not poincare_path.is_file():
        raise FileNotFoundError(f"Poincare data not found: {poincare_path}")

    poincare_data = np.load(poincare_path, mmap_mode="r")
    if poincare_data.ndim != 3 or poincare_data.shape[1] != 2:
        raise ValueError(
            "Poincare data must have shape (surface, coordinate, point); "
            f"found {poincare_data.shape}."
        )

    poincare_theta = poincare_data[:, 0, :].ravel()
    poincare_rho = poincare_data[:, 1, :].ravel()
    finite = np.isfinite(poincare_theta) & np.isfinite(poincare_rho)
    return poincare_theta[finite], poincare_rho[finite]


def main():
    data_path = ( PROJECT_ROOT / "output" / ANALYSIS_DIR / "data" / ANALYSIS_SUBDIR / INPUT_FILENAME)
    output_path = ( PROJECT_ROOT / "output" / ANALYSIS_DIR / "plots" / ANALYSIS_SUBDIR / OUTPUT_FILENAME)

    if not data_path.is_file():
        raise FileNotFoundError(f"Connection-length data not found: {data_path}")

    with np.load(data_path) as data:
        theta = data["theta"]
        rho = data["rho"]
        connection_length = data["connection_length"]
        lcfs_xz = data["lcfs_xz"]
        saved_phi_deg = float(data["phi_deg"])

    # Repeat the first angular column at 2*pi to close the contour-plot seam.
    theta_plot = np.column_stack((theta, theta[:, :1] + 2.0 * np.pi))
    rho_plot = np.column_stack((rho, rho[:, :1]))
    length_plot = np.ma.masked_invalid( np.column_stack((connection_length, connection_length[:, :1])) )

    levels, norm, value_min, value_max = make_color_scale(connection_length)
    cmap = plt.get_cmap(COLORMAP).copy()
    cmap.set_bad(MASK_COLOR)

    fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={"projection": "polar"})

    contours = ax.contourf(
        theta_plot,
        rho_plot,
        length_plot,
        levels=levels,
        norm=norm,
        cmap=cmap,
        extend=CONTOUR_EXTEND,
        antialiased=False,
    )

    if SHOW_POINCARE:
        poincare_path = (
            PROJECT_ROOT
            / "output"
            / ANALYSIS_DIR
            / "data"
            / POINCARE_SUBDIR
            / POINCARE_FILENAME
        )
        poincare_theta, poincare_rho = load_poincare_points(poincare_path)
        ax.scatter(
            poincare_theta,
            poincare_rho,
            color=POINCARE_COLOR,
            s=POINCARE_MARKER_SIZE,
            linewidths=0.0,
            label=POINCARE_LABEL,
            zorder=POINCARE_ZORDER,
        )

    if SHOW_LCFS:
        closed_lcfs = np.vstack((lcfs_xz, lcfs_xz[0]))
        lcfs_theta = np.arctan2(closed_lcfs[:, 1], closed_lcfs[:, 0])
        lcfs_rho = np.linalg.norm(closed_lcfs, axis=1)
        ax.scatter(lcfs_theta, lcfs_rho,
            color=LCFS_COLOR,
            #linewidth=LCFS_LINEWIDTH,
            s=0.25,
            label=LCFS_LABEL,
        )

    plot_title = (
        f"Field-line connection length at $\\phi={saved_phi_deg:g}^\\circ$"
        if TITLE is None
        else TITLE
    )
    ax.set_title(plot_title, fontsize=TITLE_SIZE, pad=TITLE_PAD)
    ax.set_ylim(0.0, np.nanmax(rho) if RHO_MAX is None else RHO_MAX)
    ax.set_rlabel_position(R_LABEL_POSITION_DEG)
    ax.set_ylabel(R_LABEL, labelpad=R_LABEL_PAD)
    ax.set_theta_zero_location(THETA_ZERO_LOCATION)
    ax.set_theta_direction(THETA_DIRECTION)
    if SHOW_GRID:
        ax.grid(SHOW_GRID, color=GRID_COLOR, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
    else:
        ax.grid(SHOW_GRID)

    if SHOW_LEGEND and (SHOW_LCFS or SHOW_POINCARE):
        ax.legend(loc=LEGEND_LOCATION)

    colorbar_ticks = COLORBAR_TICKS
    if colorbar_ticks is None and COLOR_SCALE == "log":
        first_decade = int(np.ceil(np.log10(value_min)))
        last_decade = int(np.floor(np.log10(value_max)))
        colorbar_ticks = 10.0 ** np.arange(first_decade, last_decade + 1)

    colorbar = fig.colorbar(
        contours,
        ax=ax,
        pad=COLORBAR_PAD,
        ticks=colorbar_ticks,
    )
    colorbar.set_label(COLORBAR_LABEL)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches=SAVE_BBOX)
    print(f"Saved plot: {output_path}")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
