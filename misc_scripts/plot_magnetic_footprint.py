"""Replot saved magnetic-footprint data without tracing or loading a field.

Run directly for DATA_FILE below, or pass a magnetic_footprint.npz path.
Angles are reconstructed from computational launch coordinates using the
Boris wall convention, including for data saved before the 180-degree fix.
Only a new plot is written; the saved NPZ is unchanged.
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from illiad.io import IOHandler
from illiad.plotting import global_plotPorts


# EDITABLE DATA AND PLOT SETTINGS
DATA_FILE = PROJECT_ROOT / "output/magnetic_footprint/IOTA4_footprint_1mm/data/magnetic_footprint.npz"
OUTPUT_FILE = None  # None writes plots/magnetic_footprint_replot.png in the run.
COLOR_SCALE = "log"  # "log" or "linear"
COLORMAP = "viridis"
N_LEVELS = 50
VMIN = 0.1  # None uses the smallest positive finite connection length.
VMAX = 500  # None uses the largest positive finite connection length.
DPI = 250
TITLE = None  # None uses the saved run's directory name.
SHOW_PORTS = True
PHI_WALL_OFFSET_DEG = 18.0  # phi_wall = (-phi_comp + 180 + offset) % 360.


class PortLoader:
    """Use the standard Boris port overlay from any working directory."""

    def loadPorts_fromCSV(self, name):
        return IOHandler.loadPorts_fromCSV(self, PROJECT_ROOT / name)


def load_footprint(data_file):
    """Return sorted wall angles and matching lengths, closing both seams."""
    with np.load(data_file, allow_pickle=False) as data:
        points = data["initial_conditions_rtp"]
        lengths = data["connection_length_m"]
    if lengths.ndim != 2 or min(lengths.shape) < 2:
        raise ValueError("Connection lengths must be a 2D grid with at least 2 points per axis.")
    if points.shape != (*lengths.shape, 3) or not np.all(np.isfinite(points)):
        raise ValueError("Expected finite initial_conditions_rtp with shape (*grid_shape, 3).")

    phi = (-np.rad2deg(points[..., 2]) + 180.0 + PHI_WALL_OFFSET_DEG) % 360.0
    phi[np.isclose(phi, 360.0, rtol=0, atol=1e-10)] = 0.0
    theta_rad = points[..., 1]
    theta = np.rad2deg(np.where(theta_rad > np.pi, theta_rad - 2 * np.pi, theta_rad))
    # Place the periodic inner-midplane sample at the lower edge; duplicate it
    # at +180 below. This is the same physical location as Boris's +180 bin.
    theta[np.isclose(theta, 180.0, rtol=0, atol=1e-10)] = -180.0
    if not (np.allclose(phi, phi[:1, :], rtol=0, atol=1e-10)
            and np.allclose(theta, theta[:, :1], rtol=0, atol=1e-10)):
        raise ValueError("Expected a separable (theta, phi) launch grid.")

    columns = np.argsort(phi[0])
    rows = np.argsort(theta[:, 0])
    phi = phi[0, columns]
    theta = theta[rows, 0]
    lengths = lengths[np.ix_(rows, columns)]
    for name, angles in (("phi", phi), ("theta", theta)):
        spacing = np.diff(np.r_[angles, angles[0] + 360.0])
        if not np.allclose(spacing, 360.0 / angles.size, rtol=0, atol=1e-8):
            raise ValueError(f"Expected a full, uniformly spaced periodic {name} grid.")

    # Extend on both sides so grids shifted off zero still cover the full map.
    phi = np.r_[phi[-1] - 360.0, phi, phi[0] + 360.0]
    theta = np.r_[theta[-1] - 360.0, theta, theta[0] + 360.0]
    lengths = np.pad(lengths, 1, mode="wrap")
    return phi, theta, lengths


def replot(data_file, output_file=None):
    data_file = Path(data_file).expanduser().resolve()

    if output_file is None:
        output_file = data_file.parent.parent / "plots/magnetic_footprint_replot.png"
    output_file = Path(output_file).expanduser().resolve()
    if output_file.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("Output must be a .png, .pdf, or .svg file.")
    
    phi, theta, lengths = load_footprint(data_file)
    positive = lengths[np.isfinite(lengths) & (lengths > 0)]
    if positive.size == 0:
        raise ValueError("No positive finite connection lengths are available.")
    vmin = positive.min() if VMIN is None else VMIN
    vmax = positive.max() if VMAX is None else VMAX
    if VMIN is None and VMAX is None and vmin == vmax:
        vmin, vmax = vmin * 0.99, vmax * 1.01
    if not (np.isfinite(vmin) and np.isfinite(vmax) and 0 < vmin < vmax):
        raise ValueError("Color limits must satisfy 0 < VMIN < VMAX and be finite.")
    if COLOR_SCALE == "log":
        levels = np.geomspace(vmin, vmax, N_LEVELS)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif COLOR_SCALE == "linear":
        levels = np.linspace(vmin, vmax, N_LEVELS)
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        raise ValueError('COLOR_SCALE must be "log" or "linear".')
    below, above = positive.min() < vmin, positive.max() > vmax
    extend = "both" if below and above else "min" if below else "max" if above else "neither"

    fig, ax = plt.subplots(figsize=(16, 6))
    try:
        contours = ax.contourf(
            phi, theta, np.ma.masked_where(~np.isfinite(lengths) | (lengths <= 0), lengths),
            levels=levels, norm=norm, cmap=COLORMAP, extend=extend,
        )
        if SHOW_PORTS:
            global_plotPorts(ax, PortLoader())
        ax.set(xlim=(0, 360), ylim=(-180, 180), aspect=0.25,
               xlabel=r"$\phi$ ($^\circ$ CCW from South-side split)",
               ylabel="Poloidal location",
               title=TITLE if TITLE is not None else f"Magnetic footprint — {data_file.parent.parent.name}")
        ax.set_xticks(np.arange(0, 361, 36))
        ax.set_yticks(np.linspace(-180, 180, 5),
                      ["Inner Midplane", "Bottom", "Outer Midplane", "Top", "Inner Midplane"])
        ax.grid(linewidth=0.5, color="0.6")
        fig.colorbar(contours, ax=ax, pad=0.02, shrink=0.6).set_label("Connection length [m]")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=DPI)
    finally:
        plt.close(fig)
    print(f"Saved plot: {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file", nargs="?", default=DATA_FILE,
                        help="Saved magnetic_footprint.npz (default: editable DATA_FILE).")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output PNG, PDF, or SVG path.")
    args = parser.parse_args()
    replot(args.data_file, args.output)
