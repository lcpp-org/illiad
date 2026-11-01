"""Overlay thresholded Boris wall histograms over phi_plot = 0--72 degrees.

Edit RUNS below, or use --runs DIR [DIR ...] for any number of runs.
The density and bins match illiad.plotting.boris_plotWallHist: normalize over
the full wall, then display only the first toroidal period (no period folding).
Below-threshold bins are transparent. Runs are drawn in list order,
with later entries on top; legend labels are inferred from run names.
Set FILL_ALPHAS or --fill-alphas to one opacity for all runs, or one per run.
Use --style contour for interpolated filled contours with opaque outlines.
"""

import argparse
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.patches import Patch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from illiad.utilities.coordtrans import wall_plot_angles


# Editable settings; run order also defines the color and legend order.
RUNS = [
    PROJECT_ROOT / "output/IOTA3_1000sp_atol1e-9/data/0mm_LCFS50_2eV_25V_Z1_APS26_IOTA3_evap_collisions_on_2ms",
    PROJECT_ROOT / "output/IOTA4_1000sp_atol1e-9/data/0mm_LCFS50_2eV_25V_Z1_APS26_IOTA4_evap_collisions_on_2ms",
    PROJECT_ROOT / "output/IOTA5_1000sp_atol1e-9/data/0mm_LCFS33_2eV_25V_Z1_APS26_IOTA5_evap_collisions_on_3ms",
]
# Colors follow list positions and cycle if there are more runs than colors.
COLORS = ["blue", "green", "red", "darkorange", "purple", "brown"]
FILL_ALPHAS = [0.35]  # One value for all runs, or one per run; range [0, 1].
THRESHOLD = 1e-5
OUTPUT = PROJECT_ROOT / "output/wall_hist_overlay/IOTA3_IOTA4_IOTA5.png"
DPI = 300
PHI_EDGES = np.linspace(0, 360, 361)
THETA_EDGES = np.linspace(-180, 180, 181)


def load_histogram(run):
    """Return full-wall density in degree^-2 with shape (theta, phi)."""
    path = Path(run).expanduser() / "Wallpt_OUTPUT.npy"
    points = np.load(path, allow_pickle=False, mmap_mode="r")
    if points.ndim != 2 or points.shape[0] not in (3, 7):
        raise ValueError(f"{path}: expected wall points shaped (3, N) or (7, N).")
    if points.shape[1] == 0 or not np.isfinite(points[:3]).all():
        raise ValueError(f"{path}: wall coordinates must be nonempty and finite.")
    phi, theta = wall_plot_angles(points[:3].T)
    if np.any((theta < -180) | (theta > 180)):
        raise ValueError(f"{path}: poloidal angles fall outside [-180, 180].")
    density, _, _ = np.histogram2d(
        phi, theta, bins=[PHI_EDGES, THETA_EDGES], density=True
    )
    return density.T


def plot_overlay(runs=RUNS, threshold=THRESHOLD, output=OUTPUT, show=False,
                 style="filled", fill_alphas=FILL_ALPHAS):
    """Save the overlay and return its path."""
    runs = list(runs)
    if not runs:
        raise ValueError("Select at least one run.")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Threshold must be finite and positive.")
    if style not in ("filled", "contour"):
        raise ValueError("Style must be 'filled' or 'contour'.")
    alphas = np.atleast_1d(np.asarray(fill_alphas, dtype=float))
    if (alphas.ndim != 1 or alphas.size not in (1, len(runs))
            or not np.isfinite(alphas).all()
            or np.any((alphas < 0) | (alphas > 1))):
        raise ValueError("Fill alphas must be one value or one per run, all in [0, 1].")
    alphas = np.broadcast_to(alphas, (len(runs),))

    # Load all data before creating output; keep full-wall normalization.
    densities = [load_histogram(run) for run in runs]
    names = [Path(run).name for run in runs]
    labels = []
    for name in names:
        match = re.search(r"(?:^|_)(IOTA\d+)(?:_|$)", name)
        labels.append(match.group(1) if match else name)
    # Distinguish different runs of the same configuration in the legend.
    labels = [name if labels.count(label) > 1 else label
              for name, label in zip(names, labels)]
    colors = [COLORS[index % len(COLORS)] for index in range(len(runs))]
    fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    try:
        for index in range(len(runs)):
            visible = densities[index][:, :72] >= threshold
            if style == "contour":
                # Interpolate the density at bin centers without smoothing.
                # Wrap both angular seams before contouring, then crop with
                # axis limits so contours reach 0/72 and +/-180 correctly.
                phi = (PHI_EDGES[:-1] + PHI_EDGES[1:]) / 2
                theta = (THETA_EDGES[:-1] + THETA_EDGES[1:]) / 2
                phi = np.r_[phi[-1] - 360, phi, phi[0] + 360]
                theta = np.r_[theta[-1] - 360, theta, theta[0] + 360]
                padded = np.pad(densities[index], 1, mode="wrap")
                ax.contourf(phi, theta, padded,
                            levels=[threshold, max(threshold, padded.max()) + threshold],
                            colors=[colors[index]], alpha=alphas[index],
                            zorder=index + 1)
                ax.contour(phi, theta, padded,
                           levels=[threshold], colors=[colors[index]],
                           linewidths=1.2, zorder=index + 1,
                           )
            else:
                layer = np.ma.masked_where(~visible, np.ones(visible.shape))
                cmap = ListedColormap([colors[index]])
                cmap.set_bad((0, 0, 0, 0))
                ax.pcolormesh(PHI_EDGES[:73], THETA_EDGES, layer,
                              cmap=cmap, vmin=0, vmax=1, shading="flat",
                              edgecolors="none", rasterized=True, zorder=index + 1,
                              alpha=alphas[index],
                              )
            print(f"{labels[index]}: {np.count_nonzero(visible)} bins at or above "
                  f"{threshold:g} in phi_plot = 0--72 degrees"
                  )
        ax.set_xlim(0, 72)
        ax.set_ylim(-180, 180)
        ax.set_aspect(0.2)
        ax.set_xticks(np.arange(0, 73, 18))
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_yticklabels(["−180°", "Bottom (−90°)", "Outer (0°)", "Top (90°)", "180°"])
        ax.set_xlabel(r"$\phi_{\mathrm{plot}}$ (degrees CCW from south split)")
        ax.set_ylabel("Poloidal location")
        relation = r"\geq"
        #ax.set_title("Wall deposition overlay\n"
        #             rf"Full-wall density ${relation}$ {threshold:g} deg$^{{-2}}$"
        #             )
        handles = [
            Patch(facecolor=to_rgba(c, a),
                  edgecolor=c if style == "contour" else "none", label=l)
            for l, c, a in zip(labels, colors, alphas)
        ]
        legend = ax.legend(handles=handles, loc="upper left", framealpha=1)
        legend.set_zorder(len(runs) + 2)
        ax.set_axisbelow(False)
        ax.grid(linewidth=0.4, alpha=0.3, zorder=len(runs) + 1)
        output = Path(output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=DPI)
        print(f"Saved {output}")
        if show:
            plt.show()
        return output
    finally:
        plt.close(fig)


def main(default_style="filled"):
    description = (
        "Overlay filled wall-density contours at one threshold over "
        "phi_plot = 0--72 degrees, using full-wall normalization."
        if default_style == "contour" else __doc__
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--runs", nargs="+", type=Path, default=RUNS, metavar="DIR",
        help="Run directories containing Wallpt_OUTPUT.npy, drawn in this order.",
    )
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--style", choices=("filled", "contour"), default=default_style)
    parser.add_argument(
        "--fill-alphas", nargs="+", type=float, default=FILL_ALPHAS, metavar="ALPHA",
        help="Fill opacity in [0, 1]: one value for all runs, or one per run.",
    )
    default_output = (
        OUTPUT.with_stem(OUTPUT.stem + "_contours")
        if default_style == "contour" else OUTPUT
    )
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    plot_overlay(args.runs, args.threshold, args.output, args.show,
                 style=args.style, fill_alphas=args.fill_alphas)


if __name__ == "__main__":
    main()
