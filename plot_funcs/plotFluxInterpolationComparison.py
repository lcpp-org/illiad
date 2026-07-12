"""Create one minimal 2-D versus periodic-3-D flux interpolation comparison."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("field_2d", type=Path, help="Legacy per-plane nField .npy file")
    parser.add_argument("field_3d", type=Path, help="Periodic 3-D nField .npy file")
    parser.add_argument("--phi-index", type=int, default=0, help="Toroidal plane index to plot")
    parser.add_argument(
        "--output", type=Path, default=Path("FluxInterpolationComparison.png"),
        help="Output image path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    field_2d = np.load(args.field_2d, mmap_mode="r")
    field_3d = np.load(args.field_3d, mmap_mode="r")
    if field_2d.shape != field_3d.shape or field_2d.ndim != 3:
        raise ValueError("Input fields must have matching (phi, theta, rho) shapes")

    phi_index = args.phi_index % field_2d.shape[0]
    plane_2d = np.asarray(field_2d[phi_index])
    plane_3d = np.asarray(field_3d[phi_index])
    difference = plane_3d - plane_2d
    difference_limit = max(float(np.nanmax(np.abs(difference))), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    extent = (0.0, 360.0, 0.0, 0.19)
    panels = (
        (plane_2d, "Per-plane 2-D", "Blues", 0.0, 1.0),
        (plane_3d, "Periodic 3-D", "Blues", 0.0, 1.0),
        (difference, "3-D minus 2-D", "coolwarm", -difference_limit, difference_limit),
    )
    for ax, (data, title, cmap, vmin, vmax) in zip(axes, panels):
        image = ax.imshow(
            data.T, origin="lower", aspect="auto", extent=extent,
            cmap=cmap, vmin=vmin, vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel(r"Poloidal angle, $\theta$ (deg)")
        ax.set_ylabel(r"Minor radius, $\rho$ (m)")
        fig.colorbar(image, ax=ax, shrink=0.85)

    fig.suptitle(f"Flux interpolation comparison, phi index {phi_index}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
