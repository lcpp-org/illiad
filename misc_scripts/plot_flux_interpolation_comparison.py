"""Create one minimal 2-D versus periodic-3-D flux interpolation diagnostic."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare saved 2d and 3d normalized-flux fields."
    )
    parser.add_argument("--field-2d", required=True, help="Saved 2d nField .npy file.")
    parser.add_argument("--field-3d", required=True, help="Saved 3d nField .npy file.")
    parser.add_argument(
        "--output", default="flux_interpolation_comparison.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--phi-index", type=int, default=-1,
        help="Toroidal plane index; default is the 360/0 seam plane.",
    )
    parser.add_argument("--theta-index", type=int, default=None)
    parser.add_argument("--rho-index", type=int, default=None)
    parser.add_argument("--rho-max", type=float, default=0.19)
    return parser.parse_args()


def load_fields(field_2d_path, field_3d_path):
    field_2d = np.load(field_2d_path)
    field_3d = np.load(field_3d_path)
    if field_2d.ndim != 3 or field_3d.ndim != 3:
        raise ValueError("Both fields must have shape (Nphi, Ntheta, Nrho)")
    if field_2d.shape != field_3d.shape:
        raise ValueError(
            f"Field shapes do not match: {field_2d.shape} and {field_3d.shape}"
        )
    return field_2d, field_3d


def main():
    args = parse_args()
    field_2d, field_3d = load_fields(args.field_2d, args.field_3d)
    nphi, ntheta, nrho = field_2d.shape
    phi_index = args.phi_index % nphi
    theta_index = ntheta // 4 if args.theta_index is None else args.theta_index % ntheta
    rho_index = nrho // 2 if args.rho_index is None else args.rho_index % nrho

    plane_2d = field_2d[phi_index]
    plane_3d = field_3d[phi_index]
    difference = plane_3d - plane_2d
    field_min = min(np.nanmin(plane_2d), np.nanmin(plane_3d))
    field_max = max(np.nanmax(plane_2d), np.nanmax(plane_3d))
    difference_limit = np.nanmax(np.abs(difference))
    if not np.isfinite(difference_limit) or difference_limit == 0:
        difference_limit = 1.0

    extent = [0, args.rho_max, 0, 360]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    image_2d = axes[0, 0].imshow(
        plane_2d, origin="lower", aspect="auto", extent=extent,
        vmin=field_min, vmax=field_max, cmap="Blues",
    )
    axes[0, 0].set_title("2d")
    image_3d = axes[0, 1].imshow(
        plane_3d, origin="lower", aspect="auto", extent=extent,
        vmin=field_min, vmax=field_max, cmap="Blues",
    )
    axes[0, 1].set_title("3d")
    image_difference = axes[1, 0].imshow(
        difference, origin="lower", aspect="auto", extent=extent,
        vmin=-difference_limit, vmax=difference_limit, cmap="coolwarm",
    )
    axes[1, 0].set_title("3d - 2d")

    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\theta$ (degrees)")
    for ax in axes.flat[:3]:
        ax.set_xlabel(r"$\rho$ (m)")
    fig.colorbar(image_2d, ax=axes[0, 0], label=r"$\hat{\psi}$")
    fig.colorbar(image_3d, ax=axes[0, 1], label=r"$\hat{\psi}$")
    fig.colorbar(image_difference, ax=axes[1, 0], label=r"$\Delta\hat{\psi}$")

    phi_degrees = np.mod(
        np.linspace(360.0 / nphi, 360.0, nphi), 360.0
    )
    phi_order = np.argsort(phi_degrees)
    joined_phi = np.append(phi_degrees[phi_order], 360.0)
    line_2d = field_2d[:, theta_index, rho_index][phi_order]
    line_3d = field_3d[:, theta_index, rho_index][phi_order]
    axes[1, 1].plot(joined_phi, np.append(line_2d, line_2d[0]), label="2d")
    axes[1, 1].plot(joined_phi, np.append(line_3d, line_3d[0]), label="3d")
    axes[1, 1].set(
        title=(
            "Toroidal lineout "
            f"(theta index {theta_index}, rho index {rho_index})"
        ),
        xlabel=r"$\phi$ (degrees)",
        ylabel=r"$\hat{\psi}$",
        xlim=(0, 360),
    )
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle(f"Flux interpolation comparison at phi index {phi_index}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
