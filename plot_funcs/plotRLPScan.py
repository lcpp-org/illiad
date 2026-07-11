#!/usr/bin/env python3
"""Plot a wall-to-center RLP scan through an interpolated flux profile."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RLP_PHI_DEG = 306.0
RLP_THETA_DEG = 0.0
WALL_RADIUS_M = 0.19

RLP_SHOT_INFO = {
    "6122": ("iota3_dflt", 16.0),
    "6123": ("iota3_dflt", 16.0),
    "6124": ("iota3_dflt", 18.0),
    "6125": ("iota3_dflt", 12.0),
    "6128": ("iota4_dflt", 12.0),
    "6129": ("iota4_dflt", 14.0),
    "6130": ("iota4_dflt", 16.0),
    "6133": ("iota4_rev", 17.0),
    "6134": ("iota4_rev", 17.0),
    "6135": ("iota4_rev", 15.0),
    "6136": ("iota4_rev", 13.0),
    "6140": ("iota3_rev", 12.0),
    "6141": ("iota3_rev", 14.0),
    "6142": ("iota3_rev", 16.0),
    "6143": ("iota3_rev", 16.0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the normalized or density-scaled RLP midplane profile and its 1D gradient."
    )
    parser.add_argument("nfield", type=Path, help="Interpolated nField .npy file with shape (phi, theta, r).")
    parser.add_argument(
        "--peak-density",
        type=float,
        default=None,
        help="Scale the normalized profile to this peak density in m^-3.",
    )
    parser.add_argument(
        "--efield",
        type=Path,
        default=None,
        help="Gradientor Efield .npy file; defaults to the matching Efield_* file beside nField_*.",
    )
    parser.add_argument("--overlay-rlp", action="store_true", help="Overlay measured RLP density data.")
    parser.add_argument(
        "--rlp-data-root",
        type=Path,
        default=Path("input_files/RLP_Results"),
        help="Directory containing RLPShot*.CSV files.",
    )
    parser.add_argument(
        "--rlp-condition",
        choices=("iota3_dflt", "iota3_rev", "iota4_dflt", "iota4_rev"),
        default="iota4_dflt",
        help="RLP coil condition to overlay; defaults to the current 486/790 forward case.",
    )
    parser.add_argument(
        "--rlp-shots",
        nargs="+",
        default=None,
        help="Specific four-digit RLP shots to overlay instead of selecting by condition.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for the two PNG files.")
    parser.add_argument("--show", action="store_true", help="Display the figures after saving them.")
    return parser.parse_args()


def periodic_distance_deg(angles, target):
    return np.abs((np.asarray(angles) - target + 180.0) % 360.0 - 180.0)


def load_scan(nfield_path):
    nfield = np.load(nfield_path, mmap_mode="r")
    if nfield.ndim != 3:
        raise ValueError(f"Expected nField shape (phi, theta, r), got {nfield.shape}.")

    nphi, ntheta, nr = nfield.shape
    phi_degrees = np.linspace(360.0 / nphi, 360.0, nphi)
    theta_degrees = np.linspace(360.0 / ntheta, 360.0, ntheta)
    radii = np.linspace(0.0, WALL_RADIUS_M, nr)

    phi_index = int(np.argmin(periodic_distance_deg(phi_degrees, RLP_PHI_DEG)))
    theta_index = int(np.argmin(periodic_distance_deg(theta_degrees, RLP_THETA_DEG)))

    distance = WALL_RADIUS_M - radii[::-1]
    profile = np.asarray(nfield[phi_index, theta_index, ::-1], dtype=float)
    return distance, profile, phi_degrees[phi_index], theta_degrees[theta_index], phi_index, theta_index, nfield.shape


def matching_efield_path(nfield_path):
    if not nfield_path.name.startswith("nField_"):
        raise ValueError("Cannot infer Efield filename: pass --efield explicitly.")
    return nfield_path.with_name(nfield_path.name.replace("nField_", "Efield_", 1))


def load_gradientor_scan(efield_path, nfield_shape, phi_index, theta_index):
    efield = np.load(efield_path, mmap_mode="r")
    expected_shape = (3, nfield_shape[2], nfield_shape[1], nfield_shape[0])
    if efield.shape != expected_shape:
        raise ValueError(f"Expected Efield shape {expected_shape}, got {efield.shape}.")
    scan_vectors = np.asarray(efield[:, ::-1, theta_index, phi_index], dtype=float)
    return np.linalg.norm(scan_vectors, axis=0)


def default_output_dir(nfield_path):
    if nfield_path.parent.parent.name == "data":
        return nfield_path.parent.parent.parent / "plots" / nfield_path.parent.name
    return nfield_path.parent


def selected_shots(condition, requested_shots):
    if requested_shots:
        shots = {str(shot) for shot in requested_shots}
        unknown = shots.difference(RLP_SHOT_INFO)
        if unknown:
            raise ValueError(f"Missing RLP position metadata for shots: {', '.join(sorted(unknown))}")
        return shots
    return {shot for shot, (shot_condition, _) in RLP_SHOT_INFO.items() if shot_condition == condition}


def load_rlp_data(data_root, condition, requested_shots):
    shots = selected_shots(condition, requested_shots)
    csv_files = sorted({*data_root.rglob("*.CSV"), *data_root.rglob("*.csv")})
    series = []

    for csv_path in csv_files:
        match = re.search(r"(\d{4})", str(csv_path))
        if not match or match.group(1) not in shots:
            continue
        shot = match.group(1)
        start_position_cm = RLP_SHOT_INFO[shot][1]
        distance = []
        density = []

        with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                try:
                    # The CSV position header says cm, but the stored values are millimeters.
                    d_m = (float(row["Position (cm)"]) / 10.0 + start_position_cm - 4.0) / 100.0
                    ne = float(row["ne (m-3)"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(d_m) and np.isfinite(ne) and ne > 0.0 and 0.0 <= d_m <= WALL_RADIUS_M:
                    distance.append(d_m)
                    density.append(ne)

        if distance:
            order = np.argsort(distance)
            series.append((shot, np.asarray(distance)[order], np.asarray(density)[order]))

    if not series:
        raise ValueError(f"No usable RLP data found in {data_root} for shots {sorted(shots)}.")
    return series


def plot_profile(distance, profile, peak_density, rlp_series, output_path):
    normalized = peak_density is None
    plotted_profile = profile if normalized else profile * peak_density

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(distance, plotted_profile, color="tab:blue", linewidth=2.0, label="Interpolated profile")

    if rlp_series:
        data_peak = max(np.max(density) for _, _, density in rlp_series)
        for index, (_, rlp_distance, density) in enumerate(rlp_series):
            plotted_density = density / data_peak if normalized else density
            label = "RLP data / data peak" if normalized and index == 0 else "RLP data" if index == 0 else None
            ax.plot(rlp_distance, plotted_density, "o", color="0.35", alpha=0.55, markersize=3.0, label=label)

    ax.set_xlim(0.0, WALL_RADIUS_M)
    ax.set_xlabel("Distance from outer wall, $d$ [m]")
    ax.set_ylabel("Normalized profile" if normalized else "$n_e$ [m$^{-3}$]")
    ax.set_title("RLP Midplane Profile ($\\phi_c=306^\\circ$, $\\theta=0^\\circ$)")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    return fig


def plot_gradient(distance, gradient, peak_density, output_path):
    normalized = peak_density is None
    plotted_gradient = gradient if normalized else gradient * peak_density

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(distance, plotted_gradient, color="tab:red", linewidth=2.0, label="$|\\nabla n|$")
    ax.set_xlim(0.0, WALL_RADIUS_M)
    ax.set_xlabel("Distance from outer wall, $d$ [m]")
    if normalized:
        ax.set_ylabel("Normalized gradient magnitude [m$^{-1}$]")
    else:
        ax.set_ylabel("Density gradient magnitude [m$^{-4}$]")
    ax.set_title("RLP Midplane Gradient Magnitude ($\\phi_c=306^\\circ$, $\\theta=0^\\circ$)")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    return fig


def main():
    args = parse_args()
    if args.peak_density is not None and args.peak_density <= 0.0:
        raise ValueError("--peak-density must be positive.")

    distance, profile, phi_used, theta_used, phi_index, theta_index, nfield_shape = load_scan(args.nfield)
    efield_path = args.efield or matching_efield_path(args.nfield)
    gradient = load_gradientor_scan(efield_path, nfield_shape, phi_index, theta_index)
    rlp_series = None
    if args.overlay_rlp:
        rlp_series = load_rlp_data(args.rlp_data_root, args.rlp_condition, args.rlp_shots)

    output_dir = args.output_dir or default_output_dir(args.nfield)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "RLP_1D_profile.png"
    gradient_path = output_dir / "RLP_1D_gradient.png"

    profile_fig = plot_profile(distance, profile, args.peak_density, rlp_series, profile_path)
    gradient_fig = plot_gradient(
        distance,
        gradient,
        args.peak_density,
        gradient_path,
    )

    print(f"RLP scan grid location: phi={phi_used:.1f} deg, theta={theta_used % 360.0:.1f} deg")
    print(f"Gradientor field: {efield_path}")
    print(f"Saved profile: {profile_path}")
    print(f"Saved gradient: {gradient_path}")
    if args.overlay_rlp:
        print("RLP measurements overlay the profile only; the CSV files contain no measured density gradient.")

    if args.show:
        plt.show()
    else:
        plt.close(profile_fig)
        plt.close(gradient_fig)


if __name__ == "__main__":
    main()
