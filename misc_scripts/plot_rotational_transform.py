"""Calculate and plot rotational transform from an existing Poincare run."""

import argparse
import ast
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from illiad.utilities.coordtrans import axisShift

# Analysis settings
MIN_CROSSINGS = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate one rotational-transform value per traced surface from "
            "an existing Poincare analysis."
        )
    )
    parser.add_argument("analysis_dir", help="Existing directory under output/.")
    parser.add_argument("--phi-deg", type=float, default=None,
        help="Toroidal plane in degrees (default: the Poincare initial plane).",
    )
    parser.add_argument("--min-crossings", type=int, default=MIN_CROSSINGS,
        help=(
            "Minimum saved crossings required for a surface "
            f"(default: {MIN_CROSSINGS})."
        ),
    )
    return parser.parse_args()

def _parse_log_value(text):
    try: return ast.literal_eval(text)
    except (SyntaxError, ValueError): return text


def load_poincare_settings(analysis_dir):
    """Read initial-condition and LCFS information from the Poincare log."""
    log_path = (
        PROJECT_ROOT
        / "output"
        / analysis_dir
        / "logs"
        / "Poincare"
        / "poincare.log"
    )
    if not log_path.is_file(): raise FileNotFoundError(f"Poincare log not found: {log_path}")

    settings = {}
    input_pattern = re.compile(r"^\|\s*([A-Z][A-Z0-9_]+):\s*(.*?)\s*$")
    lcfs_pattern = re.compile(r"LCFS_index\s*=\s*(\d+)")

    for log_line in log_path.read_text().splitlines():
        pipe_index = log_line.find("|")
        message = log_line[pipe_index:] if pipe_index >= 0 else log_line
        input_match = input_pattern.match(message)
        if input_match:
            settings[input_match.group(1)] = _parse_log_value(input_match.group(2))

        lcfs_match = lcfs_pattern.search(log_line)
        if lcfs_match:
            settings["LCFS_INDEX"] = int(lcfs_match.group(1))

    required = {"IC_PHI_DEG", "START_RADIUS", "END_RADIUS", "NLINES"}
    missing = sorted(required.difference(settings))
    if missing:
        raise ValueError(f"Missing required values in {log_path}: {', '.join(missing)}")
    if settings.get("DOUBLE_LINE", False):
        raise ValueError(
            "Rotational transform cannot be recovered from DOUBLE_LINE output "
            "because its forward and reverse crossings are combined."
        )

    return settings, log_path


def load_poincare_data(analysis_dir, phi_deg):
    data_path = (
        PROJECT_ROOT
        / "output"
        / analysis_dir
        / "data"
        / "Poincare"
        / f"Poincare_{phi_deg:03.0f}.npy")

    if not data_path.is_file(): raise FileNotFoundError(f"Poincare plane data not found: {data_path}")

    data = np.load(data_path, mmap_mode="r")
    if data.ndim != 3 or data.shape[1] != 2:
        raise ValueError(
            f"Expected Poincare data shaped (surfaces, 2, crossings), got "
            f"{data.shape} in {data_path}.")
    return data, data_path


def calculate_rotational_transform(poincare_data, min_crossings):
    """Calculate iota = -Delta(theta)/(2*pi*Delta(toroidal turns)).

    Each saved row contains [theta, rho] at one fixed toroidal plane. Adjacent
    samples are therefore successive toroidal transits. The poloidal angle is
    reconstructed around the center of each surface before it is unwrapped.
    The minus sign accounts for ILLIAD's clockwise-positive toroidal angle and
    counterclockwise-positive poloidal angle conventions.
    """
    if min_crossings < 2:
        raise ValueError("--min-crossings must be at least 2.")

    surface_count = poincare_data.shape[0]
    iota = np.full(surface_count, np.nan)
    coordinate_winding = np.full(surface_count, np.nan)
    crossing_count = np.zeros(surface_count, dtype=int)
    surface_center_xz = np.full((surface_count, 2), np.nan)

    for surface_index in range(surface_count):
        theta, rho = poincare_data[surface_index]
        finite = np.isfinite(theta) & np.isfinite(rho)
        theta = np.asarray(theta[finite], dtype=np.float64)
        rho = np.asarray(rho[finite], dtype=np.float64)
        crossing_count[surface_index] = theta.size
        if theta.size < min_crossings:
            continue

        x = rho * np.cos(theta)
        z = rho * np.sin(theta)
        center_x = 0.5 * (x.min() + x.max())
        center_z = 0.5 * (z.min() + z.max())
        surface_center_xz[surface_index] = (center_x, center_z)

        center_theta = np.arctan2(center_z, center_x)
        center_radius = np.hypot(center_x, center_z)
        centered_theta = axisShift(theta, rho, center_theta, center_radius)[0]
        unwrapped_theta = np.unwrap(centered_theta)
        toroidal_transits = theta.size - 1
        winding = (
            (unwrapped_theta[-1] - unwrapped_theta[0])
            / (2.0 * np.pi * toroidal_transits)
        )
        coordinate_winding[surface_index] = winding
        iota[surface_index] = -winding

    return iota, coordinate_winding, crossing_count, surface_center_xz


def plot_rotational_transform(initial_radii, iota, lcfs_index, phi_deg, analysis_dir, output_path):
    """Plot the rotational-transform radial profile."""
    valid = np.isfinite(iota)
    surface_index = np.arange(iota.size)
    if lcfs_index is None:
        profile_mask = valid
    else:
        profile_mask = valid & (surface_index >= lcfs_index)
    if not np.any(profile_mask):
        raise ValueError("No closed surfaces have enough crossings to calculate iota.")

    order = np.argsort(initial_radii[profile_mask])
    radius_plot = initial_radii[profile_mask][order]
    iota_plot = iota[profile_mask][order]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(radius_plot, iota_plot, "o-", color="tab:blue", markersize=4)

    if lcfs_index is not None and 0 <= lcfs_index < initial_radii.size:
        ax.axvline(initial_radii[lcfs_index], color="black", linestyle="--", linewidth=1.0, label=f"LCFS (surface {lcfs_index})")
        ax.legend()

    ax.set_xlabel(r"Initial minor radius $\rho_0$ [m]")
    ax.set_ylabel(r"Rotational transform $\iota$")
    ax.set_title(
        f"Rotational-transform profile: {analysis_dir}\n"
        rf"$\phi={phi_deg:g}^\circ$ "
        f"({np.count_nonzero(profile_mask)} closed surfaces)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    settings, log_path = load_poincare_settings(args.analysis_dir)
    phi_deg = settings["IC_PHI_DEG"] if args.phi_deg is None else args.phi_deg

    poincare_data, data_path = load_poincare_data(args.analysis_dir, phi_deg)
    expected_surfaces = int(settings["NLINES"])
    if poincare_data.shape[0] != expected_surfaces:
        raise ValueError(
            f"Poincare data contains {poincare_data.shape[0]} surfaces, but the "
            f"log reports NLINES={expected_surfaces}.")

    initial_radii = np.linspace(
        float(settings["START_RADIUS"]),
        float(settings["END_RADIUS"]),
        expected_surfaces,
    )
    iota, coordinate_winding, crossing_count, surface_center_xz = (
        calculate_rotational_transform(poincare_data, args.min_crossings)
    )

    data_dir = PROJECT_ROOT / "output" / args.analysis_dir / "data" / "Poincare"
    plot_dir = PROJECT_ROOT / "output" / args.analysis_dir / "plots" / "Poincare"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rotational_transform_phi_{phi_deg:03.0f}"
    output_data_path = data_dir / f"{stem}.npz"
    output_plot_path = plot_dir / f"{stem}.png"

    np.savez(output_data_path,
        surface_index=np.arange(expected_surfaces),
        initial_radius_m=initial_radii,
        iota=iota,
        coordinate_winding=coordinate_winding,
        crossing_count=crossing_count,
        is_closed_surface=(
            np.arange(expected_surfaces) >= settings["LCFS_INDEX"]
            if "LCFS_INDEX" in settings
            else np.full(expected_surfaces, True)
        ),
        surface_center_xz_m=surface_center_xz,
        phi_deg=phi_deg,
        min_crossings=args.min_crossings,
        source_poincare_file=str(data_path),
        source_poincare_log=str(log_path),
    )

    plot_rotational_transform(initial_radii, iota,
        settings.get("LCFS_INDEX"),
        phi_deg,
        args.analysis_dir,
        output_plot_path)

    valid = np.isfinite(iota)
    print(
        f"Calculated finite-transit iota for "
        f"{np.count_nonzero(valid)}/{iota.size} field lines."
    )
    if "LCFS_INDEX" in settings:
        closed_valid = valid & (
            np.arange(expected_surfaces) >= settings["LCFS_INDEX"]
        )
        print(f"Plotted {np.count_nonzero(closed_valid)} closed surfaces.")
    print(f"Saved data: {output_data_path}")
    print(f"Saved plot: {output_plot_path}")


if __name__ == "__main__":
    main()
