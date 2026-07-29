"""Trace and plot field-line connection lengths outside an existing LCFS."""

import argparse
import ast
import os
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.interpolate import splev, splprep


# Allow this script to be run from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from illiad.io import IOHandler
from illiad.mesh import Mesh
from illiad.poincare import Poincare
from illiad.utilities.phi_events import inVV


ANALYSIS_SUBDIR = "ConnectionLengths_double"

# Analysis settings
MAX_SPINS = 300  # Set an integer here; None uses the Poincare log value
LCFS_CLEARANCE = 0.005  # [m]
N_RHO = 120
N_THETA = 180
RHO_MIN = 0.002  # [m]
RHO_MAX = 0.189  # [m], kept just inside the r=0.19 m vessel boundary
LCFS_SPLINE_SMOOTHING = 1e-5
LCFS_BOUNDARY_POINTS = 1000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot field-line connection lengths outside an existing LCFS."
    )
    parser.add_argument("analysis_dir", help="Existing directory under output/.")
    parser.add_argument(
        "--phi-deg",
        type=float,
        default=None,
        help="Toroidal plane in degrees (default: the Poincare initial plane).",
    )
    parser.add_argument(
        "--lcfs-index",
        type=int,
        default=None,
        help="LCFS surface index (default: read from the Poincare log).",
    )
    parser.add_argument(
        "--spins",
        type=int,
        default=None,
        help="Maximum number of toroidal spins (default: read from the Poincare log).",
    )
    parser.add_argument(
        "--double-line",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Trace in both +B and -B directions and sum the two lengths "
            "(default: read from the Poincare log)."
        ),
    )
    return parser.parse_args()


def _parse_log_value(text):
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def load_poincare_settings(analysis_dir):
    """Load the magnetic configuration and LCFS index from the Poincare log."""
    log_path = ( PROJECT_ROOT / "output" / analysis_dir / "logs" / "Poincare" / "poincare.log")
    if not log_path.is_file():
        raise FileNotFoundError(f"Poincare log not found: {log_path}")

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

    required = {
        "CURRENT_TOR",
        "CURRENT_HEL",
        "CONFIG_TOR",
        "CONFIG_HEL",
        "ENABLE_ERRFIELD",
        "IC_PHI_DEG",
        "SPINS",
        "SOLVER",
        "RTOL",
        "ATOL",
        "NTHREADS",
        "DOUBLE_LINE",
    }
    missing = sorted(required.difference(settings))
    if missing:
        raise ValueError(
            f"Missing required values in {log_path}: {', '.join(missing)}"
        )

    return settings


def load_lcfs_boundary(analysis_dir, phi_deg, lcfs_index):
    """Return the LCFS as an ordered closed curve in the poloidal x-z plane."""
    poincare_path = ( PROJECT_ROOT / "output" / analysis_dir / "data" / "Poincare" / f"Poincare_{phi_deg:03.0f}.npy")
    if not poincare_path.is_file():
        raise FileNotFoundError(f"Poincare plane data not found: {poincare_path}")

    poincare_data = np.load(poincare_path, mmap_mode="r")
    if not 0 <= lcfs_index < poincare_data.shape[0]:
        raise IndexError(
            f"LCFS index {lcfs_index} is outside the available surface range "
            f"0-{poincare_data.shape[0] - 1}."
        )

    theta, rho = poincare_data[lcfs_index]
    finite = np.isfinite(theta) & np.isfinite(rho)
    theta = np.asarray(theta[finite], dtype=np.float64)
    rho = np.asarray(rho[finite], dtype=np.float64)
    # if theta.size < 3:
    #     raise ValueError(
    #         f"LCFS surface {lcfs_index} in {poincare_path} has fewer than three points."
    #     )

    boundary = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
    boundary = np.unique(boundary, axis=0)

    center = 0.5 * (boundary.min(axis=0) + boundary.max(axis=0))
    poloidal_angle = np.arctan2(
        boundary[:, 1] - center[1],
        boundary[:, 0] - center[0],
    )
    boundary = boundary[np.argsort(poloidal_angle)]
    spline, _ = splprep(
        boundary.T,
        s=LCFS_SPLINE_SMOOTHING,
        per=True,
    )
    boundary = np.column_stack(
        splev(
            np.linspace(0.0, 1.0, LCFS_BOUNDARY_POINTS, endpoint=False),
            spline,
        )
    )
    return boundary, poincare_path


def minimum_boundary_distance(points, boundary):
    """Return each point's minimum Euclidean distance to the LCFS segments."""
    minimum_squared = np.full(points.shape[0], np.inf)

    for start, stop in zip(boundary, np.roll(boundary, -1, axis=0)):
        segment = stop - start
        segment_length_squared = np.dot(segment, segment)
        if segment_length_squared == 0.0:
            continue

        projection = np.clip(
            ((points - start) @ segment) / segment_length_squared,
            0.0,
            1.0,
        )
        closest = start + projection[:, None] * segment
        distance_squared = np.sum((points - closest) ** 2, axis=1)
        minimum_squared = np.minimum(minimum_squared, distance_squared)

    return np.sqrt(minimum_squared)


def make_initial_conditions(phi_deg, boundary, clearance, n_rho, n_theta,
                            rho_min, rho_max):
    """Build a regular polar grid and retain points beyond the buffered LCFS."""
    if clearance < 0.0:
        raise ValueError("LCFS clearance must be non-negative.")
    if n_rho < 2 or n_theta < 3:
        raise ValueError("The grid requires n_rho >= 2 and n_theta >= 3.")
    if not 0.0 <= rho_min < rho_max:
        raise ValueError("Require 0 <= rho_min < rho_max.")

    rho_values = np.linspace(rho_min, rho_max, n_rho)
    theta_values = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    theta_grid, rho_grid = np.meshgrid(theta_values, rho_values)

    grid_xz = np.column_stack(
        (
            rho_grid.ravel() * np.cos(theta_grid.ravel()),
            rho_grid.ravel() * np.sin(theta_grid.ravel()),
        )
    )
    closed_boundary = np.vstack((boundary, boundary[0]))
    inside_lcfs = MplPath(closed_boundary).contains_points(grid_xz)
    near_lcfs = minimum_boundary_distance(grid_xz, boundary) <= clearance
    trace_mask = ~(inside_lcfs | near_lcfs)

    phi_rad = np.deg2rad(phi_deg)
    initial_conditions = np.column_stack(
        (
            rho_grid.ravel()[trace_mask],
            theta_grid.ravel()[trace_mask],
            np.full(np.count_nonzero(trace_mask), phi_rad),
        )
    )
    return initial_conditions, trace_mask.reshape(rho_grid.shape), theta_grid, rho_grid


def build_magnetic_field(settings):
    """Recreate the magnetic field used by the existing Poincare analysis."""
    magnetic_field = Mesh(R0=0.72, a=0.19)
    magnetic_field.loadCartesianField(
        coilCurrent=settings["CURRENT_TOR"],
        errField=settings["ENABLE_ERRFIELD"],
        att_mult=settings["CONFIG_TOR"],
    )
    magnetic_field.set_nonPer_errField()
    magnetic_field.addFieldPerturbation(
        coilCurrent=settings["CURRENT_HEL"],
        att_mult=settings["CONFIG_HEL"],
    )
    return magnetic_field


def resolve_workers(workers):
    if workers > 0:
        return workers
    return max(1, (os.cpu_count() or 1) + workers)


def trace_connection_lengths(initial_conditions, settings, magnetic_field, sim_io):
    """Trace the selected field lines without generating new Poincare planes."""
    tracer = Poincare(
        sim_io,
        solvr=settings["SOLVER"],
        r_tol=settings["RTOL"],
        a_tol=settings["ATOL"],
        workers=resolve_workers(settings["NTHREADS"]),
        double_line=settings["DOUBLE_LINE"],
        anlys_name=ANALYSIS_SUBDIR,
    )
    tracer.set_conditions(
        initial_conditions,
        spins=settings["SPINS"],
        field=magnetic_field,
        events=[inVV],
    )

    solver_output = list(tracer.parallel_solver())
    lengths = np.asarray([result[0] for result in solver_output])
    if settings["DOUBLE_LINE"]:
        n_lines = initial_conditions.shape[0]
        lengths = lengths[:n_lines] + lengths[n_lines:]
    return lengths


def plot_connection_lengths(theta_grid, rho_grid, connection_grid, boundary,
                            phi_deg, rho_max, output_path):
    """Make the polar connection-length contour plot."""
    theta_plot = np.column_stack((theta_grid, theta_grid[:, :1] + 2.0 * np.pi))
    rho_plot = np.column_stack((rho_grid, rho_grid[:, :1]))
    length_plot = np.ma.masked_invalid(
        np.column_stack((connection_grid, connection_grid[:, :1]))
    )

    positive = length_plot.compressed()
    positive = positive[positive > 0.0]
    if positive.size == 0:
        raise ValueError("No positive connection lengths are available to plot.")

    value_min = positive.min()
    value_max = positive.max()
    if np.isclose(value_min, value_max):
        delta = max(0.01 * value_min, np.finfo(float).eps)
        levels = np.linspace(value_min - delta, value_max + delta, 3)
        norm = Normalize(vmin=levels[0], vmax=levels[-1])
    else:
        levels = np.geomspace(value_min, value_max, 30)
        norm = LogNorm(vmin=value_min, vmax=value_max)

    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={"projection": "polar"})
    contours = ax.contourf(
        theta_plot,
        rho_plot,
        length_plot,
        levels=levels,
        norm=norm,
        cmap="viridis",
    )

    boundary_closed = np.vstack((boundary, boundary[0]))
    boundary_theta = np.arctan2(boundary_closed[:, 1], boundary_closed[:, 0])
    boundary_rho = np.linalg.norm(boundary_closed, axis=1)
    ax.plot(boundary_theta, boundary_rho, color="black", linewidth=1.0, label="LCFS")

    ax.set_ylim(0.0, rho_max)
    ax.set_title(f"Field-line connection length at $\\phi={phi_deg:g}^\\circ$")
    ax.set_ylabel(r"$\rho$ [m]", labelpad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.12))
    colorbar = fig.colorbar(contours, ax=ax, pad=0.10)
    colorbar.set_label("Connection length [m]")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    settings = load_poincare_settings(args.analysis_dir)

    phi_deg = settings["IC_PHI_DEG"] if args.phi_deg is None else args.phi_deg
    lcfs_index = settings.get("LCFS_INDEX") if args.lcfs_index is None else args.lcfs_index
    if lcfs_index is None:
        raise ValueError("No LCFS index was found; provide one with --lcfs-index.")
    if MAX_SPINS is not None:
        settings["SPINS"] = MAX_SPINS
    if args.spins is not None:
        settings["SPINS"] = args.spins
    if not isinstance(settings["SPINS"], int) or settings["SPINS"] <= 0:
        raise ValueError("MAX_SPINS must be a positive integer or None.")
    if args.double_line is not None:
        settings["DOUBLE_LINE"] = args.double_line

    mode_suffix = "_double_line" if settings["DOUBLE_LINE"] else ""

    boundary, poincare_path = load_lcfs_boundary(
        args.analysis_dir,
        phi_deg,
        lcfs_index,
    )
    initial_conditions, trace_mask, theta_grid, rho_grid = make_initial_conditions(
        phi_deg,
        boundary,
        LCFS_CLEARANCE,
        N_RHO,
        N_THETA,
        RHO_MIN,
        RHO_MAX,
    )
    if initial_conditions.size == 0:
        raise ValueError("The LCFS mask removed every grid point.")

    sim_io = IOHandler(args.analysis_dir)
    sim_io.startLog(
        log_name=f"connection_lengths{mode_suffix}.log",
        subdir=ANALYSIS_SUBDIR,
        logger_name=ANALYSIS_SUBDIR,
    )
    run_settings = {
        **settings,
        "PHI_DEG": phi_deg,
        "LCFS_INDEX": lcfs_index,
        "LCFS_CLEARANCE_M": LCFS_CLEARANCE,
        "N_RHO": N_RHO,
        "N_THETA": N_THETA,
        "RHO_MIN": RHO_MIN,
        "RHO_MAX": RHO_MAX,
        "TRACED_FIELD_LINES": initial_conditions.shape[0],
        "POINCARE_FILE": str(poincare_path),
    }
    sim_io.inputsBoilerplate("CONNECTION-LENGTH INPUTS", run_settings)

    magnetic_field = build_magnetic_field(settings)
    connection_lengths = trace_connection_lengths(
        initial_conditions,
        settings,
        magnetic_field,
        sim_io,
    )

    connection_grid = np.full(trace_mask.shape, np.nan)
    connection_grid[trace_mask] = connection_lengths

    sim_io.createSubDir(ANALYSIS_SUBDIR)
    stem = f"connection_lengths_phi_{phi_deg:03.0f}{mode_suffix}"
    data_path = Path(sim_io.data_dir) / ANALYSIS_SUBDIR / f"{stem}.npz"
    plot_path = Path(sim_io.plot_dir) / ANALYSIS_SUBDIR / f"{stem}.png"
    np.savez_compressed(
        data_path,
        theta=theta_grid,
        rho=rho_grid,
        connection_length=connection_grid,
        traced_mask=trace_mask,
        initial_conditions=initial_conditions,
        lcfs_xz=boundary,
        phi_deg=phi_deg,
        lcfs_index=lcfs_index,
        clearance_m=LCFS_CLEARANCE,
        spins=settings["SPINS"],
        double_line=settings["DOUBLE_LINE"],
    )
    plot_connection_lengths(
        theta_grid,
        rho_grid,
        connection_grid,
        boundary,
        phi_deg,
        RHO_MAX,
        plot_path,
    )

    sim_io.log.info("Saved connection-length data: %s", data_path)
    sim_io.log.info("Saved connection-length plot: %s", plot_path)
    sim_io.log.info("## CONNECTION-LENGTH ANALYSIS FINISHED ##")
    print(f"Saved data: {data_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
