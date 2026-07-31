"""Trace and plot the magnetic footprint on the HIDRA vessel wall.

The connection-length map is evaluated on a regular grid of launch locations
just inside the wall. Each field line is traced in both the +B and -B
directions. The summed connection length is plotted at the launch location,
while both wall intersections and directional lengths are retained in the
saved data.
"""

import argparse
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np


# Allow this script to be run from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from illiad.io import IOHandler
from illiad.mesh import Mesh
from illiad.plotting import global_plotPorts
from illiad.poincare import Poincare
from illiad.utilities.coordtrans import XYZ_to_RTP_many
from illiad.utilities.phi_events import inVV
from illiad.utilities.run_config import load_inputs_json, merge_input_params


LOGGER_NAME = "MagneticFootprint"

DEFAULT_INPUTS = {
    "ANALYSIS_NAME": "iota3_1mm",

    # Magnetic configuration
    "CURRENT_TOR": 0.486,  # [kA]
    "CURRENT_HEL": 0.900,  # [kA]
    "CONFIG_TOR": "default_toroidal",
    "CONFIG_HEL": "default_helical",
    "ENABLE_ERRFIELD": True,

    # Launch grid and tracing
    "RMAJOR": 0.72,  # [m]
    "RMINOR": 0.19,  # [m]
    "WALL_OFFSET_M": 0.001,  # [m], distance inside the circular wall
    "NPHI": 120,
    "NTHETA": 120,
    "SPINS": 200,
    "SOLVER": "LSODA",
    "RTOL": 2.49e-12,
    "ATOL": 1e-8,
    "NTHREADS": -1,

    # The current Boris wall plots use phi_wall = -phi_comp + 18 degrees.
    "PHI_WALL_OFFSET_DEG": 18.0,

    # Plot controls
    "COLOR_SCALE": "log",
    "COLORMAP": "viridis",
    "N_LEVELS": 60,
    "VMIN": None,
    "VMAX": None,
    "DPI": 300,
}

_CLI_INPUTS = object()
DIRECTION_LABELS = np.array(["+B", "-B"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trace and plot a magnetic connection-length footprint."
    )
    parser.add_argument(
        "--inputs-json",
        default=None,
        help="Optional JSON object overriding the built-in magnetic-footprint defaults.",
    )
    return parser.parse_args()


def _require_positive_int(params, name, minimum=1):
    value = params[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def validate_inputs(params):
    analysis_name = params["ANALYSIS_NAME"]
    if (
        not isinstance(analysis_name, str)
        or not analysis_name.strip()
        or Path(analysis_name).name != analysis_name
        or analysis_name in {".", ".."}
    ):
        raise ValueError("ANALYSIS_NAME must be one non-empty directory name.")

    for name in ("NPHI", "NTHETA"):
        _require_positive_int(params, name, minimum=2)
    _require_positive_int(params, "SPINS")
    _require_positive_int(params, "N_LEVELS", minimum=3)
    _require_positive_int(params, "DPI")

    for name in ("RMAJOR", "RMINOR", "WALL_OFFSET_M", "RTOL", "ATOL"):
        value = params[name]
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number.")
    if params["WALL_OFFSET_M"] >= params["RMINOR"]:
        raise ValueError("WALL_OFFSET_M must be smaller than RMINOR.")

    nthreads = params["NTHREADS"]
    if isinstance(nthreads, bool) or not isinstance(nthreads, int):
        raise ValueError("NTHREADS must be an integer.")
    resolve_workers(nthreads)

    if params["COLOR_SCALE"] not in {"log", "linear"}:
        raise ValueError('COLOR_SCALE must be either "log" or "linear".')

    vmin = params["VMIN"]
    vmax = params["VMAX"]
    if vmin is not None and (not np.isfinite(vmin) or vmin <= 0.0):
        raise ValueError("VMIN must be a positive finite number or null.")
    if vmax is not None and (not np.isfinite(vmax) or vmax <= 0.0):
        raise ValueError("VMAX must be a positive finite number or null.")
    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError("VMIN must be smaller than VMAX.")


def resolve_workers(nthreads):
    cpu_count = os.cpu_count() or 1
    workers = nthreads if nthreads > 0 else cpu_count + nthreads
    if workers < 1:
        raise ValueError(
            f"NTHREADS={nthreads} resolves to {workers} workers on this system."
        )
    return workers


def make_initial_conditions(params):
    """Return the regular wall map and corresponding computational RTP points."""
    phi_values_deg = np.linspace(0.0, 360.0, params["NPHI"], endpoint=False)
    theta_values_deg = np.linspace(
        -180.0,
        180.0,
        params["NTHETA"],
        endpoint=False,
    )
    phi_grid_deg, theta_grid_deg = np.meshgrid(
        phi_values_deg,
        theta_values_deg,
    )

    # The wall map follows the current Boris convention: phi is CCW from the
    # South-side split, while computational RTP phi is clockwise.
    phi_comp_deg = (
        params["PHI_WALL_OFFSET_DEG"] - phi_grid_deg
    ) % 360.0
    theta_comp_deg = theta_grid_deg % 360.0
    start_rho = params["RMINOR"] - params["WALL_OFFSET_M"]

    initial_conditions_rtp = np.column_stack(
        (
            np.full(phi_grid_deg.size, start_rho),
            np.deg2rad(theta_comp_deg.ravel()),
            np.deg2rad(phi_comp_deg.ravel()),
        )
    )
    return initial_conditions_rtp, phi_grid_deg, theta_grid_deg


def build_magnetic_field(params):
    """Load the configured toroidal, helical, and optional error fields."""
    magnetic_field = Mesh(R0=params["RMAJOR"], a=params["RMINOR"])
    magnetic_field.loadCartesianField(
        coilCurrent=params["CURRENT_TOR"],
        errField=params["ENABLE_ERRFIELD"],
        att_mult=params["CONFIG_TOR"],
    )
    magnetic_field.set_nonPer_errField()
    magnetic_field.addFieldPerturbation(
        coilCurrent=params["CURRENT_HEL"],
        att_mult=params["CONFIG_HEL"],
    )
    return magnetic_field


def _parse_solver_output(solver_output, nlines, magnetic_field, max_length):
    if len(solver_output) != 2 * nlines:
        raise RuntimeError(
            f"Expected {2 * nlines} directional traces; received {len(solver_output)}."
        )

    direction_length = np.full((nlines, 2), np.nan)
    wall_xyz = np.full((nlines, 2, 3), np.nan)
    hit_wall = np.zeros((nlines, 2), dtype=bool)

    for output_index, (length, _plane_output, wall_output) in enumerate(
        solver_output
    ):
        direction_index = 0 if output_index < nlines else 1
        line_index = output_index % nlines
        direction_length[line_index, direction_index] = length

        if isinstance(wall_output, np.ndarray) and wall_output.size:
            wall_xyz[line_index, direction_index] = wall_output[0]
            hit_wall[line_index, direction_index] = True

    reached_limit = np.isclose(
        direction_length,
        max_length,
        rtol=1e-7,
        atol=1e-9,
    )
    valid_trace = hit_wall | reached_limit
    connection_length = np.sum(direction_length, axis=1)
    connection_length[~np.all(valid_trace, axis=1)] = np.nan

    wall_rtp = np.full_like(wall_xyz, np.nan)
    finite_wall = np.all(np.isfinite(wall_xyz), axis=-1)
    wall_rtp[finite_wall] = XYZ_to_RTP_many(
        wall_xyz[finite_wall],
        magnetic_field.R0,
    )
    return {
        "connection_length": connection_length,
        "direction_length": direction_length,
        "wall_xyz": wall_xyz,
        "wall_rtp": wall_rtp,
        "hit_wall": hit_wall,
        "reached_limit": reached_limit,
        "valid_trace": valid_trace,
    }


def trace_field_lines(initial_conditions_rtp, params, magnetic_field, sim_io):
    """Trace every launch point in both field directions to the wall or limit."""
    tracer = Poincare(
        sim_io,
        solvr=params["SOLVER"],
        r_tol=params["RTOL"],
        a_tol=params["ATOL"],
        workers=resolve_workers(params["NTHREADS"]),
        double_line=True,
        anlys_name="",
    )

    nlines = initial_conditions_rtp.shape[0]
    max_length = 2.0 * np.pi * magnetic_field.R0 * params["SPINS"]
    tracer.set_conditions(
        initial_conditions_rtp,
        spins=params["SPINS"],
        field=magnetic_field,
        events=[inVV],
    )
    solver_output = list(tracer.parallel_solver())
    return _parse_solver_output(
        solver_output,
        nlines,
        magnetic_field,
        max_length,
    )


def wall_rtp_to_map_degrees(wall_rtp, phi_wall_offset_deg):
    """Convert computational wall RTP coordinates to the Boris wall map."""
    wall_theta_deg = np.rad2deg(wall_rtp[..., 1])
    wall_theta_deg = (wall_theta_deg + 180.0) % 360.0 - 180.0
    wall_phi_deg = (
        -np.rad2deg(wall_rtp[..., 2]) + phi_wall_offset_deg
    ) % 360.0
    return wall_phi_deg, wall_theta_deg


def make_color_scale(connection_length, params):
    positive = connection_length[
        np.isfinite(connection_length) & (connection_length > 0.0)
    ]
    if positive.size == 0:
        raise ValueError("No positive finite connection lengths are available.")

    data_min = positive.min()
    data_max = positive.max()
    value_min = data_min if params["VMIN"] is None else params["VMIN"]
    value_max = data_max if params["VMAX"] is None else params["VMAX"]
    if np.isclose(value_min, value_max):
        delta = max(0.01 * value_min, np.finfo(float).eps)
        value_min -= delta
        value_max += delta
    if value_min >= value_max:
        raise ValueError("The resolved color limits require VMIN < VMAX.")

    if params["COLOR_SCALE"] == "log":
        levels = np.geomspace(value_min, value_max, params["N_LEVELS"])
        norm = LogNorm(vmin=value_min, vmax=value_max)
    else:
        levels = np.linspace(value_min, value_max, params["N_LEVELS"])
        norm = Normalize(vmin=value_min, vmax=value_max)

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
    return levels, norm, extend


def plot_magnetic_footprint(
    phi_grid_deg,
    theta_grid_deg,
    connection_length,
    params,
    sim_io,
    plot_name,
):
    """Plot summed connection length over the regular wall launch grid."""
    connection_grid = connection_length.reshape(phi_grid_deg.shape)

    # Close both periodic seams without tracing duplicate launch points.
    phi_values = np.append(phi_grid_deg[0], 360.0)
    theta_values = np.append(theta_grid_deg[:, 0], 180.0)
    phi_plot, theta_plot = np.meshgrid(phi_values, theta_values)
    connection_plot = np.column_stack(
        (connection_grid, connection_grid[:, :1])
    )
    connection_plot = np.row_stack(
        (connection_plot, connection_plot[:1])
    )
    connection_plot = np.ma.masked_invalid(connection_plot)

    levels, norm, extend = make_color_scale(connection_length, params)
    fig, ax = plt.subplots(figsize=(16, 6))
    contours = ax.contourf(
        phi_plot,
        theta_plot,
        connection_plot,
        levels=levels,
        norm=norm,
        cmap=params["COLORMAP"],
        extend=extend,
    )
    global_plotPorts(ax, sim_io)

    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-180.0, 180.0)
    ax.set_aspect(0.25)
    ax.set_xticks(np.arange(0.0, 361.0, 36.0))
    ax.set_yticks(np.linspace(-180.0, 180.0, 5))
    ax.set_yticklabels(
        ["Inner Midplane", "Bottom", "Outer Midplane", "Top", "Inner Midplane"]
    )
    ax.set_xlabel(
        r"$\phi$ ($^\circ$ CCW from South-side split)"
    )
    ax.set_ylabel("Poloidal location")
    ax.set_title(
        "Magnetic footprint "
        f"($I_t={params['CURRENT_TOR']:g}$ kA, "
        f"$I_h={params['CURRENT_HEL']:g}$ kA)"
    )
    ax.grid(linewidth=0.5, color="0.6")

    colorbar = fig.colorbar(contours, ax=ax, pad=0.02)
    colorbar.set_label("Connection length [m]")
    sim_io.saveFig(
        plot_name,
        dpi=params["DPI"],
    )
    plt.close(fig)


def save_outputs(
    sim_io,
    params,
    initial_conditions_rtp,
    phi_grid_deg,
    theta_grid_deg,
    trace_output,
):
    data_name = "magnetic_footprint.npz"
    plot_name = "magnetic_footprint.png"
    data_path = Path(sim_io.data_dir) / data_name
    plot_path = Path(sim_io.plot_dir) / plot_name

    wall_phi_deg, wall_theta_deg = wall_rtp_to_map_degrees(
        trace_output["wall_rtp"],
        params["PHI_WALL_OFFSET_DEG"],
    )
    grid_shape = phi_grid_deg.shape
    direction_grid_shape = (*grid_shape, 2)
    wall_grid_shape = (*grid_shape, 2, 3)

    np.savez_compressed(
        data_path,
        start_phi_deg=phi_grid_deg,
        start_theta_deg=theta_grid_deg,
        initial_conditions_rtp=initial_conditions_rtp.reshape(
            *grid_shape,
            3,
        ),
        connection_length_m=trace_output["connection_length"].reshape(
            grid_shape
        ),
        direction_connection_length_m=trace_output[
            "direction_length"
        ].reshape(direction_grid_shape),
        direction_labels=DIRECTION_LABELS,
        wall_intersection_xyz=trace_output["wall_xyz"].reshape(
            wall_grid_shape
        ),
        wall_intersection_rtp=trace_output["wall_rtp"].reshape(
            wall_grid_shape
        ),
        wall_intersection_phi_deg=wall_phi_deg.reshape(direction_grid_shape),
        wall_intersection_theta_deg=wall_theta_deg.reshape(
            direction_grid_shape
        ),
        hit_wall=trace_output["hit_wall"].reshape(direction_grid_shape),
        reached_max_length=trace_output["reached_limit"].reshape(
            direction_grid_shape
        ),
        valid_trace=trace_output["valid_trace"].reshape(direction_grid_shape),
        wall_offset_m=params["WALL_OFFSET_M"],
        spins=params["SPINS"],
    )
    plot_magnetic_footprint(
        phi_grid_deg,
        theta_grid_deg,
        trace_output["connection_length"],
        params,
        sim_io,
        plot_name,
    )
    return data_path, plot_path


def main(input_overrides=_CLI_INPUTS):
    if input_overrides is _CLI_INPUTS:
        args = parse_args()
        input_overrides = (
            load_inputs_json(args.inputs_json, "Magnetic-footprint inputs")
            if args.inputs_json
            else None
        )
    params = merge_input_params(DEFAULT_INPUTS, input_overrides)
    validate_inputs(params)

    (PROJECT_ROOT / "output" / "magnetic_footprint").mkdir(
        parents=True,
        exist_ok=True,
    )
    run_name = f"magnetic_footprint/{params['ANALYSIS_NAME']}"
    sim_io = IOHandler(run_name)
    sim_io.startLog(
        log_name=os.path.join("logs", "magnetic_footprint.log"),
        logger_name=LOGGER_NAME,
    )

    initial_conditions_rtp, phi_grid_deg, theta_grid_deg = (
        make_initial_conditions(params)
    )
    run_settings = {
        **params,
        "START_RADIUS": params["RMINOR"] - params["WALL_OFFSET_M"],
        "DOUBLE_LINE": True,
        "TRACED_FIELD_LINES": initial_conditions_rtp.shape[0],
        "DIRECTIONAL_SOLVES": 2 * initial_conditions_rtp.shape[0],
        "OUTPUT_DIR": run_name,
    }
    sim_io.inputsBoilerplate(
        "MAGNETIC-FOOTPRINT INPUTS",
        run_settings,
        [
            "CURRENT_TOR",
            "CURRENT_HEL",
            "CONFIG_TOR",
            "CONFIG_HEL",
            "ENABLE_ERRFIELD",
            "RMAJOR",
            "RMINOR",
            "WALL_OFFSET_M",
            "START_RADIUS",
            "NPHI",
            "NTHETA",
            "TRACED_FIELD_LINES",
            "DIRECTIONAL_SOLVES",
            "SPINS",
            "SOLVER",
            "RTOL",
            "ATOL",
            "NTHREADS",
            "DOUBLE_LINE",
            "PHI_WALL_OFFSET_DEG",
            "COLOR_SCALE",
            "COLORMAP",
            "N_LEVELS",
            "VMIN",
            "VMAX",
            "DPI",
            "OUTPUT_DIR",
        ],
    )

    magnetic_field = build_magnetic_field(params)
    trace_output = trace_field_lines(
        initial_conditions_rtp,
        params,
        magnetic_field,
        sim_io,
    )
    data_path, plot_path = save_outputs(
        sim_io,
        params,
        initial_conditions_rtp,
        phi_grid_deg,
        theta_grid_deg,
        trace_output,
    )

    hit_count = np.count_nonzero(trace_output["hit_wall"])
    total_directions = trace_output["hit_wall"].size
    sim_io.log.info(
        "Wall intersections: %d of %d directional traces.",
        hit_count,
        total_directions,
    )
    sim_io.log.info("Saved magnetic-footprint data: %s", data_path)
    sim_io.log.info("Saved magnetic-footprint plot: %s", plot_path)
    sim_io.log.info("## MAGNETIC-FOOTPRINT ANALYSIS FINISHED ##")
    print(f"Saved data: {data_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
