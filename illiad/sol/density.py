"""Build the piecewise LCFS / connection-length plasma-density field.

This post-processing script implements
``input_files/piecewise_plasma_density_model.pdf``.  It consumes the saved
linear flux profile ``q = 1 - Psi_bar``, an existing regular SOL
connection-length field, and saved Poincare surfaces.  It performs no
field-line tracing.

The selected density exponent is applied as

    psi_n,in = 1 - (1 - q)**alpha = 1 - Psi_bar**alpha.

The output is a float64 ``(phi, theta, rho)`` scalar field.  Its default
normalization is ``n_axis = 1`` so the result can continue to be scaled by
the Boris ``PLASMA_DENSITY`` input when loaded as a density field.
"""

from contextlib import nullcontext
import gc
import os
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.path import Path as MplPath
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .tracer import load_lcfs_boundary, load_poincare_settings
from . import stitching as common


# Numerical and plotting implementation settings not exposed by the CLI.
MAJOR_RADIUS_M = 0.72

# Surface mapping and numerical differentiation
VESSEL_RADIUS_M = None  # None uses the outermost rho grid node
BOUNDARY_RESAMPLE_POINTS = 720
PATH_SAMPLES = 256
NORMAL_DERIVATIVE_STEP_M = 0.002
SURFACE_SLOPE_SMOOTHING_SIGMA = 2.0
TREE_WORKERS = -1

# Optional density-decay-length bounds
LAMBDA_N_MIN_M = None
LAMBDA_N_MAX_M = None

# Output and plot settings
FIGSIZE = (7, 6)
DPI = 300
COLORMAP = "afmhot"
N_LEVELS = 100
PLOT_VMIN = None  # Normalized density; None uses n_wall/n_axis or log floor
PLOT_VMAX = None  # Normalized density; None uses 1
LOG_PLOT_VMIN = 1e-4
CONTOUR_EXTEND = "neither"
PHYSICAL_PHI_OFFSET_DEG = 198.0
MIDPLANE_TRACE_PHI_DEG = (324.0, 360.0)
MIDPLANE_TRACE_FIGSIZE = (8, 5)

OUTPUT_FIELD_FILENAME = "stitched_density_connection_length.npy"
MODEL_METADATA_FILENAME = "piecewise_density_metadata.npz"
OUTPUT_PLOT_FILENAME = "stitched_density_{phi_deg:03.0f}.png"
MIDPLANE_TRACE_FILENAME = "midplane_density_trace.png"

def validate_settings(n_axis, n_lcfs, n_wall, alpha, sol_beta):
    if not all(np.isfinite(value) for value in (n_axis, n_lcfs, n_wall)):
        raise ValueError("N_AXIS, N_LCFS, and N_WALL must be finite.")
    if not n_axis > n_lcfs > n_wall >= 0.0:
        raise ValueError("Require N_AXIS > N_LCFS > N_WALL >= 0.")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("ALPHA must be positive and finite.")
    if not 0.0 < sol_beta <= 1.0:
        raise ValueError("SOL_BETA must lie in (0, 1].")
    if BOUNDARY_RESAMPLE_POINTS < 32:
        raise ValueError("BOUNDARY_RESAMPLE_POINTS must be at least 32.")
    if PATH_SAMPLES < 3:
        raise ValueError("PATH_SAMPLES must be at least 3.")
    if NORMAL_DERIVATIVE_STEP_M <= 0.0:
        raise ValueError("NORMAL_DERIVATIVE_STEP_M must be positive.")
    if SURFACE_SLOPE_SMOOTHING_SIGMA < 0.0:
        raise ValueError("SURFACE_SLOPE_SMOOTHING_SIGMA cannot be negative.")
    if LAMBDA_N_MIN_M is not None and LAMBDA_N_MIN_M <= 0.0:
        raise ValueError("LAMBDA_N_MIN_M must be positive or None.")
    if LAMBDA_N_MAX_M is not None and LAMBDA_N_MAX_M <= 0.0:
        raise ValueError("LAMBDA_N_MAX_M must be positive or None.")
    if (
        LAMBDA_N_MIN_M is not None
        and LAMBDA_N_MAX_M is not None
        and LAMBDA_N_MIN_M > LAMBDA_N_MAX_M
    ):
        raise ValueError("LAMBDA_N_MIN_M cannot exceed LAMBDA_N_MAX_M.")
    if LOG_PLOT_VMIN <= 0.0:
        raise ValueError("LOG_PLOT_VMIN must be positive.")
    if CONTOUR_EXTEND not in {"neither", "both", "min", "max"}:
        raise ValueError("Invalid CONTOUR_EXTEND setting.")

def construct_density_plane(sol_plane, linear_profile_plane,
                            theta, rho, grid_points, boundary,
                            vessel_radius, l_parallel_0,
                            n_axis, n_lcfs, n_wall,
                            alpha, sol_beta):

    boundary = common.resample_closed_curve(boundary, BOUNDARY_RESAMPLE_POINTS)

    normals = common.outward_normals(boundary)

    closed = np.vstack((boundary, boundary[0]))
    shape = (theta.size, rho.size)
    inside = MplPath(closed).contains_points(grid_points).reshape(shape)
    if not np.any(inside) or not np.any(~inside):
        raise ValueError("LCFS mask does not divide the computational mesh.")


    delta_n_core = n_axis - n_lcfs
    delta_n_sol = n_lcfs - n_wall
    density_profile = 1.0 - (1.0 - linear_profile_plane) ** alpha

    output = np.empty((theta.size, rho.size), dtype=np.float64)

    output[inside] = n_lcfs + delta_n_core * density_profile[inside]

    chi, diagnostics = common.construct_path_attenuation(sol_plane, density_profile,
                                                         theta, rho, boundary, normals,
                                                         vessel_radius, l_parallel_0, delta_n_core, delta_n_sol, sol_beta,
                                                         path_samples=PATH_SAMPLES, derivative_step=NORMAL_DERIVATIVE_STEP_M,
                                                         smoothing_sigma=SURFACE_SLOPE_SMOOTHING_SIGMA,
                                                         lambda_min=LAMBDA_N_MIN_M, lambda_max=LAMBDA_N_MAX_M)

    output[~inside] = common.evaluate_exterior_profile(grid_points[~inside.ravel()], boundary, normals, chi,
                                                       vessel_radius, n_wall, delta_n_sol, tree_workers=TREE_WORKERS)

    if not np.all(np.isfinite(output)):
        raise ValueError("Constructed density contains non-finite values.")

    diagnostics["boundary"] = boundary
    diagnostics["normal"] = normals
    diagnostics["inside_cells"] = int(np.count_nonzero(inside))
    diagnostics["outside_cells"] = int(np.count_nonzero(~inside))
    diagnostics["density_min"] = float(np.min(output))
    diagnostics["density_max"] = float(np.max(output))
    return output, diagnostics

def build_density_field(analysis_dir, sol, linear_profile, rho, theta, phi_deg,
                        lcfs_index, vessel_radius, l_parallel_0,
                        n_axis, n_lcfs, n_wall,
                        alpha, sol_beta,
                        output_path, sim_io, show_progress):

    diagnostic_shape = (phi_deg.size, BOUNDARY_RESAMPLE_POINTS)
    boundaries = np.empty(diagnostic_shape + (2,), dtype=np.float64)
    normals = np.empty_like(boundaries)
    lambda_n_0 = np.empty(diagnostic_shape, dtype=np.float64)
    lambda_n_min = np.empty(diagnostic_shape, dtype=np.float64)
    lambda_n_max = np.empty(diagnostic_shape, dtype=np.float64)
    slopes = np.empty(diagnostic_shape, dtype=np.float64)
    wall_distance = np.empty(diagnostic_shape, dtype=np.float64)
    bridge_width = np.empty(diagnostic_shape, dtype=np.float64)
    chi_wall = np.empty(diagnostic_shape, dtype=np.float64)


    temporary_path = output_path.with_name(f".{output_path.stem}.building.npy")
    output = np.lib.format.open_memmap(temporary_path, mode="w+", dtype=np.float64, shape=sol.shape)

    _, _, _, grid_points = common.make_grid(rho, theta)

    ## SET UP PHI LOOP
    start_time = perf_counter()
    progress = tqdm(range(phi_deg.size), desc="Constructing piecewise density", unit="plane", dynamic_ncols=True, disable=not show_progress)
    log_context = (logging_redirect_tqdm(loggers=[sim_io.log]) if show_progress else nullcontext())
    try:
        with log_context:
            for plane_index in progress:

                boundary, _ = load_lcfs_boundary(analysis_dir, float(phi_deg[plane_index]), lcfs_index)
                plane, diagnostics = construct_density_plane(sol[plane_index], linear_profile[plane_index], theta, rho,
                                                             grid_points, boundary, vessel_radius, l_parallel_0,
                                                             n_axis, n_lcfs, n_wall, alpha, sol_beta)

                output[plane_index] = plane
                boundaries[plane_index] = diagnostics["boundary"]
                normals[plane_index] = diagnostics["normal"]
                lambda_n_0[plane_index] = diagnostics["lambda_0"]
                lambda_n_min[plane_index] = diagnostics["lambda_min"]
                lambda_n_max[plane_index] = diagnostics["lambda_max"]
                slopes[plane_index] = diagnostics["slope"]
                wall_distance[plane_index] = diagnostics["path_wall_distance"]
                bridge_width[plane_index] = diagnostics["bridge_width"]
                chi_wall[plane_index] = diagnostics["chi_wall"]
                sim_io.log.info("Constructed density phi=%03.0f: %d interior/%d exterior cells, lambda_n,0 %.6g/%.6g/%.6g m, bridge "
                    "%.6g/%.6g/%.6g m, chi_n,w min %.6g, density %.6g to %.6g.",
                    phi_deg[plane_index],
                    diagnostics["inside_cells"],
                    diagnostics["outside_cells"],
                    np.min(diagnostics["lambda_0"]),
                    np.median(diagnostics["lambda_0"]),
                    np.max(diagnostics["lambda_0"]),
                    np.min(diagnostics["bridge_width"]),
                    np.median(diagnostics["bridge_width"]),
                    np.max(diagnostics["bridge_width"]),
                    np.min(diagnostics["chi_wall"]),
                    diagnostics["density_min"],
                    diagnostics["density_max"],
                )
                if plane_index % 10 == 0:
                    output.flush()
                    gc.collect()
        output.flush()
        del output
        os.replace(temporary_path, output_path)
    except BaseException:
        del output
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    elapsed = perf_counter() - start_time
    sim_io.log.info("Constructed %d density planes in %.3f s (%.3f s/plane).",
                    phi_deg.size, elapsed, elapsed / phi_deg.size)

    metadata = {
        "phi_grid_deg": phi_deg,
        "lcfs_boundary_xz_m": boundaries,
        "lcfs_outward_normal_xz": normals,
        "surface_density_profile_inward_slope_per_m": slopes,
        "lambda_n_0_m": lambda_n_0,
        "lambda_n_min_along_path_m": lambda_n_min,
        "lambda_n_max_along_path_m": lambda_n_max,
        "path_wall_distance_m": wall_distance,
        "connection_length_bridge_width_m": bridge_width,
        "chi_n_wall": chi_wall,
        "l_parallel_0_m": np.array(l_parallel_0),
        "n_axis": np.array(n_axis),
        "n_lcfs": np.array(n_lcfs),
        "n_wall": np.array(n_wall),
        "alpha": np.array(alpha),
        "sol_beta": np.array(sol_beta),
        "lcfs_index": np.array(lcfs_index),
        "vessel_radius_m": np.array(vessel_radius),
        "nfield_input_profile": np.array("1 - Psi_bar (alpha=1)"),
    }
    return np.load(output_path, mmap_mode="r"), metadata

def plot_density_plane(density_plane, grid_x, grid_z, boundary, vessel_radius, phi_value,
                       n_axis, color_scale, plot_vmin, plot_vmax, show_lcfs,
                       sim_io, output_subdir):

    plot_x = np.vstack((grid_x[-1], grid_x))
    plot_z = np.vstack((grid_z[-1], grid_z))
    plot_density = np.vstack((density_plane[-1], density_plane)) / n_axis
    if color_scale == "log":
        plot_density = np.maximum(plot_density, plot_vmin)
        levels = np.geomspace(plot_vmin, plot_vmax, N_LEVELS)
        color_norm = LogNorm(vmin=plot_vmin, vmax=plot_vmax)
    else:
        levels = np.linspace(plot_vmin, plot_vmax, N_LEVELS)
        color_norm = None

    fig, ax = plt.subplots(figsize=FIGSIZE)
    contour = ax.contourf(plot_x, plot_z, plot_density, levels=levels,
                          cmap=COLORMAP, norm=color_norm, extend=CONTOUR_EXTEND)
    if show_lcfs:
        closed_boundary = np.vstack((boundary, boundary[0]))
        ax.plot(closed_boundary[:, 0], closed_boundary[:, 1], color="white", linewidth=1.0, label="LCFS")
    wall_angle = np.linspace(0.0, 2.0 * np.pi, 720)
    ax.plot(vessel_radius * np.cos(wall_angle), vessel_radius * np.sin(wall_angle), color="0.35", linewidth=1.0, label="Vessel wall")
    phi_phys = (phi_value + PHYSICAL_PHI_OFFSET_DEG) % 360.0
    ax.set_title(f"Piecewise plasma density\n$\\phi_{{comp}}={phi_value:.0f}^\\circ$, $\\phi_{{phys}}={phi_phys:.0f}^\\circ$")
    ax.set_xlabel(r"$x=\rho\cos\theta$ [m]")
    ax.set_ylabel(r"$z=\rho\sin\theta$ [m]")
    ax.set_aspect("equal")
    ax.set_xlim(-vessel_radius, vessel_radius)
    ax.set_ylim(-vessel_radius, vessel_radius)
    ax.grid(color="0.75", linewidth=0.4)
    ax.legend(loc="upper right")
    colorbar = fig.colorbar(contour, ax=ax, pad=0.03)
    colorbar.set_label(r"Normalized density $n/n_0$")
    fig.tight_layout()
    sim_io.saveFig(OUTPUT_PLOT_FILENAME.format(phi_deg=phi_value), subdir=output_subdir, dpi=DPI)
    plt.close(fig)

def generate_density_plots(field, rho, theta, phi_deg, boundaries, vessel_radius,
                           n_axis, color_scale, plot_vmin, plot_vmax, show_lcfs,
                           sim_io, output_subdir, show_progress):

    _, grid_x, grid_z, _ = common.make_grid(rho, theta)
    progress = tqdm(  range(phi_deg.size), desc="Plotting piecewise density", unit="plane", dynamic_ncols=True, disable=not show_progress)
    log_context = (logging_redirect_tqdm(loggers=[sim_io.log]) if show_progress else nullcontext())
    with log_context:
        for plane_index in progress:
            plot_density_plane(np.asarray(field[plane_index]), grid_x, grid_z, boundaries[plane_index], vessel_radius, float(phi_deg[plane_index]),
                               n_axis, color_scale, plot_vmin, plot_vmax, show_lcfs,
                               sim_io, output_subdir)
            if plane_index % 10 == 0:
                gc.collect()

def generate_midplane_density_plot(field, rho, theta, phi_deg,
                                   vessel_radius, n_axis, n_lcfs, sim_io, output_subdir):

    theta_lfs_index = common.nearest_coordinate_index(theta, 2.0 * np.pi, "theta_LFS")
    theta_hfs_index = common.nearest_coordinate_index(theta, np.pi, "theta_HFS")
    distance_from_lfs = np.concatenate( (vessel_radius - rho[::-1], vessel_radius + rho[1:]) )

    fig, ax = plt.subplots(figsize=MIDPLANE_TRACE_FIGSIZE)
    for requested_phi in MIDPLANE_TRACE_PHI_DEG:
        phi_index = common.nearest_coordinate_index(phi_deg, requested_phi, "phi_comp")
        density = np.concatenate( (np.asarray(field[phi_index, theta_lfs_index, ::-1]), np.asarray(field[phi_index, theta_hfs_index, 1:])) )
        ax.plot(distance_from_lfs, density / n_axis, linewidth=1.5, label=rf"$\phi_{{comp}}={phi_deg[phi_index]:.0f}^\circ$")

    lcfs_fraction = n_lcfs / n_axis
    ax.axhline(lcfs_fraction, color="black", linestyle="--", linewidth=1.2, label=rf"$n_{{LCFS}}/n_0={lcfs_fraction:g}$")
    ax.set_title("Horizontal-midplane plasma-density profile")
    ax.set_xlabel("Distance from low-field-side wall [m]")
    ax.set_ylabel(r"Normalized density $n/n_0$")
    ax.set_xlim(0.0, 2.0 * vessel_radius)
    ax.set_ylim(0.0, 1.02)
    ax.grid(color="0.75", linewidth=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    sim_io.saveFig(MIDPLANE_TRACE_FILENAME, subdir=output_subdir, dpi=DPI)
    plt.close(fig)

def _run_analysis(args, sim_io):
    validate_settings(args.n_axis, args.n_lcfs, args.n_wall, args.alpha, args.sol_beta)


    output_subdir = (args.output_subdir if args.output_subdir is not None else f"{args.sol_subdir}_Density_v2")

    poincare_settings = load_poincare_settings(args.analysis_dir)
    lcfs_index, lcfs_index_source = common.resolve_lcfs_index(args.lcfs_index, args.nfield_file, poincare_settings)

    l_parallel_0, l_parallel_0_source = common.resolve_l_parallel_0(args.l_parallel_0_m, poincare_settings, major_radius_m=MAJOR_RADIUS_M)
    print(f"Using LCFS surface {lcfs_index} ({lcfs_index_source}) and L_parallel,0={l_parallel_0:.6g} m")

    input_data = common.load_inputs(sim_io.data_dir, args.sol_subdir, args.nfield_subdir, args.nfield_file, sol_field_filename=args.sol_field_file)
    sol, linear_profile, rho, theta, phi_deg, sol_path, profile_path = input_data

    vessel_radius = (float(rho[-1]) if VESSEL_RADIUS_M is None else float(VESSEL_RADIUS_M))
    if PLOT_VMIN is None:
        plot_vmin = (LOG_PLOT_VMIN if args.color_scale == "log" else args.n_wall / args.n_axis)
    else:
        plot_vmin = float(PLOT_VMIN)
    plot_vmax = 1.0 if PLOT_VMAX is None else float(PLOT_VMAX)

    if plot_vmin >= plot_vmax:
        raise ValueError("Resolved plot limits require PLOT_VMIN < PLOT_VMAX.")
    if args.color_scale == "log" and plot_vmin <= 0.0:
        raise ValueError("Logarithmic plot limits require PLOT_VMIN > 0.")
    if not np.isclose(vessel_radius, rho[-1], rtol=0.0, atol=1e-12):
        raise ValueError("The exact wall boundary requires VESSEL_RADIUS_M to equal the outermost rho grid node.")

    output_data_dir = Path(sim_io.data_dir) / output_subdir
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_data_dir / OUTPUT_FIELD_FILENAME
    for coordinate, filename in ((rho, common.RHO_FILENAME), (theta, common.THETA_FILENAME), (phi_deg, common.PHI_FILENAME)):
        sim_io.saveNumpyData(coordinate, filename.removesuffix(".npy"), subdir=output_subdir)

    ## CALCULATIONS
    finite_sol = np.asarray(sol[np.isfinite(sol) & (sol > 0.0)])
    run_settings = {
        "ANALYSIS_DIR": args.analysis_dir,
        "SOL_SUBDIR": args.sol_subdir,
        "SOL_FIELD": str(sol_path),
        "NFIELD_SUBDIR": args.nfield_subdir,
        "LINEAR_PROFILE_FILE": str(profile_path),
        "OUTPUT_SUBDIR": output_subdir,
        "OUTPUT_FIELD_FILENAME": OUTPUT_FIELD_FILENAME,
        "FIELD_SHAPE": sol.shape,
        "LCFS_INDEX": lcfs_index,
        "LCFS_INDEX_SOURCE": lcfs_index_source,
        "N_AXIS": args.n_axis,
        "N_LCFS": args.n_lcfs,
        "N_WALL": args.n_wall,
        "DELTA_N_CORE": args.n_axis - args.n_lcfs,
        "DELTA_N_SOL": args.n_lcfs - args.n_wall,
        "ALPHA": args.alpha,
        "NFIELD_INPUT_PROFILE": "1 - Psi_bar (alpha=1)",
        "SOL_BETA": args.sol_beta,
        "L_PARALLEL_0_M": l_parallel_0,
        "L_PARALLEL_0_SOURCE": l_parallel_0_source,
        "MAJOR_RADIUS_M": MAJOR_RADIUS_M,
        "SOL_CONNECTION_LENGTH_MIN_M": float(np.min(finite_sol)),
        "SOL_CONNECTION_LENGTH_MAX_M": float(np.max(finite_sol)),
        "VESSEL_RADIUS_M": vessel_radius,
        "BOUNDARY_RESAMPLE_POINTS": BOUNDARY_RESAMPLE_POINTS,
        "PATH_SAMPLES": PATH_SAMPLES,
        "NORMAL_DERIVATIVE_STEP_M": NORMAL_DERIVATIVE_STEP_M,
        "SURFACE_SLOPE_SMOOTHING_SIGMA": SURFACE_SLOPE_SMOOTHING_SIGMA,
        "TREE_WORKERS": TREE_WORKERS,
        "LAMBDA_N_MIN_M": LAMBDA_N_MIN_M,
        "LAMBDA_N_MAX_M": LAMBDA_N_MAX_M,
        "COLOR_SCALE": args.color_scale,
        "PLOT_VMIN": plot_vmin,
        "PLOT_VMAX": plot_vmax,
        "SHOW_LCFS": args.show_lcfs,
        "MIDPLANE_TRACE_PHI_DEG": MIDPLANE_TRACE_PHI_DEG,
        "MIDPLANE_TRACE_FILENAME": MIDPLANE_TRACE_FILENAME,
        "GENERATE_PLOTS": args.plots,
        "SHOW_PROGRESS": args.progress,
    }
    sim_io.inputsBoilerplate("PIECEWISE FLUX / CONNECTION-LENGTH DENSITY INPUTS", run_settings, list(run_settings))
    if run_settings["SOL_CONNECTION_LENGTH_MAX_M"] >= l_parallel_0:
        sim_io.log.warning("The exterior field reaches %.6g m, at or above L_parallel_0 %.6g m; values will be capped at the LCFS reference.",
            run_settings["SOL_CONNECTION_LENGTH_MAX_M"], l_parallel_0)

    field, metadata = build_density_field(args.analysis_dir, sol, linear_profile,
                                          rho, theta, phi_deg, lcfs_index,
                                          vessel_radius, l_parallel_0,
                                          args.n_axis, args.n_lcfs, args.n_wall,
                                          args.alpha, args.sol_beta,
                                          output_path, sim_io, args.progress)

    ## SAVE OUTPUT
    metadata_path = output_data_dir / MODEL_METADATA_FILENAME
    np.savez_compressed(metadata_path, **metadata)
    sim_io.log.info("Saved piecewise density: %s", output_path)
    sim_io.log.info("Saved density model metadata: %s", metadata_path)
    sim_io.log.info("Finite-wall LCFS density-slope factor 1/(1-exp(-chi_n,w)): %.6g to %.6g.",
        np.min(1.0 / (-np.expm1(-metadata["chi_n_wall"]))),
        np.max(1.0 / (-np.expm1(-metadata["chi_n_wall"]))))

    ## PLOTTING
    if args.plots:
        generate_midplane_density_plot(field, rho, theta, phi_deg,
                                       vessel_radius, args.n_axis, args.n_lcfs, sim_io, output_subdir)
        generate_density_plots(field, rho, theta, phi_deg, metadata["lcfs_boundary_xz_m"], vessel_radius,
                               args.n_axis, args.color_scale, plot_vmin, plot_vmax, args.show_lcfs,
                               sim_io, output_subdir, args.progress)
        sim_io.log.info( "Saved density midplane trace and %d contour plots under %s.", phi_deg.size, Path(sim_io.plot_dir) / output_subdir)

    sim_io.log.info("## PIECEWISE PLASMA-DENSITY MODEL FINISHED ##")

    return field, metadata, output_path, metadata_path


class SOLDensity:
    """Construct a piecewise core/SOL plasma-density field.

    The analysis consumes an existing linear interior profile, a regular-grid
    SOL connection-length field, and saved Poincare surfaces. It does not
    perform field-line tracing.
    """

    def __init__(self, IO_handler, input_params):
        self.simIO = IO_handler
        self.input_params = dict(input_params)
        self.field = None
        self.metadata = None
        self.output_path = None
        self.metadata_path = None

    def run(self):
        """Build, save, and optionally plot the density field."""
        params = self.input_params
        for key in ("GENERATE_PLOTS", "SHOW_LCFS", "SHOW_PROGRESS"):
            if not isinstance(params[key], bool):
                raise ValueError(f"{key} must be a boolean")
        if params["COLOR_SCALE"] not in {"linear", "log"}:
            raise ValueError("COLOR_SCALE must be 'linear' or 'log'")
        args = SimpleNamespace(
            analysis_dir=params["ANLYS_DIR"],
            sol_subdir=params["SOL_SUBDIR"],
            sol_field_file=params["SOL_FIELD_FILENAME"],
            nfield_subdir=params["NFIELD_SUBDIR"],
            nfield_file=params["NFIELD_FILENAME"],
            output_subdir=params["ANLYS_SUBDIR"],
            lcfs_index=params["LCFS_INDEX"],
            n_axis=params["N_AXIS"],
            n_lcfs=params["N_LCFS"],
            n_wall=params["N_WALL"],
            alpha=params["ALPHA"],
            sol_beta=params["SOL_BETA"],
            l_parallel_0_m=params["L_PARALLEL_0_M"],
            plots=params["GENERATE_PLOTS"],
            show_lcfs=params["SHOW_LCFS"],
            color_scale=params["COLOR_SCALE"],
            progress=params["SHOW_PROGRESS"],
        )
        result = _run_analysis(args, self.simIO)
        self.field, self.metadata, self.output_path, self.metadata_path = result
        return self.field
