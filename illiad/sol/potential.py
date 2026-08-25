"""Build the piecewise LCFS / connection-length electrostatic potential.

This post-processing script implements
``input_files/piecewise_electrostatic_potential_model.pdf``.  It consumes an
existing interior nField, an existing regular SOL connection-length field,
and saved Poincare surfaces; it performs no field-line tracing.

The input nField is the linear interior profile

    q = 1 - Psi_tor.

The selected profile exponent is applied here using the same transformation
as ``FluxInterpolator``:

    psi_in = 1 - (1 - q)**alpha = 1 - Psi_tor**alpha.

At every toroidal plane the script derives the local LCFS potential scale
length from the transformed profile's inward slope, traces outward-normal
mapping paths from the LCFS to the wall, bridges any unsampled gap between
the LCFS and the first valid SOL sample, and integrates
``chi = integral(ds / lambda_phi)`` along those paths.

The saved scalar field retains the original stitcher's gradientor-compatible
``(phi, theta, rho)`` float64 layout and filename.  With the default potential
settings it is normalized to one at the magnetic axis and zero at the wall;
the Boris workflow may therefore continue to apply ``PLASMA_POTENTIAL`` when
loading the resulting electric field.
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
VESSEL_RADIUS_M = None  # None uses the outermost rho grid node

# Surface mapping and numerical differentiation.  The nField is sampled one
# and two steps inward from the LCFS for a second-order one-sided derivative.
BOUNDARY_RESAMPLE_POINTS = 720
PATH_SAMPLES = 256
NORMAL_DERIVATIVE_STEP_M = 0.002
SURFACE_SLOPE_SMOOTHING_SIGMA = 2.0
TREE_WORKERS = -1

# Optional bounds from the model PDF.  None leaves that side unbounded.
LAMBDA_PHI_MIN_M = None
LAMBDA_PHI_MAX_M = None

# Output and plot settings
FIGSIZE = (7, 6)
DPI = 250
COLORMAP = "afmhot"
N_LEVELS = 12
PLOT_VMIN = None
PLOT_VMAX = None
LOG_PLOT_VMIN = 1e-5  # Positive floor when log scale uses default limits
CONTOUR_EXTEND = "neither"
PHYSICAL_PHI_OFFSET_DEG = 198.0
MIDPLANE_TRACE_PHI_DEG = (324.0, 360.0)
MIDPLANE_TRACE_FIGSIZE = (8, 5)

# Output file names
OUTPUT_FIELD_FILENAME = "stitched_nfield_connection_length.npy"
MODEL_METADATA_FILENAME = "piecewise_potential_metadata.npz"
OUTPUT_PLOT_FILENAME = "stitched_potential_{phi_deg:03.0f}.png"
MIDPLANE_TRACE_FILENAME = "midplane_potential_trace.png"

def validate_settings(phi_wall, delta_phi_0w, delta_phi_sol, alpha, sol_beta):
    if not np.isfinite(phi_wall):
        raise ValueError("PHI_WALL must be finite.")
    if not 0.0 < delta_phi_sol < delta_phi_0w:
        raise ValueError("Require 0 < DELTA_PHI_SOL < DELTA_PHI_0W.")
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
    if LAMBDA_PHI_MIN_M is not None and LAMBDA_PHI_MIN_M <= 0.0:
        raise ValueError("LAMBDA_PHI_MIN_M must be positive or None.")
    if LAMBDA_PHI_MAX_M is not None and LAMBDA_PHI_MAX_M <= 0.0:
        raise ValueError("LAMBDA_PHI_MAX_M must be positive or None.")
    if (LAMBDA_PHI_MIN_M is not None and LAMBDA_PHI_MAX_M is not None and LAMBDA_PHI_MIN_M > LAMBDA_PHI_MAX_M):
        raise ValueError("LAMBDA_PHI_MIN_M cannot exceed LAMBDA_PHI_MAX_M.")
    if (PLOT_VMIN is not None and PLOT_VMAX is not None and PLOT_VMIN >= PLOT_VMAX):
        raise ValueError("Require PLOT_VMIN < PLOT_VMAX.")
    if CONTOUR_EXTEND not in {"neither", "both", "min", "max"}:
        raise ValueError("Invalid CONTOUR_EXTEND setting.")
    if LOG_PLOT_VMIN <= 0.0:
        raise ValueError("LOG_PLOT_VMIN must be positive.")

def construct_plane(sol_plane, core_plane,
                    theta, rho, grid_points, lcfs_points,
                    vessel_radius, l_parallel_0,
                    phi_wall,
                    delta_phi_0w, delta_phi_sol,
                    alpha, sol_beta):

    boundary = common.resample_closed_curve(lcfs_points, BOUNDARY_RESAMPLE_POINTS)
    normals = common.outward_normals(boundary)

    closed = np.vstack((boundary, boundary[0]))
    shape = (theta.size, rho.size)
    inside = MplPath(closed).contains_points(grid_points).reshape(shape)
    if not np.any(inside) or not np.any(~inside):
        raise ValueError("LCFS mask does not divide the computational mesh.")


    delta_phi_core = delta_phi_0w - delta_phi_sol
    output = np.empty((theta.size, rho.size), dtype=np.float64)

    linear_profile = np.clip(np.asarray(core_plane, dtype=np.float64), 0.0, 1.0)
    interior_plane_data = 1.0 - (1.0 - linear_profile) ** alpha

    output[inside] = (phi_wall + delta_phi_sol + delta_phi_core * interior_plane_data[inside])

    chi, diagnostics = common.construct_path_attenuation( sol_plane, interior_plane_data,
                                                         theta, rho, boundary, normals,
                                                         vessel_radius, l_parallel_0, delta_phi_core, delta_phi_sol, sol_beta,
                                                         path_samples=PATH_SAMPLES, derivative_step=NORMAL_DERIVATIVE_STEP_M,
                                                         smoothing_sigma=SURFACE_SLOPE_SMOOTHING_SIGMA,
                                                         lambda_min=LAMBDA_PHI_MIN_M, lambda_max=LAMBDA_PHI_MAX_M)

    output[~inside] = common.evaluate_exterior_profile(grid_points[~inside.ravel()], boundary, normals, chi,
                                                       vessel_radius, phi_wall, delta_phi_sol, tree_workers=TREE_WORKERS)
    if not np.all(np.isfinite(output)):
        raise ValueError("Constructed potential contains non-finite values.")

    diagnostics["boundary"] = boundary
    diagnostics["normal"] = normals
    diagnostics["inside_cells"] = int(np.count_nonzero(inside))
    diagnostics["outside_cells"] = int(np.count_nonzero(~inside))
    diagnostics["potential_min"] = float(np.min(output))
    diagnostics["potential_max"] = float(np.max(output))
    return output, diagnostics

def build_piecewise_field(analysis_dir, sol_data, core_data,
                          rho, theta, phi_deg, lcfs_index,
                          vessel_radius, l_parallel_0,
                          phi_wall,
                          delta_phi_0w, delta_phi_sol,
                          alpha, sol_beta,
                          output_path, sim_io, show_progress):

    diagnostic_shape = (phi_deg.size, BOUNDARY_RESAMPLE_POINTS)
    boundary_all = np.empty(diagnostic_shape + (2,), dtype=np.float64)
    normal_all = np.empty_like(boundary_all)
    lambda_phi_0_all = np.empty(diagnostic_shape, dtype=np.float64)
    lambda_phi_min_all = np.empty(diagnostic_shape, dtype=np.float64)
    lambda_phi_max_all = np.empty(diagnostic_shape, dtype=np.float64)
    slope_all = np.empty(diagnostic_shape, dtype=np.float64)
    wall_distance_all = np.empty(diagnostic_shape, dtype=np.float64)
    bridge_width_all = np.empty(diagnostic_shape, dtype=np.float64)
    chi_wall_all = np.empty(diagnostic_shape, dtype=np.float64)


    temporary_path = output_path.with_name(f".{output_path.stem}.building.npy")
    output = np.lib.format.open_memmap(temporary_path, mode="w+", dtype=np.float64, shape=sol_data.shape)

    _, _, _, grid_points = common.make_grid(rho, theta)

    ## SET UP PHI LOOP
    start_time = perf_counter()
    progress = tqdm( range(phi_deg.size), desc="Constructing piecewise potential", unit="plane", dynamic_ncols=True, disable=not show_progress)
    log_context = (logging_redirect_tqdm(loggers=[sim_io.log]) if show_progress else nullcontext())
    try:
        with log_context:
            for phi_index in progress:

                lcfs_points, _ = load_lcfs_boundary(analysis_dir, float(phi_deg[phi_index]), lcfs_index, spline_smoothing=1e-5, boundary_points=1000)
                plane, diagnostics = construct_plane(sol_data[phi_index], core_data[phi_index], theta, rho,
                                                     grid_points, lcfs_points, vessel_radius, l_parallel_0,
                                                     phi_wall,
                                                     delta_phi_0w, delta_phi_sol, alpha, sol_beta)
                output[phi_index] = plane
                boundary_all[phi_index] = diagnostics["boundary"]
                normal_all[phi_index] = diagnostics["normal"]
                lambda_phi_0_all[phi_index] = diagnostics["lambda_0"]
                lambda_phi_min_all[phi_index] = diagnostics["lambda_min"]
                lambda_phi_max_all[phi_index] = diagnostics["lambda_max"]
                slope_all[phi_index] = diagnostics["slope"]
                wall_distance_all[phi_index] = diagnostics["path_wall_distance"]
                bridge_width_all[phi_index] = diagnostics["bridge_width"]
                chi_wall_all[phi_index] = diagnostics["chi_wall"]
                sim_io.log.info(f"Constructed phi=%03.0f: %d interior/%d exterior cells, lambda_phi_0 %.6g/%.6g/%.6g m, bridge "
                                "%.6g/%.6g/%.6g m, chi_w min %.6g, potential %.6g to %.6g.",
                                phi_deg[phi_index], diagnostics["inside_cells"], diagnostics["outside_cells"],
                                np.min(diagnostics["lambda_0"]), np.median(diagnostics["lambda_0"]), np.max(diagnostics["lambda_0"]),
                                np.min(diagnostics["bridge_width"]), np.median(diagnostics["bridge_width"]), np.max(diagnostics["bridge_width"]),
                                np.min(diagnostics["chi_wall"]),
                                diagnostics["potential_min"], diagnostics["potential_max"])

                if phi_index % 10 == 0:
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
    sim_io.log.info("Constructed %d planes in %.3f s (%.3f s/plane).", phi_deg.size, elapsed, elapsed / phi_deg.size)
    metadata = {
        "phi_grid_deg": phi_deg,
        "lcfs_boundary_xz_m": boundary_all,
        "lcfs_outward_normal_xz": normal_all,
        "surface_profile_inward_slope_per_m": slope_all,
        "lambda_phi_0_m": lambda_phi_0_all,
        "lambda_phi_min_along_path_m": lambda_phi_min_all,
        "lambda_phi_max_along_path_m": lambda_phi_max_all,
        "path_wall_distance_m": wall_distance_all,
        "connection_length_bridge_width_m": bridge_width_all,
        "chi_wall": chi_wall_all,
        "l_parallel_0_m": np.array(l_parallel_0),
        "delta_phi_0w": np.array(delta_phi_0w),
        "delta_phi_sol": np.array(delta_phi_sol),
        "alpha": np.array(alpha),
        "nfield_input_profile": np.array("1 - Psi_bar (alpha=1)"),
        "phi_wall": np.array(phi_wall),
        "sol_beta": np.array(sol_beta),
        "lcfs_index": np.array(lcfs_index),
        "vessel_radius_m": np.array(vessel_radius),
    }
    return np.load(output_path, mmap_mode="r"), metadata

def plot_plane(field_plane, grid_x, grid_z, boundary, vessel_radius, phi_deg, color_scale, plot_vmin, plot_vmax, show_lcfs, sim_io, output_subdir):
    plot_x = np.vstack((grid_x[-1], grid_x))
    plot_z = np.vstack((grid_z[-1], grid_z))
    plot_field = np.vstack((field_plane[-1], field_plane))
    if color_scale == "log":
        plot_field = np.maximum(plot_field, plot_vmin)
        levels = np.geomspace(plot_vmin, plot_vmax, N_LEVELS)
        color_norm = LogNorm(vmin=plot_vmin, vmax=plot_vmax)
    else:
        levels = np.linspace(plot_vmin, plot_vmax, N_LEVELS)
        color_norm = None

    fig, ax = plt.subplots(figsize=FIGSIZE)
    contour = ax.contourf(plot_x, plot_z, plot_field, levels=levels, cmap=COLORMAP, norm=color_norm, extend=CONTOUR_EXTEND)
    if show_lcfs:
        closed_boundary = np.vstack((boundary, boundary[0]))
        ax.plot(closed_boundary[:, 0], closed_boundary[:, 1], color="white", linewidth=1.0, label="LCFS")

    wall_angle = np.linspace(0.0, 2.0 * np.pi, 720)
    ax.plot( vessel_radius * np.cos(wall_angle), vessel_radius * np.sin(wall_angle), color="0.35", linewidth=1.0, label="Vessel wall")

    phi_phys = (phi_deg + PHYSICAL_PHI_OFFSET_DEG) % 360.0
    ax.set_title(f"Piecewise electrostatic potential\n$\\phi_{{comp}}={phi_deg:.0f}^\\circ$, $\\phi_{{phys}}={phi_phys:.0f}^\\circ$")
    ax.set_xlabel(r"$x=\rho\cos\theta$ [m]")
    ax.set_ylabel(r"$z=\rho\sin\theta$ [m]")
    ax.set_aspect("equal")
    ax.set_xlim(-vessel_radius, vessel_radius)
    ax.set_ylim(-vessel_radius, vessel_radius)
    ax.grid(color="0.75", linewidth=0.4)
    ax.legend(loc="upper right")
    colorbar = fig.colorbar(contour, ax=ax, pad=0.03)
    colorbar.set_label("Electrostatic potential")
    fig.tight_layout()

    plot_name = OUTPUT_PLOT_FILENAME.format(phi_deg=phi_deg)
    sim_io.saveFig(plot_name, subdir=output_subdir, dpi=DPI)
    plt.close(fig)

def generate_plots(field, rho, theta, phi_deg, boundaries, vessel_radius, color_scale, plot_vmin, plot_vmax, show_lcfs, sim_io, output_subdir, show_progress,):
    _, grid_x, grid_z, _ = common.make_grid(rho, theta)
    progress = tqdm(range(phi_deg.size), desc="Plotting piecewise potential", unit="plane", dynamic_ncols=True, disable=not show_progress,)
    log_context = (logging_redirect_tqdm(loggers=[sim_io.log]) if show_progress else nullcontext())
    with log_context:
        for phi_index in progress:
            plot_plane(np.asarray(field[phi_index]), grid_x, grid_z, boundaries[phi_index], vessel_radius,
                       float(phi_deg[phi_index]), color_scale, plot_vmin, plot_vmax, show_lcfs, sim_io, output_subdir)
            if phi_index % 10 == 0:
                gc.collect()

def generate_midplane_trace_plot(field, rho, theta, phi_deg, vessel_radius,
                                 phi_wall, delta_phi_0w, delta_phi_sol,
                                 sim_io, output_subdir,):
    """Plot LFS-to-HFS horizontal-midplane potential at selected phi."""
    theta_lfs_index = common.nearest_coordinate_index(theta, 2.0 * np.pi, "theta_LFS")
    theta_hfs_index = common.nearest_coordinate_index(theta, np.pi, "theta_HFS")
    distance_from_lfs = np.concatenate( (vessel_radius - rho[::-1], vessel_radius + rho[1:]) )

    fig, ax = plt.subplots(figsize=MIDPLANE_TRACE_FIGSIZE)
    for requested_phi in MIDPLANE_TRACE_PHI_DEG:
        phi_index = common.nearest_coordinate_index(phi_deg, requested_phi, "phi_comp")
        potential = np.concatenate( (np.asarray(field[phi_index, theta_lfs_index, ::-1]), np.asarray(field[phi_index, theta_hfs_index, 1:])) )
        normalized_potential = (potential - phi_wall) / delta_phi_0w
        ax.plot(distance_from_lfs, normalized_potential, linewidth=1.5, label=rf"$\phi_{{comp}}={phi_deg[phi_index]:.0f}^\circ$")

    phi_sol_normalized = delta_phi_sol / delta_phi_0w
    ax.axhline(phi_sol_normalized, color="black", linestyle="--", linewidth=1.2, label=rf"$\Phi_{{SOL}}={phi_sol_normalized:g}$",)
    ax.set_title("Horizontal-midplane electrostatic-potential profile")
    ax.set_xlabel("Distance from low-field-side wall [m]")
    ax.set_ylabel("Normalized potential")
    ax.set_xlim(0.0, 2.0 * vessel_radius)
    ax.set_ylim(0.0, 1.02)
    ax.grid(color="0.75", linewidth=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    sim_io.saveFig(MIDPLANE_TRACE_FILENAME, subdir=output_subdir, dpi=DPI)
    plt.close(fig)

def _run_analysis(args, sim_io):
    validate_settings(
        args.phi_wall,
        args.delta_phi_0w,
        args.delta_phi_sol,
        args.alpha,
        args.sol_beta,
    )


    output_subdir = (args.output_subdir if args.output_subdir is not None else f"{args.sol_subdir}_Stitched_v2")

    poincare_settings = load_poincare_settings(args.analysis_dir)
    lcfs_index, lcfs_index_source = common.resolve_lcfs_index(args.lcfs_index, args.nfield_file, poincare_settings)

    l_parallel_0, l_parallel_0_source = common.resolve_l_parallel_0(args.l_parallel_0_m, poincare_settings, major_radius_m=MAJOR_RADIUS_M)
    print(f"Using LCFS surface {lcfs_index} ({lcfs_index_source}) and "f"L_parallel,0={l_parallel_0:.6g} m")

    input_data = common.load_inputs(sim_io.data_dir, args.sol_subdir, args.nfield_subdir, args.nfield_file, sol_field_filename=args.sol_field_file)
    sol_data, core_data, rho, theta, phi_deg, sol_path, core_path = input_data

    vessel_radius = (float(rho[-1]) if VESSEL_RADIUS_M is None else float(VESSEL_RADIUS_M))
    if PLOT_VMIN is None:
        plot_vmin = (LOG_PLOT_VMIN if args.color_scale == "log" else args.phi_wall)
    else:
        plot_vmin = float(PLOT_VMIN)
    plot_vmax = (args.phi_wall + args.delta_phi_0w if PLOT_VMAX is None else float(PLOT_VMAX))

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
    delta_phi_core = args.delta_phi_0w - args.delta_phi_sol
    finite_sol = np.asarray(sol_data[np.isfinite(sol_data) & (sol_data > 0.0)])
    run_settings = {
        "ANALYSIS_DIR": args.analysis_dir,
        "SOL_SUBDIR": args.sol_subdir,
        "SOL_FIELD": str(sol_path),
        "NFIELD_SUBDIR": args.nfield_subdir,
        "NFIELD_FILE": str(core_path),
        "OUTPUT_SUBDIR": output_subdir,
        "OUTPUT_FIELD_FILENAME": OUTPUT_FIELD_FILENAME,
        "FIELD_SHAPE": sol_data.shape,
        "LCFS_INDEX": lcfs_index,
        "LCFS_INDEX_SOURCE": lcfs_index_source,
        "PHI_WALL": args.phi_wall,
        "DELTA_PHI_0W": args.delta_phi_0w,
        "DELTA_PHI_SOL": args.delta_phi_sol,
        "DELTA_PHI_CORE": delta_phi_core,
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
        "LAMBDA_PHI_MIN_M": LAMBDA_PHI_MIN_M,
        "LAMBDA_PHI_MAX_M": LAMBDA_PHI_MAX_M,
        "NFIELD_DATA_MIN": float(np.min(core_data)),
        "NFIELD_DATA_MAX": float(np.max(core_data)),
        "COLOR_SCALE": args.color_scale,
        "PLOT_VMIN": plot_vmin,
        "PLOT_VMAX": plot_vmax,
        "SHOW_LCFS": args.show_lcfs,
        "MIDPLANE_TRACE_PHI_DEG": MIDPLANE_TRACE_PHI_DEG,
        "MIDPLANE_TRACE_FILENAME": MIDPLANE_TRACE_FILENAME,
        "GENERATE_PLOTS": args.plots,
        "SHOW_PROGRESS": args.progress,
    }
    sim_io.inputsBoilerplate("PIECEWISE NFIELD / CONNECTION-LENGTH POTENTIAL INPUTS", run_settings, list(run_settings),)
    if run_settings["SOL_CONNECTION_LENGTH_MAX_M"] >= l_parallel_0:
        sim_io.log.warning("The exterior field (%.6g m), exceeds L_parallel_0 (%.6g m); values will be capped at the LCFS reference.",
                           run_settings["SOL_CONNECTION_LENGTH_MAX_M"], l_parallel_0)

    field, metadata = build_piecewise_field(args.analysis_dir, sol_data, core_data,
                                            rho, theta, phi_deg, lcfs_index,
                                            vessel_radius, l_parallel_0,
                                            args.phi_wall,
                                            args.delta_phi_0w, args.delta_phi_sol,
                                            args.alpha, args.sol_beta,
                                            output_path, sim_io, args.progress)

    ## SAVE OUTPUT
    metadata_path = output_data_dir / MODEL_METADATA_FILENAME
    np.savez_compressed(metadata_path, **metadata)
    sim_io.log.info("Saved piecewise potential: %s", output_path)
    sim_io.log.info("Saved piecewise model metadata: %s", metadata_path)
    sim_io.log.info("Finite-wall LCFS slope factor 1/(1-exp(-chi_w)): %.6g to %.6g.",
                    np.min(1.0 / (-np.expm1(-metadata["chi_wall"]))),
                    np.max(1.0 / (-np.expm1(-metadata["chi_wall"]))))

    ## PLOTTING
    if args.plots:
        generate_midplane_trace_plot(field, rho, theta, phi_deg, vessel_radius,
                                     args.phi_wall, args.delta_phi_0w,
                                     args.delta_phi_sol, sim_io, output_subdir)
        sim_io.log.info("Saved horizontal-midplane potential trace: %s/%s.", output_subdir, MIDPLANE_TRACE_FILENAME)
        generate_plots(field, rho, theta, phi_deg, metadata["lcfs_boundary_xz_m"], vessel_radius,
                       args.color_scale, plot_vmin, plot_vmax, args.show_lcfs,
                       sim_io, output_subdir, args.progress)
        sim_io.log.info("Saved %d piecewise-potential plots under %s.", phi_deg.size, Path(sim_io.plot_dir) / output_subdir)

    sim_io.log.info("## PIECEWISE NFIELD / CONNECTION-LENGTH POTENTIAL FINISHED ##")

    return field, metadata, output_path, metadata_path


class SOLPotential:
    """Construct a piecewise core/SOL electrostatic-potential field.

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
        """Build, save, and optionally plot the potential field."""
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
            phi_wall=params["PHI_WALL"],
            delta_phi_0w=params["DELTA_PHI_0W"],
            delta_phi_sol=params["DELTA_PHI_SOL"],
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
