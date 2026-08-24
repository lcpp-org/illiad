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

import argparse
from contextlib import nullcontext
import gc
import os
from pathlib import Path
import re
import sys
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from illiad.io import IOHandler
from illiad.sol import (
    load_lcfs_boundary,
    load_poincare_settings,
)


# Analysis settings
ANALYSIS_DIR = "IOTA3_1000sp_atol1e-9"
SOL_SUBDIR = "ConLenVolume_REDO_250spins_rk1mm_RegularGrid"
NFIELD_SUBDIR = "LCFS41"
#NFIELD_FILENAME = "nField_LCFS40alpha1p0.npy"
NFIELD_FILENAME = "nField_LCFS41_linear.npy"
SOL_FIELD_FILENAME = "connection_length_field_m.npy"

# None first infers the surface from NFIELD_FILENAME (LCFS<number>), then
# falls back to LCFS_INDEX in the Poincare log.  This prevents an appended
# Poincare run from silently changing the boundary paired with the nField.
LCFS_INDEX = 41

# Piecewise-potential inputs.  These defaults preserve the normalized scalar
# convention used by the original stitcher and by PLASMA_POTENTIAL scaling.
PHI_WALL = 0.0
DELTA_PHI_0W = 1.0
DELTA_PHI_SOL = 0.2
ALPHA = 0.85
SOL_BETA = 0.5

# Numerical LCFS connection-length reference.  None derives the single-line
# trace limit 2*pi*R0*SPINS from the Poincare log.  It is an intentional model
# reference, not a claim that the physical LCFS connection length is finite.
L_PARALLEL_0_M = None
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
GENERATE_PLOTS = True
SHOW_PROGRESS = True
FIGSIZE = (7, 6)
DPI = 250
COLORMAP = "afmhot"
COLOR_SCALE = "log"  # "linear" or "log"
N_LEVELS = 12
PLOT_VMIN = None  # None uses PHI_WALL
PLOT_VMAX = None  # None uses PHI_WALL + the selected DELTA_PHI_0W
LOG_PLOT_VMIN = 1e-5  # Positive floor when log scale uses default limits
CONTOUR_EXTEND = "neither"
SHOW_LCFS = True
PHYSICAL_PHI_OFFSET_DEG = 198.0
MIDPLANE_TRACE_PHI_DEG = (324.0, 360.0)
MIDPLANE_TRACE_FIGSIZE = (8, 5)

# Output file names
OUTPUT_SUBDIR = "ConLenVolume_REDO_250spins_rk1mm_RegularGrid_Stitched_v2-3"  # None uses f"{SOL_SUBDIR}_Stitched_v2"
OUTPUT_FIELD_FILENAME = "stitched_nfield_connection_length.npy"
RHO_FILENAME = "rho_grid_m.npy"
THETA_FILENAME = "theta_grid_rad.npy"
PHI_FILENAME = "phi_grid_deg.npy"
MODEL_METADATA_FILENAME = "piecewise_potential_metadata.npz"
OUTPUT_PLOT_FILENAME = "stitched_potential_{phi_deg:03.0f}.png"
MIDPLANE_TRACE_FILENAME = "midplane_potential_trace.png"

# common function
def resolve_lcfs_index(requested_index, nfield_filename, poincare_settings):
    if requested_index is not None:
        if requested_index < 0:
            raise ValueError("LCFS index cannot be negative.")
        return requested_index, "explicit setting"

    filename_match = re.search(r"LCFS(\d+)", nfield_filename, re.IGNORECASE)
    if filename_match:
        return int(filename_match.group(1)), "nField filename"

    logged_index = poincare_settings.get("LCFS_INDEX")
    if logged_index is None:
        raise ValueError("Could not infer an LCFS index from the nField filename or ""Poincare log; provide --lcfs-index.")
    return int(logged_index), "Poincare log"

# common function
def resolve_l_parallel_0(requested_value, poincare_settings):
    if requested_value is not None:
        value = float(requested_value)
        source = "explicit setting"
    else:
        spins = poincare_settings.get("SPINS")
        if not isinstance(spins, int) or spins <= 0:
            raise ValueError("Cannot derive L_PARALLEL_0_M without a positive integer SPINS value in the Poincare log; provide --l-parallel-0-m.")
        
        value = 2.0 * np.pi * MAJOR_RADIUS_M * spins
        source = f"2*pi*{MAJOR_RADIUS_M:g} m*{spins} logged spins"
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("L_PARALLEL_0_M must be positive and finite.")
    return value, source

# "load_inputs" method
def require_file(path, description):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path

# common function
def load_inputs(analysis_dir, sol_subdir, coreField_subdir, coreField_filename):
    base_data_dir = PROJECT_ROOT / "output" / analysis_dir / "data"
    sol_data_dir = base_data_dir / sol_subdir
    coreField_path = base_data_dir / coreField_subdir / coreField_filename

    sol_path = require_file(sol_data_dir / SOL_FIELD_FILENAME, "SOL-solved field")
    rho_path = require_file(sol_data_dir / RHO_FILENAME, "rho grid")
    theta_path = require_file(sol_data_dir / THETA_FILENAME, "theta grid")
    phi_path = require_file(sol_data_dir / PHI_FILENAME, "phi grid")
    require_file(coreField_path, "Core-solved Field")

    print(f"Reading regular SOL field: {sol_path}")
    sol_data = np.load(sol_path, mmap_mode="r")
    print(f"Output field shape (phi, theta, rho): {sol_data.shape}")
    print(f"Reading interior field: {coreField_path}")
    core_data = np.load(coreField_path, mmap_mode="r")
    rho = np.load(rho_path)
    theta = np.load(theta_path)
    phi_deg = np.load(phi_path)

    expected_shape = (phi_deg.size, theta.size, rho.size)
    if sol_data.shape != expected_shape:
        raise ValueError(f"SOL field shape {sol_data.shape} does not match coordinates {expected_shape}.")
    if core_data.shape != expected_shape:
        raise ValueError(f"nField shape {core_data.shape} does not match SOL field shape {expected_shape}. Regen fields on the same mesh.")
    if sol_data.ndim != 3:
        raise ValueError("Input fields must use the (phi, theta, rho) layout.")
    if np.any(np.diff(rho) <= 0.0):
        raise ValueError("rho_grid_m.npy must be strictly increasing.")
    if np.any(np.diff(theta) <= 0.0):
        raise ValueError("theta_grid_rad.npy must be strictly increasing.")
    if np.any(np.diff(phi_deg) <= 0.0):
        raise ValueError("phi_grid_deg.npy must be strictly increasing.")
    if not np.all(np.isfinite(core_data)):
        raise ValueError("nField contains non-finite values.")

    finite_sol = np.isfinite(sol_data) & (sol_data > 0.0)
    if not np.any(finite_sol):
        raise ValueError("Connection-length field has no positive finite samples.")
    
    return sol_data, core_data, rho, theta, phi_deg, sol_path, coreField_path

# common function
def make_grid(rho, theta):
    grid_theta, grid_rho = np.meshgrid(theta, rho, indexing="ij")
    grid_x = grid_rho * np.cos(grid_theta)
    grid_z = grid_rho * np.sin(grid_theta)
    grid_points = np.column_stack((grid_x.ravel(), grid_z.ravel()))
    return grid_rho, grid_x, grid_z, grid_points

# common function
def resample_closed_curve(vertices, point_count):
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError("An LCFS boundary requires at least three x-z points.")
    if np.allclose(vertices[0], vertices[-1]):
        vertices = vertices[:-1]

    closed = np.vstack((vertices, vertices[0]))
    segment_length = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    keep_segment = segment_length > 0.0
    if np.count_nonzero(keep_segment) < 3:
        raise ValueError("LCFS boundary has fewer than three unique points.")
    
    kept_vertices = closed[:-1][keep_segment]
    closed = np.vstack((kept_vertices, kept_vertices[0]))
    segment_length = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_length)))
    target = np.linspace(0.0, cumulative[-1], point_count, endpoint=False)
    return np.column_stack( (np.interp(target, cumulative, closed[:, 0]), np.interp(target, cumulative, closed[:, 1])) )

# common function
def outward_normals(boundary):
    tangent = np.roll(boundary, -1, axis=0) - np.roll(boundary, 1, axis=0)
    tangent_norm = np.linalg.norm(tangent, axis=1)
    if np.any(tangent_norm == 0.0):
        raise ValueError("Cannot construct normals from repeated LCFS points.")
    tangent = tangent / tangent_norm[:, None]

    signed_area = 0.5 * np.sum(boundary[:, 0] * np.roll(boundary[:, 1], -1) - boundary[:, 1] * np.roll(boundary[:, 0], -1))
    if signed_area > 0.0:
        normals = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    else:
        normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return normals

# common function (via "construct_path_attenuation" and "surface_profile_slope")
def xz_to_theta_rho(points):
    return np.column_stack(( np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi), np.linalg.norm(points, axis=1)) )

# common function (via "construct_path_attenuation" and many more...)
def wall_distance(origins, directions, vessel_radius):
    origin_dot_direction = np.sum(origins * directions, axis=1)
    discriminant = ( origin_dot_direction**2 + vessel_radius**2 - np.sum(origins**2, axis=1))
    if np.any(discriminant < -1e-12):
        raise ValueError("An LCFS mapping ray does not intersect the vessel.")
    
    distance = -origin_dot_direction + np.sqrt(np.maximum(discriminant, 0.0))
    if np.any(distance <= 0.0):
        raise ValueError("An outward LCFS mapping ray misses the vessel wall.")
    
    return distance

# common function (via "construct_path_attenuation")
def surface_profile_slope(nfield_interpolator, boundary, normals, derivative_step=NORMAL_DERIVATIVE_STEP_M, smoothing_sigma=SURFACE_SLOPE_SMOOTHING_SIGMA):

    step = derivative_step
    inward_one = boundary - step * normals
    #inward_two = boundary - 2.0 * step * normals
    inward_two = inward_one - step * normals
    profile_one = nfield_interpolator(xz_to_theta_rho(inward_one))
    profile_two = nfield_interpolator(xz_to_theta_rho(inward_two))
    if not np.all(np.isfinite(profile_one)) or not np.all(np.isfinite(profile_two)):
        raise ValueError("Could not evaluate the nField on both inward derivative shells.")

    # psi_in(LCFS) = 0 is imposed analytically.  This is the positive inward
    # slope -d(psi_in)/dn_out from a second-order one-sided derivative.
    slope = (4.0 * profile_one - profile_two) / (2.0 * step)
    if smoothing_sigma > 0.0:
        slope = gaussian_filter1d(slope, smoothing_sigma, mode="wrap")

    if not np.all(np.isfinite(slope)) or np.any(slope <= 0.0):
        bad_count = int(np.count_nonzero(~np.isfinite(slope) | (slope <= 0.0)))
        raise ValueError(f"Derived nonpositive LCFS nField slope at {bad_count} surface points. Verify that the selected LCFS matches the nField.")
    
    return slope, profile_one, profile_two

# "bridge_connection_length_paths" method
def smoothstep(values):
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)

# common function (via "construct_path_attenuation")
def bridge_connection_length_paths(raw_paths, path_distance, l_parallel_0):
    """Join L_parallel_0 to the first valid exterior sample on every path."""
    # sanitize the inputs, and set points on LCFS to False
    valid = np.isfinite(raw_paths) & (raw_paths > 0.0)
    valid[:, 0] = False
    if np.any(~np.any(valid, axis=1)):
        count = int(np.count_nonzero(~np.any(valid, axis=1)))
        raise ValueError(f"No positive exterior connection length was found on {count} LCFS-to-wall paths.")

    first_valid = np.argmax(valid, axis=1)
    row_index = np.arange(raw_paths.shape[0])
    first_value = np.minimum(raw_paths[row_index, first_valid], l_parallel_0)
    first_distance = path_distance[row_index, first_valid]

    fraction = np.divide(path_distance, first_distance[:, None], out=np.ones_like(path_distance), where=first_distance[:, None] > 0.0)

    bridged = l_parallel_0 + (first_value[:, None] - l_parallel_0) * smoothstep(fraction)
    after_bridge = np.arange(raw_paths.shape[1])[None, :] > first_valid[:, None]
    bridged[after_bridge] = raw_paths[after_bridge]
    bridged[:, 0] = l_parallel_0

    # The regular field should only be missing next to its own LCFS mask.  If
    # isolated invalid cells remain farther out, interpolate them along the
    # same physical mapping path rather than across neighboring paths.
    invalid_after = after_bridge & (~np.isfinite(bridged) | (bridged <= 0.0))
    for row in np.flatnonzero(np.any(invalid_after, axis=1)):
        good = np.isfinite(bridged[row]) & (bridged[row] > 0.0)
        bridged[row, ~good] = np.interp(path_distance[row, ~good], path_distance[row, good], bridged[row, good])

    bridged = np.minimum(bridged, l_parallel_0)
    if not np.all(np.isfinite(bridged)) or np.any(bridged <= 0.0):
        raise ValueError("Bridged connection-length paths are not positive.")

    return bridged, first_distance

# common function
def construct_path_attenuation(sol_plane, profile_plane,
                               theta, rho, boundary, normals,
                               vessel_radius, l_parallel_0,
                               delta_phi_core, delta_phi_sol, sol_beta,
                               path_samples=PATH_SAMPLES,
                               derivative_step=NORMAL_DERIVATIVE_STEP_M, smoothing_sigma=SURFACE_SLOPE_SMOOTHING_SIGMA,
                               lambda_min=LAMBDA_PHI_MIN_M, lambda_max=LAMBDA_PHI_MAX_M,):

    ## GET INTERIOR SLOPE, CALCULATE LAMBDA AT LCFS
    theta_extended = np.concatenate( ([theta[-1] - 2.0 * np.pi], theta, [theta[0] + 2.0 * np.pi]) )
    profile_extended = np.concatenate( (profile_plane[-1:, :], profile_plane, profile_plane[:1, :]), axis=0 )
    nfield_interpolator = RegularGridInterpolator( (theta_extended, rho), profile_extended, bounds_error=False, fill_value=np.nan, )
    slope, profile_one, profile_two = surface_profile_slope(nfield_interpolator, boundary, normals, derivative_step, smoothing_sigma)
    lambda_phi_0 = delta_phi_sol / (delta_phi_core * slope)

    ## CREATE GRID OF L_PARALLEL AT EVERY SAMPLE POINT FOR EVERY LCFS RAY
    path_wall_distance = wall_distance(boundary, normals, vessel_radius)
    path_fraction = np.linspace(0.0, 1.0, path_samples)
    path_distance = path_wall_distance[:, None] * path_fraction[None, :]
    path_points = boundary[:, None, :] + (path_distance[:, :, None] * normals[:, None, :])
    sol_array = np.asarray(sol_plane, dtype=np.float64)
    sol_extended = np.concatenate( (sol_array[-1:, :], sol_array, sol_array[:1, :]), axis=0 )
    sol_interpolator = RegularGridInterpolator( (theta_extended, rho), sol_extended, bounds_error=False, fill_value=np.nan, )
    raw_connection = sol_interpolator(xz_to_theta_rho(path_points.reshape(-1, 2))).reshape(path_distance.shape)


    connection, bridge_width = bridge_connection_length_paths(raw_connection, path_distance, l_parallel_0)
    lambda_phi = lambda_phi_0[:, None] * (connection / l_parallel_0) ** sol_beta

    if lambda_min is not None:
        lambda_phi = np.maximum(lambda_phi, lambda_min)
    if lambda_max is not None:
        lambda_phi = np.minimum(lambda_phi, lambda_max)
    if not np.all(np.isfinite(lambda_phi)) or np.any(lambda_phi <= 0.0):
        raise ValueError("Potential scale lengths are not positive and finite.")

    inverse_lambda = 1.0 / lambda_phi
    increments = 0.5 * (inverse_lambda[:, 1:] + inverse_lambda[:, :-1]) * np.diff(path_distance, axis=1)

    chi = np.zeros_like(lambda_phi)
    chi[:, 1:] = np.cumsum(increments, axis=1)
    if not np.all(np.isfinite(chi)) or np.any(chi[:, -1] <= 0.0):
        raise ValueError("Attenuation integrals are not positive and finite.")

    diagnostics = {
        "slope": slope,
        "profile_one": profile_one,
        "profile_two": profile_two,
        "lambda_phi_0": lambda_phi_0,
        "lambda_phi_min": np.min(lambda_phi, axis=1),
        "lambda_phi_max": np.max(lambda_phi, axis=1),
        "path_wall_distance": path_wall_distance,
        "bridge_width": bridge_width,
        "chi_wall": chi[:, -1],
    }
    return chi, diagnostics

# common function
def evaluate_exterior_profile(exterior_points, lcfs_points, normals,
                              chi, vessel_radius,
                              outer_value, profile_difference):
    
    # find nearest lcfs point to each exterior point and calc distance
    tree = cKDTree(lcfs_points)
    _, surface_index = tree.query(exterior_points, workers=TREE_WORKERS)
    origins = lcfs_points[surface_index]
    displacement = exterior_points - origins
    distance = np.linalg.norm(displacement, axis=1)
    if np.any(distance == 0.0):
        directions = normals[surface_index].copy()
        directions[distance > 0.0] = (displacement[distance > 0.0] / distance[distance > 0.0, None])
    else:
        directions = displacement / distance[:, None]

    # A mesh point extremely close to the LCFS can select a neighboring
    # resampled vertex whose point-to-vertex vector has a tiny inward normal
    # projection.  Its physical mapping is the vertex normal; use that limit
    # instead of rejecting an otherwise valid plane.
    outward_projection = np.sum(directions * normals[surface_index], axis=1)    
    nonoutward = outward_projection <= 0.0
    directions[nonoutward] = normals[surface_index[nonoutward]]

    # fractional distance to wall at each exterior point
    target_wall_distance = wall_distance(origins, directions, vessel_radius)
    target_fraction = np.clip(distance / target_wall_distance, 0.0, 1.0)

    # linearly interpolate chi from path sample points to exterior points
    path_samples = chi.shape[1]
    sample_position = target_fraction * (path_samples - 1)
    lower = np.floor(sample_position).astype(np.int64)
    upper = np.minimum(lower + 1, path_samples - 1)
    weight = sample_position - lower
    chi_target = ( (1.0 - weight) * chi[surface_index, lower] + weight * chi[surface_index, upper])
    chi_wall = chi[surface_index, -1]

    exp_target = np.exp(-np.minimum(chi_target, 745.0))
    exp_wall = np.exp(-np.minimum(chi_wall, 745.0))
    denominator = -np.expm1(-chi_wall)
    if np.any(denominator <= 0.0):
        raise ValueError("A wall attenuation denominator is nonpositive.")
    profile = outer_value + profile_difference * (exp_target - exp_wall) / denominator

    return profile





# unique function
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Construct the piecewise nField / connection-length potential."
        )
    )
    parser.add_argument(
        "analysis_dir",
        nargs="?",
        default=ANALYSIS_DIR,
        help=f"Existing directory under output/ (default: {ANALYSIS_DIR}).",
    )
    parser.add_argument(
        "--sol-subdir",
        default=SOL_SUBDIR,
        help=f"Regular SOL field data subdirectory (default: {SOL_SUBDIR}).",
    )
    parser.add_argument(
        "--nfield-subdir",
        default=NFIELD_SUBDIR,
        help=f"Interior nField data subdirectory (default: {NFIELD_SUBDIR}).",
    )
    parser.add_argument(
        "--nfield-file",
        default=NFIELD_FILENAME,
        help=f"Interior nField filename (default: {NFIELD_FILENAME}).",
    )
    parser.add_argument(
        "--output-subdir",
        default=OUTPUT_SUBDIR,
        help="Output subdirectory (default: <sol-subdir>_Stitched_v2).",
    )
    parser.add_argument(
        "--lcfs-index",
        type=int,
        default=LCFS_INDEX,
        help=(
            "LCFS Poincare surface index (default: infer from nField filename, "
            "then fall back to the Poincare log)."
        ),
    )
    parser.add_argument(
        "--delta-phi-0w",
        type=float,
        default=DELTA_PHI_0W,
        help=f"Axis-to-wall potential difference (default: {DELTA_PHI_0W}).",
    )
    parser.add_argument(
        "--delta-phi-sol",
        type=float,
        default=DELTA_PHI_SOL,
        help=f"LCFS-to-wall potential difference (default: {DELTA_PHI_SOL}).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help=f"Interior profile exponent applied to linear nField (default: {ALPHA}).",
    )
    parser.add_argument(
        "--sol-beta",
        type=float,
        default=SOL_BETA,
        help=f"Connection-length exponent beta (default: {SOL_BETA}).",
    )
    parser.add_argument(
        "--l-parallel-0-m",
        type=float,
        default=L_PARALLEL_0_M,
        help=(
            "Finite LCFS connection-length reference in meters (default: "
            "derive 2*pi*R0*SPINS from the Poincare log)."
        ),
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=GENERATE_PLOTS,
        help="Generate one potential contour plot per toroidal plane.",
    )
    parser.add_argument(
        "--show-lcfs",
        action=argparse.BooleanOptionalAction,
        default=SHOW_LCFS,
        help="Draw the LCFS boundary on generated potential plots.",
    )
    parser.add_argument(
        "--color-scale",
        choices=("linear", "log"),
        default=COLOR_SCALE,
        help=f"Potential-plot color scale (default: {COLOR_SCALE}).",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=SHOW_PROGRESS,
        help="Show construction and plotting progress bars.",
    )
    return parser.parse_args()

# unique function
def validate_settings(delta_phi_0w, delta_phi_sol, alpha, sol_beta):
    if not np.isfinite(PHI_WALL):
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
    if COLOR_SCALE not in {"linear", "log"}:
        raise ValueError("COLOR_SCALE must be 'linear' or 'log'.")
    if LOG_PLOT_VMIN <= 0.0:
        raise ValueError("LOG_PLOT_VMIN must be positive.")

# unique function
def construct_plane(sol_plane, core_plane,
                    theta, rho, grid_points, lcfs_points,
                    vessel_radius, l_parallel_0,
                    delta_phi_0w, delta_phi_sol,
                    alpha, sol_beta):
    
    boundary = resample_closed_curve(lcfs_points, BOUNDARY_RESAMPLE_POINTS)
    normals = outward_normals(boundary)

    closed = np.vstack((boundary, boundary[0]))
    shape = (theta.size, rho.size)
    inside = MplPath(closed).contains_points(grid_points).reshape(shape)
    if not np.any(inside) or not np.any(~inside):
        raise ValueError("LCFS mask does not divide the computational mesh.")


    delta_phi_core = delta_phi_0w - delta_phi_sol
    output = np.empty((theta.size, rho.size), dtype=np.float64)

    linear_profile = np.clip(np.asarray(core_plane, dtype=np.float64), 0.0, 1.0)
    interior_plane_data = 1.0 - (1.0 - linear_profile) ** alpha

    output[inside] = (PHI_WALL + delta_phi_sol + delta_phi_core * interior_plane_data[inside])

    chi, diagnostics = construct_path_attenuation(sol_plane, interior_plane_data,
                                                  theta, rho, boundary, normals,
                                                  vessel_radius, l_parallel_0,
                                                  delta_phi_core, delta_phi_sol, sol_beta)

    output[~inside] = evaluate_exterior_profile(grid_points[~inside.ravel()], boundary, normals, chi, vessel_radius, PHI_WALL, delta_phi_sol)
    if not np.all(np.isfinite(output)):
        raise ValueError("Constructed potential contains non-finite values.")

    diagnostics["boundary"] = boundary
    diagnostics["normal"] = normals
    diagnostics["inside_cells"] = int(np.count_nonzero(inside))
    diagnostics["outside_cells"] = int(np.count_nonzero(~inside))
    diagnostics["potential_min"] = float(np.min(output))
    diagnostics["potential_max"] = float(np.max(output))
    return output, diagnostics

# unique function
def build_piecewise_field(analysis_dir, sol_data, core_data,
                          rho, theta, phi_deg, lcfs_index,
                          vessel_radius, l_parallel_0,
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

    _, _, _, grid_points = make_grid(rho, theta)

    ## SET UP PHI LOOP
    start_time = perf_counter()
    progress = tqdm( range(phi_deg.size), desc="Constructing piecewise potential", unit="plane", dynamic_ncols=True, disable=not show_progress)
    log_context = (logging_redirect_tqdm(loggers=[sim_io.log]) if show_progress else nullcontext())
    try:
        with log_context:
            for phi_index in progress:

                lcfs_points, _ = load_lcfs_boundary(analysis_dir, float(phi_deg[phi_index]), lcfs_index, spline_smoothing=1e-5, boundary_points=1000)
                plane, diagnostics = construct_plane(sol_data[phi_index], core_data[phi_index],
                                                     theta, rho,
                                                     grid_points, lcfs_points,
                                                     vessel_radius, l_parallel_0,
                                                     delta_phi_0w, delta_phi_sol,
                                                     alpha, sol_beta)
                output[phi_index] = plane
                boundary_all[phi_index] = diagnostics["boundary"]
                normal_all[phi_index] = diagnostics["normal"]
                lambda_phi_0_all[phi_index] = diagnostics["lambda_phi_0"]
                lambda_phi_min_all[phi_index] = diagnostics["lambda_phi_min"]
                lambda_phi_max_all[phi_index] = diagnostics["lambda_phi_max"]
                slope_all[phi_index] = diagnostics["slope"]
                wall_distance_all[phi_index] = diagnostics["path_wall_distance"]
                bridge_width_all[phi_index] = diagnostics["bridge_width"]
                chi_wall_all[phi_index] = diagnostics["chi_wall"]
                sim_io.log.info(f"Constructed phi=%03.0f: %d interior/%d exterior cells, lambda_phi_0 %.6g/%.6g/%.6g m, bridge "
                                "%.6g/%.6g/%.6g m, chi_w min %.6g, potential %.6g to %.6g.",
                                phi_deg[phi_index], diagnostics["inside_cells"], diagnostics["outside_cells"],
                                np.min(diagnostics["lambda_phi_0"]), np.median(diagnostics["lambda_phi_0"]), np.max(diagnostics["lambda_phi_0"]),
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
        "phi_wall": np.array(PHI_WALL),
        "sol_beta": np.array(sol_beta),
        "lcfs_index": np.array(lcfs_index),
        "vessel_radius_m": np.array(vessel_radius),
    }
    return np.load(output_path, mmap_mode="r"), metadata

# unique function (assumed)
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

# unique function (assumed)
def generate_plots(field, rho, theta, phi_deg, boundaries, vessel_radius, color_scale, plot_vmin, plot_vmax, show_lcfs, sim_io, output_subdir, show_progress,):
    _, grid_x, grid_z, _ = make_grid(rho, theta)
    progress = tqdm(range(phi_deg.size), desc="Plotting piecewise potential", unit="plane", dynamic_ncols=True, disable=not show_progress,)
    log_context = (logging_redirect_tqdm(loggers=[sim_io.log]) if show_progress else nullcontext())
    with log_context:
        for phi_index in progress:
            plot_plane(np.asarray(field[phi_index]), grid_x, grid_z, boundaries[phi_index], vessel_radius,
                       float(phi_deg[phi_index]), color_scale, plot_vmin, plot_vmax, show_lcfs, sim_io, output_subdir)
            if phi_index % 10 == 0:
                gc.collect()

# "generate_midplane_trace_plot" method
def nearest_coordinate_index(coordinates, target, name):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    index = int(np.argmin(np.abs(coordinates - target)))
    if coordinates.size > 1:
        tolerance = 0.5 * np.min(np.diff(coordinates)) + 1e-12
    else:
        tolerance = 1e-12
    if abs(coordinates[index] - target) > tolerance:
        raise ValueError(f"Requested {name}={target:g} is not represented by the grid.")
    
    return index

# unique function (assumed)
def generate_midplane_trace_plot(field, rho, theta, phi_deg, vessel_radius, delta_phi_0w, delta_phi_sol, sim_io, output_subdir,):
    """Plot LFS-to-HFS horizontal-midplane potential at selected phi."""
    theta_lfs_index = nearest_coordinate_index(theta, 2.0 * np.pi, "theta_LFS")
    theta_hfs_index = nearest_coordinate_index(theta, np.pi, "theta_HFS")
    distance_from_lfs = np.concatenate( (vessel_radius - rho[::-1], vessel_radius + rho[1:]) )

    fig, ax = plt.subplots(figsize=MIDPLANE_TRACE_FIGSIZE)
    for requested_phi in MIDPLANE_TRACE_PHI_DEG:
        phi_index = nearest_coordinate_index(phi_deg, requested_phi, "phi_comp",)
        potential = np.concatenate( (np.asarray(field[phi_index, theta_lfs_index, ::-1]), np.asarray(field[phi_index, theta_hfs_index, 1:])) )
        normalized_potential = (potential - PHI_WALL) / delta_phi_0w
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

# unique function-ish
def main():
    ## INPUT PREAMBLE
    args = parse_args()

    validate_settings(args.delta_phi_0w, args.delta_phi_sol, args.alpha, args.sol_beta)

    
    output_subdir = (args.output_subdir if args.output_subdir is not None else f"{args.sol_subdir}_Stitched_v2")

    poincare_settings = load_poincare_settings(args.analysis_dir)
    lcfs_index, lcfs_index_source = resolve_lcfs_index(args.lcfs_index, args.nfield_file, poincare_settings)

    l_parallel_0, l_parallel_0_source = resolve_l_parallel_0(args.l_parallel_0_m, poincare_settings)
    print(f"Using LCFS surface {lcfs_index} ({lcfs_index_source}) and "f"L_parallel,0={l_parallel_0:.6g} m")

    input_data = load_inputs(args.analysis_dir, args.sol_subdir, args.nfield_subdir, args.nfield_file)
    sol_data, core_data, rho, theta, phi_deg, sol_path, core_path = input_data

    # double-check: why are we re-saving input rho,theta,phi grids?
    for coordinate, filename in ((rho, RHO_FILENAME), (theta, THETA_FILENAME), (phi_deg, PHI_FILENAME)):
        sim_io.saveNumpyData(coordinate, filename.removesuffix(".npy"), subdir=output_subdir)

    vessel_radius = (float(rho[-1]) if VESSEL_RADIUS_M is None else float(VESSEL_RADIUS_M)) 
    if PLOT_VMIN is None:
        plot_vmin = (LOG_PLOT_VMIN if args.color_scale == "log" else PHI_WALL)
    else:
        plot_vmin = float(PLOT_VMIN)
    plot_vmax = (PHI_WALL + args.delta_phi_0w if PLOT_VMAX is None else float(PLOT_VMAX))

    if plot_vmin >= plot_vmax:
        raise ValueError("Resolved plot limits require PLOT_VMIN < PLOT_VMAX.")
    if args.color_scale == "log" and plot_vmin <= 0.0:
        raise ValueError("Logarithmic plot limits require PLOT_VMIN > 0.")
    if not np.isclose(vessel_radius, rho[-1], rtol=0.0, atol=1e-12):
        raise ValueError("The exact wall boundary requires VESSEL_RADIUS_M to equal the outermost rho grid node.")

    sim_io = IOHandler(args.analysis_dir)
    sim_io.startLog(log_name="solPotential.log", subdir=output_subdir, logger_name=output_subdir)
    output_data_dir = Path(sim_io.data_dir) / output_subdir
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_data_dir / OUTPUT_FIELD_FILENAME

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
        "PHI_WALL": PHI_WALL,
        "DELTA_PHI_0W": args.delta_phi_0w,
        "DELTA_PHI_SOL": args.delta_phi_sol,
        "DELTA_PHI_CORE": delta_phi_core,
        "ALPHA": args.alpha,
        "NFIELD_INPUT_PROFILE": "1 - Psi_bar (alpha=1)",
        "SOL_BETA": args.sol_beta,
        "L_PARALLEL_0_M": l_parallel_0,
        "L_PARALLEL_0_SOURCE": l_parallel_0_source,
        "SOL_CONNECTION_LENGTH_MIN_M": float(np.min(finite_sol)),
        "SOL_CONNECTION_LENGTH_MAX_M": float(np.max(finite_sol)),
        "VESSEL_RADIUS_M": vessel_radius,
        "BOUNDARY_RESAMPLE_POINTS": BOUNDARY_RESAMPLE_POINTS,
        "PATH_SAMPLES": PATH_SAMPLES,
        "NORMAL_DERIVATIVE_STEP_M": NORMAL_DERIVATIVE_STEP_M,
        "SURFACE_SLOPE_SMOOTHING_SIGMA": SURFACE_SLOPE_SMOOTHING_SIGMA,
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
                                     args.delta_phi_0w, args.delta_phi_sol, sim_io, output_subdir)
        sim_io.log.info("Saved horizontal-midplane potential trace: %s/%s.", output_subdir, MIDPLANE_TRACE_FILENAME)
        generate_plots(field, rho, theta, phi_deg, metadata["lcfs_boundary_xz_m"], vessel_radius,
                       args.color_scale, plot_vmin, plot_vmax, args.show_lcfs,
                       sim_io, output_subdir, args.progress)
        sim_io.log.info("Saved %d piecewise-potential plots under %s.", phi_deg.size, Path(sim_io.plot_dir) / output_subdir)

    sim_io.log.info("## PIECEWISE NFIELD / CONNECTION-LENGTH POTENTIAL FINISHED ##")


if __name__ == "__main__":
    main()
