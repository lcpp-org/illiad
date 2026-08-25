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


# dummy
def main():
    print("This file contains common function definitions for stitching connection-length potentials.")

if __name__ == "__main__":
    main()
