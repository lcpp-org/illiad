"""Batched PyTorch scrape-off-layer field-line tracing analysis.

Sparse grids outside the last-closed flux surface are launched from one or
more toroidal planes. Each field line is traced in both directions, and its
wall-to-wall connection length is attached to every captured toroidal-plane
crossing. The retained raw output stores canonical RTP crossings plus
field-line IDs in append-only plane shards; per-crossing values are resolved
from compact field-line metadata.

CUDA is used when available, with a CPU fallback for validation. The analysis
uses :class:`illiad.mesh.TorchMesh` field interpolation while owning its
fixed-step integration, wall detection, crossing capture, output, and plots.
"""

import ast
from contextlib import nullcontext
import gc
from pathlib import Path
import re
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.path import Path as MplPath
import matplotlib.tri as mtri
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from illiad.mesh import TorchMesh
import illiad.mesh.torch_mesh as torch_mesh_module
import illiad.utilities.coordtrans as coordtrans_module
from illiad.utilities.coordtrans import RTP_to_XYZ_many, XYZ_to_RTP_many
from .crossings import (
    TRACE_LCFS_INDEX_FILENAME,
    TRACE_LENGTH_LIMIT_FILENAME,
    TRACE_SPINS_FILENAME,
    TRACE_VESSEL_RADIUS_FILENAME,
    PlaneShardWriter,
    open_plane_crossing_source,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LCFS_SPLINE_SMOOTHING = 1.0e-5
LCFS_BOUNDARY_POINTS = 1000
COLOR_RANGE_CHUNK_SIZE = 1_000_000


def _parse_log_value(text):
    try: return ast.literal_eval(text)
    except (SyntaxError, ValueError): return text


def load_poincare_settings(analysis_dir, project_root=PROJECT_ROOT):
    """Load saved Poincare inputs and the identified LCFS index."""
    log_path = (Path(project_root) / "output" / analysis_dir
                 / "logs" / "Poincare" / "poincare.log")

    if not log_path.is_file(): raise FileNotFoundError(f"Poincare log not found: {log_path}")

    settings = {}
    input_pattern = re.compile(r"^\|\s*([A-Z][A-Z0-9_]+):\s*(.*?)\s*$")
    lcfs_pattern = re.compile(r"LCFS_index\s*=\s*(\d+)")
    for log_line in log_path.read_text(encoding="utf-8").splitlines():
        pipe_index = log_line.find("|")
        message = log_line[pipe_index:] if pipe_index >= 0 else log_line
        input_match = input_pattern.match(message)
        if input_match:
            settings[input_match.group(1)] = _parse_log_value(
                input_match.group(2)
            )

        lcfs_match = lcfs_pattern.search(log_line)
        if lcfs_match:
            settings["LCFS_INDEX"] = int(lcfs_match.group(1))
    return settings


def load_lcfs_boundary(analysis_dir, phi_deg, lcfs_index,
                       project_root=PROJECT_ROOT, spline_smoothing=LCFS_SPLINE_SMOOTHING, boundary_points=LCFS_BOUNDARY_POINTS):
    """Return an existing LCFS as an ordered closed poloidal curve."""
    poincare_path = (
        Path(project_root)
        / "output"
        / analysis_dir
        / "data"
        / "Poincare"
        / f"Poincare_{phi_deg:03.0f}.npy"
    )

    if not poincare_path.is_file():
        raise FileNotFoundError(f"Poincare plane data not found: {poincare_path}")

    poincare_data = np.load(poincare_path, mmap_mode="r")
    if not 0 <= lcfs_index < poincare_data.shape[0]:
        raise IndexError(f"LCFS index {lcfs_index} is outside the available surface range 0-{poincare_data.shape[0] - 1}.")

    theta, rho = poincare_data[lcfs_index]
    finite = np.isfinite(theta) & np.isfinite(rho)
    theta = np.asarray(theta[finite], dtype=np.float64)
    rho = np.asarray(rho[finite], dtype=np.float64)
    boundary = np.unique(np.column_stack((rho * np.cos(theta), rho * np.sin(theta))), axis=0)

    if boundary.shape[0] < 4:
        raise ValueError(f"LCFS surface {lcfs_index} in {poincare_path} has fewer than four unique finite points.")

    center = 0.5 * (boundary.min(axis=0) + boundary.max(axis=0))
    poloidal_angle = np.arctan2(boundary[:, 1] - center[1], boundary[:, 0] - center[0])

    boundary = boundary[np.argsort(poloidal_angle)]
    spline, _ = splprep(boundary.T, s=float(spline_smoothing), per=True)
    boundary = np.column_stack(splev(np.linspace(0.0, 1.0, int(boundary_points), endpoint=False), spline))

    return boundary, poincare_path


def minimum_boundary_distance(points, boundary):
    """Return each point's minimum Euclidean distance to LCFS segments."""
    minimum_squared = np.full(points.shape[0], np.inf)
    for start, stop in zip(boundary, np.roll(boundary, -1, axis=0)):
        segment = stop - start
        segment_length_squared = np.dot(segment, segment)
        if segment_length_squared == 0.0:
            continue

        projection = np.clip(
            ((points - start) @ segment) / segment_length_squared, 0.0, 1.0)
        closest = start + projection[:, None] * segment
        distance_squared = np.sum((points - closest) ** 2, axis=1)
        minimum_squared = np.minimum(minimum_squared, distance_squared)
    return np.sqrt(minimum_squared)


def seed_plane_degrees(n_seed_planes, seed_phi_deg, n_planes):
    """Return equally spaced seed planes selected from the output planes."""
    if (isinstance(n_seed_planes, bool) or not isinstance(n_seed_planes, int) or n_seed_planes <= 0):
        raise ValueError("N_SEED_PLANES must be a positive integer.")

    if n_seed_planes > n_planes or n_planes % n_seed_planes:
        raise ValueError(f"N_SEED_PLANES must be a positive divisor of N_PLANES={n_planes}.")

    plane_step_deg = 360.0 / n_planes
    normalized_seed_phi = seed_phi_deg % 360.0
    if np.isclose(normalized_seed_phi, 0.0):
        normalized_seed_phi = 360.0
    first_plane_number = int(np.rint(normalized_seed_phi / plane_step_deg))
    first_plane_phi = first_plane_number * plane_step_deg
    if not np.isclose(normalized_seed_phi, first_plane_phi):
        raise ValueError(f"SEED_PHI_DEG must lie on the {plane_step_deg:g}-degree output-plane grid.")

    plane_spacing = n_planes // n_seed_planes
    first_plane_index = (first_plane_number - 1) % n_planes
    plane_indices = (first_plane_index + np.arange(n_seed_planes, dtype=np.int32) * plane_spacing) % n_planes
    phi_degrees = (plane_indices + 1) * plane_step_deg
    return plane_indices, phi_degrees


def make_seed_initial_conditions(
    analysis_dir,
    lcfs_index,
    n_seed_planes,
    seed_phi_deg,
    n_rho,
    n_theta,
    rho_min,
    rho_max,
    lcfs_clearance,
    vessel_radius,
    n_planes,
    spline_smoothing=LCFS_SPLINE_SMOOTHING,
    boundary_points=LCFS_BOUNDARY_POINTS,
    project_root=PROJECT_ROOT,
):
    """Build sparse polar seed grids outside the LCFS."""
    if n_rho < 2 or n_theta < 3:
        raise ValueError("The sparse grid requires N_RHO >= 2 and N_THETA >= 3.")
    if not 0.0 <= rho_min < rho_max <= vessel_radius:
        raise ValueError("Require 0 <= RHO_MIN < RHO_MAX <= VESSEL_RADIUS_M.")
    if lcfs_clearance < 0.0:
        raise ValueError("LCFS_CLEARANCE_M must be non-negative.")

    seed_plane_indices, seed_phi_degrees = seed_plane_degrees(n_seed_planes, seed_phi_deg, n_planes)
    rho_values = np.linspace(rho_min, rho_max, n_rho)
    theta_values = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    theta_grid, rho_grid = np.meshgrid(theta_values, rho_values)
    grid_xz = np.column_stack((rho_grid.ravel() * np.cos(theta_grid.ravel()), rho_grid.ravel() * np.sin(theta_grid.ravel()),))

    condition_blocks = []
    seed_id_blocks = []
    seed_counts = []
    poincare_paths = []
    for seed_id, phi_deg in enumerate(seed_phi_degrees):
        boundary, poincare_path = load_lcfs_boundary(
            analysis_dir,
            phi_deg,
            lcfs_index,
            project_root=project_root,
            spline_smoothing=spline_smoothing,
            boundary_points=boundary_points,
        )
        closed_boundary = np.vstack((boundary, boundary[0]))
        inside_lcfs = MplPath(closed_boundary).contains_points(grid_xz)
        if lcfs_clearance > 0.0:
            near_lcfs = (minimum_boundary_distance(grid_xz, boundary) <= lcfs_clearance)
        else:
            near_lcfs = np.zeros(grid_xz.shape[0], dtype=bool)
        trace_mask = ~(inside_lcfs | near_lcfs)

        count = int(np.count_nonzero(trace_mask))
        seed_counts.append(count)
        poincare_paths.append(poincare_path)
        if count == 0:
            continue

        condition_blocks.append(
            np.column_stack(
                (rho_grid.ravel()[trace_mask], theta_grid.ravel()[trace_mask], np.full(count, np.deg2rad(phi_deg)) )
            )
        )
        seed_id_blocks.append(np.full(count, seed_id, dtype=np.int16))

    if not condition_blocks:
        raise ValueError("The LCFS masks removed every sparse-grid seed point.")

    return {
        "initial_conditions_rtp": np.vstack(condition_blocks),
        "seed_id": np.concatenate(seed_id_blocks),
        "seed_plane_index": seed_plane_indices,
        "seed_phi_deg": seed_phi_degrees,
        "seed_counts": np.asarray(seed_counts, dtype=np.int64),
        "poincare_paths": poincare_paths,
    }


def resolve_device(requested):
    """Resolve and configure the device used by TorchMesh and transforms."""
    if requested == "auto":
        selected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        selected = torch.device("cuda")
    elif requested == "cpu":
        selected = torch.device("cpu")
    else:
        raise ValueError('DEVICE must be one of "auto", "cuda", or "cpu".')

    # TorchMesh and XYZ_to_RTP2 use module-level device selectors. Updating
    # both keeps an explicit --device choice internally consistent.
    torch_mesh_module.device = selected
    coordtrans_module.device = selected
    return selected


def build_torch_magnetic_field(settings, device):
    """Build the configured magnetic field on the selected Torch device."""
    torch_mesh_module.device = device
    coordtrans_module.device = device
    magnetic_field = TorchMesh(
        R0=settings["MAJOR_RADIUS_M"],
        a=settings["VESSEL_RADIUS_M"],
    )
    magnetic_field.loadCartesianField(
        coilCurrent=settings["CURRENT_TOR"],
        errField=settings["ENABLE_ERRFIELD"],
        att_mult=settings["CONFIG_TOR"],
    )
    if settings["ENABLE_ERRFIELD"]:
        magnetic_field.setErrorField()
    magnetic_field.addFieldPerturbation(
        coilCurrent=settings["CURRENT_HEL"],
        att_mult=settings["CONFIG_HEL"],
    )
    magnetic_field._sol_error_enabled = bool(
        settings["ENABLE_ERRFIELD"]
    )
    return magnetic_field


def minor_radius(points_xyz, major_radius):
    cylindrical_radius = torch.linalg.vector_norm(points_xyz[..., :2], dim=-1)
    return torch.sqrt((cylindrical_radius - major_radius) ** 2 + points_xyz[..., 2] ** 2)


def wrapped_phi(points_xyz):
    phi = -torch.atan2(points_xyz[..., 1], points_xyz[..., 0])
    return torch.remainder(phi, 2.0 * torch.pi)


def compile_safe_weights(magnetic_field, positions):
    """Reproduce ``TorchMesh.get_weights`` with compile-safe tensor shapes."""
    x, y, z = positions.unbind(dim=1)
    cylindrical_radius = torch.sqrt(x * x + y * y)
    radius = torch.sqrt(x * x + y * y + z * z + magnetic_field.R0 * magnetic_field.R0 - 2.0 * magnetic_field.R0 * cylindrical_radius)
    theta = torch.remainder(torch.atan2(z, cylindrical_radius - magnetic_field.R0), 2.0 * torch.pi)
    phi = torch.remainder(-torch.atan2(y, x), 2.0 * torch.pi)

    theta_local = torch.remainder(theta, magnetic_field.theta_max)
    phi_local = torch.remainder(phi, magnetic_field.phi_max)
    phi_period = torch.div(phi, magnetic_field.phi_max, rounding_mode="floor")
    radius_index = torch.where(radius >= magnetic_field.r_max, magnetic_field.nr - 2, torch.div(radius, magnetic_field.dr, rounding_mode="floor"))
    radius_element = torch.remainder(radius, magnetic_field.dr)
    theta_index = torch.div(theta_local, magnetic_field.dtheta, rounding_mode="floor")
    theta_element = torch.remainder(theta_local, magnetic_field.dtheta)
    phi_index = torch.div(phi_local, magnetic_field.dphi, rounding_mode="floor")
    phi_element = torch.remainder(phi_local, magnetic_field.dphi)

    radius_low = radius_index * magnetic_field.dr
    theta_low = (theta_index + 1.0) * magnetic_field.dtheta
    inverse_radius_element = magnetic_field.dr - radius_element
    inverse_theta_element = magnetic_field.dtheta - theta_element
    inverse_phi_element = magnetic_field.dphi - phi_element
    radius_low_element = (radius_low + 0.5 * radius_element) * radius_element
    radius_inverse_element = (radius + 0.5 * inverse_radius_element) * inverse_radius_element

    low_theta_factor = magnetic_field.R0 + radius_low * torch.cos(theta_low)
    high_theta_factor = magnetic_field.R0 + radius * torch.cos(theta_low)
    low_local_factor = magnetic_field.R0 + radius_low * torch.cos(theta)
    high_local_factor = magnetic_field.R0 + radius * torch.cos(theta)
    weights = torch.stack(
        (
            low_theta_factor * radius_low_element * theta_element * phi_element,
            high_theta_factor * radius_inverse_element * theta_element * phi_element,
            low_local_factor * radius_low_element * inverse_theta_element * phi_element,
            high_local_factor * radius_inverse_element * inverse_theta_element * phi_element,
            low_theta_factor * radius_low_element * theta_element * inverse_phi_element,
            high_theta_factor * radius_inverse_element * theta_element * inverse_phi_element,
            low_local_factor * radius_low_element * inverse_theta_element * inverse_phi_element,
            high_local_factor * radius_inverse_element * inverse_theta_element * inverse_phi_element,
        )
    )

    radius_high = (radius_index + 1).to(torch.int32)
    radius_low_index = radius_index.to(torch.int32)
    theta_high = theta_index.to(torch.int32)
    theta_low_index = (theta_index - 1).to(torch.int32)
    phi_high = phi_index.to(torch.int32)
    phi_low = (phi_index - 1).to(torch.int32)
    corner_indices = torch.stack(
        (
            torch.stack(
                (
                    radius_high,
                    radius_low_index,
                    radius_high,
                    radius_low_index,
                    radius_high,
                    radius_low_index,
                    radius_high,
                    radius_low_index,
                )
            ),
            torch.stack(
                (
                    theta_high,
                    theta_high,
                    theta_low_index,
                    theta_low_index,
                    theta_high,
                    theta_high,
                    theta_low_index,
                    theta_low_index,
                )
            ),
            torch.stack(
                (
                    phi_high,
                    phi_high,
                    phi_high,
                    phi_high,
                    phi_low,
                    phi_low,
                    phi_low,
                    phi_low,
                )
            ),
        )
    )
    return weights, corner_indices, phi_period


def compile_safe_interp_field(magnetic_field, positions):
    """Evaluate ``TorchMesh`` without data-dependent Python branches."""
    weights, corner_indices, phi_period = compile_safe_weights(magnetic_field, positions)
    phi_high = corner_indices[2, 0]
    corner_vectors = torch.movedim(magnetic_field.B[corner_indices[0], corner_indices[1], corner_indices[2],], -1, 1)

    lower = corner_vectors[4:]
    cos_phi = torch.cos(magnetic_field.phi_max)
    sin_phi = torch.sin(magnetic_field.phi_max)
    rotated_lower = torch.stack(
        (
            cos_phi * lower[:, 0] - sin_phi * lower[:, 1],
            sin_phi * lower[:, 0] + cos_phi * lower[:, 1],
            lower[:, 2],
        ),
        dim=1,
    )
    corner_vectors = torch.cat(
        (
            corner_vectors[:4],
            torch.where(
                (phi_high == 0)[None, None, :],
                rotated_lower,
                lower,
            ),
        ),
        dim=0,
    )

    total_volume = weights.sum(dim=0)
    vectors = ((corner_vectors * weights[:, None]).sum(dim=0) / total_volume[None, :])

    phi_rotation = -phi_period * magnetic_field.phi_max
    cos_rotation = torch.cos(phi_rotation)
    sin_rotation = torch.sin(phi_rotation)
    vectors = torch.stack(
        (
            cos_rotation * vectors[0] - sin_rotation * vectors[1],
            sin_rotation * vectors[0] + cos_rotation * vectors[1],
            vectors[2],
        ),
        dim=0,
    )
    if magnetic_field._sol_error_enabled:
        vectors = vectors + magnetic_field.err_adder[:, None]
    return vectors


def normalized_field_direction(
    magnetic_field,
    positions,
    direction_sign,
    minimum_field_magnitude,
):
    """Evaluate batched +/- B/|B| and identify invalid field samples."""
    field_vectors = compile_safe_interp_field(
        magnetic_field,
        positions,
    ).transpose(0, 1)
    field_magnitude = torch.linalg.vector_norm(field_vectors, dim=1)
    valid = (
        torch.all(torch.isfinite(field_vectors), dim=1)
        & torch.isfinite(field_magnitude)
        & (field_magnitude > minimum_field_magnitude)
    )
    safe_magnitude = torch.where(
        valid,
        field_magnitude,
        torch.ones_like(field_magnitude),
    )
    derivative = (
        field_vectors / safe_magnitude[:, None] * direction_sign[:, None]
    )
    derivative = torch.where(valid[:, None], derivative, 0.0)
    return derivative, valid


def fieldline_step(
    magnetic_field,
    positions,
    direction_sign,
    step_size,
    integrator,
    minimum_field_magnitude,
):
    """Advance a batch by one fixed path-length step."""
    if integrator == "midpoint":
        k1, valid1 = normalized_field_direction(
            magnetic_field,
            positions,
            direction_sign,
            minimum_field_magnitude,
        )
        k2, valid2 = normalized_field_direction(
            magnetic_field,
            positions + 0.5 * step_size * k1,
            direction_sign,
            minimum_field_magnitude,
        )
        next_positions = positions + step_size * k2
        valid = valid1 & valid2
    elif integrator == "rk4":
        k1, valid1 = normalized_field_direction(
            magnetic_field,
            positions,
            direction_sign,
            minimum_field_magnitude,
        )
        k2, valid2 = normalized_field_direction(
            magnetic_field,
            positions + 0.5 * step_size * k1,
            direction_sign,
            minimum_field_magnitude,
        )
        k3, valid3 = normalized_field_direction(
            magnetic_field,
            positions + 0.5 * step_size * k2,
            direction_sign,
            minimum_field_magnitude,
        )
        k4, valid4 = normalized_field_direction(
            magnetic_field,
            positions + step_size * k3,
            direction_sign,
            minimum_field_magnitude,
        )
        next_positions = positions + (step_size / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
        valid = valid1 & valid2 & valid3 & valid4
    else:
        raise ValueError('INTEGRATOR must be either "rk4" or "midpoint".')

    valid &= torch.all(torch.isfinite(next_positions), dim=1)
    return next_positions, valid


def make_step_chunk_function(
    magnetic_field,
    chunk_size,
    integrator,
    minimum_field_magnitude,
):
    """Create a fixed-size stepping kernel for execution between host checks."""
    def step_chunk(positions, direction_sign, step_sizes):
        current = positions
        alive = torch.ones(
            positions.shape[0],
            dtype=torch.bool,
            device=positions.device,
        )
        starts = []
        candidates = []
        valid_segments = []
        wall_hits = []
        invalid_segments = []

        for local_step in range(chunk_size):
            step_size = step_sizes[local_step]
            enabled = step_size > 0.0
            start = current
            candidate, field_valid = fieldline_step(
                magnetic_field,
                start,
                direction_sign,
                step_size,
                integrator,
                minimum_field_magnitude,
            )
            evaluated = alive & enabled
            segment_valid = evaluated & field_valid
            invalid = evaluated & ~field_valid
            hit_wall = segment_valid & (
                minor_radius(candidate, magnetic_field.R0) >= magnetic_field.a
            )
            survives = segment_valid & ~hit_wall
            current = torch.where(survives[:, None], candidate, current)
            alive = torch.where(enabled, survives, alive)

            starts.append(start)
            candidates.append(candidate)
            valid_segments.append(segment_valid)
            wall_hits.append(hit_wall)
            invalid_segments.append(invalid)

        return (
            torch.stack(starts),
            torch.stack(candidates),
            torch.stack(valid_segments),
            torch.stack(wall_hits),
            torch.stack(invalid_segments),
            current,
            alive,
        )

    return step_chunk


class StepChunkRunner:
    """Compile a chunk kernel when possible and fall back cleanly to eager."""

    def __init__(
        self,
        magnetic_field,
        chunk_size,
        compile_chunks,
        integrator,
        minimum_field_magnitude,
        sim_io,
    ):
        self.eager = make_step_chunk_function(
            magnetic_field,
            chunk_size,
            integrator,
            minimum_field_magnitude,
        )
        self.function = self.eager
        self.compiled = False
        self.sim_io = sim_io
        if compile_chunks and magnetic_field.B.device.type == "cuda":
            self.function = torch.compile(
                self.eager,
                dynamic=True,
                fullgraph=True,
            )
            self.compiled = True

    def __call__(self, positions, direction_sign, step_sizes):
        try:
            return self.function(positions, direction_sign, step_sizes)
        except Exception as exc:
            if not self.compiled:
                raise
            self.sim_io.log.warning("Compiled step chunks are unavailable; falling back to eager "
                "chunk execution: %s", exc)
            self.function = self.eager
            self.compiled = False
            return self.function(positions, direction_sign, step_sizes)


def wall_intersections(
    start_xyz,
    stop_xyz,
    major_radius,
    vessel_radius,
    bisection_steps,
):
    """Refine line-segment wall intersections with batched bisection."""
    low = torch.zeros(start_xyz.shape[0], dtype=start_xyz.dtype, device=start_xyz.device)
    high = torch.ones_like(low)
    segment = stop_xyz - start_xyz
    for _ in range(bisection_steps):
        middle = 0.5 * (low + high)
        middle_xyz = start_xyz + middle[:, None] * segment
        middle_inside = minor_radius(middle_xyz, major_radius) < vessel_radius
        low = torch.where(middle_inside, middle, low)
        high = torch.where(middle_inside, high, middle)
    intersections = start_xyz + high[:, None] * segment
    return intersections, high


class PlaneCrossingStore:
    """Buffer GPU plane crossings and stream flushed blocks to a CPU sink."""

    def __init__(self, n_planes, capacity, device, sink, dtype=torch.float64):
        if capacity <= 0:
            raise ValueError("CROSSING_BUFFER_SIZE must be positive.")
        self.n_planes = n_planes
        self.capacity = capacity
        self.device = device
        self.sink = sink
        self.xyz_buffer = torch.empty((capacity, 3), dtype=dtype, device=device)
        self.plane_buffer = torch.empty(capacity, dtype=torch.int16, device=device)
        self.fieldline_buffer = torch.empty(
            capacity,
            dtype=torch.int64,
            device=device,
        )
        self.direction_buffer = torch.empty(
            capacity,
            dtype=torch.int8,
            device=device,
        )
        self.cursor = 0
        self.counts = np.zeros(n_planes, dtype=np.int64)

    @property
    def count(self):
        return int(self.counts.sum()) + self.cursor

    def append(self, xyz, plane_index, fieldline_id, source_direction):
        input_cursor = 0
        input_count = xyz.shape[0]
        while input_cursor < input_count:
            available = self.capacity - self.cursor
            take = min(available, input_count - input_cursor)
            target = slice(self.cursor, self.cursor + take)
            source = slice(input_cursor, input_cursor + take)
            self.xyz_buffer[target] = xyz[source]
            self.plane_buffer[target] = plane_index[source]
            self.fieldline_buffer[target] = fieldline_id[source]
            self.direction_buffer[target] = source_direction[source]
            self.cursor += take
            input_cursor += take
            if self.cursor == self.capacity:
                self.flush()

    def flush(self):
        if self.cursor == 0:
            return

        xyz = self.xyz_buffer[: self.cursor].cpu().numpy()
        plane = self.plane_buffer[: self.cursor].cpu().numpy()
        fieldline_id = self.fieldline_buffer[: self.cursor].cpu().numpy()
        source_direction = self.direction_buffer[: self.cursor].cpu().numpy()

        order = np.argsort(plane, kind="stable")
        xyz = xyz[order]
        plane = plane[order]
        fieldline_id = fieldline_id[order]
        source_direction = source_direction[order]
        unique_planes, starts, counts = np.unique(
            plane,
            return_index=True,
            return_counts=True,
        )
        for plane_index, start, count in zip(unique_planes, starts, counts):
            stop = start + count
            plane_index = int(plane_index)
            self.sink.append_xyz(
                plane_index,
                xyz[start:stop],
                fieldline_id[start:stop],
                source_direction[start:stop],
            )
            self.counts[plane_index] += count
        self.cursor = 0

    def finish(self):
        self.flush()
        return self


def capture_plane_crossings(
    start_xyz,
    stop_xyz,
    start_phi_unwrapped,
    stop_phi_unwrapped,
    fieldline_id,
    source_direction,
    crossing_store,
    valid_segment=None,
):
    """Linearly reconstruct at most one plane crossing per short step."""
    plane_spacing = (2.0 * torch.pi) / crossing_store.n_planes
    epsilon = 1.0e-10
    delta_phi = stop_phi_unwrapped - start_phi_unwrapped

    positive_id = torch.floor(
        (start_phi_unwrapped + epsilon) / plane_spacing
    ).to(torch.int64) + 1
    positive_target = positive_id * plane_spacing
    positive_crossing = (
        (delta_phi > epsilon)
        & (stop_phi_unwrapped + epsilon >= positive_target)
    )

    negative_id = torch.ceil(
        (start_phi_unwrapped - epsilon) / plane_spacing
    ).to(torch.int64) - 1
    negative_target = negative_id * plane_spacing
    negative_crossing = (
        (delta_phi < -epsilon)
        & (stop_phi_unwrapped - epsilon <= negative_target)
    )

    crossing = positive_crossing | negative_crossing
    if valid_segment is not None:
        crossing &= valid_segment

    crossing_id = torch.where(positive_crossing, positive_id, negative_id)
    crossing_target = crossing_id * plane_spacing
    safe_delta_phi = torch.where(
        torch.abs(delta_phi) > epsilon,
        delta_phi,
        torch.ones_like(delta_phi),
    )
    alpha = (crossing_target - start_phi_unwrapped) / safe_delta_phi
    alpha = torch.clamp(alpha, 0.0, 1.0)
    crossing_xyz = start_xyz + alpha[:, None] * (stop_xyz - start_xyz)
    plane_index = torch.remainder(
        crossing_id - 1,
        crossing_store.n_planes,
    ).to(torch.int16)
    crossing_store.append(
        crossing_xyz[crossing],
        plane_index[crossing],
        fieldline_id[crossing],
        source_direction[crossing],
    )


def integrate_directional_batch(
    initial_xyz,
    initial_phi,
    fieldline_id,
    direction_column,
    direction_sign,
    magnetic_field,
    max_length,
    step_size,
    crossing_store,
    sim_io,
    batch_number,
    batch_count,
    show_progress,
    chunk_size,
    compile_chunks,
    integrator,
    minimum_field_magnitude,
    wall_bisection_steps,
    progress_refresh_steps,
    progress_interval_steps,
    chunk_runner=None,
):
    """Integrate one batch and return directional trace metadata."""
    device = magnetic_field.B.device
    positions = torch.as_tensor(
        initial_xyz,
        dtype=torch.float64,
        device=device,
    ).clone()
    phi_unwrapped = torch.as_tensor(
        initial_phi,
        dtype=torch.float64,
        device=device,
    ).clone()
    fieldline_id = torch.as_tensor(fieldline_id, dtype=torch.int64, device=device)
    direction_column = torch.as_tensor(
        direction_column,
        dtype=torch.int64,
        device=device,
    )
    direction_sign = torch.as_tensor(
        direction_sign,
        dtype=torch.float64,
        device=device,
    )
    source_direction = direction_sign.to(torch.int8)

    batch_size = positions.shape[0]
    path_length = torch.zeros(batch_size, dtype=torch.float64, device=device)
    wall_xyz = torch.full(
        (batch_size, 3),
        torch.nan,
        dtype=torch.float64,
        device=device,
    )
    hit_wall = torch.zeros(batch_size, dtype=torch.bool, device=device)
    reached_limit = torch.zeros(batch_size, dtype=torch.bool, device=device)
    failed = torch.zeros(batch_size, dtype=torch.bool, device=device)
    running = torch.arange(batch_size, dtype=torch.int64, device=device)
    max_steps = int(np.ceil(max_length / step_size))
    if chunk_runner is None:
        chunk_runner = StepChunkRunner(
            magnetic_field,
            chunk_size,
            compile_chunks,
            integrator,
            minimum_field_magnitude,
            sim_io,
        )

    sim_io.log.info(
        "Torch batch %d/%d: %d directional traces, up to %d integration "
        "steps in %d-step GPU chunks.",
        batch_number,
        batch_count,
        batch_size,
        max_steps,
        chunk_size,
    )
    start_time = perf_counter()
    progress_bar = tqdm(
        total=max_steps,
        desc=f"Torch batch {batch_number}/{batch_count}",
        unit="step",
        dynamic_ncols=True,
        mininterval=5.0,
        disable=not show_progress,
    )
    log_context = (
        logging_redirect_tqdm(loggers=[sim_io.log])
        if show_progress
        else nullcontext()
    )
    with torch.inference_mode(), log_context:
        for chunk_start in range(0, max_steps, chunk_size):
            if running.numel() == 0:
                break

            active_global = running
            active_xyz = positions[active_global]
            active_direction = direction_sign[active_global]
            steps_this_chunk = min(chunk_size, max_steps - chunk_start)
            step_sizes_cpu = np.zeros(chunk_size, dtype=np.float64)
            for local_step in range(steps_this_chunk):
                distance_before_step = (chunk_start + local_step) * step_size
                step_sizes_cpu[local_step] = min(
                    step_size,
                    max_length - distance_before_step,
                )
            step_sizes = torch.as_tensor(
                step_sizes_cpu,
                dtype=torch.float64,
                device=device,
            )
            (
                segment_start,
                segment_stop,
                segment_valid,
                segment_hit_wall,
                segment_invalid,
                final_positions,
                survives_chunk,
            ) = chunk_runner(
                active_xyz,
                active_direction,
                step_sizes,
            )

            hit_step, hit_row = torch.nonzero(
                segment_hit_wall,
                as_tuple=True,
            )
            wall_points, wall_fraction = wall_intersections(
                segment_start[hit_step, hit_row],
                segment_stop[hit_step, hit_row],
                magnetic_field.R0,
                magnetic_field.a,
                wall_bisection_steps,
            )
            segment_endpoint = segment_stop.clone()
            segment_endpoint[hit_step, hit_row] = wall_points
            final_positions[hit_row] = wall_points

            start_phi_wrapped = wrapped_phi(segment_start)
            endpoint_phi_wrapped = wrapped_phi(segment_endpoint)
            delta_phi = torch.atan2(
                torch.sin(endpoint_phi_wrapped - start_phi_wrapped),
                torch.cos(endpoint_phi_wrapped - start_phi_wrapped),
            )
            delta_phi = torch.where(segment_valid, delta_phi, 0.0)
            cumulative_delta = torch.cumsum(delta_phi, dim=0)
            start_phi_unwrapped = (
                phi_unwrapped[active_global][None, :]
                + cumulative_delta
                - delta_phi
            )
            endpoint_phi_unwrapped = start_phi_unwrapped + delta_phi

            segment_fieldline = fieldline_id[active_global][None, :].expand(
                chunk_size,
                -1,
            )
            segment_direction = source_direction[active_global][None, :].expand(
                chunk_size,
                -1,
            )
            capture_plane_crossings(
                segment_start.reshape(-1, 3),
                segment_endpoint.reshape(-1, 3),
                start_phi_unwrapped.reshape(-1),
                endpoint_phi_unwrapped.reshape(-1),
                segment_fieldline.reshape(-1),
                segment_direction.reshape(-1),
                crossing_store,
                valid_segment=segment_valid.reshape(-1),
            )

            length_increment = (
                segment_valid.to(torch.float64) * step_sizes[:, None]
            )
            length_increment[hit_step, hit_row] = (
                step_sizes[hit_step] * wall_fraction
            )
            path_length[active_global] += length_increment.sum(dim=0)
            positions[active_global] = final_positions
            phi_unwrapped[active_global] += cumulative_delta[-1]

            wall_global = active_global[hit_row]
            hit_wall[wall_global] = True
            wall_xyz[wall_global] = wall_points

            invalid_rows = torch.any(segment_invalid, dim=0)
            invalid_global = active_global[invalid_rows]
            failed[invalid_global] = True

            completed_steps = chunk_start + steps_this_chunk
            if completed_steps == max_steps:
                reached_limit[active_global[survives_chunk]] = True
                running = running[:0]
            else:
                running = active_global[survives_chunk]

            if show_progress:
                progress_bar.update(steps_this_chunk)

            if show_progress and (
                completed_steps % progress_refresh_steps < steps_this_chunk
                or running.numel() == 0
            ):
                progress_bar.set_postfix(
                    active=running.numel(),
                    walls=torch.count_nonzero(hit_wall).item(),
                    crossings=crossing_store.count,
                    refresh=False,
                )

            if (
                completed_steps % progress_interval_steps < steps_this_chunk
                or running.numel() == 0
            ):
                elapsed = perf_counter() - start_time
                sim_io.log.info(
                    "Torch batch %d/%d step %d/%d: %d active, "
                    "%d wall hits, %d failed, %.1f sec.",
                    batch_number,
                    batch_count,
                    completed_steps,
                    max_steps,
                    running.numel(),
                    torch.count_nonzero(hit_wall).item(),
                    torch.count_nonzero(failed).item(),
                    elapsed,
                )

            if running.numel() == 0 and completed_steps < max_steps:
                if show_progress:
                    progress_bar.total = progress_bar.n
                break

    progress_bar.close()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - start_time
    sim_io.log.info(
        "Torch batch %d/%d finished in %.3f sec.",
        batch_number,
        batch_count,
        elapsed,
    )
    return {
        "fieldline_id": fieldline_id.cpu().numpy(),
        "direction_column": direction_column.cpu().numpy(),
        "path_length": path_length.cpu().numpy(),
        "wall_xyz": wall_xyz.cpu().numpy(),
        "hit_wall": hit_wall.cpu().numpy(),
        "reached_limit": reached_limit.cpu().numpy(),
        "valid_trace": (hit_wall | reached_limit).cpu().numpy(),
    }


def trace_connection_length_volume(
    seed_data,
    settings,
    magnetic_field,
    step_size,
    batch_size,
    crossing_store,
    sim_io,
    show_progress,
    chunk_size,
    compile_chunks,
    integrator,
    minimum_field_magnitude,
    wall_bisection_steps,
    progress_refresh_steps,
    progress_interval_steps,
):
    """Trace every seed in both directions in batched Torch integrations."""
    initial_conditions_rtp = seed_data["initial_conditions_rtp"]
    n_fieldlines = initial_conditions_rtp.shape[0]
    if n_fieldlines > np.iinfo(np.int32).max:
        raise ValueError("The fieldline count exceeds the int32 raw-ID range.")
    fieldline_id = np.concatenate( (np.arange(n_fieldlines, dtype=np.int64), np.arange(n_fieldlines, dtype=np.int64)) )
    direction_column = np.concatenate( (np.zeros(n_fieldlines, dtype=np.int64), np.ones(n_fieldlines, dtype=np.int64)) )
    direction_sign = np.concatenate( (np.ones(n_fieldlines, dtype=np.float64), -np.ones(n_fieldlines, dtype=np.float64)) )
    directional_rtp = initial_conditions_rtp[fieldline_id]
    directional_xyz = RTP_to_XYZ_many(directional_rtp, magnetic_field.R0)
    directional_phi = directional_rtp[:, 2]
    max_length = 2.0 * np.pi * magnetic_field.R0 * settings["SPINS"]

    direction_length = np.full((n_fieldlines, 2), np.nan)
    wall_xyz = np.full((n_fieldlines, 2, 3), np.nan)
    hit_wall = np.zeros((n_fieldlines, 2), dtype=bool)
    reached_limit = np.zeros((n_fieldlines, 2), dtype=bool)
    valid_trace = np.zeros((n_fieldlines, 2), dtype=bool)

    directional_count = directional_xyz.shape[0]
    batch_count = int(np.ceil(directional_count / batch_size))
    chunk_runner = StepChunkRunner(magnetic_field, chunk_size, compile_chunks, integrator, minimum_field_magnitude, sim_io)

    trace_start = perf_counter()
    for batch_index, start in enumerate(range(0, directional_count, batch_size), start=1):
        stop = min(start + batch_size, directional_count)
        result = integrate_directional_batch(
            directional_xyz[start:stop],
            directional_phi[start:stop],
            fieldline_id[start:stop],
            direction_column[start:stop],
            direction_sign[start:stop],
            magnetic_field,
            max_length,
            step_size,
            crossing_store,
            sim_io,
            batch_index,
            batch_count,
            show_progress,
            chunk_size,
            compile_chunks,
            integrator,
            minimum_field_magnitude,
            wall_bisection_steps,
            progress_refresh_steps,
            progress_interval_steps,
            chunk_runner,
        )
        result_index = (result["fieldline_id"], result["direction_column"])
        direction_length[result_index] = result["path_length"]
        wall_xyz[result_index] = result["wall_xyz"]
        hit_wall[result_index] = result["hit_wall"]
        reached_limit[result_index] = result["reached_limit"]
        valid_trace[result_index] = result["valid_trace"]

    crossing_store.finish()
    connection_length = np.sum(direction_length, axis=1)
    connection_length[~np.all(valid_trace, axis=1)] = np.nan
    wall_rtp = np.full_like(wall_xyz, np.nan)
    finite_wall = np.all(np.isfinite(wall_xyz), axis=-1)
    wall_rtp[finite_wall] = XYZ_to_RTP_many(wall_xyz[finite_wall], magnetic_field.R0)
    trace_elapsed = perf_counter() - trace_start
    sim_io.log.info("ALL TORCH SOLVERS FINISHED IN %.3f seconds; captured %d crossings.", trace_elapsed, crossing_store.counts.sum())
    return {
        "direction_length": direction_length,
        "connection_length": connection_length,
        "wall_xyz": wall_xyz,
        "wall_rtp": wall_rtp,
        "hit_wall": hit_wall,
        "reached_limit": reached_limit,
        "valid_trace": valid_trace,
        "trace_length_limit_m": float(max_length),
        "trace_spins": int(settings["SPINS"]),
        "trace_lcfs_index": int(settings["LCFS_INDEX"]),
        "trace_vessel_radius_m": float(settings["VESSEL_RADIUS_M"]),
        "plane_phi_deg": np.linspace(360.0 / crossing_store.n_planes, 360.0, crossing_store.n_planes),
    }


def trace_connection_length_volume_paired(
    seed_data,
    settings,
    magnetic_field,
    step_size,
    batch_size,
    crossing_buffer_size,
    sink_factory,
    accumulator,
    raw_chunk_size,
    sim_io,
    show_progress,
    chunk_size,
    compile_chunks,
    integrator,
    minimum_field_magnitude,
    wall_bisection_steps,
    progress_refresh_steps,
    progress_interval_steps,
):
    """Trace paired directions and resolve each bounded crossing spool."""
    initial_conditions_rtp = seed_data["initial_conditions_rtp"]
    n_fieldlines = initial_conditions_rtp.shape[0]
    if n_fieldlines > np.iinfo(np.int32).max:
        raise ValueError("The fieldline count exceeds the int32 spool-ID range.")
    max_length = 2.0 * np.pi * magnetic_field.R0 * settings["SPINS"]
    fieldlines_per_batch = max(1, batch_size // 2)
    batch_count = int(np.ceil(n_fieldlines / fieldlines_per_batch))
    chunk_runner = StepChunkRunner(
        magnetic_field,
        chunk_size,
        compile_chunks,
        integrator,
        minimum_field_magnitude,
        sim_io,
    )

    direction_length = np.full((n_fieldlines, 2), np.nan)
    connection_length = np.full(n_fieldlines, np.nan)
    wall_xyz = np.full((n_fieldlines, 2, 3), np.nan)
    hit_wall = np.zeros((n_fieldlines, 2), dtype=bool)
    reached_limit = np.zeros((n_fieldlines, 2), dtype=bool)
    valid_trace = np.zeros((n_fieldlines, 2), dtype=bool)
    seed_plane_for_fieldline = seed_data["seed_plane_index"][
        seed_data["seed_id"]
    ]
    total_crossings = 0
    trace_start = perf_counter()

    for batch_number, start in enumerate(
        range(0, n_fieldlines, fieldlines_per_batch),
        start=1,
    ):
        stop = min(start + fieldlines_per_batch, n_fieldlines)
        batch_fieldline = np.arange(start, stop, dtype=np.int64)
        fieldline_id = np.concatenate((batch_fieldline, batch_fieldline))
        direction_column = np.concatenate(
            (
                np.zeros(batch_fieldline.size, dtype=np.int64),
                np.ones(batch_fieldline.size, dtype=np.int64),
            )
        )
        direction_sign = np.concatenate(
            (
                np.ones(batch_fieldline.size, dtype=np.float64),
                -np.ones(batch_fieldline.size, dtype=np.float64),
            )
        )
        directional_rtp = initial_conditions_rtp[fieldline_id]
        directional_xyz = RTP_to_XYZ_many(
            directional_rtp,
            magnetic_field.R0,
        )
        sink = sink_factory(batch_number)
        crossing_store = PlaneCrossingStore(
            settings["N_PLANES"],
            crossing_buffer_size,
            magnetic_field.B.device,
            sink,
        )
        try:
            result = integrate_directional_batch(
                directional_xyz,
                directional_rtp[:, 2],
                fieldline_id,
                direction_column,
                direction_sign,
                magnetic_field,
                max_length,
                step_size,
                crossing_store,
                sim_io,
                batch_number,
                batch_count,
                show_progress,
                chunk_size,
                compile_chunks,
                integrator,
                minimum_field_magnitude,
                wall_bisection_steps,
                progress_refresh_steps,
                progress_interval_steps,
                chunk_runner,
            )
            crossing_store.finish()
            result_index = (
                result["fieldline_id"],
                result["direction_column"],
            )
            direction_length[result_index] = result["path_length"]
            wall_xyz[result_index] = result["wall_xyz"]
            hit_wall[result_index] = result["hit_wall"]
            reached_limit[result_index] = result["reached_limit"]
            valid_trace[result_index] = result["valid_trace"]
            batch_length = np.sum(direction_length[batch_fieldline], axis=1)
            batch_valid = np.all(valid_trace[batch_fieldline], axis=1)
            batch_length[~batch_valid] = np.nan
            connection_length[batch_fieldline] = batch_length

            for plane_index in np.unique(
                seed_plane_for_fieldline[batch_fieldline]
            ):
                on_plane = (
                    seed_plane_for_fieldline[batch_fieldline] == plane_index
                )
                seed_fieldline = batch_fieldline[on_plane]
                sink.append_rtp(
                    int(plane_index),
                    initial_conditions_rtp[seed_fieldline],
                    seed_fieldline,
                    np.zeros(seed_fieldline.size, dtype=np.int8),
                )
            sink.consume(
                accumulator,
                connection_length,
                raw_chunk_size,
            )
            total_crossings += int(crossing_store.counts.sum())
        except Exception:
            sink.abort()
            raise

    wall_rtp = np.full_like(wall_xyz, np.nan)
    finite_wall = np.all(np.isfinite(wall_xyz), axis=-1)
    wall_rtp[finite_wall] = XYZ_to_RTP_many(
        wall_xyz[finite_wall],
        magnetic_field.R0,
    )
    sim_io.log.info(
        "ALL PAIRED TORCH SOLVERS FINISHED IN %.3f seconds; processed %d "
        "crossings through bounded batch spools.",
        perf_counter() - trace_start,
        total_crossings,
    )
    return {
        "direction_length": direction_length,
        "connection_length": connection_length,
        "wall_xyz": wall_xyz,
        "wall_rtp": wall_rtp,
        "hit_wall": hit_wall,
        "reached_limit": reached_limit,
        "valid_trace": valid_trace,
        "trace_length_limit_m": float(max_length),
        "trace_spins": int(settings["SPINS"]),
        "trace_lcfs_index": int(settings["LCFS_INDEX"]),
        "trace_vessel_radius_m": float(settings["VESSEL_RADIUS_M"]),
        "plane_phi_deg": np.linspace(
            360.0 / settings["N_PLANES"],
            360.0,
            settings["N_PLANES"],
        ),
    }


def save_trace_metadata(
    sim_io,
    seed_data,
    trace_data,
    major_radius,
    analysis_subdir,
):
    """Save seed and per-field-line trace metadata."""
    data_dir = Path(sim_io.data_dir) / analysis_subdir
    data_dir.mkdir(parents=True, exist_ok=True)
    small_outputs = {
        "plane_phi_deg": trace_data["plane_phi_deg"],
        "seed_initial_conditions_rtp": seed_data["initial_conditions_rtp"],
        "seed_id": seed_data["seed_id"],
        "seed_plane_index": seed_data["seed_plane_index"],
        "seed_phi_deg": seed_data["seed_phi_deg"],
        "seed_counts": seed_data["seed_counts"],
        "major_radius_m": np.asarray(major_radius),
        TRACE_LENGTH_LIMIT_FILENAME.removesuffix(".npy"): np.asarray(
            trace_data["trace_length_limit_m"]
        ),
        TRACE_SPINS_FILENAME.removesuffix(".npy"): np.asarray(
            trace_data["trace_spins"], dtype=np.int64
        ),
        TRACE_LCFS_INDEX_FILENAME.removesuffix(".npy"): np.asarray(
            trace_data["trace_lcfs_index"], dtype=np.int64
        ),
        TRACE_VESSEL_RADIUS_FILENAME.removesuffix(".npy"): np.asarray(
            trace_data["trace_vessel_radius_m"]
        ),
        "fieldline_connection_length_m": trace_data["connection_length"],
        "direction_connection_length_m": trace_data["direction_length"],
        "wall_intersection_rtp": trace_data["wall_rtp"],
        "hit_wall": trace_data["hit_wall"],
        "reached_max_length": trace_data["reached_limit"],
        "valid_trace": trace_data["valid_trace"],
    }
    for name, values in small_outputs.items():
        sim_io.saveNumpyData(values, name, subdir=analysis_subdir)
    return data_dir


def save_torch_outputs(
    sim_io,
    seed_data,
    trace_data,
    crossing_writer,
    major_radius,
    analysis_subdir,
):
    """Finish append-only raw shards and save compact trace metadata."""
    data_dir = Path(sim_io.data_dir) / analysis_subdir
    for obsolete_name in (
        "raw_points_xyz.npy",
        "raw_points_rtp.npy",
        "raw_connection_length_m.npy",
        "raw_plane_index.npy",
        "raw_fieldline_id.npy",
        "raw_source_direction.npy",
        "plane_offsets.npy",
        "wall_intersection_xyz.npy",
    ):
        (data_dir / obsolete_name).unlink(missing_ok=True)
    seed_plane_for_fieldline = seed_data["seed_plane_index"][
        seed_data["seed_id"]
    ]
    if trace_data["connection_length"].size > np.iinfo(np.int32).max:
        raise ValueError("The fieldline count exceeds the int32 raw-ID range.")

    for plane_index in range(crossing_writer.n_planes):
        seed_fieldlines = np.flatnonzero(seed_plane_for_fieldline == plane_index)
        if seed_fieldlines.size:
            crossing_writer.append_rtp(
                plane_index,
                seed_data["initial_conditions_rtp"][seed_fieldlines],
                seed_fieldlines.astype(np.int32, copy=False),
                np.zeros(seed_fieldlines.size, dtype=np.int8),
            )

    save_trace_metadata(
        sim_io,
        seed_data,
        trace_data,
        major_radius,
        analysis_subdir,
    )
    crossing_writer.finish(trace_data["connection_length"])
    trace_data["crossing_source"] = open_plane_crossing_source(data_dir)
    trace_data["fieldline_connection_length"] = trace_data[
        "connection_length"
    ]
    sim_io.log.info(
        "Saved %d raw Torch samples across %d toroidal planes: %s",
        crossing_writer.counts.sum(),
        crossing_writer.n_planes,
        data_dir,
    )
    return data_dir


def make_color_scale(values, color_scale, n_levels, value_min, value_max):
    """Resolve a shared color scale from field-line connection lengths."""
    data_min = np.inf
    data_max = -np.inf
    for start in range(0, values.size, COLOR_RANGE_CHUNK_SIZE):
        chunk = np.asarray(values[start : start + COLOR_RANGE_CHUNK_SIZE])
        positive = chunk[np.isfinite(chunk) & (chunk > 0.0)]
        if positive.size:
            data_min = min(data_min, float(positive.min()))
            data_max = max(data_max, float(positive.max()))
    if not np.isfinite(data_min):
        raise ValueError("No positive finite connection lengths are available.")

    value_min = data_min if value_min is None else float(value_min)
    value_max = data_max if value_max is None else float(value_max)
    if np.isclose(value_min, value_max):
        delta = max(0.01 * value_min, np.finfo(float).eps)
        value_min -= delta
        value_max += delta
    if value_min >= value_max:
        raise ValueError("The resolved color limits require VMIN < VMAX.")

    if color_scale == "log":
        levels = np.geomspace(value_min, value_max, n_levels)
        norm = LogNorm(vmin=value_min, vmax=value_max)
    elif color_scale == "linear":
        levels = np.linspace(value_min, value_max, n_levels)
        norm = Normalize(vmin=value_min, vmax=value_max)
    else:
        raise ValueError('COLOR_SCALE must be either "log" or "linear".')

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


def _unique_plane_samples(points_rtp, values, max_samples, sample_seed):
    finite = (
        np.all(np.isfinite(points_rtp), axis=1)
        & np.isfinite(values)
        & (values > 0.0)
    )
    points_rtp = points_rtp[finite]
    values = values[finite]
    if not points_rtp.size:
        return points_rtp, values

    if max_samples is not None and points_rtp.shape[0] > max_samples:
        sample_indices = np.sort(
            np.random.default_rng(sample_seed).choice(
                points_rtp.shape[0],
                max_samples,
                replace=False,
            )
        )
        points_rtp = points_rtp[sample_indices]
        values = values[sample_indices]

    x = points_rtp[:, 0] * np.cos(points_rtp[:, 1])
    z = points_rtp[:, 0] * np.sin(points_rtp[:, 1])
    _, unique_indices = np.unique(
        np.column_stack((x, z)),
        axis=0,
        return_index=True,
    )
    unique_indices.sort()
    return points_rtp[unique_indices], values[unique_indices]


def extend_plot_samples_to_wall(x, z, values, vessel_radius):
    """Append a plot-only wall ring using nearest-sample values in x-z space.

    Inputs must be finite, unique samples. Empty slices remain empty.
    The 720-point ring approximates the circular wall without tracing new lines.
    """
    if not len(values):
        return x, z, values
    angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    wall_points = vessel_radius * np.column_stack((np.cos(angles), np.sin(angles)))
    distances, indices = cKDTree(np.column_stack((x, z))).query(wall_points)
    # Avoid duplicate vertices when saved samples already lie on the ring.
    missing = distances > np.finfo(np.float64).eps * max(1.0, vessel_radius)
    return (
        np.concatenate((x, wall_points[missing, 0])),
        np.concatenate((z, wall_points[missing, 1])),
        np.concatenate((values, values[indices[missing]])),
    )


def plot_plane_samples(
    points_rtp,
    values,
    boundary,
    phi_deg,
    levels,
    norm,
    extend,
    sim_io,
    params,
):
    """Plot one unstructured slice with LCFS-interior triangles masked."""
    points_rtp, values = _unique_plane_samples(
        points_rtp,
        values,
        params["PLOT_MAX_SAMPLES"],
        params["PLOT_SAMPLE_SEED"],
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    color_artist = None
    if points_rtp.shape[0] >= 3:
        x = points_rtp[:, 0] * np.cos(points_rtp[:, 1])
        z = points_rtp[:, 0] * np.sin(points_rtp[:, 1])
        try:
            plot_values = values
            if params.get("PLOT_EXTEND_TO_WALL", False):
                x, z, plot_values = extend_plot_samples_to_wall(
                    x, z, values, params["VESSEL_RADIUS_M"]
                )
            triangulation = mtri.Triangulation(x, z)
            triangle_x = x[triangulation.triangles]
            triangle_z = z[triangulation.triangles]
            triangle_probes = np.stack(
                (
                    np.column_stack(
                        (triangle_x.mean(axis=1), triangle_z.mean(axis=1))
                    ),
                    np.column_stack(
                        (
                            0.5 * (triangle_x[:, 0] + triangle_x[:, 1]),
                            0.5 * (triangle_z[:, 0] + triangle_z[:, 1]),
                        )
                    ),
                    np.column_stack(
                        (
                            0.5 * (triangle_x[:, 1] + triangle_x[:, 2]),
                            0.5 * (triangle_z[:, 1] + triangle_z[:, 2]),
                        )
                    ),
                    np.column_stack(
                        (
                            0.5 * (triangle_x[:, 2] + triangle_x[:, 0]),
                            0.5 * (triangle_z[:, 2] + triangle_z[:, 0]),
                        )
                    ),
                ),
                axis=1,
            )
            closed_boundary = np.vstack((boundary, boundary[0]))
            inside_lcfs = MplPath(closed_boundary).contains_points(
                triangle_probes.reshape(-1, 2)
            ).reshape(-1, triangle_probes.shape[1]).any(axis=1)
            triangulation.set_mask(inside_lcfs)
            if np.all(inside_lcfs):
                raise ValueError("Every triangulation element is inside the LCFS.")

            color_artist = ax.tricontourf(
                triangulation,
                plot_values,
                levels=levels,
                norm=norm,
                cmap=params["COLORMAP"],
                extend=extend,
            )
        except (RuntimeError, ValueError) as exc:
            sim_io.log.warning(
                "Falling back to scatter at phi=%g deg: %s",
                phi_deg,
                exc,
            )

    if color_artist is None and points_rtp.size:
        x = points_rtp[:, 0] * np.cos(points_rtp[:, 1])
        z = points_rtp[:, 0] * np.sin(points_rtp[:, 1])
        color_artist = ax.scatter(
            x,
            z,
            c=values,
            s=4.0,
            linewidths=0.0,
            norm=norm,
            cmap=params["COLORMAP"],
        )
    elif color_artist is None:
        ax.text(
            0.5,
            0.5,
            "No connection-length samples",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    closed_boundary = np.vstack((boundary, boundary[0]))
    ax.plot(
        closed_boundary[:, 0],
        closed_boundary[:, 1],
        color="black",
        linewidth=1.0,
        label="LCFS",
    )
    vessel_radius = params["VESSEL_RADIUS_M"]
    vessel_angle = np.linspace(0.0, 2.0 * np.pi, 720)
    ax.plot(
        vessel_radius * np.cos(vessel_angle),
        vessel_radius * np.sin(vessel_angle),
        color="0.35",
        linewidth=1.0,
        label="Vessel wall",
    )

    phi_phys_deg = (phi_deg + params["PHYSICAL_PHI_OFFSET_DEG"]) % 360.0
    ax.set_title(
        "Connection length samples\n"
        f"$\\phi_{{phy}}={phi_phys_deg:03.0f}^\\circ$ CW from North split, "
        f"$\\phi_c={phi_deg:03.0f}^\\circ$"
    )
    ax.set_xlabel(r"$x=\rho\cos\theta$ [m]")
    ax.set_ylabel(r"$z=\rho\sin\theta$ [m]")
    ax.set_xlim(-vessel_radius, vessel_radius)
    ax.set_ylim(-vessel_radius, vessel_radius)
    ax.set_aspect("equal")
    ax.grid(linewidth=0.4, color="0.75")
    ax.legend(loc="upper right")
    if color_artist is not None:
        colorbar = fig.colorbar(color_artist, ax=ax, pad=0.03)
        colorbar.set_label("Connection length [m]")

    plot_name = f"connection_length_{phi_deg:03.0f}.png"
    sim_io.saveFig(
        plot_name,
        dpi=params["DPI"],
        subdir=params["ANLYS_SUBDIR"],
    )
    sim_io.log.info(
        "Saved figure with %d samples: %s/%s",
        points_rtp.shape[0],
        params["ANLYS_SUBDIR"],
        plot_name,
    )
    plt.close(fig)
    gc.collect()


def plot_all_planes(analysis_dir, lcfs_index, trace_data, sim_io, params):
    """Produce one unstructured filled-contour plot per toroidal plane."""
    fieldline_values = trace_data["fieldline_connection_length"]
    source = trace_data["crossing_source"]
    levels, norm, extend = make_color_scale(
        fieldline_values,
        params["COLOR_SCALE"],
        params["N_LEVELS"],
        params["VMIN"],
        params["VMAX"],
    )
    for plane_index, phi_deg in enumerate(trace_data["plane_phi_deg"]):
        point_blocks = []
        value_blocks = []
        for chunk in source.iter_plane_chunks(
            plane_index,
            COLOR_RANGE_CHUNK_SIZE,
        ):
            point_blocks.append(chunk.points_rtp)
            value_blocks.append(chunk.connection_length_m)
        if point_blocks:
            plane_points = np.concatenate(point_blocks)
            plane_values = np.concatenate(value_blocks)
        else:
            plane_points = np.empty((0, 3), dtype=np.float64)
            plane_values = np.empty(0, dtype=np.float64)
        boundary, _ = load_lcfs_boundary(
            analysis_dir,
            phi_deg,
            lcfs_index,
            spline_smoothing=params["LCFS_SPLINE_SMOOTHING"],
            boundary_points=params["LCFS_BOUNDARY_POINTS"],
        )
        plot_plane_samples(
            plane_points,
            plane_values,
            boundary,
            phi_deg,
            levels,
            norm,
            extend,
            sim_io,
            params,
        )


def _require_positive_integer(params, key):
    value = params[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer.")


def validate_runtime_settings(params):
    """Validate connection-length inputs before allocating large tensors."""
    if not isinstance(params.get("PLOT_EXTEND_TO_WALL", False), bool):
        raise ValueError("PLOT_EXTEND_TO_WALL must be a boolean.")
    for key in (
        "SPINS",
        "N_PLANES",
        "N_SEED_PLANES",
        "N_RHO",
        "N_THETA",
        "BATCH_SIZE",
        "CROSSING_BUFFER_SIZE",
        "STEP_CHUNK_SIZE",
        "WALL_BISECTION_STEPS",
        "PROGRESS_INTERVAL_STEPS",
        "PROGRESS_REFRESH_STEPS",
        "LCFS_BOUNDARY_POINTS",
        "N_LEVELS",
        "DPI",
    ):
        _require_positive_integer(params, key)
    if params["N_PLANES"] > np.iinfo(np.int16).max:
        raise ValueError("N_PLANES exceeds the int16 crossing-index capacity.")
    if params["N_RHO"] < 2 or params["N_THETA"] < 3:
        raise ValueError("The sparse grid requires N_RHO >= 2 and N_THETA >= 3.")
    if params["LCFS_BOUNDARY_POINTS"] < 4:
        raise ValueError("LCFS_BOUNDARY_POINTS must be at least four.")
    if params["N_LEVELS"] < 2:
        raise ValueError("N_LEVELS must be at least two.")
    if (
        isinstance(params["LCFS_INDEX"], bool)
        or not isinstance(params["LCFS_INDEX"], int)
        or params["LCFS_INDEX"] < 0
    ):
        raise ValueError("LCFS_INDEX must be a non-negative integer.")
    if (
        isinstance(params["PLOT_SAMPLE_SEED"], bool)
        or not isinstance(params["PLOT_SAMPLE_SEED"], int)
    ):
        raise ValueError("PLOT_SAMPLE_SEED must be an integer.")

    for key in (
        "COMPILE_STEP_CHUNKS",
        "SHOW_PROGRESS",
        "GENERATE_PLOTS",
        "ENABLE_ERRFIELD",
    ):
        if not isinstance(params[key], bool):
            raise ValueError(f"{key} must be a boolean.")
    if params["INTEGRATOR"] not in {"midpoint", "rk4"}:
        raise ValueError('INTEGRATOR must be either "midpoint" or "rk4".')
    if params["COLOR_SCALE"] not in {"log", "linear"}:
        raise ValueError('COLOR_SCALE must be either "log" or "linear".')

    step_size = float(params["STEP_SIZE_M"])
    major_radius = float(params["MAJOR_RADIUS_M"])
    vessel_radius = float(params["VESSEL_RADIUS_M"])
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("STEP_SIZE_M must be positive and finite.")
    if not major_radius > vessel_radius > 0.0:
        raise ValueError(
            "Require MAJOR_RADIUS_M > VESSEL_RADIUS_M > 0."
        )
    if float(params["MIN_FIELD_MAGNITUDE"]) <= 0.0:
        raise ValueError("MIN_FIELD_MAGNITUDE must be positive.")
    if float(params["LCFS_SPLINE_SMOOTHING"]) < 0.0:
        raise ValueError("LCFS_SPLINE_SMOOTHING must be non-negative.")
    if not np.isfinite(float(params["SEED_PHI_DEG"])):
        raise ValueError("SEED_PHI_DEG must be finite.")
    if not np.isfinite(float(params["PHYSICAL_PHI_OFFSET_DEG"])):
        raise ValueError("PHYSICAL_PHI_OFFSET_DEG must be finite.")
    if not 0.0 <= float(params["RHO_MIN"]) < float(params["RHO_MAX"]):
        raise ValueError("Require 0 <= RHO_MIN < RHO_MAX.")
    if float(params["RHO_MAX"]) > vessel_radius:
        raise ValueError("RHO_MAX must not exceed VESSEL_RADIUS_M.")
    if float(params["LCFS_CLEARANCE_M"]) < 0.0:
        raise ValueError("LCFS_CLEARANCE_M must be non-negative.")

    seed_plane_degrees(
        params["N_SEED_PLANES"],
        float(params["SEED_PHI_DEG"]),
        params["N_PLANES"],
    )

    plot_max_samples = params["PLOT_MAX_SAMPLES"]
    if plot_max_samples is not None and (
        isinstance(plot_max_samples, bool)
        or not isinstance(plot_max_samples, int)
        or plot_max_samples <= 0
    ):
        raise ValueError("PLOT_MAX_SAMPLES must be null or a positive integer.")

    plane_spacing = 2.0 * np.pi / params["N_PLANES"]
    conservative_limit = (
        0.5 * (major_radius - vessel_radius) * plane_spacing
    )
    if step_size > conservative_limit:
        raise ValueError(
            f"STEP_SIZE_M={step_size:g} is too large for single-crossing "
            f"reconstruction; use <= {conservative_limit:g} m."
        )


class SOLTracer:
    """Trace and save LCFS-exterior field-line connection lengths."""

    def __init__(self, io_handler, magnetic_field, input_params):
        """Initialize the analysis from a Torch field and merged inputs."""
        self.simIO = io_handler
        self.field = magnetic_field
        self.input_params = dict(input_params)
        self.analysis_dir = self.input_params["ANLYS_DIR"]
        self.analysis_subdir = self.input_params["ANLYS_SUBDIR"]
        self.device = self.field.B.device
        self.compile_chunks = (self.input_params["COMPILE_STEP_CHUNKS"] and self.device.type == "cuda")
        if not hasattr(self.field, "_sol_error_enabled"):
            self.field._sol_error_enabled = bool(self.input_params["ENABLE_ERRFIELD"])

        self.seed_data = None
        self.trace_data = None
        self.data_dir = None

        validate_runtime_settings(self.input_params)

    def build_initial_conditions(self):
        """Build and retain all LCFS-masked seed planes."""
        params = self.input_params
        self.seed_data = make_seed_initial_conditions(
            self.analysis_dir,
            params["LCFS_INDEX"],
            params["N_SEED_PLANES"],
            params["SEED_PHI_DEG"],
            params["N_RHO"],
            params["N_THETA"],
            params["RHO_MIN"],
            params["RHO_MAX"],
            params["LCFS_CLEARANCE_M"],
            params["VESSEL_RADIUS_M"],
            params["N_PLANES"],
            spline_smoothing=params["LCFS_SPLINE_SMOOTHING"],
            boundary_points=params["LCFS_BOUNDARY_POINTS"],
        )
        return self.seed_data

    def log_inputs(self):
        """Log configured and derived settings for reproducibility."""
        params = self.input_params
        directional_solves = 2 * self.seed_data["initial_conditions_rtp"].shape[0]
        max_length = 2.0 * np.pi * self.field.R0 * params["SPINS"]
        cuda_device_name = None
        if self.device.type == "cuda":
            cuda_device_name = torch.cuda.get_device_name(self.device)
        run_settings = {
            **params,
            "SEED_PHI_DEG_RESOLVED": self.seed_data["seed_phi_deg"].tolist(),
            "SEED_COUNTS": self.seed_data["seed_counts"].tolist(),
            "SEED_POINCARE_FILES": [str(path) for path in self.seed_data["poincare_paths"]],
            "TRACED_FIELD_LINES": self.seed_data["initial_conditions_rtp"].shape[0],
            "DIRECTIONAL_SOLVES": directional_solves,
            "DOUBLE_LINE": True,
            "TORCH_VERSION": torch.__version__,
            "CUDA_AVAILABLE": torch.cuda.is_available(),
            "DEVICE_RESOLVED": str(self.device),
            "CUDA_DEVICE_NAME": cuda_device_name,
            "DTYPE": "torch.float64",
            "MAX_STEPS": int(np.ceil(max_length / params["STEP_SIZE_M"])),
            "COMPILE_STEP_CHUNKS_RESOLVED": self.compile_chunks,
        }
        self.simIO.inputsBoilerplate("SOL TRACE INPUTS", run_settings)

    def trace(self):
        """Run both field-line directions for every seed."""
        params = self.input_params
        self.data_dir = Path(self.simIO.data_dir) / self.analysis_subdir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for obsolete_name in (
            "raw_points_xyz.npy",
            "raw_points_rtp.npy",
            "raw_connection_length_m.npy",
            "raw_plane_index.npy",
            "raw_fieldline_id.npy",
            "raw_source_direction.npy",
            "plane_offsets.npy",
            "wall_intersection_xyz.npy",
        ):
            (self.data_dir / obsolete_name).unlink(missing_ok=True)
        plane_phi_deg = np.linspace(
            360.0 / params["N_PLANES"],
            360.0,
            params["N_PLANES"],
        )
        crossing_writer = PlaneShardWriter(
            self.data_dir,
            plane_phi_deg,
            self.field.R0,
        )
        crossing_store = PlaneCrossingStore(
            params["N_PLANES"],
            params["CROSSING_BUFFER_SIZE"],
            self.device,
            crossing_writer,
        )
        try:
            self.trace_data = trace_connection_length_volume(
                self.seed_data,
                params,
                self.field,
                params["STEP_SIZE_M"],
                params["BATCH_SIZE"],
                crossing_store,
                self.simIO,
                params["SHOW_PROGRESS"],
                params["STEP_CHUNK_SIZE"],
                self.compile_chunks,
                params["INTEGRATOR"],
                params["MIN_FIELD_MAGNITUDE"],
                params["WALL_BISECTION_STEPS"],
                params["PROGRESS_REFRESH_STEPS"],
                params["PROGRESS_INTERVAL_STEPS"])

            self.data_dir = save_torch_outputs(
                self.simIO,
                self.seed_data,
                self.trace_data,
                crossing_writer,
                self.field.R0,
                self.analysis_subdir)
        except Exception:
            crossing_writer.abort()
            raise
        return self.trace_data

    def plot(self):
        """Generate the configured per-plane contour diagnostics."""
        plot_all_planes(self.analysis_dir, self.input_params["LCFS_INDEX"], self.trace_data, self.simIO, self.input_params)

    def run(self):
        """Build seeds, trace the volume, save compact arrays, and plot."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.build_initial_conditions()
        self.log_inputs()
        self.trace()
        if self.input_params["GENERATE_PLOTS"]:
            self.plot()

        hit_count = np.count_nonzero(self.trace_data["hit_wall"])
        total_directions = self.trace_data["hit_wall"].size
        self.simIO.log.info("Wall intersections: %d of %d directional traces.",
            hit_count, total_directions)
        self.simIO.log.info("Saved raw connection-length data: %s", self.data_dir)
        if self.input_params["GENERATE_PLOTS"]:
            self.simIO.log.info("Saved %d contour plots: %s",
                                self.input_params["N_PLANES"], Path(self.simIO.plot_dir) / self.analysis_subdir)
        if self.device.type == "cuda":
            peak_gib = torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 3)
            self.simIO.log.info("PEAK CUDA MEMORY ALLOCATED: %.3f GiB", peak_gib)
        self.simIO.log.info("## SOL TRACE ANALYSIS FINISHED ##")

        return self.trace_data
