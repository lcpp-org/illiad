"""Prototype batched PyTorch connection-length volume tracer.

This script reproduces the contour-plot outputs from
``connection_length_volume.py`` while replacing the independent SciPy
``solve_ivp`` calls with a fixed-step, batched PyTorch integrator. Its raw
output contract stores canonical RTP crossings plus field-line IDs, avoiding
expanded Cartesian coordinates and per-crossing values that can be rebuilt
from the compact arrays. CUDA is used when available; the same code falls
back to CPU for validation.

The prototype deliberately remains outside the production Poincare classes.
It uses the existing ``TorchMesh`` field interpolation but owns its stepping,
wall detection, and toroidal-plane crossing capture here.
"""

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from illiad.io import IOHandler
from illiad.mesh import TorchMesh
import illiad.mesh.torch_mesh as torch_mesh_module
import illiad.utilities.coordtrans as coordtrans_module
from illiad.utilities.coordtrans import RTP_to_XYZ_many, XYZ_to_RTP_many
from misc_scripts import connection_length_volume as volume
from misc_scripts.connection_lengths_outside_lcfs import (
    load_poincare_settings,
)


ANALYSIS_SUBDIR = "ConLenVolume_REDO_250spins_rk2mm"

# Sampling and tracing defaults. These match connection_length_volume.py.
N_PLANES = volume.N_PLANES
N_SEED_PLANES = volume.N_SEED_PLANES
MAX_SPINS = volume.MAX_SPINS
N_RHO = volume.N_RHO
N_THETA = volume.N_THETA
RHO_MIN = volume.RHO_MIN
RHO_MAX = volume.RHO_MAX
LCFS_CLEARANCE = volume.LCFS_CLEARANCE

# Prototype PyTorch settings.
DEVICE = "auto"              # "auto", "cuda", or "cpu"
INTEGRATOR = "midpoint"           # "rk4" or "midpoint"
STEP_SIZE_M = 0.002           # Fixed field-line path-length step [m]
BATCH_SIZE = 4096*128             # Directional traces integrated together
CROSSING_BUFFER_SIZE = 1_000_000
STEP_CHUNK_SIZE = 8           # GPU steps launched between Python bookkeeping
COMPILE_STEP_CHUNKS = True    # Use torch.compile on CUDA, with eager fallback
WALL_BISECTION_STEPS = 24
MIN_FIELD_MAGNITUDE = 1.0e-14
PROGRESS_INTERVAL_STEPS = 5000
PROGRESS_REFRESH_STEPS = 100
SHOW_PROGRESS = True
GENERATE_PLOTS = True
PLOT_MAX_SAMPLES = 150_000  # Per plane; None plots every raw crossing
PLOT_SAMPLE_SEED = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prototype batched PyTorch connection-length volume tracer."
        )
    )
    parser.add_argument("analysis_dir", help="Existing directory under output/.")
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
        help="Maximum spins in each direction (default: MAX_SPINS in script).",
    )
    parser.add_argument(
        "--seed-planes",
        type=int,
        default=None,
        help=(
            "Number of equally spaced seed planes "
            f"(default: {N_SEED_PLANES}; must divide {N_PLANES})."
        ),
    )
    parser.add_argument(
        "--seed-phi-deg",
        type=float,
        default=None,
        help=(
            "First seed plane in computational degrees "
            "(default: IC_PHI_DEG from the Poincare log)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default=None,
        help=f"Torch device (default: {DEVICE}).",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=None,
        help=f"Fixed path-length step in meters (default: {STEP_SIZE_M}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Directional traces per batch (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=None,
        help=(
            "Integration steps accumulated per GPU chunk "
            f"(default: {STEP_CHUNK_SIZE})."
        ),
    )
    parser.add_argument(
        "--compile-chunks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compile fixed-size GPU step chunks (default: enabled on CUDA).",
    )
    parser.add_argument(
        "--rho-count",
        type=int,
        default=None,
        help=f"Sparse seed-grid radial count (default: {N_RHO}).",
    )
    parser.add_argument(
        "--theta-count",
        type=int,
        default=None,
        help=f"Sparse seed-grid poloidal count (default: {N_THETA}).",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Generate the 360 contour plots (default: enabled).",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show tqdm integration progress bars (default: enabled).",
    )
    return parser.parse_args()


def resolve_device(requested):
    """Resolve and configure the device used by TorchMesh and transforms."""
    requested = DEVICE if requested is None else requested
    if requested == "auto":
        selected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        selected = torch.device("cuda")
    else:
        selected = torch.device("cpu")

    # TorchMesh and XYZ_to_RTP2 use module-level device selectors. Updating
    # both keeps an explicit --device choice internally consistent.
    torch_mesh_module.device = selected
    coordtrans_module.device = selected
    return selected


def build_torch_magnetic_field(settings, device):
    """Recreate the Poincare magnetic field on the selected Torch device."""
    torch_mesh_module.device = device
    coordtrans_module.device = device
    magnetic_field = TorchMesh(R0=0.72, a=0.19)
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
    magnetic_field._prototype_error_enabled = bool(settings["ENABLE_ERRFIELD"])
    return magnetic_field


def minor_radius(points_xyz, major_radius):
    cylindrical_radius = torch.linalg.vector_norm(points_xyz[..., :2], dim=-1)
    return torch.sqrt(
        (cylindrical_radius - major_radius) ** 2 + points_xyz[..., 2] ** 2
    )


def wrapped_phi(points_xyz):
    phi = -torch.atan2(points_xyz[..., 1], points_xyz[..., 0])
    return torch.remainder(phi, 2.0 * torch.pi)


def compile_safe_weights(magnetic_field, positions):
    """Reproduce ``TorchMesh.get_weights`` with compile-safe tensor shapes."""
    x, y, z = positions.unbind(dim=1)
    cylindrical_radius = torch.sqrt(x * x + y * y)
    radius = torch.sqrt(
        x * x
        + y * y
        + z * z
        + magnetic_field.R0 * magnetic_field.R0
        - 2.0 * magnetic_field.R0 * cylindrical_radius
    )
    theta = torch.remainder(
        torch.atan2(z, cylindrical_radius - magnetic_field.R0),
        2.0 * torch.pi,
    )
    phi = torch.remainder(-torch.atan2(y, x), 2.0 * torch.pi)

    theta_local = torch.remainder(theta, magnetic_field.theta_max)
    phi_local = torch.remainder(phi, magnetic_field.phi_max)
    phi_period = torch.div(
        phi,
        magnetic_field.phi_max,
        rounding_mode="floor",
    )
    radius_index = torch.where(
        radius >= magnetic_field.r_max,
        magnetic_field.nr - 2,
        torch.div(radius, magnetic_field.dr, rounding_mode="floor"),
    )
    radius_element = torch.remainder(radius, magnetic_field.dr)
    theta_index = torch.div(
        theta_local,
        magnetic_field.dtheta,
        rounding_mode="floor",
    )
    theta_element = torch.remainder(theta_local, magnetic_field.dtheta)
    phi_index = torch.div(
        phi_local,
        magnetic_field.dphi,
        rounding_mode="floor",
    )
    phi_element = torch.remainder(phi_local, magnetic_field.dphi)

    radius_low = radius_index * magnetic_field.dr
    theta_low = (theta_index + 1.0) * magnetic_field.dtheta
    inverse_radius_element = magnetic_field.dr - radius_element
    inverse_theta_element = magnetic_field.dtheta - theta_element
    inverse_phi_element = magnetic_field.dphi - phi_element
    radius_low_element = (
        radius_low + 0.5 * radius_element
    ) * radius_element
    radius_inverse_element = (
        radius + 0.5 * inverse_radius_element
    ) * inverse_radius_element

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
    weights, corner_indices, phi_period = compile_safe_weights(
        magnetic_field,
        positions,
    )
    phi_high = corner_indices[2, 0]
    corner_vectors = torch.movedim(
        magnetic_field.B[
            corner_indices[0],
            corner_indices[1],
            corner_indices[2],
        ],
        -1,
        1,
    )

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
    vectors = (
        (corner_vectors * weights[:, None]).sum(dim=0)
        / total_volume[None, :]
    )
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
    if magnetic_field._prototype_error_enabled:
        vectors = vectors + magnetic_field.err_adder[:, None]
    return vectors


def normalized_field_direction(magnetic_field, positions, direction_sign):
    """Evaluate batched +/- B/|B| and identify invalid field samples."""
    field_vectors = compile_safe_interp_field(
        magnetic_field,
        positions,
    ).transpose(0, 1)
    field_magnitude = torch.linalg.vector_norm(field_vectors, dim=1)
    valid = (
        torch.all(torch.isfinite(field_vectors), dim=1)
        & torch.isfinite(field_magnitude)
        & (field_magnitude > MIN_FIELD_MAGNITUDE)
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
    integrator=INTEGRATOR,
):
    """Advance a batch by one fixed path-length step."""
    if integrator == "midpoint":
        k1, valid1 = normalized_field_direction(
            magnetic_field,
            positions,
            direction_sign,
        )
        k2, valid2 = normalized_field_direction(
            magnetic_field,
            positions + 0.5 * step_size * k1,
            direction_sign,
        )
        next_positions = positions + step_size * k2
        valid = valid1 & valid2
    elif integrator == "rk4":
        k1, valid1 = normalized_field_direction(
            magnetic_field,
            positions,
            direction_sign,
        )
        k2, valid2 = normalized_field_direction(
            magnetic_field,
            positions + 0.5 * step_size * k1,
            direction_sign,
        )
        k3, valid3 = normalized_field_direction(
            magnetic_field,
            positions + 0.5 * step_size * k2,
            direction_sign,
        )
        k4, valid4 = normalized_field_direction(
            magnetic_field,
            positions + step_size * k3,
            direction_sign,
        )
        next_positions = positions + (step_size / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
        valid = valid1 & valid2 & valid3 & valid4
    else:
        raise ValueError('INTEGRATOR must be either "rk4" or "midpoint".')

    valid &= torch.all(torch.isfinite(next_positions), dim=1)
    return next_positions, valid


def make_step_chunk_function(magnetic_field, chunk_size):
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

    def __init__(self, magnetic_field, chunk_size, compile_chunks, sim_io):
        self.eager = make_step_chunk_function(magnetic_field, chunk_size)
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
            self.sim_io.log.warning(
                "Compiled step chunks are unavailable; falling back to eager "
                "chunk execution: %s",
                exc,
            )
            self.function = self.eager
            self.compiled = False
            return self.function(positions, direction_sign, step_sizes)


def wall_intersections(
    start_xyz,
    stop_xyz,
    major_radius,
    vessel_radius,
):
    """Refine line-segment wall intersections with batched bisection."""
    low = torch.zeros(start_xyz.shape[0], dtype=start_xyz.dtype, device=start_xyz.device)
    high = torch.ones_like(low)
    segment = stop_xyz - start_xyz
    for _ in range(WALL_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        middle_xyz = start_xyz + middle[:, None] * segment
        middle_inside = minor_radius(middle_xyz, major_radius) < vessel_radius
        low = torch.where(middle_inside, middle, low)
        high = torch.where(middle_inside, high, middle)
    intersections = start_xyz + high[:, None] * segment
    return intersections, high


class PlaneCrossingStore:
    """Buffer GPU plane crossings and group flushed chunks by output plane."""

    def __init__(self, n_planes, capacity, device, dtype=torch.float64):
        if capacity <= 0:
            raise ValueError("CROSSING_BUFFER_SIZE must be positive.")
        self.n_planes = n_planes
        self.capacity = capacity
        self.device = device
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
        self.blocks = [[] for _ in range(n_planes)]
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
            self.blocks[plane_index].append(
                (
                    xyz[start:stop],
                    fieldline_id[start:stop],
                    source_direction[start:stop],
                )
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
        mininterval=1.0,
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
                completed_steps % PROGRESS_REFRESH_STEPS < steps_this_chunk
                or running.numel() == 0
            ):
                progress_bar.set_postfix(
                    active=running.numel(),
                    walls=torch.count_nonzero(hit_wall).item(),
                    crossings=crossing_store.count,
                    refresh=False,
                )

            if (
                completed_steps % PROGRESS_INTERVAL_STEPS < steps_this_chunk
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


def trace_connection_length_volume_torch(
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
):
    """Trace every seed in both directions in batched Torch integrations."""
    initial_conditions_rtp = seed_data["initial_conditions_rtp"]
    n_fieldlines = initial_conditions_rtp.shape[0]
    fieldline_id = np.concatenate(
        (
            np.arange(n_fieldlines, dtype=np.int64),
            np.arange(n_fieldlines, dtype=np.int64),
        )
    )
    direction_column = np.concatenate(
        (
            np.zeros(n_fieldlines, dtype=np.int64),
            np.ones(n_fieldlines, dtype=np.int64),
        )
    )
    direction_sign = np.concatenate(
        (
            np.ones(n_fieldlines, dtype=np.float64),
            -np.ones(n_fieldlines, dtype=np.float64),
        )
    )
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
    chunk_runner = StepChunkRunner(
        magnetic_field,
        chunk_size,
        compile_chunks,
        sim_io,
    )
    trace_start = perf_counter()
    for batch_index, start in enumerate(
        range(0, directional_count, batch_size),
        start=1,
    ):
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
    wall_rtp[finite_wall] = XYZ_to_RTP_many(
        wall_xyz[finite_wall],
        magnetic_field.R0,
    )
    trace_elapsed = perf_counter() - trace_start
    sim_io.log.info(
        "ALL TORCH SOLVERS FINISHED IN %.3f seconds; captured %d crossings.",
        trace_elapsed,
        crossing_store.counts.sum(),
    )
    return {
        "direction_length": direction_length,
        "connection_length": connection_length,
        "wall_xyz": wall_xyz,
        "wall_rtp": wall_rtp,
        "hit_wall": hit_wall,
        "reached_limit": reached_limit,
        "valid_trace": valid_trace,
        "plane_phi_deg": np.linspace(
            360.0 / N_PLANES,
            360.0,
            N_PLANES,
        ),
    }


def _open_output_memmap(data_dir, name, dtype, shape):
    return np.lib.format.open_memmap(
        data_dir / f"{name}.npy",
        mode="w+",
        dtype=dtype,
        shape=shape,
    )


def save_torch_outputs(
    sim_io,
    seed_data,
    trace_data,
    crossing_store,
    major_radius,
):
    """Save compact plane-sorted raw arrays without duplicating them in RAM."""
    data_dir = Path(sim_io.data_dir) / ANALYSIS_SUBDIR
    data_dir.mkdir(parents=True, exist_ok=True)
    for obsolete_name in (
        "raw_points_xyz.npy",
        "raw_connection_length_m.npy",
        "raw_plane_index.npy",
        "wall_intersection_xyz.npy",
    ):
        (data_dir / obsolete_name).unlink(missing_ok=True)
    plane_phi_deg = trace_data["plane_phi_deg"]
    seed_plane_for_fieldline = seed_data["seed_plane_index"][
        seed_data["seed_id"]
    ]
    seed_counts_per_plane = np.bincount(
        seed_plane_for_fieldline,
        minlength=N_PLANES,
    )
    plane_counts = crossing_store.counts + seed_counts_per_plane
    plane_offsets = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(plane_counts))
    )
    total_samples = int(plane_offsets[-1])
    if trace_data["connection_length"].size > np.iinfo(np.int32).max:
        raise ValueError("The fieldline count exceeds the int32 raw-ID range.")

    small_outputs = {
        "plane_offsets": plane_offsets,
        "plane_phi_deg": plane_phi_deg,
        "seed_initial_conditions_rtp": seed_data["initial_conditions_rtp"],
        "seed_id": seed_data["seed_id"],
        "seed_plane_index": seed_data["seed_plane_index"],
        "seed_phi_deg": seed_data["seed_phi_deg"],
        "seed_counts": seed_data["seed_counts"],
        "major_radius_m": np.asarray(major_radius),
        "fieldline_connection_length_m": trace_data["connection_length"],
        "direction_connection_length_m": trace_data["direction_length"],
        "wall_intersection_rtp": trace_data["wall_rtp"],
        "hit_wall": trace_data["hit_wall"],
        "reached_max_length": trace_data["reached_limit"],
        "valid_trace": trace_data["valid_trace"],
    }
    for name, values in small_outputs.items():
        sim_io.saveNumpyData(values, name, subdir=ANALYSIS_SUBDIR)

    raw_rtp = _open_output_memmap(
        data_dir,
        "raw_points_rtp",
        np.float64,
        (total_samples, 3),
    )
    raw_fieldline = _open_output_memmap(
        data_dir,
        "raw_fieldline_id",
        np.int32,
        (total_samples,),
    )
    raw_direction = _open_output_memmap(
        data_dir,
        "raw_source_direction",
        np.int8,
        (total_samples,),
    )

    for plane_index in range(N_PLANES):
        cursor = int(plane_offsets[plane_index])
        for xyz_block, fieldline_block, direction_block in crossing_store.blocks[
            plane_index
        ]:
            stop = cursor + xyz_block.shape[0]
            raw_rtp[cursor:stop] = XYZ_to_RTP_many(xyz_block, major_radius)
            raw_rtp[cursor:stop, 2] = np.deg2rad(plane_phi_deg[plane_index])
            raw_fieldline[cursor:stop] = fieldline_block
            raw_direction[cursor:stop] = direction_block
            cursor = stop

        seed_fieldlines = np.flatnonzero(seed_plane_for_fieldline == plane_index)
        if seed_fieldlines.size:
            stop = cursor + seed_fieldlines.size
            raw_rtp[cursor:stop] = seed_data["initial_conditions_rtp"][
                seed_fieldlines
            ]
            raw_rtp[cursor:stop, 2] = np.deg2rad(plane_phi_deg[plane_index])
            raw_fieldline[cursor:stop] = seed_fieldlines
            raw_direction[cursor:stop] = 0
            cursor = stop

        expected_stop = int(plane_offsets[plane_index + 1])
        if cursor != expected_stop:
            raise RuntimeError(
                f"Plane {plane_index} wrote {cursor} samples; "
                f"expected offset {expected_stop}."
            )
        crossing_store.blocks[plane_index].clear()

    for array in (
        raw_rtp,
        raw_fieldline,
        raw_direction,
    ):
        array.flush()

    trace_data.update(
        {
            "raw_points_rtp": np.load(
                data_dir / "raw_points_rtp.npy",
                mmap_mode="r",
            ),
            "raw_fieldline_id": np.load(
                data_dir / "raw_fieldline_id.npy",
                mmap_mode="r",
            ),
            "raw_source_direction": np.load(
                data_dir / "raw_source_direction.npy",
                mmap_mode="r",
            ),
            "fieldline_connection_length": trace_data["connection_length"],
            "plane_offsets": plane_offsets,
        }
    )
    sim_io.log.info(
        "Saved %d raw Torch samples across %d toroidal planes: %s",
        total_samples,
        N_PLANES,
        data_dir,
    )
    return data_dir


def validate_runtime_settings(
    step_size,
    batch_size,
    chunk_size,
    n_rho,
    n_theta,
):
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("STEP_SIZE_M must be positive and finite.")
    if batch_size <= 0:
        raise ValueError("BATCH_SIZE must be a positive integer.")
    if chunk_size <= 0:
        raise ValueError("STEP_CHUNK_SIZE must be a positive integer.")
    if n_rho < 2 or n_theta < 3:
        raise ValueError("The sparse grid requires N_RHO >= 2 and N_THETA >= 3.")

    plane_spacing = 2.0 * np.pi / N_PLANES
    conservative_limit = 0.5 * (0.72 - volume.VESSEL_RADIUS) * plane_spacing
    if step_size > conservative_limit:
        raise ValueError(
            f"STEP_SIZE_M={step_size:g} is too large for single-crossing "
            f"reconstruction; use <= {conservative_limit:g} m."
        )


def main():
    args = parse_args()
    settings = load_poincare_settings(args.analysis_dir)
    lcfs_index = (
        settings.get("LCFS_INDEX")
        if args.lcfs_index is None
        else args.lcfs_index
    )
    if lcfs_index is None:
        raise ValueError("No LCFS index was found; provide --lcfs-index.")

    if MAX_SPINS is not None:
        settings["SPINS"] = MAX_SPINS
    if args.spins is not None:
        settings["SPINS"] = args.spins
    if (
        isinstance(settings["SPINS"], bool)
        or not isinstance(settings["SPINS"], int)
        or settings["SPINS"] <= 0
    ):
        raise ValueError("SPINS must be a positive integer.")

    n_seed_planes = (
        N_SEED_PLANES if args.seed_planes is None else args.seed_planes
    )
    seed_phi_deg = (
        settings["IC_PHI_DEG"]
        if args.seed_phi_deg is None
        else args.seed_phi_deg
    )
    step_size = STEP_SIZE_M if args.step_size is None else args.step_size
    batch_size = BATCH_SIZE if args.batch_size is None else args.batch_size
    chunk_size = (
        STEP_CHUNK_SIZE if args.chunk_steps is None else args.chunk_steps
    )
    compile_chunks = (
        COMPILE_STEP_CHUNKS
        if args.compile_chunks is None
        else args.compile_chunks
    )
    n_rho = N_RHO if args.rho_count is None else args.rho_count
    n_theta = N_THETA if args.theta_count is None else args.theta_count
    generate_plots = GENERATE_PLOTS if args.plots is None else args.plots
    show_progress = SHOW_PROGRESS if args.progress is None else args.progress
    validate_runtime_settings(
        step_size,
        batch_size,
        chunk_size,
        n_rho,
        n_theta,
    )
    device = resolve_device(args.device)
    compile_chunks = compile_chunks and device.type == "cuda"

    seed_data = volume.make_seed_initial_conditions(
        args.analysis_dir,
        lcfs_index,
        n_seed_planes,
        seed_phi_deg=seed_phi_deg,
        n_rho=n_rho,
        n_theta=n_theta,
    )
    directional_solves = 2 * seed_data["initial_conditions_rtp"].shape[0]
    max_length = 2.0 * np.pi * 0.72 * settings["SPINS"]
    max_steps = int(np.ceil(max_length / step_size))

    sim_io = IOHandler(args.analysis_dir)
    sim_io.startLog(
        log_name="connection_length_volume_torch.log",
        subdir=ANALYSIS_SUBDIR,
        logger_name=ANALYSIS_SUBDIR,
    )
    cuda_device_name = None
    if device.type == "cuda":
        cuda_device_name = torch.cuda.get_device_name(device)
        torch.cuda.reset_peak_memory_stats(device)
    run_settings = {
        **settings,
        "LCFS_INDEX": lcfs_index,
        "LCFS_CLEARANCE_M": LCFS_CLEARANCE,
        "N_PLANES": N_PLANES,
        "N_SEED_PLANES": n_seed_planes,
        "SEED_PHI_OFFSET_DEG": seed_phi_deg,
        "SEED_PHI_DEG": seed_data["seed_phi_deg"].tolist(),
        "SEED_COUNTS": seed_data["seed_counts"].tolist(),
        "SEED_POINCARE_FILES": [
            str(path) for path in seed_data["poincare_paths"]
        ],
        "N_RHO": n_rho,
        "N_THETA": n_theta,
        "RHO_MIN": RHO_MIN,
        "RHO_MAX": RHO_MAX,
        "TRACED_FIELD_LINES": seed_data["initial_conditions_rtp"].shape[0],
        "DIRECTIONAL_SOLVES": directional_solves,
        "DOUBLE_LINE": True,
        "TORCH_VERSION": torch.__version__,
        "CUDA_AVAILABLE": torch.cuda.is_available(),
        "DEVICE": str(device),
        "CUDA_DEVICE_NAME": cuda_device_name,
        "DTYPE": "torch.float64",
        "INTEGRATOR": INTEGRATOR,
        "STEP_SIZE_M": step_size,
        "MAX_STEPS": max_steps,
        "BATCH_SIZE": batch_size,
        "STEP_CHUNK_SIZE": chunk_size,
        "COMPILE_STEP_CHUNKS": compile_chunks,
        "CROSSING_BUFFER_SIZE": CROSSING_BUFFER_SIZE,
        "WALL_BISECTION_STEPS": WALL_BISECTION_STEPS,
        "MIN_FIELD_MAGNITUDE": MIN_FIELD_MAGNITUDE,
        "PROGRESS_INTERVAL_STEPS": PROGRESS_INTERVAL_STEPS,
        "PROGRESS_REFRESH_STEPS": PROGRESS_REFRESH_STEPS,
        "SHOW_PROGRESS": show_progress,
        "GENERATE_PLOTS": generate_plots,
        "COLOR_SCALE": volume.COLOR_SCALE,
        "COLORMAP": volume.COLORMAP,
        "N_LEVELS": volume.N_LEVELS,
        "VMIN": volume.VMIN,
        "VMAX": volume.VMAX,
        "DPI": volume.DPI,
        "PLOT_MAX_SAMPLES": PLOT_MAX_SAMPLES,
        "PLOT_SAMPLE_SEED": PLOT_SAMPLE_SEED,
    }
    sim_io.inputsBoilerplate(
        "PYTORCH CONNECTION-LENGTH VOLUME INPUTS",
        run_settings,
        [
            "CURRENT_TOR",
            "CURRENT_HEL",
            "CONFIG_TOR",
            "CONFIG_HEL",
            "ENABLE_ERRFIELD",
            "LCFS_INDEX",
            "LCFS_CLEARANCE_M",
            "N_PLANES",
            "N_SEED_PLANES",
            "SEED_PHI_OFFSET_DEG",
            "SEED_PHI_DEG",
            "SEED_COUNTS",
            "SEED_POINCARE_FILES",
            "N_RHO",
            "N_THETA",
            "RHO_MIN",
            "RHO_MAX",
            "TRACED_FIELD_LINES",
            "DIRECTIONAL_SOLVES",
            "SPINS",
            "DOUBLE_LINE",
            "TORCH_VERSION",
            "CUDA_AVAILABLE",
            "DEVICE",
            "CUDA_DEVICE_NAME",
            "DTYPE",
            "INTEGRATOR",
            "STEP_SIZE_M",
            "MAX_STEPS",
            "BATCH_SIZE",
            "STEP_CHUNK_SIZE",
            "COMPILE_STEP_CHUNKS",
            "CROSSING_BUFFER_SIZE",
            "WALL_BISECTION_STEPS",
            "MIN_FIELD_MAGNITUDE",
            "PROGRESS_INTERVAL_STEPS",
            "PROGRESS_REFRESH_STEPS",
            "SHOW_PROGRESS",
            "GENERATE_PLOTS",
            "COLOR_SCALE",
            "COLORMAP",
            "N_LEVELS",
            "VMIN",
            "VMAX",
            "DPI",
            "PLOT_MAX_SAMPLES",
            "PLOT_SAMPLE_SEED",
        ],
    )

    magnetic_field = build_torch_magnetic_field(settings, device)
    crossing_store = PlaneCrossingStore(
        N_PLANES,
        CROSSING_BUFFER_SIZE,
        device,
    )
    trace_data = trace_connection_length_volume_torch(
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
    )
    data_dir = save_torch_outputs(
        sim_io,
        seed_data,
        trace_data,
        crossing_store,
        magnetic_field.R0,
    )
    if generate_plots:
        volume.PLOT_MAX_SAMPLES = PLOT_MAX_SAMPLES
        volume.PLOT_SAMPLE_SEED = PLOT_SAMPLE_SEED
        volume.plot_all_planes(
            args.analysis_dir,
            lcfs_index,
            trace_data,
            sim_io,
            analysis_subdir=ANALYSIS_SUBDIR,
        )

    hit_count = np.count_nonzero(trace_data["hit_wall"])
    total_directions = trace_data["hit_wall"].size
    sim_io.log.info(
        "Wall intersections: %d of %d directional traces.",
        hit_count,
        total_directions,
    )
    sim_io.log.info("Saved raw connection-length data: %s", data_dir)
    if generate_plots:
        sim_io.log.info(
            "Saved %d contour plots: %s",
            N_PLANES,
            Path(sim_io.plot_dir) / ANALYSIS_SUBDIR,
        )
    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
        sim_io.log.info("PEAK CUDA MEMORY ALLOCATED: %.3f GiB", peak_gib)
    sim_io.log.info("## PYTORCH CONNECTION-LENGTH VOLUME ANALYSIS FINISHED ##")
    print(f"Saved raw data: {data_dir}")
    if generate_plots:
        print(
            "Saved contour plots: "
            f"{Path(sim_io.plot_dir) / ANALYSIS_SUBDIR}"
        )


if __name__ == "__main__":
    main()
