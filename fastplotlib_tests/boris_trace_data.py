"""Small helpers for standalone fastplotlib Boris trace experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TraceSelection:
    traces: np.ndarray
    valid_lengths: np.ndarray
    particle_indices: np.ndarray


def load_boris_trace_file(path: str | Path, mmap: bool = True) -> np.ndarray:
    """Load a Boris trace array with shape ``(time, particle, xyz)``."""
    mmap_mode = "r" if mmap else None
    traces = np.load(Path(path).expanduser(), mmap_mode=mmap_mode)
    validate_trace_shape(traces)
    return traces


def normalize_trace_paths(paths: str | Path | list[str | Path] | tuple[str | Path, ...]) -> list[Path]:
    """Normalize one or more Boris trace paths."""
    if isinstance(paths, (str, Path)):
        return [Path(paths).expanduser()]
    return [Path(path).expanduser() for path in paths]


def load_boris_trace_files(
    paths: str | Path | list[str | Path] | tuple[str | Path, ...],
    mmap: bool = True,
) -> np.ndarray:
    """Load one or more Boris trace arrays and combine them along the particle axis."""
    trace_paths = normalize_trace_paths(paths)
    if not trace_paths:
        raise ValueError("At least one Boris trace file is required.")
    if len(trace_paths) == 1:
        return load_boris_trace_file(trace_paths[0], mmap=mmap)

    traces_by_file = [load_boris_trace_file(path, mmap=mmap) for path in trace_paths]
    max_steps = max(traces.shape[0] for traces in traces_by_file)
    total_particles = sum(traces.shape[1] for traces in traces_by_file)
    dtype = np.result_type(*(traces.dtype for traces in traces_by_file))
    combined = np.zeros((max_steps, total_particles, 3), dtype=dtype)

    particle_offset = 0
    for traces in traces_by_file:
        particle_stop = particle_offset + traces.shape[1]
        combined[: traces.shape[0], particle_offset:particle_stop, :] = traces
        particle_offset = particle_stop

    return combined


def validate_trace_shape(traces: np.ndarray) -> None:
    if traces.ndim != 3 or traces.shape[2] != 3:
        raise ValueError(
            "Expected ion_traces with shape (n_steps, n_particles, 3); "
            f"got {traces.shape!r}."
        )


def infer_valid_lengths(traces: np.ndarray) -> np.ndarray:
    """Match the existing Matplotlib helper: count non-zero samples per particle."""
    validate_trace_shape(traces)
    valid_lengths = np.zeros(traces.shape[1], dtype=np.int64)
    for particle_idx in range(traces.shape[1]):
        particle_trace = traces[:, particle_idx, :]
        valid_lengths[particle_idx] = int(np.count_nonzero(np.any(particle_trace != 0.0, axis=1)))
    return valid_lengths


def select_trace_particles(
    traces: np.ndarray,
    valid_lengths: np.ndarray | None = None,
    max_particles: int | None = None,
    skip_indices: set[int] | None = None,
) -> TraceSelection:
    """Return a compact view/list of usable particles for plotting."""
    validate_trace_shape(traces)
    if valid_lengths is None:
        valid_lengths = infer_valid_lengths(traces)

    skip_indices = skip_indices or set()
    selected = [
        idx
        for idx, valid_length in enumerate(valid_lengths)
        if idx not in skip_indices and int(valid_length) >= 2
    ]
    if max_particles is not None:
        selected = selected[: max(0, int(max_particles))]
    if not selected:
        raise ValueError("No valid particles selected for plotting.")

    particle_indices = np.asarray(selected, dtype=np.int64)
    if np.all(np.diff(particle_indices) == 1):
        selected_traces = traces[:, int(particle_indices[0]) : int(particle_indices[-1]) + 1, :]
    else:
        selected_traces = traces[:, particle_indices, :]

    return TraceSelection(
        traces=selected_traces,
        valid_lengths=np.asarray(valid_lengths, dtype=np.int64)[particle_indices],
        particle_indices=particle_indices,
    )


def make_synthetic_boris_traces(
    n_steps: int = 4_000,
    n_particles: int = 256,
    R0: float = 0.72,
    a: float = 0.19,
    turns: float = 8.0,
) -> np.ndarray:
    """Create torus-like synthetic traces with the same shape as Boris output."""
    t = np.linspace(0.0, 2.0 * np.pi * turns, int(n_steps), dtype=np.float32)
    phases = np.linspace(0.0, 2.0 * np.pi, int(n_particles), endpoint=False, dtype=np.float32)
    minor = np.float32(0.55 * a)
    theta = t[:, None] * np.float32(1.7) + phases[None, :]
    phi = t[:, None] + phases[None, :] * np.float32(0.37)
    radial = np.float32(R0) + minor * np.cos(theta)

    traces = np.empty((int(n_steps), int(n_particles), 3), dtype=np.float32)
    traces[:, :, 0] = radial * np.cos(phi)
    traces[:, :, 1] = radial * np.sin(phi)
    traces[:, :, 2] = minor * np.sin(theta)
    return traces


def make_torus_mesh(
    R0: float = 0.72,
    a: float = 0.19,
    nphi: int = 144,
    ntheta: int = 48,
    half_shell: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return triangle vertices/indices for a HIDRA-like torus shell."""
    theta_min = -np.pi if not half_shell else -np.pi
    theta_max = np.pi if not half_shell else 0.0
    phi = np.linspace(0.0, 2.0 * np.pi, int(nphi), endpoint=False, dtype=np.float32)
    theta = np.linspace(theta_min, theta_max, int(ntheta), dtype=np.float32)
    pp, tt = np.meshgrid(phi, theta, indexing="ij")

    x = (np.float32(R0) + np.float32(a) * np.cos(tt)) * np.cos(pp)
    y = (np.float32(R0) + np.float32(a) * np.cos(tt)) * np.sin(pp)
    z = np.float32(a) * np.sin(tt)
    positions = np.column_stack([x.ravel(), y.ravel(), z.ravel()]).astype(np.float32)

    indices = []
    for i in range(int(nphi)):
        i_next = (i + 1) % int(nphi)
        for j in range(int(ntheta) - 1):
            p00 = i * int(ntheta) + j
            p01 = i * int(ntheta) + j + 1
            p10 = i_next * int(ntheta) + j
            p11 = i_next * int(ntheta) + j + 1
            indices.append((p00, p10, p01))
            indices.append((p10, p11, p01))

    return positions, np.asarray(indices, dtype=np.uint32)


def frame_count(valid_lengths: np.ndarray, stride: int, steps_per_frame: int) -> int:
    max_strided = int(np.max((valid_lengths + stride - 1) // stride))
    return max(1, (max_strided + steps_per_frame - 1) // steps_per_frame)
