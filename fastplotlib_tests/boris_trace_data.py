"""Small helpers for standalone fastplotlib Boris trace experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PORT_CSV_TO_REPO_ZERO_DEG = 180.0 - 18.0


@dataclass(frozen=True)
class TraceSelection:
    traces: np.ndarray
    valid_lengths: np.ndarray
    particle_indices: np.ndarray


@dataclass(frozen=True)
class HidraPort:
    phi_csv_deg: float
    theta_deg: float
    width_deg: float
    height_deg: float
    component: str
    port_id: str


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
    traces[:, :, 1] = -radial * np.sin(phi)
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
    y = -(np.float32(R0) + np.float32(a) * np.cos(tt)) * np.sin(pp)
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


def _resolve_relative_repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute() or path.exists():
        return path
    repo_path = Path(__file__).resolve().parents[1] / path
    return repo_path if repo_path.exists() else path


def port_csv_phi_to_repo_phi(
    phi_csv_deg: np.ndarray | float,
    phi_zero_offset_deg: float = PORT_CSV_TO_REPO_ZERO_DEG,
) -> np.ndarray:
    """Convert CSV counterclockwise-from-north phi to repo clockwise phi."""
    return (float(phi_zero_offset_deg) - np.asarray(phi_csv_deg, dtype=np.float32)) % 360.0


def load_hidra_port_records(
    path: str | Path = "input_files/HIDRA_ports.csv",
) -> list[HidraPort]:
    """Load HIDRA port records and compute angular port extents."""
    records: list[HidraPort] = []
    with _resolve_relative_repo_path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            p_rmaj = float(row["Major radius [m]"])
            p_rmin = float(row["Minor radius [m]"])
            p_dia = float(row["Port Dia. [mm]"]) / 1000.0
            records.append(
                HidraPort(
                    phi_csv_deg=float(row["Phi (toroidal) [deg.]"]),
                    theta_deg=float(row["Theta (poloidal)[deg.]"]),
                    width_deg=float(np.degrees(np.arcsin(p_dia / p_rmaj))),
                    height_deg=float(np.degrees(np.arcsin(p_dia / p_rmin))),
                    component=row.get("Component Installed", "").strip(),
                    port_id=row.get("Port ID", "").strip(),
                )
            )
    return records


def load_hidra_ports(
    path: str | Path = "input_files/HIDRA_ports.csv",
) -> np.ndarray:
    """Load HIDRA port centers and angular sizes as ``phi, theta, width, height`` degrees."""
    records = load_hidra_port_records(path)
    return np.asarray(
        [
            [port.phi_csv_deg, port.theta_deg, port.width_deg, port.height_deg]
            for port in records
        ],
        dtype=np.float32,
    )


def torus_surface_points(
    phi: np.ndarray,
    theta: np.ndarray,
    R0: float,
    a: float,
    surface_offset: float = 0.0,
) -> np.ndarray:
    """Map toroidal/poloidal angles onto the torus surface."""
    phi = np.asarray(phi, dtype=np.float32)
    theta = np.asarray(theta, dtype=np.float32)
    radius = np.float32(R0) + np.float32(a) * np.cos(theta)

    x = radius * np.cos(phi)
    y = -radius * np.sin(phi)
    z = np.float32(a) * np.sin(theta)
    points = np.column_stack([x, y, z]).astype(np.float32)

    if surface_offset:
        normals = np.column_stack(
            [
                np.cos(theta) * np.cos(phi),
                -np.cos(theta) * np.sin(phi),
                np.sin(theta),
            ]
        ).astype(np.float32)
        points += normals * np.float32(surface_offset)

    return points


def _split_visible_closed_curve(points: np.ndarray, visible: np.ndarray) -> list[np.ndarray]:
    """Split a sampled closed curve into visible contiguous line segments."""
    visible = np.asarray(visible, dtype=bool)
    if np.all(visible):
        return [np.vstack([points, points[:1]]).astype(np.float32)]
    if not np.any(visible):
        return []

    false_idx = np.flatnonzero(~visible)
    start = int((false_idx[0] + 1) % len(visible))
    order = (np.arange(len(visible)) + start) % len(visible)
    ordered_points = points[order]
    ordered_visible = visible[order]

    lines: list[np.ndarray] = []
    idx = 0
    while idx < len(ordered_visible):
        if not ordered_visible[idx]:
            idx += 1
            continue
        stop = idx + 1
        while stop < len(ordered_visible) and ordered_visible[stop]:
            stop += 1
        if stop - idx >= 2:
            lines.append(ordered_points[idx:stop].astype(np.float32))
        idx = stop
    return lines


def make_port_boundary_lines(
    port_file: str | Path = "input_files/HIDRA_ports.csv",
    R0: float = 0.72,
    a: float = 0.19,
    samples: int = 96,
    half_shell: bool = False,
    surface_offset: float = 0.003,
    phi_zero_offset_deg: float = PORT_CSV_TO_REPO_ZERO_DEG,
) -> list[np.ndarray]:
    """Return HIDRA port outlines wrapped onto the torus surface."""
    ports = load_hidra_port_records(port_file)
    angle = np.linspace(0.0, 2.0 * np.pi, max(12, int(samples)), endpoint=False, dtype=np.float32)
    lines: list[np.ndarray] = []

    for port in ports:
        phi_csv = port.phi_csv_deg + 0.5 * port.width_deg * np.cos(angle)
        phi = np.radians(port_csv_phi_to_repo_phi(phi_csv, phi_zero_offset_deg))
        theta = np.radians(port.theta_deg) + 0.5 * np.radians(port.height_deg) * np.sin(angle)
        points = torus_surface_points(phi, theta, R0=R0, a=a, surface_offset=surface_offset)

        if half_shell:
            visible = (theta >= -np.pi) & (theta <= 0.0)
            lines.extend(_split_visible_closed_curve(points, visible))
        else:
            lines.append(np.vstack([points, points[:1]]).astype(np.float32))

    return lines


def parse_label_filters(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        raw_items = value.replace(",", ";").split(";")
    else:
        raw_items = value
    return [str(item).strip() for item in raw_items if str(item).strip()]


def make_port_label_positions(
    port_file: str | Path = "input_files/HIDRA_ports.csv",
    R0: float = 0.72,
    a: float = 0.19,
    label_filters: str | list[str] | tuple[str, ...] = ("HIDRA-MAT", "RLP"),
    surface_offset: float = 0.045,
    phi_zero_offset_deg: float = PORT_CSV_TO_REPO_ZERO_DEG,
) -> list[tuple[str, np.ndarray]]:
    """Return labels and 3D positions for matching HIDRA port components."""
    filters = parse_label_filters(label_filters)
    labels: list[tuple[str, np.ndarray]] = []
    for port in load_hidra_port_records(port_file):
        component = port.component
        if not component:
            continue
        label = next((item for item in filters if item.lower() in component.lower()), None)
        if label is None:
            continue
        phi = np.radians(port_csv_phi_to_repo_phi(port.phi_csv_deg, phi_zero_offset_deg))
        theta = np.radians(port.theta_deg)
        point = torus_surface_points(
            np.asarray([phi], dtype=np.float32),
            np.asarray([theta], dtype=np.float32),
            R0=R0,
            a=a,
            surface_offset=surface_offset,
        )[0]
        labels.append((label, point))
    return labels


def repo_phi_unit_vector(phi_deg: float) -> np.ndarray:
    phi = np.radians(float(phi_deg))
    return np.asarray([np.cos(phi), -np.sin(phi), 0.0], dtype=np.float32)


def make_compass_rose_lines(
    size: float = 0.18,
    z: float = -0.19,
    north_phi_deg: float = PORT_CSV_TO_REPO_ZERO_DEG,
    samples: int = 96,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return floor-plane compass rose lines and the north-label position."""
    size = float(size)
    z = float(z)
    north = repo_phi_unit_vector(north_phi_deg)
    east = repo_phi_unit_vector(north_phi_deg + 90.0)
    center = np.asarray([0.0, 0.0, z], dtype=np.float32)

    angle = np.linspace(0.0, 2.0 * np.pi, max(16, int(samples)), endpoint=True, dtype=np.float32)
    circle = center + size * (
        np.cos(angle)[:, None] * north[None, :] + np.sin(angle)[:, None] * east[None, :]
    )

    lines = [
        circle.astype(np.float32),
        np.vstack([center - 0.78 * size * north, center + size * north]).astype(np.float32),
        np.vstack([center - 0.70 * size * east, center + 0.70 * size * east]).astype(np.float32),
    ]

    tip = center + size * north
    left = tip - 0.23 * size * north + 0.13 * size * east
    right = tip - 0.23 * size * north - 0.13 * size * east
    lines.append(np.vstack([left, tip, right]).astype(np.float32))

    return lines, (center + 1.22 * size * north).astype(np.float32)


def frame_count(valid_lengths: np.ndarray, stride: int, steps_per_frame: int) -> int:
    max_strided = int(np.max((valid_lengths + stride - 1) // stride))
    return max(1, (max_strided + steps_per_frame - 1) // steps_per_frame)
