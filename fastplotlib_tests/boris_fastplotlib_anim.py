"""Standalone fastplotlib prototype for Boris trace animation.

This is intentionally separate from ``plot_funcs.plotFuncs.boris_plotTraceAnim``.
It loads the same ``Ion_traces_*.npy`` arrays and renders them with fastplotlib
so you can compare interactive redraw performance against the Matplotlib path.

Example:
    python3 fastplotlib_tests/boris_fastplotlib_anim.py \
        --trace-file output/.../data/Ion_traces_....npy \
        --max-particles 256 --mode trails --line-window 300

If no trace file is provided, the script uses synthetic torus-like traces.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np

from boris_trace_data import (
    frame_count,
    infer_valid_lengths,
    load_boris_trace_file,
    make_compass_rose_lines,
    make_port_boundary_lines,
    make_port_label_positions,
    make_synthetic_boris_traces,
    make_torus_mesh,
    select_trace_particles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="fastplotlib benchmark/prototype for Boris ion trace animation."
    )
    parser.add_argument("--trace-file", type=Path, default=None, help="Path to Ion_traces_*.npy.")
    parser.add_argument("--no-mmap", action="store_true", help="Load the entire trace file into RAM.")
    parser.add_argument("--max-particles", type=int, default=256, help="Limit plotted particles.")
    parser.add_argument("--skip-indices", default="", help="Comma-separated particle indices to skip.")
    parser.add_argument("--stride", type=int, default=1, help="Subsample stored trace points.")
    parser.add_argument("--steps-per-frame", type=int, default=1, help="Trace samples advanced per rendered frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop advancing after this many frames.")
    parser.add_argument(
        "--mode",
        choices=("trails", "grow-lines", "markers"),
        default="trails",
        help=(
            "trails: animated GPU scatter trail/current positions; "
            "grow-lines: closer Matplotlib-style growing line per particle; "
            "markers: current particle positions only."
        ),
    )
    parser.add_argument("--line-window", type=int, default=300, help="Recent points kept for grow-lines mode.")
    parser.add_argument("--trail-length", type=int, default=25, help="Recent points kept in trails mode.")
    parser.add_argument("--marker-size", type=float, default=5.0, help="Scatter marker size.")
    parser.add_argument("--line-thickness", type=float, default=1.25, help="Line thickness in grow-lines mode.")
    parser.add_argument("--show-static-lines", action="store_true", help="Draw full selected traces once.")
    parser.add_argument("--R0", type=float, default=0.72, help="Major radius for the torus shell.")
    parser.add_argument("--a", type=float, default=0.19, help="Minor radius for the torus shell.")
    parser.add_argument("--torus-half", choices=("bottom", "full"), default="bottom", help="Render only the bottom half of the torus or the full torus.")
    parser.add_argument("--show-ports", action=argparse.BooleanOptionalAction, default=True, help="Show HIDRA port outlines on the torus wall.")
    parser.add_argument("--port-file", type=Path, default=Path("input_files/HIDRA_ports.csv"), help="CSV file with HIDRA port locations.")
    parser.add_argument("--port-color", default="#F5F7FA", help="Port outline color.")
    parser.add_argument("--port-alpha", type=float, default=0.85, help="Port outline alpha.")
    parser.add_argument("--port-line-thickness", type=float, default=1.0, help="Port outline line thickness.")
    parser.add_argument("--port-samples", type=int, default=96, help="Samples per wrapped port outline.")
    parser.add_argument("--port-surface-offset", type=float, default=0.003, help="Outward offset from the torus surface for port outlines.")
    parser.add_argument("--port-phi-zero-offset", type=float, default=162.0, help="CSV phi, in degrees, corresponding to repo phi=0.")
    parser.add_argument("--label-ports", action=argparse.BooleanOptionalAction, default=False, help="Label matching HIDRA ports.")
    parser.add_argument("--port-label-filter", default="HIDRA-MAT;RLP", help="Semicolon-separated component substrings to label.")
    parser.add_argument("--port-label-color", default="#F5F7FA", help="Port label color.")
    parser.add_argument("--port-label-font-size", type=float, default=16.0, help="Port label font size.")
    parser.add_argument("--port-label-surface-offset", type=float, default=0.045, help="Outward offset from the torus surface for port labels.")
    parser.add_argument("--show-compass", action=argparse.BooleanOptionalAction, default=True, help="Show a floor-plane north compass in the torus center.")
    parser.add_argument("--compass-size", type=float, default=0.18, help="Compass rose radius in meters.")
    parser.add_argument("--compass-z", type=float, default=None, help="Compass z location. Defaults to -a - 0.015.")
    parser.add_argument("--compass-color", default="#F5F7FA", help="Compass rose color.")
    parser.add_argument("--compass-alpha", type=float, default=0.80, help="Compass rose alpha.")
    parser.add_argument("--compass-line-thickness", type=float, default=2.0, help="Compass line thickness.")
    parser.add_argument("--compass-label-color", default="#F5F7FA", help="Compass north label color.")
    parser.add_argument("--compass-font-size", type=float, default=18.0, help="Compass north label font size.")
    parser.add_argument("--size", default="1280x720", help="Window size as WIDTHxHEIGHT.")
    parser.add_argument(
        "--present-method",
        choices=("bitmap", "screen"),
        default="bitmap",
        help=(
            "Canvas presentation method. 'screen' may improve FPS but is less "
            "portable than the default bitmap path."
        ),
    )
    parser.add_argument("--synthetic-steps", type=int, default=4_000, help="Synthetic trace steps.")
    parser.add_argument("--synthetic-particles", type=int, default=512, help="Synthetic particle count.")
    parser.add_argument("--synthetic-turns", type=float, default=8.0, help="Synthetic trace toroidal turns.")
    parser.add_argument(
        "--benchmark-window",
        type=int,
        default=180,
        help="Print FPS over this many animation updates.",
    )
    return parser.parse_args()


def parse_skip_indices(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def import_fastplotlib():
    try:
        return importlib.import_module("fastplotlib")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "fastplotlib is not importable in this Python environment. "
            "Run this with the environment where you installed fastplotlib."
        ) from exc


def load_or_make_traces(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.trace_file is None:
        traces = make_synthetic_boris_traces(
            n_steps=args.synthetic_steps,
            n_particles=args.synthetic_particles,
            R0=args.R0,
            a=args.a,
            turns=args.synthetic_turns,
        )
        return traces, "synthetic"

    traces = load_boris_trace_file(args.trace_file, mmap=not args.no_mmap)
    return traces, str(args.trace_file)


def particle_colors(n_particles: int, alpha: float = 1.0) -> np.ndarray:
    """Small HSV-like palette without pulling in Matplotlib."""
    hue = np.linspace(0.0, 1.0, int(n_particles), endpoint=False, dtype=np.float32)
    r = 0.5 + 0.5 * np.cos(2.0 * np.pi * (hue + 0.00))
    g = 0.5 + 0.5 * np.cos(2.0 * np.pi * (hue + 0.66))
    b = 0.5 + 0.5 * np.cos(2.0 * np.pi * (hue + 0.33))
    return np.column_stack([r, g, b, np.full_like(r, float(alpha))]).astype(np.float32)


def parse_rgba(value: str, alpha: float = 1.0) -> np.ndarray:
    named = {
        "white": (1.0, 1.0, 1.0),
        "black": (0.0, 0.0, 0.0),
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 0.8, 0.0),
        "blue": (0.0, 0.25, 1.0),
        "cyan": (0.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "orange": (1.0, 0.55, 0.0),
        "purple": (0.55, 0.25, 0.95),
    }
    text = value.strip().lower()
    if text in named:
        return np.asarray((*named[text], alpha), dtype=np.float32)

    if text.startswith("#") and len(text) in (7, 9):
        channels = [int(text[i : i + 2], 16) / 255.0 for i in range(1, len(text), 2)]
        if len(channels) == 3:
            channels.append(alpha)
        else:
            channels[3] *= alpha
        return np.asarray(channels, dtype=np.float32)

    parts = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) not in (3, 4):
        raise ValueError(f"Could not parse color {value!r}.")
    if max(parts) > 1.0:
        parts = [channel / 255.0 for channel in parts]
    if len(parts) == 3:
        parts.append(alpha)
    else:
        parts[3] *= alpha
    return np.asarray(parts, dtype=np.float32)


def make_nan_buffer(shape: tuple[int, ...]) -> np.ndarray:
    out = np.empty(shape, dtype=np.float32)
    out[:] = np.nan
    return out


def add_torus(subplot, R0: float, a: float, half_shell: bool = True) -> None:
    positions, indices = make_torus_mesh(R0=R0, a=a, half_shell=half_shell)
    if hasattr(subplot, "add_mesh"):
        try:
            subplot.add_mesh(
                positions=positions,
                indices=indices,
                colors=(0.18, 0.42, 0.72, 0.18),
                mode="basic",
                alpha=0.18,
            )
            return
        except TypeError:
            subplot.add_mesh(positions, indices, colors=(0.18, 0.42, 0.72, 0.18), mode="basic")
            return

    # Older fastplotlib versions may not expose add_mesh. Use sparse rings as a fallback.
    ring_count = 16
    theta_min, theta_max = (-np.pi, 0.0) if half_shell else (-np.pi, np.pi)
    theta = np.linspace(theta_min, theta_max, 128, dtype=np.float32)
    phi_values = np.linspace(0.0, 2.0 * np.pi, ring_count, endpoint=False, dtype=np.float32)
    rings = []
    for phi in phi_values:
        x = (R0 + a * np.cos(theta)) * np.cos(phi)
        y = -(R0 + a * np.cos(theta)) * np.sin(phi)
        z = a * np.sin(theta)
        rings.append(np.column_stack([x, y, z]).astype(np.float32))
    subplot.add_line_collection(rings, colors=(0.18, 0.42, 0.72, 0.45), thickness=1.0)


def add_ports(subplot, args: argparse.Namespace) -> None:
    if not args.show_ports and not args.label_ports:
        return

    if args.show_ports:
        lines = make_port_boundary_lines(
            args.port_file,
            R0=args.R0,
            a=args.a,
            samples=args.port_samples,
            half_shell=(args.torus_half == "bottom"),
            surface_offset=args.port_surface_offset,
            phi_zero_offset_deg=args.port_phi_zero_offset,
        )
        if lines:
            subplot.add_line_collection(
                lines,
                colors=parse_rgba(args.port_color, args.port_alpha),
                thickness=float(args.port_line_thickness),
            )

    if args.label_ports:
        labels = make_port_label_positions(
            args.port_file,
            R0=args.R0,
            a=args.a,
            label_filters=args.port_label_filter,
            surface_offset=args.port_label_surface_offset,
            phi_zero_offset_deg=args.port_phi_zero_offset,
        )
        for text, position in labels:
            subplot.add_text(
                text,
                font_size=float(args.port_label_font_size),
                face_color=parse_rgba(args.port_label_color, 1.0),
                outline_color=parse_rgba("#02070D", 1.0),
                outline_thickness=0.15,
                screen_space=True,
                offset=tuple(float(value) for value in position),
                anchor="middle-center",
            )


def add_compass(subplot, args: argparse.Namespace) -> None:
    if not args.show_compass:
        return

    compass_z = -float(args.a) - 0.015 if args.compass_z is None else float(args.compass_z)
    lines, label_pos = make_compass_rose_lines(
        size=args.compass_size,
        z=compass_z,
        north_phi_deg=args.port_phi_zero_offset,
    )
    subplot.add_line_collection(
        lines,
        colors=parse_rgba(args.compass_color, args.compass_alpha),
        thickness=float(args.compass_line_thickness),
    )
    subplot.add_text(
        "N",
        font_size=float(args.compass_font_size),
        face_color=parse_rgba(args.compass_label_color, 1.0),
        outline_color=parse_rgba("#02070D", 1.0),
        outline_thickness=0.15,
        screen_space=True,
        offset=tuple(float(value) for value in label_pos),
        anchor="middle-center",
    )


def add_static_lines(subplot, traces: np.ndarray, valid_lengths: np.ndarray, stride: int) -> None:
    lines = [
        np.asarray(traces[: int(valid_length) : stride, idx, :], dtype=np.float32)
        for idx, valid_length in enumerate(valid_lengths)
        if int(valid_length) >= 2
    ]
    if lines:
        subplot.add_line_collection(lines, colors=(0.6, 0.72, 0.9, 0.22), thickness=0.75)


def add_trail_graphics(subplot, n_particles: int, trail_length: int, marker_size: float):
    colors = np.repeat(particle_colors(n_particles, alpha=1.0), int(trail_length), axis=0)
    if trail_length > 1:
        alphas = np.linspace(0.08, 1.0, int(trail_length), dtype=np.float32)
        colors[:, 3] = np.tile(alphas, int(n_particles))

    data = make_nan_buffer((int(n_particles) * int(trail_length), 3))
    scatter = subplot.add_scatter(
        data=data,
        sizes=float(marker_size),
        colors=colors,
        edge_width=0.0,
        mode="simple",
    )
    return scatter, data


def add_marker_graphic(subplot, n_particles: int, marker_size: float):
    data = make_nan_buffer((int(n_particles), 3))
    scatter = subplot.add_scatter(
        data=data,
        sizes=float(marker_size),
        colors=particle_colors(n_particles, alpha=1.0),
        edge_width=0.0,
        mode="simple",
    )
    return scatter, data


def add_growing_lines(subplot, n_particles: int, line_window: int, thickness: float):
    colors = particle_colors(n_particles, alpha=0.95)
    buffers = []
    graphics = []
    for idx in range(int(n_particles)):
        buffer = make_nan_buffer((int(line_window), 3))
        graphic = subplot.add_line(
            data=buffer,
            thickness=float(thickness),
            colors=colors[idx],
            uniform_color=True,
        )
        buffers.append(buffer)
        graphics.append(graphic)
    return graphics, buffers


def setup_camera(subplot, R0: float, a: float) -> None:
    extent = float(R0 + a + 0.06)
    try:
        subplot.camera.show_rect(-extent, extent, -extent, extent)
    except Exception:
        pass
    try:
        #subplot.camera.local.position = (1.45, -1.85, 0.75)
        subplot.camera.position.set = (-1.85, -1.85, 0.75)
        subplot.camera.look_at((0.0, 0.0, 0.0))
        subplot.camera.fov = 60.0
    except Exception:
        pass
    try:
        subplot.axes.grids.xy.visible = True
        subplot.axes.grids.xz.visible = False
        subplot.axes.grids.yz.visible = False
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    if args.stride < 1 or args.steps_per_frame < 1:
        raise SystemExit("--stride and --steps-per-frame must be positive.")
    if args.line_window < 2:
        raise SystemExit("--line-window must be at least 2.")
    if args.trail_length < 1:
        raise SystemExit("--trail-length must be at least 1.")

    fpl = import_fastplotlib()

    #def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = args.size.lower().split("x", 1)
        size = int(width), int(height)
    except Exception as exc:
        raise argparse.ArgumentTypeError("--size must look like 1280x720") from exc
    #size = parse_size(args.size)



    raw_traces, source_label = load_or_make_traces(args)
    valid_lengths = infer_valid_lengths(raw_traces)
    selection = select_trace_particles(
        raw_traces,
        valid_lengths=valid_lengths,
        max_particles=args.max_particles,
        skip_indices=parse_skip_indices(args.skip_indices),
    )
    traces = selection.traces
    valid_lengths = selection.valid_lengths
    n_particles = traces.shape[1]
    n_frames = frame_count(valid_lengths, args.stride, args.steps_per_frame)
    if args.max_frames is not None:
        n_frames = min(n_frames, int(args.max_frames))

    print(f"fastplotlib: {getattr(fpl, '__version__', 'unknown')}")
    print(f"source: {source_label}")
    print(f"selected trace shape: {traces.shape}, dtype={traces.dtype}")
    print(f"selected particles: {n_particles}; animation frames: {n_frames}")
    print(f"mode: {args.mode}; stride={args.stride}; steps_per_frame={args.steps_per_frame}")

    figure = fpl.Figure(
        cameras="3d",
        controller_types="orbit",
        canvas_kwargs={"present_method": args.present_method},
        size=size,
    )
    subplot = figure[0, 0]
    try:
        subplot.set_title(f"Boris traces via fastplotlib ({args.mode})")
    except Exception:
        pass

    add_torus(subplot, args.R0, args.a, half_shell=(args.torus_half == "bottom"))
    add_ports(subplot, args)
    add_compass(subplot, args)
    if args.show_static_lines:
        add_static_lines(subplot, traces, valid_lengths, args.stride)

    marker_graphic = marker_data = None
    trail_graphic = trail_data = None
    line_graphics = line_buffers = None

    if args.mode in {"markers", "grow-lines"}:
        marker_graphic, marker_data = add_marker_graphic(subplot, n_particles, args.marker_size)
    if args.mode == "trails":
        trail_graphic, trail_data = add_trail_graphics(
            subplot, n_particles, args.trail_length, args.marker_size
        )
    if args.mode == "grow-lines":
        line_graphics, line_buffers = add_growing_lines(
            subplot, n_particles, args.line_window, args.line_thickness
        )

    setup_camera(subplot, args.R0, args.a)

    state = {
        "frame": 0,
        "updates": 0,
        "t0": time.perf_counter(),
        "window_t0": time.perf_counter(),
    }

    def update_animation():
        frame_idx = state["frame"]
        strided_end = frame_idx * args.steps_per_frame + 1

        if args.mode == "markers":
            for particle_idx in range(n_particles):
                strided_len = (int(valid_lengths[particle_idx]) + args.stride - 1) // args.stride
                sample_idx = min(strided_end - 1, strided_len - 1) * args.stride
                marker_data[particle_idx] = traces[sample_idx, particle_idx, :]
            marker_graphic.data = marker_data

        elif args.mode == "trails":
            for particle_idx in range(n_particles):
                strided_len = (int(valid_lengths[particle_idx]) + args.stride - 1) // args.stride
                end = min(strided_end, strided_len)
                start = max(0, end - args.trail_length)
                raw_indices = np.arange(start, end, dtype=np.int64) * args.stride
                offset = particle_idx * args.trail_length
                trail_data[offset : offset + args.trail_length, :] = np.nan
                if raw_indices.size:
                    dest_start = offset + args.trail_length - raw_indices.size
                    trail_data[dest_start : offset + args.trail_length, :] = traces[
                        raw_indices, particle_idx, :
                    ]
            trail_graphic.data = trail_data

        elif args.mode == "grow-lines":
            for particle_idx in range(n_particles):
                strided_len = (int(valid_lengths[particle_idx]) + args.stride - 1) // args.stride
                end = min(strided_end, strided_len)
                start = max(0, end - args.line_window)
                raw_indices = np.arange(start, end, dtype=np.int64) * args.stride
                buffer = line_buffers[particle_idx]
                buffer[:] = np.nan
                if raw_indices.size:
                    buffer[: raw_indices.size, :] = traces[raw_indices, particle_idx, :]
                line_graphics[particle_idx].data = buffer

                sample_idx = max(0, end - 1) * args.stride
                marker_data[particle_idx] = traces[sample_idx, particle_idx, :]
            marker_graphic.data = marker_data

        state["frame"] = (state["frame"] + 1) % max(1, n_frames)
        state["updates"] += 1

        if state["updates"] % args.benchmark_window == 0:
            now = time.perf_counter()
            dt = now - state["window_t0"]
            total_dt = now - state["t0"]
            print(
                f"updates={state['updates']:6d} "
                f"window_fps={args.benchmark_window / dt:8.2f} "
                f"avg_fps={state['updates'] / total_dt:8.2f} "
                f"frame={state['frame']:6d}/{n_frames}",
                flush=True,
            )
            state["window_t0"] = now

    figure.add_animations(update_animation)
    figure.show()

    if __name__ == "__main__":
        fpl.loop.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
