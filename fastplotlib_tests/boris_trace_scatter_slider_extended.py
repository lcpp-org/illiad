"""Interactive fastplotlib scatter viewer for Boris ion traces.

The viewer shows one scatter point per particle at a selected stored trace
timestep. A small PySide6/Qt control panel provides a time slider plus
step/play controls.

Example:
    conda run -n testenv python fastplotlib_tests/boris_trace_scatter_slider_extended.py \
        --inputs-json animation_inputs.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# import UIUC colors for consistency with other plots
try:
    from plot_funcs.plotFuncs import UIUC
except ImportError:
    UIUC = {
        "il_blue": '#13294B',
        "il_orange": '#FF5F05',
        "il_storm": '#707372',  # Added a default color for il_storm
        "il_stormdark1": '#4A4C4B',  # Added a default color for il_storm
        "il_stormdark2": '#252525',  # Added a default color for il_storm
    }

colors = [

    (0.0, UIUC["il_storm"]),
    (0.01, UIUC["il_stormdark2"]),
    (0.05, UIUC["il_blue"]),
    (0.5, UIUC["il_blue"]),
    (0.75, UIUC["il_orange"]),
    (1.0, UIUC["il_orange"]),
    ]
#colors = [UIUC["il_blue"], UIUC["il_orange"]]
custom_cmap = LinearSegmentedColormap.from_list("my_gradient", colors, N=256)
mpl.colormaps.register(cmap=custom_cmap, name="my_registered_cmap")



from boris_trace_data import (
    infer_valid_lengths,
    load_boris_trace_files,
    normalize_trace_paths,
    make_compass_rose_lines,
    make_port_boundary_lines,
    make_port_label_positions,
    make_synthetic_boris_traces,
    make_torus_mesh,
    select_trace_particles,
)





class ArgFileParser(argparse.ArgumentParser):
    """Allow @args-file syntax with shell-like quoting and comments."""

    def convert_arg_line_to_args(self, arg_line: str):
        if not arg_line.strip():
            return []
        return shlex.split(arg_line, comments=True, posix=True)


def json_config_to_args(path: Path, parser: argparse.ArgumentParser) -> list[str]:
    """Translate a JSON config object into argparse tokens."""
    try:
        with path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
    except OSError as exc:
        raise SystemExit(f"Could not read --inputs-json file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse --inputs-json file {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise SystemExit("--inputs-json must contain a JSON object.")

    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest != argparse.SUPPRESS and action.dest != "help"
    }
    tokens: list[str] = []

    for raw_key, value in config.items():
        dest = raw_key.replace("-", "_")
        if dest == "inputs_json":
            continue
        if dest not in actions:
            valid = ", ".join(sorted(key for key in actions if key != "inputs_json"))
            raise SystemExit(f"Unknown JSON input {raw_key!r}. Valid inputs are: {valid}")
        if value is None:
            continue

        action = actions[dest]
        long_option = next(
            (item for item in action.option_strings if item.startswith("--") and not item.startswith("--no-")),
            None,
        )
        if long_option is None:
            continue

        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                raise SystemExit(f"JSON input {raw_key!r} must be true or false.")
            tokens.append(long_option if value else f"--no-{long_option[2:]}")
            continue

        if getattr(action, "nargs", None) == 0 and isinstance(getattr(action, "const", None), bool):
            if not isinstance(value, bool):
                raise SystemExit(f"JSON input {raw_key!r} must be true or false.")
            if value:
                tokens.append(long_option)
            continue

        if isinstance(value, list):
            for item in value:
                tokens.extend([long_option, str(item)])
            continue

        tokens.extend([long_option, str(value)])

    return tokens


def build_arg_parser() -> ArgFileParser:
    parser = ArgFileParser(
        description="Scatter all Boris particles at a selected trace timestep using fastplotlib.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--inputs-json", type=Path, default=None, help="Load animation inputs from a JSON object.")
    parser.add_argument(
        "--trace-file",
        type=Path,
        action="append",
        default=None,
        help="Path to Ion_traces_*.npy. Repeat or pass a JSON list to combine trace files.",
    )
    parser.add_argument("--no-mmap", action="store_true", help="Load the whole trace file into RAM.")
    parser.add_argument("--max-particles", type=int, default=None, help="Optional cap for testing.")
    parser.add_argument("--skip-indices", default="", help="Comma-separated particle indices to skip.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Slider frame increment in stored samples.")
    parser.add_argument("--initial-frame", type=int, default=0, help="Initial stored sample index.")
    parser.add_argument("--marker-size", type=float, default=6.0, help="Scatter marker size in screen pixels.")
    parser.add_argument(
        "--color-mode",
        choices=("solid", "particle", "speed", "energy"),
        default="particle",
        help=(
            "Marker coloring mode. speed/energy are estimated from finite differences "
            "between stored trace positions."
        ),
    )
    parser.add_argument(
        "--marker-color",
        default="cyan",
        help="Solid marker color for --color-mode solid. Accepts names, hex, or r,g,b[,a].",
    )
    parser.add_argument("--marker-alpha", type=float, default=1.0, help="Marker alpha for solid/particle colors.")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap for speed/energy colors.")
    parser.add_argument(
        "--cmap-colors",
        default=None,
        help=(
            "Custom colormap as semicolon-separated colors, e.g. "
            "'#0011ff;cyan;yellow;red'. Overrides --cmap for speed/energy."
        ),
    )
    parser.add_argument("--cmap-reverse", action="store_true", help="Reverse --cmap or --cmap-colors.")
    parser.add_argument("--color-vmin", type=float, default=None, help="Fixed lower color limit for speed/energy.")
    parser.add_argument("--color-vmax", type=float, default=None, help="Fixed upper color limit for speed/energy.")
    parser.add_argument(
        "--sample-dt",
        type=float,
        default=1.0,
        help="Time between stored trace samples, seconds. Use DT * TRACE_STRIDE for physical speed/energy.",
    )
    parser.add_argument(
        "--ion-mass-amu",
        type=float,
        default=6.941,
        help="Ion mass in amu for --color-mode energy. Default is lithium.",
    )
    parser.add_argument("--trail-length", type=int, default=0, help="Number of previous samples to show as particle trails.")
    parser.add_argument("--trail-stride", type=int, default=1, help="Stored-sample spacing between trail points.")
    parser.add_argument("--trail-alpha-min", type=float, default=0.05, help="Alpha for oldest trail points.")
    parser.add_argument(
        "--trail-color",
        default="same",
        help="Trail color: 'same' follows particle/solid colors where possible, or pass a color spec.",
    )
    parser.add_argument("--trail-marker-size", type=float, default=None, help="Trail marker size. Defaults to 0.7 times marker size.")
    parser.add_argument("--histogram-bins", type=int, default=40, help="Number of energy histogram bins.")
    parser.add_argument("--histogram-height", type=int, default=220, help="Energy histogram panel height in pixels.")
    parser.add_argument("--histogram-log-y", action=argparse.BooleanOptionalAction, default=False, help="Use a log-scaled histogram y-axis.")
    parser.add_argument("--hide-histogram", action="store_true", help="Hide the energy histogram panel.")
    parser.add_argument("--histogram-color-vmin", type=float, default=None, help="Fixed lower color limit for histogram bar colors, eV.")
    parser.add_argument("--histogram-color-vmax", type=float, default=None, help="Fixed upper color limit for histogram bar colors, eV.")
    parser.add_argument(
        "--histogram-xmax",
        type=float,
        default=None,
        help="Fixed histogram x-axis maximum, eV. Defaults to 5x the fixed histogram color max when available.",
    )
    parser.add_argument(
        "--histogram-auto-xmax",
        action="store_true",
        help="Autoscale the histogram x-axis from each frame instead of using a fixed maximum.",
    )
    parser.add_argument("--hide-running-fraction", action="store_true", help="Hide the running-particle fraction side plot.")
    parser.add_argument("--side-panel-width", type=int, default=380, help="Width of the stacked side diagnostic plots in pixels.")
    parser.add_argument("--plot-background", default="#02070D", help="Shared background color for the viewer and side plots.")
    parser.add_argument("--plot-foreground", default="#E8EDF2", help="Shared foreground/text color for the side plots.")
    parser.add_argument("--R0", type=float, default=0.72, help="Major radius for the torus shell.")
    parser.add_argument("--a", type=float, default=0.19, help="Minor radius for the torus shell.")
    parser.add_argument("--size", default="1280x760", help="Window size as WIDTHxHEIGHT.")
    parser.add_argument("--hide-zero-rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-torus", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument(
        "--torus-style",
        choices=("mesh", "wireframe", "both"),
        default="wireframe",
        help="Torus rendering style.",
    )
    parser.add_argument("--torus-color", default="#d40707", help="Translucent mesh torus color.")
    parser.add_argument("--torus-alpha", type=float, default=0.16, help="Translucent mesh torus alpha.")
    parser.add_argument("--torus-wire-color", default="#9bb7d4", help="Wireframe torus color.")
    parser.add_argument("--torus-wire-alpha", type=float, default=0.42, help="Wireframe torus alpha.")
    parser.add_argument("--torus-wire-thickness", type=float, default=1.0, help="Wireframe line thickness.")
    parser.add_argument(
        "--torus-half",
        choices=("bottom", "full"),
        default="bottom",
        help="Render only the bottom half of the torus or the full torus.",
    )
    parser.add_argument("--torus-nphi", type=int, default=144, help="Torus mesh/ring toroidal resolution.")
    parser.add_argument("--torus-ntheta", type=int, default=64, help="Torus mesh/ring poloidal resolution.")
    parser.add_argument("--torus-wire-phi", type=int, default=24, help="Number of toroidal wire rings.")
    parser.add_argument("--torus-wire-theta", type=int, default=9, help="Number of poloidal wire rings.")
    parser.add_argument("--axes", action=argparse.BooleanOptionalAction, default=False, help="Show or hide fastplotlib axes.")
    parser.add_argument("--export-mp4", type=Path, default=None, help="Render an offscreen MP4 instead of opening the GUI.")
    parser.add_argument("--export-start", type=int, default=0, help="First stored frame for --export-mp4.")
    parser.add_argument("--export-stop", type=int, default=None, help="Exclusive stop frame for --export-mp4.")
    parser.add_argument("--export-step", type=int, default=1, help="Stored frame step for --export-mp4.")
    parser.add_argument("--export-fps", type=float, default=30.0, help="Output frames per second for --export-mp4.")
    parser.add_argument("--export-size", default=None, help="Export canvas size WIDTHxHEIGHT. Defaults to --size.")
    parser.add_argument("--synthetic-steps", type=int, default=4_000, help="Synthetic trace steps.")
    parser.add_argument("--synthetic-particles", type=int, default=2_400, help="Synthetic particle count.")
    parser.add_argument("--synthetic-turns", type=float, default=8.0, help="Synthetic toroidal turns.")
    parser.add_argument("--play-fps", type=float, default=60.0, help="Playback frame rate for Play.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    argv = list(sys.argv[1:] if argv is None else argv)

    pre_parser = ArgFileParser(add_help=False, fromfile_prefix_chars="@")
    pre_parser.add_argument("--inputs-json", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    json_args = []
    if pre_args.inputs_json is not None:
        json_args = json_config_to_args(pre_args.inputs_json, parser)

    return parser.parse_args(json_args + argv)


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = value.lower().split("x", 1)
        return int(width), int(height)
    except Exception as exc:
        raise argparse.ArgumentTypeError("--size must look like 1280x760") from exc


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
            "Run with the conda environment where fastplotlib is installed, e.g. "
            "`conda run -n testenv python ...`."
        ) from exc


def import_qt_and_canvas():
    try:
        from PySide6 import QtCore, QtWidgets
        from rendercanvas.pyside6 import QRenderWidget
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This script requires PySide6 and rendercanvas' PySide6 backend. "
            "They appear to be available in your `testenv`; run with "
            "`conda run -n testenv python ...`."
        ) from exc
    return QtCore, QtWidgets, QRenderWidget


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

    trace_paths = normalize_trace_paths(args.trace_file)
    traces = load_boris_trace_files(trace_paths, mmap=not args.no_mmap)
    if len(trace_paths) == 1:
        return traces, str(trace_paths[0])
    return traces, f"{len(trace_paths)} trace files: " + ", ".join(str(path) for path in trace_paths)


def particle_colors(n_particles: int) -> np.ndarray:
    hue = np.linspace(0.0, 1.0, int(n_particles), endpoint=False, dtype=np.float32)
    r = 0.5 + 0.5 * np.cos(2.0 * np.pi * (hue + 0.00))
    g = 0.5 + 0.5 * np.cos(2.0 * np.pi * (hue + 0.66))
    b = 0.5 + 0.5 * np.cos(2.0 * np.pi * (hue + 0.33))
    return np.column_stack([r, g, b, np.ones_like(r)]).astype(np.float32)


def parse_rgba(value: str, alpha: float = 1.0) -> np.ndarray:
    """Parse a simple color spec into float32 RGBA."""
    try:
        from matplotlib.colors import to_rgba

        rgba = to_rgba(value, alpha=alpha)
        return np.asarray(rgba, dtype=np.float32)
    except Exception:
        pass

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


def solid_colors(n_particles: int, color: str, alpha: float) -> np.ndarray:
    rgba = parse_rgba(color, alpha)
    return np.repeat(rgba[None, :], int(n_particles), axis=0).astype(np.float32)


def apply_alpha(colors: np.ndarray, alpha: float) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float32).copy()
    colors[:, 3] *= float(alpha)
    return colors


def parse_custom_cmap(value: str | None, reverse: bool = False) -> np.ndarray | None:
    if value is None:
        return None

    colors = [parse_rgba(item.strip()) for item in value.split(";") if item.strip()]
    if len(colors) < 2:
        raise ValueError("--cmap-colors requires at least two semicolon-separated colors.")

    cmap = np.asarray(colors, dtype=np.float32)
    if reverse:
        cmap = cmap[::-1].copy()
    return cmap


def interpolate_custom_cmap(normed: np.ndarray, cmap_colors: np.ndarray) -> np.ndarray:
    normed = np.clip(np.asarray(normed, dtype=np.float32), 0.0, 1.0)
    scaled = normed * float(len(cmap_colors) - 1)
    left = np.floor(scaled).astype(np.int64)
    right = np.clip(left + 1, 0, len(cmap_colors) - 1)
    frac = (scaled - left).astype(np.float32)[:, None]
    return cmap_colors[left] * (1.0 - frac) + cmap_colors[right] * frac


def scalar_to_colors(
    values: np.ndarray,
    cmap_name: str,
    out: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    custom_cmap: np.ndarray | None = None,
    reverse: bool = False,
) -> tuple[float, float, float, float]:
    """Map scalar values to RGBA colors and return finite min/max plus color limits."""
    finite = np.isfinite(values)
    out[:] = (0.0, 0.0, 0.0, 0.0)
    if not np.any(finite):
        return np.nan, np.nan, np.nan, np.nan

    finite_values = values[finite]
    data_min = float(np.nanmin(finite_values))
    data_max = float(np.nanmax(finite_values))
    color_min = data_min if vmin is None else float(vmin)
    color_max = data_max if vmax is None else float(vmax)
    if not np.isfinite(color_min) or not np.isfinite(color_max) or color_max <= color_min:
        color_max = color_min + 1.0

    normed = np.clip((values - color_min) / (color_max - color_min), 0.0, 1.0)
    if custom_cmap is not None:
        out[finite] = interpolate_custom_cmap(normed[finite], custom_cmap)
    else:
        try:
            from matplotlib import colormaps

            cmap = colormaps[cmap_name].reversed() if reverse else colormaps[cmap_name]
            mapped = cmap(normed[finite])
            out[finite] = np.asarray(mapped, dtype=np.float32)
        except Exception:
            grey = normed[finite].astype(np.float32)
            if reverse:
                grey = 1.0 - grey
            out[finite] = np.column_stack([grey, grey, grey, np.ones_like(grey)])

    return data_min, data_max, color_min, color_max

def instantaneous_speed(
    traces: np.ndarray,
    frame_idx: int,
    valid_lengths: np.ndarray,
    sample_dt: float,
    out: np.ndarray,
) -> np.ndarray:
    """Estimate per-particle speed from neighboring stored positions."""
    if sample_dt <= 0.0:
        raise ValueError("--sample-dt must be positive.")

    frame_idx = max(0, min(int(frame_idx), traces.shape[0] - 1))
    particle_idx = np.arange(traces.shape[1])
    last_valid = np.maximum(valid_lengths.astype(np.int64) - 1, 0)
    prev_idx = np.minimum(np.maximum(frame_idx - 1, 0), last_valid)
    next_idx = np.minimum(frame_idx + 1, last_valid)

    prev_pos = np.asarray(traces[prev_idx, particle_idx, :], dtype=np.float32)
    next_pos = np.asarray(traces[next_idx, particle_idx, :], dtype=np.float32)
    denom = (next_idx - prev_idx).astype(np.float32) * float(sample_dt)

    out[:] = np.nan
    valid = denom > 0.0
    if np.any(valid):
        displacement = next_pos[valid] - prev_pos[valid]
        out[valid] = np.linalg.norm(displacement, axis=1) / denom[valid]
    return out


def instantaneous_energy_ev(
    traces: np.ndarray,
    frame_idx: int,
    valid_lengths: np.ndarray,
    sample_dt: float,
    ion_mass_amu: float,
    out: np.ndarray,
) -> np.ndarray:
    """Estimate per-particle kinetic energy in eV at one stored trace sample."""
    instantaneous_speed(traces, frame_idx, valid_lengths, sample_dt, out)
    kg_per_amu = 1.660_539_068e-27
    joules_per_ev = 1.602_176_634e-19
    mass_kg = float(ion_mass_amu) * kg_per_amu
    out[:] = 0.5 * mass_kg * out**2 / joules_per_ev
    return out


def running_fraction_over_time(
    traces: np.ndarray,
    valid_lengths: np.ndarray,
    max_frame: int,
    sample_dt: float,
    ion_mass_amu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stored frame indices and fraction of selected particles with energy > 0."""
    frame_indices = np.arange(int(max_frame) + 1, dtype=np.int64)
    fractions = np.zeros(frame_indices.shape, dtype=np.float32)
    energy = np.empty(traces.shape[1], dtype=np.float32)
    denominator = max(1, int(traces.shape[1]))

    for output_idx, frame_idx in enumerate(frame_indices):
        instantaneous_energy_ev(traces, int(frame_idx), valid_lengths, sample_dt, ion_mass_amu, energy)
        fractions[output_idx] = float(np.count_nonzero(np.isfinite(energy) & (energy > 0.0))) / denominator

    return frame_indices, fractions


def frame_positions(
    traces: np.ndarray,
    frame_idx: int,
    valid_lengths: np.ndarray,
    hide_zero_rows: bool,
    out: np.ndarray,
) -> np.ndarray:
    clamped = max(0, min(int(frame_idx), traces.shape[0] - 1))
    out[:] = np.asarray(traces[clamped, :, :], dtype=np.float32)

    inactive = valid_lengths <= clamped
    if hide_zero_rows:
        inactive |= ~np.any(out != 0.0, axis=1)
    if np.any(inactive):
        out[inactive] = np.nan
    return out


def trail_positions(traces: np.ndarray,
                    frame_idx: int,
                    valid_lengths: np.ndarray,
                    trail_length: int,
                    trail_stride: int,
                    hide_zero_rows: bool,
                    out: np.ndarray,
                    ) -> np.ndarray:
    
    """Fill an ``(n_particles * trail_length, 3)`` buffer with recent positions."""
    trail_length = max(1, int(trail_length))
    trail_stride = max(1, int(trail_stride))
    frame_idx = max(0, min(int(frame_idx), traces.shape[0] - 1))
    out[:] = np.nan

    for particle_idx in range(traces.shape[1]):
        valid_length = int(valid_lengths[particle_idx])
        if valid_length < 2:
            continue

        end = min(frame_idx, valid_length - 1)
        raw_indices = end - np.arange(trail_length - 1, -1, -1, dtype=np.int64) * trail_stride
        raw_indices = raw_indices[raw_indices >= 0]
        if raw_indices.size == 0:
            continue

        values = np.asarray(traces[raw_indices, particle_idx, :], dtype=np.float32)
        if hide_zero_rows:
            keep = np.any(values != 0.0, axis=1)
            values = values[keep]
        if values.size == 0:
            continue

        offset = particle_idx * trail_length
        out[offset + trail_length - values.shape[0] : offset + trail_length, :] = values

    return out


def make_trail_colors(n_particles: int,
                      trail_length: int,
                      marker_colors: np.ndarray,
                      trail_color: str,
                      alpha_min: float,
                      ) -> np.ndarray:
    
    """Create static RGBA colors for the trail scatter buffer."""
    trail_length = max(1, int(trail_length))
    alphas = np.linspace(float(alpha_min), 0.75, trail_length, dtype=np.float32)

    if trail_color.strip().lower() == "same":
        base = np.asarray(marker_colors, dtype=np.float32).copy()
    else:
        base = solid_colors(n_particles, trail_color, 1.0)

    colors = np.repeat(base, trail_length, axis=0)
    colors[:, 3] *= np.tile(alphas, int(n_particles))
    return colors.astype(np.float32)


def torus_wire_lines(R0: float, a: float,
                    nphi: int, ntheta: int,
                    wire_phi: int, wire_theta: int,
                    half_shell: bool = True,
                    )-> list[np.ndarray]:
    
    """Build torus latitude/longitude curves as 3D line arrays."""
    theta_min, theta_max = (-np.pi, 0.0) if half_shell else (-np.pi, np.pi)
    theta_curve = np.linspace(theta_min, theta_max, max(8, int(ntheta)), dtype=np.float32)
    phi_curve = np.linspace(0.0, 2.0 * np.pi, max(16, int(nphi)), dtype=np.float32)
    lines: list[np.ndarray] = []

    for phi in np.linspace(0.0, 2.0 * np.pi, max(1, int(wire_phi)), endpoint=False, dtype=np.float32):
        x = (R0 + a * np.cos(theta_curve)) * np.cos(phi)
        y = -(R0 + a * np.cos(theta_curve)) * np.sin(phi)
        z = a * np.sin(theta_curve)
        lines.append(np.column_stack([x, y, z]).astype(np.float32))

    for theta in np.linspace(theta_min, theta_max, max(1, int(wire_theta)), dtype=np.float32):
        x = (R0 + a * np.cos(theta)) * np.cos(phi_curve)
        y = -(R0 + a * np.cos(theta)) * np.sin(phi_curve)
        z = np.full_like(phi_curve, a * np.sin(theta))
        lines.append(np.column_stack([x, y, z]).astype(np.float32))

    return lines


def add_torus(subplot,
              R0: float,
              a: float,
              style: str = "wireframe",
              mesh_color: str = "#3f7fbf",
              mesh_alpha: float = 0.16,
              wire_color: str = "#9bb7d4",
              wire_alpha: float = 0.42,
              wire_thickness: float = 1.0,
              half_shell: bool = True,
              nphi: int = 144,
              ntheta: int = 64,
              wire_phi: int = 24,
              wire_theta: int = 9,
              ) -> None:

    """Add a translucent and/or wireframe torus scene reference."""
    if style in ("mesh", "both"):
        positions, indices = make_torus_mesh(
            R0=R0,
            a=a,
            nphi=max(8, nphi),
            ntheta=max(4, ntheta),
            half_shell=half_shell,
        )
        rgba = parse_rgba(mesh_color, mesh_alpha)
        if hasattr(subplot, "add_mesh"):
            try:
                subplot.add_mesh(
                    positions=positions,
                    indices=indices,
                    colors=rgba,
                    mode="basic",
                    alpha=float(mesh_alpha),
                )
            except TypeError:
                subplot.add_mesh(positions, indices, colors=rgba, mode="basic")

    if style in ("wireframe", "both") or not hasattr(subplot, "add_mesh"):
        lines = torus_wire_lines(R0, a, nphi, ntheta, wire_phi, wire_theta, half_shell=half_shell)
        subplot.add_line_collection(
            lines,
            colors=parse_rgba(wire_color, wire_alpha),
            thickness=float(wire_thickness),
        )


def setup_camera(subplot, R0: float, a: float) -> None:
    extent = float(R0 + a)
    try:
        subplot.camera.show_rect(-extent, extent, -extent, extent)
    except Exception:
        pass
    try:
        #subplot.camera.local.position = (1.45, -1.85, 0.75)
        subplot.camera.local.position = (0.0, -1.0, 0.35)
        subplot.camera.look_at((0.0, -0.0, -0.0))
        subplot.camera.set_state({'fov': 60.0, 'zoom': 0.01})

    except Exception:
        pass


def set_subplot_background(subplot, color: str) -> None:
    try:
        subplot.background_color = color
    except Exception:
        pass


def style_mpl_axis(axis, background_color: str, foreground_color: str) -> None:
    axis.figure.patch.set_facecolor(background_color)
    axis.set_facecolor(background_color)
    axis.title.set_color(foreground_color)
    axis.xaxis.label.set_color(foreground_color)
    axis.yaxis.label.set_color(foreground_color)
    axis.tick_params(colors=foreground_color)
    for spine in axis.spines.values():
        spine.set_color(foreground_color)


class EnergyHistogramPanel:
    """Matplotlib panel showing per-frame particle energy distribution."""

    def __init__(
        self,
        axis,
        canvas,
        bins: int,
        cmap: str,
        custom_cmap: np.ndarray | None,
        cmap_reverse: bool,
        color_vmin: float | None,
        color_vmax: float | None,
        x_max: float | None,
        log_y: bool,
        background_color: str,
        foreground_color: str,
    ):
        self.axis = axis
        self.canvas = canvas
        self.bins = max(1, int(bins))
        self.cmap = cmap
        self.custom_cmap = custom_cmap
        self.cmap_reverse = bool(cmap_reverse)
        self.color_vmin = color_vmin
        self.color_vmax = color_vmax
        self.x_max = x_max
        self.log_y = bool(log_y)
        self.background_color = background_color
        self.foreground_color = foreground_color
        self._color_buffer = np.empty((self.bins, 4), dtype=np.float32)

    def update(self, energies_ev: np.ndarray, frame_idx: int) -> None:
        finite = np.asarray(energies_ev[np.isfinite(energies_ev) & (energies_ev > 0.0)], dtype=np.float32)
        self.axis.clear()
        style_mpl_axis(self.axis, self.background_color, self.foreground_color)

        if finite.size == 0:
            self.axis.set_title(f"Particle energy distribution: frame {frame_idx}")
            self.axis.set_xlabel("Energy (eV)")
            self.axis.set_ylabel("Particles")
            self.axis.text(
                0.5,
                0.5,
                "No nonzero finite energies",
                ha="center",
                va="center",
                color=self.foreground_color,
                transform=self.axis.transAxes,
            )
            if self.x_max is not None:
                self.axis.set_xlim(0.0, self.x_max)
            self._draw()
            return

        if self.x_max is not None:
            hist_range = (0.0, self.x_max)
        else:
            data_min = float(np.nanmin(finite))
            data_max = float(np.nanmax(finite))
            if not np.isfinite(data_min) or not np.isfinite(data_max):
                data_min, data_max = 0.0, 1.0
            if data_max <= data_min:
                pad = max(abs(data_min) * 0.05, 1.0)
                hist_range = (max(0.0, data_min - pad), data_max + pad)
            else:
                hist_range = (0.0, data_max)

        counts, edges = np.histogram(finite, bins=self.bins, range=hist_range)
        centers = 0.5 * (edges[:-1] + edges[1:])
        if self._color_buffer.shape[0] != centers.shape[0]:
            self._color_buffer = np.empty((centers.shape[0], 4), dtype=np.float32)
        scalar_to_colors(
            centers,
            self.cmap,
            self._color_buffer,
            self.color_vmin,
            self.color_vmax,
            self.custom_cmap,
            self.cmap_reverse,
        )

        self.axis.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            color=self._color_buffer,
            edgecolor="none",
        )
        if self.log_y:
            self.axis.set_yscale("log")
            self.axis.set_ylim(bottom=0.8)
        if self.x_max is not None:
            self.axis.set_xlim(0.0, self.x_max)
        self.axis.set_title(f"Particle energy distribution: frame {frame_idx}")
        self.axis.set_xlabel("Energy (eV)")
        self.axis.set_ylabel("Particles")
        self.axis.grid(True, axis="y", color=self.foreground_color, alpha=0.25)
        self._draw()

    def _draw(self) -> None:
        try:
            self.axis.figure.tight_layout()
        except Exception:
            pass
        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()
        else:
            self.canvas.draw()


class RunningFractionPanel:
    """Matplotlib panel showing the full running-particle fraction history."""

    def __init__(
        self,
        axis,
        canvas,
        frame_indices: np.ndarray,
        fractions: np.ndarray,
        background_color: str,
        foreground_color: str,
        line_color: str,
        cursor_color: str,
    ):
        self.axis = axis
        self.canvas = canvas
        self.frame_indices = np.asarray(frame_indices, dtype=np.int64)
        self.values = np.asarray(fractions, dtype=np.float32) * 100.0
        self.background_color = background_color
        self.foreground_color = foreground_color
        self.line_color = line_color
        self.cursor_color = cursor_color
        self.cursor = None
        self._build()

    def _build(self) -> None:
        self.axis.clear()
        style_mpl_axis(self.axis, self.background_color, self.foreground_color)
        self.axis.plot(self.frame_indices, self.values, color=self.line_color, linewidth=1.8)
        x0 = int(self.frame_indices[0]) if self.frame_indices.size else 0
        x1 = int(self.frame_indices[-1]) if self.frame_indices.size else 1
        if x1 <= x0:
            x1 = x0 + 1
        self.cursor = self.axis.axvline(x0, color=self.cursor_color, linewidth=1.4)
        self.axis.set_xlim(x0, x1)
        self.axis.set_ylim(0.0, 100.0)
        self.axis.set_title("Particles still running")
        self.axis.set_xlabel("Stored timestep")
        self.axis.set_ylabel("Running (%)")
        self.axis.grid(True, color=self.foreground_color, alpha=0.25)
        self._draw()

    def update(self, frame_idx: int) -> None:
        if self.cursor is not None:
            self.cursor.set_xdata([frame_idx, frame_idx])
        self._draw()

    def _draw(self) -> None:
        try:
            self.axis.figure.tight_layout()
        except Exception:
            pass
        if hasattr(self.canvas, "draw_idle"):
            self.canvas.draw_idle()
        else:
            self.canvas.draw()


class TraceTimeState:
    """State holder used by the Qt controls."""

    def __init__(
        self,
        scatter,
        traces: np.ndarray,
        valid_lengths: np.ndarray,
        frame_buffer: np.ndarray,
        color_buffer: np.ndarray,
        scalar_buffer: np.ndarray,
        energy_buffer: np.ndarray,
        histogram_panel: EnergyHistogramPanel | None,
        running_panel: RunningFractionPanel | None,
        trail_scatter,
        trail_buffer: np.ndarray | None,
        trail_length: int,
        trail_stride: int,
        max_frame: int,
        frame_stride: int,
        hide_zero_rows: bool,
        play_fps: float,
        color_mode: str,
        cmap: str,
        custom_cmap: np.ndarray | None,
        cmap_reverse: bool,
        color_vmin: float | None,
        color_vmax: float | None,
        sample_dt: float,
        ion_mass_amu: float,
    ):
        self.scatter = scatter
        self.traces = traces
        self.valid_lengths = valid_lengths
        self.frame_buffer = frame_buffer
        self.color_buffer = color_buffer
        self.scalar_buffer = scalar_buffer
        self.energy_buffer = energy_buffer
        self.histogram_panel = histogram_panel
        self.running_panel = running_panel
        self.trail_scatter = trail_scatter
        self.trail_buffer = trail_buffer
        self.trail_length = max(0, int(trail_length))
        self.trail_stride = max(1, int(trail_stride))
        self.max_frame = int(max_frame)
        self.frame_stride = max(1, int(frame_stride))
        self.hide_zero_rows = bool(hide_zero_rows)
        self.play_fps = max(1.0, float(play_fps))
        self.color_mode = color_mode
        self.cmap = cmap
        self.custom_cmap = custom_cmap
        self.cmap_reverse = bool(cmap_reverse)
        self.color_vmin = color_vmin
        self.color_vmax = color_vmax
        self.sample_dt = float(sample_dt)
        self.ion_mass_amu = float(ion_mass_amu)
        self.current_frame = 0
        self.playing = False
        self.loop = True
        self._last_play_time = time.perf_counter()
        self.color_status = ""

    def set_frame(self, frame_idx: int) -> None:
        frame_idx = max(0, min(int(frame_idx), self.max_frame))
        if frame_idx == self.current_frame:
            return
        self.current_frame = frame_idx
        self.update_scatter()

    def step(self, direction: int) -> None:
        next_frame = self.current_frame + int(direction) * self.frame_stride
        if next_frame > self.max_frame:
            next_frame = 0 if self.loop else self.max_frame
        elif next_frame < 0:
            next_frame = self.max_frame if self.loop else 0
        self.set_frame(next_frame)

    def update_scatter(self) -> None:
        frame_positions(
            self.traces,
            self.current_frame,
            self.valid_lengths,
            self.hide_zero_rows,
            self.frame_buffer,
        )
        self.scatter.data = self.frame_buffer
        self.update_trails()
        self.update_colors()
        self.update_histogram()
        self.update_running_fraction()
        canvas = getattr(self.scatter, "figure", None)
        if canvas is not None and hasattr(canvas, "canvas"):
            canvas.canvas.request_draw()

    def update_trails(self) -> None:
        if self.trail_scatter is None or self.trail_buffer is None or self.trail_length <= 0:
            return
        trail_positions(
            self.traces,
            self.current_frame,
            self.valid_lengths,
            self.trail_length,
            self.trail_stride,
            self.hide_zero_rows,
            self.trail_buffer,
        )
        self.trail_scatter.data = self.trail_buffer

    def update_colors(self) -> None:
        if self.color_mode not in ("speed", "energy"):
            return

        instantaneous_speed(
            self.traces,
            self.current_frame,
            self.valid_lengths,
            self.sample_dt,
            self.scalar_buffer,
        )
        if self.color_mode == "energy":
            instantaneous_energy_ev(
                self.traces,
                self.current_frame,
                self.valid_lengths,
                self.sample_dt,
                self.ion_mass_amu,
                self.scalar_buffer,
            )

        inactive = ~np.isfinite(self.frame_buffer).all(axis=1)
        self.scalar_buffer[inactive] = np.nan
        data_min, data_max, color_min, color_max = scalar_to_colors(
            self.scalar_buffer,
            self.cmap,
            self.color_buffer,
            self.color_vmin,
            self.color_vmax,
            self.custom_cmap,
            self.cmap_reverse,
        )
        self.scatter.colors = self.color_buffer
        units = "m/s" if self.color_mode == "speed" else "eV"
        self.color_status = (
            f"{self.color_mode}: {data_min:.3g}..{data_max:.3g} {units} "
            f"(scale {color_min:.3g}..{color_max:.3g})"
        )

    def update_histogram(self) -> None:
        if self.histogram_panel is None:
            return
        instantaneous_energy_ev(
            self.traces,
            self.current_frame,
            self.valid_lengths,
            self.sample_dt,
            self.ion_mass_amu,
            self.energy_buffer,
        )
        inactive = ~np.isfinite(self.frame_buffer).all(axis=1)
        self.energy_buffer[inactive] = np.nan
        self.histogram_panel.update(self.energy_buffer, self.current_frame)

    def update_running_fraction(self) -> None:
        if self.running_panel is None:
            return
        self.running_panel.update(self.current_frame)

    def update_playback(self) -> None:
        if not self.playing:
            return
        now = time.perf_counter()
        if now - self._last_play_time >= 1.0 / self.play_fps:
            self.step(1)
            self._last_play_time = now


class TraceSliderWindow:
    """Thin Qt window around a fastplotlib canvas and a horizontal time slider."""

    def __init__(
        self,
        QtCore,
        QtWidgets,
        canvas,
        state: TraceTimeState,
        side_canvases: list | None = None,
        side_panel_width: int = 380,
        background_color: str = "#02070D",
        foreground_color: str = "#E8EDF2",
    ):
        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.canvas = canvas
        self.state = state
        self.side_panel_width = int(side_panel_width) if side_canvases else 0

        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("Boris ion trace scatter slider")
        self.window.setStyleSheet(f"background-color: {background_color}; color: {foreground_color};")

        root = QtWidgets.QVBoxLayout(self.window)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        content = QtWidgets.QHBoxLayout()
        content.setSpacing(6)
        root.addLayout(content, stretch=1)
        content.addWidget(canvas, stretch=4)
        if side_canvases:
            side = QtWidgets.QVBoxLayout()
            side.setSpacing(6)
            content.addLayout(side, stretch=1)
            for side_canvas in side_canvases:
                side_canvas.setMinimumWidth(int(side_panel_width))
                side_canvas.setMaximumWidth(int(side_panel_width))
                side.addWidget(side_canvas, stretch=1)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(6)
        root.addLayout(controls)

        self.frame_label = QtWidgets.QLabel()
        self.frame_label.setMinimumWidth(150)
        controls.addWidget(self.frame_label)
        self.color_label = QtWidgets.QLabel()
        self.color_label.setMinimumWidth(260)
        controls.addWidget(self.color_label)

        self.first_button = QtWidgets.QPushButton("<<")
        self.back_button = QtWidgets.QPushButton("<")
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.setCheckable(True)
        self.forward_button = QtWidgets.QPushButton(">")
        self.last_button = QtWidgets.QPushButton(">>")
        for button in (
            self.first_button,
            self.back_button,
            self.play_button,
            self.forward_button,
            self.last_button,
        ):
            controls.addWidget(button)

        self.loop_box = QtWidgets.QCheckBox("Loop")
        self.loop_box.setChecked(state.loop)
        controls.addWidget(self.loop_box)

        controls.addWidget(QtWidgets.QLabel("FPS"))
        self.fps_spin = QtWidgets.QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 120.0)
        self.fps_spin.setDecimals(1)
        self.fps_spin.setSingleStep(5.0)
        self.fps_spin.setValue(state.play_fps)
        self.fps_spin.setMaximumWidth(80)
        controls.addWidget(self.fps_spin)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, state.max_frame)
        self.slider.setSingleStep(state.frame_stride)
        self.slider.setPageStep(max(state.frame_stride, state.frame_stride * 10))
        root.addWidget(self.slider)

        self.timer = QtCore.QTimer(self.window)
        self.timer.timeout.connect(self._play_tick)

        self.first_button.clicked.connect(lambda: self.set_frame(0))
        self.back_button.clicked.connect(lambda: self.step(-1))
        self.forward_button.clicked.connect(lambda: self.step(1))
        self.last_button.clicked.connect(lambda: self.set_frame(state.max_frame))
        self.play_button.toggled.connect(self._toggle_play)
        self.loop_box.toggled.connect(self._set_loop)
        self.fps_spin.valueChanged.connect(self._set_fps)
        self.slider.valueChanged.connect(self.set_frame)

        self.set_frame(state.current_frame)

    def _set_loop(self, value: bool) -> None:
        self.state.loop = bool(value)

    def _set_fps(self, value: float) -> None:
        self.state.play_fps = max(1.0, float(value))
        if self.timer.isActive():
            self.timer.start(int(round(1000.0 / self.state.play_fps)))

    def _toggle_play(self, checked: bool) -> None:
        self.state.playing = bool(checked)
        self.play_button.setText("Pause" if checked else "Play")
        if checked:
            self.timer.start(int(round(1000.0 / self.state.play_fps)))
        else:
            self.timer.stop()

    def _play_tick(self) -> None:
        self.step(1)

    def set_frame(self, frame_idx: int) -> None:
        self.state.set_frame(int(frame_idx))
        if self.slider.value() != self.state.current_frame:
            self.slider.blockSignals(True)
            self.slider.setValue(self.state.current_frame)
            self.slider.blockSignals(False)
        self.frame_label.setText(f"Frame {self.state.current_frame} / {self.state.max_frame}")
        self.color_label.setText(self.state.color_status)
        self.canvas.request_draw()

    def step(self, direction: int) -> None:
        self.state.step(direction)
        self.set_frame(self.state.current_frame)

    def show(self) -> None:
        self.window.resize(self.canvas.width() + self.side_panel_width + 28, self.canvas.height() + 92)
        self.window.show()


def initial_marker_colors(args: argparse.Namespace, n_particles: int) -> np.ndarray:
    if args.color_mode == "solid":
        return solid_colors(n_particles, args.marker_color, args.marker_alpha)
    if args.color_mode == "particle":
        return apply_alpha(particle_colors(n_particles), args.marker_alpha)

    colors = np.zeros((n_particles, 4), dtype=np.float32)
    colors[:, 3] = 1.0
    return colors


def add_scene_torus(subplot, args: argparse.Namespace) -> None:
    if not args.show_torus:
        return

    add_torus(
        subplot,
        args.R0,
        args.a,
        style=args.torus_style,
        mesh_color=args.torus_color,
        mesh_alpha=args.torus_alpha,
        wire_color=args.torus_wire_color,
        wire_alpha=args.torus_wire_alpha,
        wire_thickness=args.torus_wire_thickness,
        half_shell=(args.torus_half == "bottom"),
        nphi=args.torus_nphi,
        ntheta=args.torus_ntheta,
        wire_phi=args.torus_wire_phi,
        wire_theta=args.torus_wire_theta,
    )


def add_scene_ports(subplot, args: argparse.Namespace) -> None:
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
                outline_color=parse_rgba(args.plot_background, 1.0),
                outline_thickness=0.15,
                screen_space=True,
                offset=tuple(float(value) for value in position),
                anchor="middle-center",
            )


def add_scene_compass(subplot, args: argparse.Namespace) -> None:
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
        outline_color=parse_rgba(args.plot_background, 1.0),
        outline_thickness=0.15,
        screen_space=True,
        offset=tuple(float(value) for value in label_pos),
        anchor="middle-center",
    )


def add_scene_particles(
    subplot,
    args: argparse.Namespace,
    traces: np.ndarray,
    valid_lengths: np.ndarray,
    initial_frame: int,
    frame_buffer: np.ndarray,
    initial_colors: np.ndarray,
) -> tuple[object, object, np.ndarray | None]:
    trail_length = max(0, int(args.trail_length))
    trail_marker_size = (
        float(args.trail_marker_size)
        if args.trail_marker_size is not None
        else float(args.marker_size) * 0.7
    )

    trail_scatter = None
    trail_buffer = None
    if trail_length > 0:
        trail_buffer = np.empty((traces.shape[1] * trail_length, 3), dtype=np.float32)
        trail_positions(
            traces,
            initial_frame,
            valid_lengths,
            trail_length,
            args.trail_stride,
            args.hide_zero_rows,
            trail_buffer,
        )
        if args.trail_color.strip().lower() == "same" and args.color_mode in ("speed", "energy"):
            trail_base_colors = solid_colors(traces.shape[1], "#d0d0d0", 0.8)
        else:
            trail_base_colors = initial_colors
        trail_colors = make_trail_colors(
            traces.shape[1],
            trail_length,
            trail_base_colors,
            args.trail_color,
            args.trail_alpha_min,
        )
        trail_scatter = subplot.add_scatter(
            data=trail_buffer,
            sizes=trail_marker_size,
            colors=trail_colors,
            edge_width=0.0,
            mode="simple",
        )

    scatter = subplot.add_scatter(
        data=frame_buffer,
        sizes=float(args.marker_size),
        colors=initial_colors,
        edge_width=0.0,
        mode="simple",
    )

    return scatter, trail_scatter, trail_buffer


def histogram_color_limits(args: argparse.Namespace) -> tuple[float | None, float | None]:
    vmin = args.histogram_color_vmin
    vmax = args.histogram_color_vmax
    if args.color_mode == "energy":
        if vmin is None:
            vmin = args.color_vmin
        if vmax is None:
            vmax = args.color_vmax
    return vmin, vmax


def histogram_xmax(args: argparse.Namespace) -> float | None:
    if args.histogram_auto_xmax:
        return None
    if args.histogram_xmax is not None:
        return float(args.histogram_xmax)

    _, vmax = histogram_color_limits(args)
    if vmax is None:
        return None
    return 5.0 * float(vmax)


def make_energy_histogram_panel(
    args: argparse.Namespace,
    custom_cmap_colors: np.ndarray | None,
    canvas,
    axis,
) -> EnergyHistogramPanel:
    vmin, vmax = histogram_color_limits(args)
    x_max = histogram_xmax(args)
    return EnergyHistogramPanel(
        axis=axis,
        canvas=canvas,
        bins=args.histogram_bins,
        cmap=args.cmap,
        custom_cmap=custom_cmap_colors,
        cmap_reverse=args.cmap_reverse,
        color_vmin=vmin,
        color_vmax=vmax,
        x_max=x_max,
        log_y=args.histogram_log_y,
        background_color=args.plot_background,
        foreground_color=args.plot_foreground,
    )


def make_running_fraction_panel(
    args: argparse.Namespace,
    canvas,
    axis,
    frame_indices: np.ndarray,
    running_fractions: np.ndarray,
) -> RunningFractionPanel:
    return RunningFractionPanel(
        axis=axis,
        canvas=canvas,
        frame_indices=frame_indices,
        fractions=running_fractions,
        background_color=args.plot_background,
        foreground_color=args.plot_foreground,
        line_color=UIUC["il_orange"],
        cursor_color=args.plot_foreground,
    )


def match_image_height(image: np.ndarray, target_height: int) -> np.ndarray:
    """Nearest-neighbor resize along height/width to match another exported frame."""
    height = int(image.shape[0])
    target_height = int(target_height)
    if height == target_height:
        return image
    if height <= 0 or target_height <= 0:
        return image

    scale = target_height / height
    target_width = max(1, int(round(image.shape[1] * scale)))
    y_idx = np.clip((np.arange(target_height) / scale).astype(np.int64), 0, height - 1)
    x_idx = np.clip((np.arange(target_width) / scale).astype(np.int64), 0, image.shape[1] - 1)
    return image[y_idx][:, x_idx]


def export_mp4(
    args: argparse.Namespace,
    fpl,
    traces: np.ndarray,
    valid_lengths: np.ndarray,
    custom_cmap_colors: np.ndarray | None,
    max_frame: int,
) -> int:
    try:
        import imageio
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure as MplFigure
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "--export-mp4 requires imageio, imageio-ffmpeg, and matplotlib.\n"
            "Install them in testenv with:\n"
            "  conda install -n testenv -c conda-forge imageio imageio-ffmpeg"
        ) from exc

    if args.export_fps <= 0.0:
        raise SystemExit("--export-fps must be positive.")
    if args.export_step < 1:
        raise SystemExit("--export-step must be positive.")

    export_size = parse_size(args.export_size or args.size)
    start = max(0, min(int(args.export_start), max_frame))
    stop = max_frame + 1 if args.export_stop is None else max(0, min(int(args.export_stop), max_frame + 1))
    if stop <= start:
        raise SystemExit("--export-stop must be greater than --export-start.")
    frame_indices = list(range(start, stop, int(args.export_step)))
    if not frame_indices:
        raise SystemExit("No export frames selected.")

    n_particles = traces.shape[1]
    frame_buffer = np.empty((n_particles, 3), dtype=np.float32)
    color_buffer = np.empty((n_particles, 4), dtype=np.float32)
    scalar_buffer = np.empty(n_particles, dtype=np.float32)
    energy_buffer = np.empty(n_particles, dtype=np.float32)

    frame_positions(traces, frame_indices[0], valid_lengths, args.hide_zero_rows, frame_buffer)
    initial_colors = initial_marker_colors(args, n_particles)

    side_canvas_items = []
    side_panel_count = int(not args.hide_running_fraction) + int(not args.hide_histogram)
    side_width = max(1, int(args.side_panel_width))
    side_heights = []
    if side_panel_count > 0:
        base_height = export_size[1] // side_panel_count
        side_heights = [base_height] * side_panel_count
        side_heights[-1] += export_size[1] - sum(side_heights)

    side_idx = 0
    running_panel = None
    if not args.hide_running_fraction:
        print("Computing running-particle fraction history...")
        running_frames, running_fractions = running_fraction_over_time(
            traces,
            valid_lengths,
            max_frame,
            args.sample_dt,
            args.ion_mass_amu,
        )
        running_figure = MplFigure(figsize=(side_width / 100, side_heights[side_idx] / 100), dpi=100)
        running_canvas = FigureCanvasAgg(running_figure)
        running_axis = running_figure.add_subplot(111)
        running_panel = make_running_fraction_panel(args, running_canvas, running_axis, running_frames, running_fractions)
        side_canvas_items.append(running_canvas)
        side_idx += 1

    histogram_panel = None
    histogram_canvas = None
    if not args.hide_histogram:
        histogram_figure = MplFigure(figsize=(side_width / 100, side_heights[side_idx] / 100), dpi=100)
        histogram_canvas = FigureCanvasAgg(histogram_figure)
        histogram_axis = histogram_figure.add_subplot(111)
        histogram_figure.tight_layout()
        histogram_panel = make_energy_histogram_panel(args, custom_cmap_colors, histogram_canvas, histogram_axis)
        side_canvas_items.append(histogram_canvas)

    figure = fpl.Figure(cameras="3d", controller_types="orbit", canvas="offscreen", size=export_size)
    subplot = figure[0, 0]
    set_subplot_background(subplot, args.plot_background)
    try:
        subplot.set_title("Boris ion positions")
    except Exception:
        pass
    add_scene_torus(subplot, args)
    add_scene_ports(subplot, args)
    add_scene_compass(subplot, args)
    scatter, trail_scatter, trail_buffer = add_scene_particles(
        subplot,
        args,
        traces,
        valid_lengths,
        frame_indices[0],
        frame_buffer,
        initial_colors,
    )

    state = TraceTimeState(
        scatter=scatter,
        traces=traces,
        valid_lengths=valid_lengths,
        frame_buffer=frame_buffer,
        color_buffer=color_buffer,
        scalar_buffer=scalar_buffer,
        energy_buffer=energy_buffer,
        histogram_panel=histogram_panel,
        running_panel=running_panel,
        trail_scatter=trail_scatter,
        trail_buffer=trail_buffer,
        trail_length=max(0, int(args.trail_length)),
        trail_stride=args.trail_stride,
        max_frame=max_frame,
        frame_stride=args.frame_stride,
        hide_zero_rows=args.hide_zero_rows,
        play_fps=args.play_fps,
        color_mode=args.color_mode,
        cmap=args.cmap,
        custom_cmap=custom_cmap_colors,
        cmap_reverse=args.cmap_reverse,
        color_vmin=args.color_vmin,
        color_vmax=args.color_vmax,
        sample_dt=args.sample_dt,
        ion_mass_amu=args.ion_mass_amu,
    )

    setup_camera(subplot, args.R0, args.a)
    figure.show(axes_visible=args.axes)

    args.export_mp4.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {len(frame_indices)} frames to {args.export_mp4}")
    with imageio.get_writer(
        args.export_mp4,
        fps=float(args.export_fps),
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=2,
    ) as writer:
        for output_idx, frame_idx in enumerate(frame_indices):
            state.current_frame = int(frame_idx)
            state.update_scatter()
            figure._render(draw=False)
            frame_image = figure.export_numpy(rgb=True)
            if side_canvas_items:
                side_images = []
                for side_canvas in side_canvas_items:
                    side_canvas.draw()
                    side_images.append(np.asarray(side_canvas.buffer_rgba())[:, :, :3].copy())
                side_stack = match_image_height(np.vstack(side_images), frame_image.shape[0])
                frame_image = np.hstack([frame_image, side_stack])
            writer.append_data(frame_image)
            if output_idx == 0 or (output_idx + 1) % 50 == 0 or output_idx + 1 == len(frame_indices):
                print(f"  frame {output_idx + 1}/{len(frame_indices)} source={frame_idx}", flush=True)

    print(f"Wrote {args.export_mp4}")
    return 0


def main() -> int:
    args = parse_args()
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be positive.")
    if args.sample_dt <= 0.0:
        raise SystemExit("--sample-dt must be positive.")
    if args.color_vmin is not None and args.color_vmax is not None and args.color_vmax <= args.color_vmin:
        raise SystemExit("--color-vmax must be greater than --color-vmin.")
    if args.trail_length < 0:
        raise SystemExit("--trail-length must be >= 0.")
    if args.trail_stride < 1:
        raise SystemExit("--trail-stride must be positive.")
    if not 0.0 <= args.trail_alpha_min <= 1.0:
        raise SystemExit("--trail-alpha-min must be between 0 and 1.")
    if args.histogram_bins < 1:
        raise SystemExit("--histogram-bins must be positive.")
    if args.histogram_height < 100:
        raise SystemExit("--histogram-height must be at least 100.")
    hist_vmin, hist_vmax = histogram_color_limits(args)
    if hist_vmin is not None and hist_vmax is not None and hist_vmax <= hist_vmin:
        raise SystemExit("--histogram-color-vmax must be greater than --histogram-color-vmin.")
    hist_xmax = histogram_xmax(args)
    if hist_xmax is not None and hist_xmax <= 0.0:
        raise SystemExit("--histogram-xmax must be positive.")
    if args.side_panel_width < 100:
        raise SystemExit("--side-panel-width must be at least 100.")
    try:
        custom_cmap_colors = parse_custom_cmap(args.cmap_colors, args.cmap_reverse)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    fpl = import_fastplotlib()
    size = parse_size(args.size)

    raw_traces, source_label = load_or_make_traces(args)
    raw_valid_lengths = infer_valid_lengths(raw_traces)
    selection = select_trace_particles(
        raw_traces,
        valid_lengths=raw_valid_lengths,
        max_particles=args.max_particles,
        skip_indices=parse_skip_indices(args.skip_indices),
    )
    traces = selection.traces
    valid_lengths = selection.valid_lengths
    n_particles = traces.shape[1]
    max_frame = int(min(traces.shape[0] - 1, np.max(valid_lengths) - 1))

    frame_buffer = np.empty((n_particles, 3), dtype=np.float32)
    color_buffer = np.empty((n_particles, 4), dtype=np.float32)
    scalar_buffer = np.empty(n_particles, dtype=np.float32)
    energy_buffer = np.empty(n_particles, dtype=np.float32)
    trail_length = max(0, int(args.trail_length))
    trail_marker_size = float(args.trail_marker_size) if args.trail_marker_size is not None else float(args.marker_size) * 0.7
    trail_buffer = np.empty((n_particles * max(1, trail_length), 3), dtype=np.float32) if trail_length > 0 else None
    initial_frame = max(0, min(int(args.initial_frame), max_frame))

    print(f"fastplotlib: {getattr(fpl, '__version__', 'unknown')}")
    print(f"source: {source_label}")
    print(f"trace shape: {traces.shape}, dtype={traces.dtype}")
    print(f"particles displayed: {n_particles}; frame range: 0..{max_frame}")
    print(f"color mode: {args.color_mode}")
    print(f"trail length: {trail_length}")
    print("Use the bottom slider or step/play controls to move through time.")

    if args.export_mp4 is not None:
        return export_mp4(args, fpl, traces, valid_lengths, custom_cmap_colors, max_frame)

    QtCore, QtWidgets, QRenderWidget = import_qt_and_canvas()
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    side_canvases = []
    histogram_panel = None
    running_panel = None
    if not args.hide_histogram or not args.hide_running_fraction:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure as MplFigure
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "The side plots require Matplotlib's Qt backend. "
                "Run with the same PySide6 environment used for fastplotlib."
            ) from exc

    dpi = 100
    side_plot_count = int(not args.hide_running_fraction) + int(not args.hide_histogram)
    side_plot_height = max(int(args.histogram_height), size[1] // max(1, side_plot_count))
    if not args.hide_running_fraction:
        print("Computing running-particle fraction history...")
        running_frames, running_fractions = running_fraction_over_time(
            traces,
            valid_lengths,
            max_frame,
            args.sample_dt,
            args.ion_mass_amu,
        )
        running_figure = MplFigure(figsize=(args.side_panel_width / dpi, side_plot_height / dpi), dpi=dpi)
        running_canvas = FigureCanvasQTAgg(running_figure)
        running_canvas.setMinimumHeight(side_plot_height)
        running_axis = running_figure.add_subplot(111)
        running_panel = make_running_fraction_panel(args, running_canvas, running_axis, running_frames, running_fractions)
        side_canvases.append(running_canvas)

    if not args.hide_histogram:
        histogram_figure = MplFigure(figsize=(args.side_panel_width / dpi, side_plot_height / dpi), dpi=dpi)
        histogram_canvas = FigureCanvasQTAgg(histogram_figure)
        histogram_canvas.setMinimumHeight(int(args.histogram_height))
        histogram_axis = histogram_figure.add_subplot(111)
        histogram_figure.tight_layout()
        histogram_panel = make_energy_histogram_panel(args, custom_cmap_colors, histogram_canvas, histogram_axis)
        side_canvases.append(histogram_canvas)

    canvas = QRenderWidget(size=size, title="Boris ion positions")
    figure = fpl.Figure(cameras="3d", controller_types="orbit", canvas=canvas, size=size)
    subplot = figure[0, 0]
    set_subplot_background(subplot, args.plot_background)
    try:
        subplot.set_title("Boris ion positions")
    except Exception:
        pass

    add_scene_torus(subplot, args)
    add_scene_ports(subplot, args)
    add_scene_compass(subplot, args)

    frame_positions(traces, initial_frame, valid_lengths, args.hide_zero_rows, frame_buffer)
    if args.color_mode == "solid":
        initial_colors = solid_colors(n_particles, args.marker_color, args.marker_alpha)
    elif args.color_mode == "particle":
        initial_colors = apply_alpha(particle_colors(n_particles), args.marker_alpha)
    else:
        initial_colors = np.zeros((n_particles, 4), dtype=np.float32)
        initial_colors[:, 3] = 1.0

    trail_scatter = None
    if trail_length > 0:
        trail_positions(
            traces,
            initial_frame,
            valid_lengths,
            trail_length,
            args.trail_stride,
            args.hide_zero_rows,
            trail_buffer,
        )
        if args.trail_color.strip().lower() == "same" and args.color_mode in ("speed", "energy"):
            trail_base_colors = solid_colors(n_particles, "#d0d0d0", 0.8)
        else:
            trail_base_colors = initial_colors
        trail_colors = make_trail_colors(
            n_particles,
            trail_length,
            trail_base_colors,
            args.trail_color,
            args.trail_alpha_min,
        )
        trail_scatter = subplot.add_scatter(
            data=trail_buffer,
            sizes=trail_marker_size,
            colors=trail_colors,
            edge_width=0.0,
            mode="simple",
        )

    scatter = subplot.add_scatter(
        data=frame_buffer,
        sizes=float(args.marker_size),
        colors=initial_colors,
        edge_width=0.0,
        mode="simple",
    )

    state = TraceTimeState(
        scatter=scatter,
        traces=traces,
        valid_lengths=valid_lengths,
        frame_buffer=frame_buffer,
        color_buffer=color_buffer,
        scalar_buffer=scalar_buffer,
        energy_buffer=energy_buffer,
        histogram_panel=histogram_panel,
        running_panel=running_panel,
        trail_scatter=trail_scatter,
        trail_buffer=trail_buffer,
        trail_length=trail_length,
        trail_stride=args.trail_stride,
        max_frame=max_frame,
        frame_stride=args.frame_stride,
        hide_zero_rows=args.hide_zero_rows,
        play_fps=args.play_fps,
        color_mode=args.color_mode,
        cmap=args.cmap,
        custom_cmap=custom_cmap_colors,
        cmap_reverse=args.cmap_reverse,
        color_vmin=args.color_vmin,
        color_vmax=args.color_vmax,
        sample_dt=args.sample_dt,
        ion_mass_amu=args.ion_mass_amu,
    )
    state.current_frame = initial_frame
    state.update_colors()
    state.update_histogram()
    state.update_running_fraction()
    setup_camera(subplot, args.R0, args.a)
    figure.show(axes_visible=args.axes)
    viewer = TraceSliderWindow(
        QtCore,
        QtWidgets,
        canvas,
        state,
        side_canvases=side_canvases,
        side_panel_width=args.side_panel_width,
        background_color=args.plot_background,
        foreground_color=args.plot_foreground,
    )
    viewer.show()

    if __name__ == "__main__":
        app.exec()

    return 0


if __name__ == "__main__":
    sys.exit(main())
