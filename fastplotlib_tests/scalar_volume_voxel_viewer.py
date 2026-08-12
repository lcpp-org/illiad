"""Interactive fastplotlib voxel view of a toroidal scalar field.

The input field follows ILLIAD's regular-field convention ``(phi, theta,
rho)``. It is sampled onto a Cartesian lattice and displayed with world-sized
square voxel glyphs whose color and alpha both increase with the scalar value.
The surrounding scene intentionally matches ``boris_trace_scatter_slider``.

Example:
    conda run -n testenv python \
        fastplotlib_tests/scalar_volume_voxel_viewer.py

Use ``--dry-run`` to load and prepare the voxels without opening a GUI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from boris_trace_scatter_slider import (
    add_scene_compass,
    add_scene_ports,
    add_scene_torus,
    import_fastplotlib,
    import_qt_and_canvas,
    parse_size,
    set_subplot_background,
    setup_camera,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIELD_DIR = (
    PROJECT_ROOT
    / "output"
    / "IOTA3_1000sp_atol1e-9"
    / "data"
    / "ConLenVolume_REDO_500spins_rk1mm_RegularGrid"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Display a regular toroidal scalar field as translucent Cartesian "
            "voxel glyphs using fastplotlib."
        ),
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "field_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_FIELD_DIR,
        help=f"Directory containing the regular field (default: {DEFAULT_FIELD_DIR}).",
    )
    parser.add_argument("--field-file", default="connection_length_field_m.npy")
    parser.add_argument("--rho-file", default="rho_grid_m.npy")
    parser.add_argument("--theta-file", default="theta_grid_rad.npy")
    parser.add_argument("--phi-file", default="phi_grid_deg.npy")
    parser.add_argument("--value-label", default="Connection length [m]")

    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.015,
        help="Requested Cartesian voxel edge length in meters.",
    )
    parser.add_argument(
        "--voxel-scale",
        type=float,
        default=0.90,
        help="Rendered glyph size divided by the actual Cartesian cell spacing.",
    )
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=200_000,
        help="Maximum displayed voxels; excess cells are thinned deterministically.",
    )
    parser.add_argument(
        "--max-lattice-cells",
        type=int,
        default=5_000_000,
        help="Safety limit for the Cartesian lattice before torus masking.",
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--value-threshold",
        type=float,
        default=None,
        help="Only show finite values greater than or equal to this threshold.",
    )
    parser.add_argument(
        "--value-percentile",
        type=float,
        default=0.0,
        help="Only show values at or above this percentile of sampled finite cells.",
    )
    parser.add_argument("--color-scale", choices=("log", "linear"), default="log")
    parser.add_argument("--cmap", default="afmhot")
    parser.add_argument("--cmap-reverse", action="store_true")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--alpha-min", type=float, default=0.03)
    parser.add_argument("--alpha-max", type=float, default=0.92)
    parser.add_argument(
        "--alpha-power",
        type=float,
        default=1.35,
        help="Exponent applied to normalized values before alpha mapping.",
    )

    parser.add_argument("--plot-background", default="#02070D")
    parser.add_argument("--plot-foreground", default="#E8EDF2")
    parser.add_argument("--R0", type=float, default=0.72)
    parser.add_argument("--a", type=float, default=0.19)
    parser.add_argument("--size", default="1280x760")
    parser.add_argument(
        "--present-method",
        choices=("bitmap", "screen"),
        default="bitmap",
        help=(
            "Interactive Qt canvas presentation method. 'screen' may improve "
            "FPS but is less portable."
        ),
    )
    parser.add_argument("--axes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--show-torus", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--torus-style",
        choices=("mesh", "wireframe", "both"),
        default="wireframe",
    )
    parser.add_argument("--torus-color", default="#d40707")
    parser.add_argument("--torus-alpha", type=float, default=0.16)
    parser.add_argument("--torus-wire-color", default="#9bb7d4")
    parser.add_argument("--torus-wire-alpha", type=float, default=0.42)
    parser.add_argument("--torus-wire-thickness", type=float, default=1.0)
    parser.add_argument("--torus-half", choices=("bottom", "full"), default="bottom")
    parser.add_argument("--torus-nphi", type=int, default=144)
    parser.add_argument("--torus-ntheta", type=int, default=64)
    parser.add_argument("--torus-wire-phi", type=int, default=24)
    parser.add_argument("--torus-wire-theta", type=int, default=9)

    parser.add_argument("--show-ports", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--port-file",
        type=Path,
        default=Path("input_files/HIDRA_ports.csv"),
    )
    parser.add_argument("--port-color", default="#F5F7FA")
    parser.add_argument("--port-alpha", type=float, default=0.85)
    parser.add_argument("--port-line-thickness", type=float, default=1.0)
    parser.add_argument("--port-samples", type=int, default=96)
    parser.add_argument("--port-surface-offset", type=float, default=0.003)
    parser.add_argument("--port-phi-zero-offset", type=float, default=162.0)
    parser.add_argument("--label-ports", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--port-label-filter", default="HIDRA-MAT;RLP")
    parser.add_argument("--port-label-color", default="#F5F7FA")
    parser.add_argument("--port-label-font-size", type=float, default=16.0)
    parser.add_argument("--port-label-surface-offset", type=float, default=0.045)

    parser.add_argument("--show-compass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compass-size", type=float, default=0.18)
    parser.add_argument("--compass-z", type=float, default=None)
    parser.add_argument("--compass-color", default="#F5F7FA")
    parser.add_argument("--compass-alpha", type=float, default=0.80)
    parser.add_argument("--compass-line-thickness", type=float, default=2.0)
    parser.add_argument("--compass-label-color", default="#F5F7FA")
    parser.add_argument("--compass-font-size", type=float, default=18.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.voxel_size <= 0.0:
        raise SystemExit("--voxel-size must be positive.")
    if args.voxel_scale <= 0.0:
        raise SystemExit("--voxel-scale must be positive.")
    if args.max_voxels < 1:
        raise SystemExit("--max-voxels must be positive.")
    if args.max_lattice_cells < 1:
        raise SystemExit("--max-lattice-cells must be positive.")
    if not 0.0 <= args.value_percentile <= 100.0:
        raise SystemExit("--value-percentile must be between 0 and 100.")
    if not 0.0 <= args.alpha_min <= args.alpha_max <= 1.0:
        raise SystemExit("Require 0 <= --alpha-min <= --alpha-max <= 1.")
    if args.alpha_power <= 0.0:
        raise SystemExit("--alpha-power must be positive.")
    if args.vmin is not None and args.vmax is not None and args.vmax <= args.vmin:
        raise SystemExit("--vmax must be greater than --vmin.")
    if args.color_scale == "log" and args.vmin is not None and args.vmin <= 0.0:
        raise SystemExit("A logarithmic color scale requires --vmin > 0.")


def load_regular_field(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    field_dir = args.field_dir.expanduser()
    if not field_dir.is_absolute():
        field_dir = (PROJECT_ROOT / field_dir).resolve()

    paths = {
        "field": field_dir / args.field_file,
        "rho": field_dir / args.rho_file,
        "theta": field_dir / args.theta_file,
        "phi": field_dir / args.phi_file,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing regular-field input(s):\n" + "\n".join(missing))

    field = np.load(paths["field"], mmap_mode="r")
    rho = np.load(paths["rho"])
    theta = np.load(paths["theta"])
    phi_deg = np.load(paths["phi"])
    if field.ndim != 3:
        raise ValueError(f"Expected a three-dimensional scalar field; found {field.shape}.")
    expected_shape = (phi_deg.size, theta.size, rho.size)
    if field.shape != expected_shape:
        raise ValueError(
            "Field/coordinate shape mismatch: expected "
            f"{expected_shape} from (phi, theta, rho), found {field.shape}."
        )
    for name, coordinate in (("rho", rho), ("theta", theta), ("phi", phi_deg)):
        if coordinate.ndim != 1 or coordinate.size < 2:
            raise ValueError(f"{name} coordinates must be a one-dimensional array of length >= 2.")
        if np.any(np.diff(coordinate) <= 0.0):
            raise ValueError(f"{name} coordinates must be strictly increasing.")
    return field, rho, theta, phi_deg, field_dir


def periodic_uniform_indices(
    values: np.ndarray,
    coordinates: np.ndarray,
    period: float,
    name: str,
) -> np.ndarray:
    """Return nearest indices for a periodic grid stored as step..period."""
    step = float(period) / coordinates.size
    expected = np.linspace(step, period, coordinates.size)
    if not np.allclose(coordinates, expected, rtol=1.0e-6, atol=1.0e-8):
        raise ValueError(
            f"{name} must be a uniform periodic grid stored from one step through {period:g}."
        )
    return (np.rint(np.remainder(values, period) / step).astype(np.int64) - 1) % coordinates.size


def nearest_sorted_indices(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    right = np.searchsorted(coordinates, values, side="left")
    right = np.clip(right, 0, coordinates.size - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(values - coordinates[left]) <= np.abs(coordinates[right] - values)
    return np.where(choose_left, left, right).astype(np.int64)


def cartesian_centers(limit: float, requested_size: float) -> tuple[np.ndarray, float]:
    count = max(2, int(np.ceil(2.0 * float(limit) / float(requested_size))))
    spacing = 2.0 * float(limit) / count
    centers = np.linspace(
        -float(limit) + 0.5 * spacing,
        float(limit) - 0.5 * spacing,
        count,
        dtype=np.float32,
    )
    return centers, spacing


def sample_cartesian_voxels(
    field: np.ndarray,
    rho_grid: np.ndarray,
    theta_grid: np.ndarray,
    phi_grid_deg: np.ndarray,
    R0: float,
    requested_size: float,
    max_lattice_cells: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Nearest-sample a toroidal regular field at Cartesian voxel centers."""
    radial_extent = float(R0) + float(rho_grid[-1])
    x, dx = cartesian_centers(radial_extent, requested_size)
    y, dy = cartesian_centers(radial_extent, requested_size)
    z, dz = cartesian_centers(float(rho_grid[-1]), requested_size)
    lattice_count = int(x.size * y.size * z.size)
    if lattice_count > max_lattice_cells:
        raise ValueError(
            f"Requested voxel size creates {lattice_count:,} Cartesian cells, "
            f"above --max-lattice-cells={max_lattice_cells:,}. Increase "
            "--voxel-size or explicitly raise the safety limit."
        )
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    major_radius = np.sqrt(xx * xx + yy * yy)
    minor_x = major_radius - float(R0)
    rho = np.sqrt(minor_x * minor_x + zz * zz)
    inside_torus = rho <= float(rho_grid[-1])

    positions = np.column_stack(
        (xx[inside_torus], yy[inside_torus], zz[inside_torus])
    ).astype(np.float32)
    rho = np.asarray(rho[inside_torus], dtype=np.float64)
    theta = np.remainder(
        np.arctan2(positions[:, 2], np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2) - R0),
        2.0 * np.pi,
    )
    # Repository phi increases clockwise, hence atan2(-y, x).
    phi_deg = np.remainder(
        np.degrees(np.arctan2(-positions[:, 1], positions[:, 0])),
        360.0,
    )

    rho_index = nearest_sorted_indices(rho, rho_grid)
    theta_index = periodic_uniform_indices(theta, theta_grid, 2.0 * np.pi, "theta")
    phi_index = periodic_uniform_indices(phi_deg, phi_grid_deg, 360.0, "phi")
    values = np.asarray(field[phi_index, theta_index, rho_index], dtype=np.float64)
    return positions, values, min(dx, dy, dz), lattice_count


def finite_data_range(
    field: np.ndarray,
    positive_only: bool,
) -> tuple[float, float]:
    data_min = np.inf
    data_max = -np.inf
    for plane_index in range(field.shape[0]):
        plane = np.asarray(field[plane_index])
        finite = plane[np.isfinite(plane)]
        if positive_only:
            finite = finite[finite > 0.0]
        if finite.size:
            data_min = min(data_min, float(finite.min()))
            data_max = max(data_max, float(finite.max()))
    if not np.isfinite(data_min):
        qualifier = "positive finite" if positive_only else "finite"
        raise ValueError(f"The field contains no {qualifier} scalar values.")
    return data_min, data_max


def filter_voxels(
    positions: np.ndarray,
    values: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    keep = np.isfinite(values)
    if args.color_scale == "log":
        keep &= values > 0.0
    if args.value_threshold is not None:
        keep &= values >= float(args.value_threshold)
    positions = positions[keep]
    values = values[keep]
    if values.size == 0:
        raise ValueError("No voxels remain after applying the value filters.")

    percentile_threshold = None
    if args.value_percentile > 0.0:
        percentile_threshold = float(np.percentile(values, args.value_percentile))
        keep = values >= percentile_threshold
        positions = positions[keep]
        values = values[keep]
    if values.size > args.max_voxels:
        selection = np.sort(
            np.random.default_rng(args.sample_seed).choice(
                values.size,
                size=args.max_voxels,
                replace=False,
            )
        )
        positions = positions[selection]
        values = values[selection]
    return positions, values, percentile_threshold


def value_colors(
    values: np.ndarray,
    args: argparse.Namespace,
    data_min: float,
    data_max: float,
) -> tuple[np.ndarray, float, float]:
    color_min = data_min if args.vmin is None else float(args.vmin)
    color_max = data_max if args.vmax is None else float(args.vmax)
    if args.color_scale == "log":
        transformed = np.log10(values)
        transformed_min = np.log10(color_min)
        transformed_max = np.log10(color_max)
    else:
        transformed = values
        transformed_min = color_min
        transformed_max = color_max
    if not transformed_max > transformed_min:
        raise ValueError("Resolved color limits require vmin < vmax.")

    normalized = np.clip(
        (transformed - transformed_min) / (transformed_max - transformed_min),
        0.0,
        1.0,
    )
    try:
        from matplotlib import colormaps

        cmap = colormaps[args.cmap]
        if args.cmap_reverse:
            cmap = cmap.reversed()
        colors = np.asarray(cmap(normalized), dtype=np.float32)
    except KeyError as exc:
        raise ValueError(f"Unknown Matplotlib colormap {args.cmap!r}.") from exc
    colors[:, 3] = args.alpha_min + (
        args.alpha_max - args.alpha_min
    ) * normalized.astype(np.float32) ** float(args.alpha_power)
    return colors, color_min, color_max


def make_window(QtWidgets, canvas, args: argparse.Namespace, status: str):
    window = QtWidgets.QWidget()
    window.setWindowTitle("Toroidal scalar volume voxels")
    window.setStyleSheet(
        f"background-color: {args.plot_background}; color: {args.plot_foreground};"
    )
    layout = QtWidgets.QVBoxLayout(window)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)
    layout.addWidget(canvas, stretch=1)
    label = QtWidgets.QLabel(status)
    label.setWordWrap(True)
    layout.addWidget(label)
    width, height = parse_size(args.size)
    window.resize(width, height + 38)
    return window


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    validate_args(args)
    field, rho, theta, phi_deg, field_dir = load_regular_field(args)
    positions, values, actual_spacing, lattice_count = sample_cartesian_voxels(
        field,
        rho,
        theta,
        phi_deg,
        args.R0,
        args.voxel_size,
        args.max_lattice_cells,
    )
    candidate_count = values.size
    positions, values, percentile_threshold = filter_voxels(positions, values, args)
    data_min, data_max = finite_data_range(
        field,
        positive_only=(args.color_scale == "log"),
    )
    colors, color_min, color_max = value_colors(values, args, data_min, data_max)
    glyph_size = actual_spacing * float(args.voxel_scale)

    scale_name = "log10" if args.color_scale == "log" else "linear"
    print(f"field: {field_dir / args.field_file}")
    print(f"field shape (phi, theta, rho): {field.shape}, dtype={field.dtype}")
    print(f"Cartesian lattice cells: {lattice_count:,}")
    print(f"cells inside torus: {candidate_count:,}")
    print(f"voxels displayed: {values.size:,}")
    print(f"actual voxel spacing: {actual_spacing:.6g} m; glyph size: {glyph_size:.6g} m")
    print(f"field range: {data_min:.6g}..{data_max:.6g}")
    print(f"color/alpha range ({scale_name}): {color_min:.6g}..{color_max:.6g}")
    if percentile_threshold is not None:
        print(
            f"percentile threshold ({args.value_percentile:g}%): "
            f"{percentile_threshold:.6g}"
        )
    if args.dry_run:
        return 0

    fpl = import_fastplotlib()
    QtCore, QtWidgets, QRenderWidget = import_qt_and_canvas()
    del QtCore
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    size = parse_size(args.size)
    canvas = QRenderWidget(
        size=size,
        title="Toroidal scalar volume voxels",
        present_method=args.present_method,
    )
    figure = fpl.Figure(
        cameras="3d",
        controller_types="orbit",
        canvas=canvas,
        size=size,
    )
    subplot = figure[0, 0]
    set_subplot_background(subplot, args.plot_background)
    try:
        subplot.set_title(args.value_label)
    except Exception:
        pass

    add_scene_torus(subplot, args)
    add_scene_ports(subplot, args)
    add_scene_compass(subplot, args)
    subplot.add_scatter(
        data=positions,
        colors=colors,
        sizes=float(glyph_size),
        uniform_size=True,
        size_space="world",
        mode="markers",
        markers="s",
        uniform_marker=True,
        edge_width=0.0,
    )

    setup_camera(subplot, args.R0, args.a)
    figure.show(axes_visible=args.axes)
    status = (
        f"{args.value_label}: {color_min:.3g} to {color_max:.3g} ({scale_name}); "
        f"{values.size:,} voxels; drag to orbit, scroll to zoom"
    )
    window = make_window(QtWidgets, canvas, args, status)
    window.show()
    if __name__ == "__main__":
        app.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())
