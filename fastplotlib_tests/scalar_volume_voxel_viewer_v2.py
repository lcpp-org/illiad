"""Render a toroidal scalar field with linear mesh-conformal cells.

Each cell is a hexahedron whose eight base vertices lie on the saved
``(phi, theta, rho)`` grid. Straight edges and planar triangular faces provide
a piecewise-linear approximation to the toroidal coordinate mesh. The default
``shrink`` opacity mode uses opaque, scalar-scaled cells; ``alpha`` retains
weighted blended transparency as a fallback.

Example:
    conda run -n testenv python \
        fastplotlib_tests/scalar_volume_voxel_viewer_v2.py

Use ``--dry-run`` to construct and validate the cell geometry without opening
the GUI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

from scalar_volume_voxel_viewer import (
    DEFAULT_FIELD_DIR,
    add_scene_compass,
    add_scene_ports,
    add_scene_torus,
    finite_data_range,
    import_fastplotlib,
    import_qt_and_canvas,
    load_regular_field,
    make_window,
    parse_size,
    set_subplot_background,
    setup_camera,
    value_colors,
)


@dataclass(frozen=True)
class CellSelection:
    phi_lower: np.ndarray
    phi_upper: np.ndarray
    theta_lower: np.ndarray
    theta_upper: np.ndarray
    rho_lower: np.ndarray
    rho_upper: np.ndarray
    values: np.ndarray
    coarse_cell_count: int
    valid_cell_count: int
    percentile_threshold: float | None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Display a regular toroidal scalar field with opaque shrunken or "
            "alpha-blended linear mesh cells."
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
        "--phi-stride",
        type=int,
        default=6,
        help="Native toroidal grid intervals represented by one coarse cell.",
    )
    parser.add_argument(
        "--theta-stride",
        type=int,
        default=6,
        help="Native poloidal grid intervals represented by one coarse cell.",
    )
    parser.add_argument(
        "--rho-stride",
        type=int,
        default=4,
        help="Native radial grid intervals represented by one coarse cell.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=50_000,
        help="Maximum rendered cells; excess valid cells are thinned deterministically.",
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--value-threshold",
        type=float,
        default=None,
        help="Only show cell values greater than or equal to this threshold.",
    )
    parser.add_argument(
        "--value-percentile",
        type=float,
        default=0.0,
        help="Only show values at or above this percentile of valid cells.",
    )

    parser.add_argument(
        "--opacity-mode",
        choices=("shrink", "alpha"),
        default="shrink",
        help="Opaque scalar-dependent shrinking, or weighted alpha blending.",
    )
    parser.add_argument(
        "--shrink-min",
        type=float,
        default=0.25,
        help="Cell scale assigned to the low end of the color range.",
    )
    parser.add_argument(
        "--shrink-max",
        type=float,
        default=0.95,
        help="Cell scale assigned to the high end of the color range.",
    )
    parser.add_argument(
        "--shrink-power",
        type=float,
        default=0.70,
        help="Exponent applied to normalized values before shrink mapping.",
    )
    parser.add_argument(
        "--alpha-cell-scale",
        type=float,
        default=0.92,
        help="Uniform centroid scale used by the alpha fallback.",
    )
    parser.add_argument(
        "--cell-style",
        choices=("faces", "wireframe", "both"),
        default="faces",
        help="Render planar faces, mesh edges, or both.",
    )
    parser.add_argument("--edge-color", default="#25313D")
    parser.add_argument("--edge-thickness", type=float, default=1.0)

    parser.add_argument("--color-scale", choices=("log", "linear"), default="log")
    parser.add_argument("--cmap", default="afmhot")
    parser.add_argument("--cmap-reverse", action="store_true")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--alpha-min", type=float, default=0.03)
    parser.add_argument("--alpha-max", type=float, default=0.92)
    parser.add_argument("--alpha-power", type=float, default=1.35)

    parser.add_argument("--plot-background", default="#02070D")
    parser.add_argument("--plot-foreground", default="#E8EDF2")
    parser.add_argument("--R0", type=float, default=0.72)
    parser.add_argument("--a", type=float, default=0.19)
    parser.add_argument("--size", default="1280x760")
    parser.add_argument(
        "--present-method",
        choices=("bitmap", "screen"),
        default="bitmap",
        help="Qt presentation method; screen may improve desktop FPS.",
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
    if min(args.phi_stride, args.theta_stride, args.rho_stride) < 1:
        raise SystemExit("All grid strides must be positive.")
    if args.max_cells < 1:
        raise SystemExit("--max-cells must be positive.")
    if not 0.0 <= args.value_percentile <= 100.0:
        raise SystemExit("--value-percentile must be between 0 and 100.")
    if not 0.0 < args.shrink_min <= args.shrink_max <= 1.0:
        raise SystemExit("Require 0 < --shrink-min <= --shrink-max <= 1.")
    if args.shrink_power <= 0.0:
        raise SystemExit("--shrink-power must be positive.")
    if not 0.0 < args.alpha_cell_scale <= 1.0:
        raise SystemExit("--alpha-cell-scale must be in (0, 1].")
    if not 0.0 <= args.alpha_min <= args.alpha_max <= 1.0:
        raise SystemExit("Require 0 <= --alpha-min <= --alpha-max <= 1.")
    if args.alpha_power <= 0.0:
        raise SystemExit("--alpha-power must be positive.")
    if args.edge_thickness <= 0.0:
        raise SystemExit("--edge-thickness must be positive.")
    if args.vmin is not None and args.vmax is not None and args.vmax <= args.vmin:
        raise SystemExit("--vmax must be greater than --vmin.")
    if args.color_scale == "log":
        if args.vmin is not None and args.vmin <= 0.0:
            raise SystemExit("A logarithmic color scale requires --vmin > 0.")
        if args.vmax is not None and args.vmax <= 0.0:
            raise SystemExit("A logarithmic color scale requires --vmax > 0.")


def periodic_cell_indices(size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.arange(0, int(size), int(stride), dtype=np.int64)
    upper = np.roll(lower, -1)
    return lower, upper


def radial_cell_indices(size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.arange(0, int(size) - 1, int(stride), dtype=np.int64)
    upper = np.minimum(lower + int(stride), int(size) - 1)
    return lower, upper


def cell_corner_values(
    field: np.ndarray,
    phi_lower: np.ndarray,
    phi_upper: np.ndarray,
    theta_lower: np.ndarray,
    theta_upper: np.ndarray,
    rho_lower: np.ndarray,
    rho_upper: np.ndarray,
) -> np.ndarray:
    indices = (
        (phi_lower, theta_lower, rho_lower),
        (phi_upper, theta_lower, rho_lower),
        (phi_upper, theta_upper, rho_lower),
        (phi_lower, theta_upper, rho_lower),
        (phi_lower, theta_lower, rho_upper),
        (phi_upper, theta_lower, rho_upper),
        (phi_upper, theta_upper, rho_upper),
        (phi_lower, theta_upper, rho_upper),
    )
    return np.column_stack(
        [np.asarray(field[p, t, r], dtype=np.float64) for p, t, r in indices]
    )


def select_cells(field: np.ndarray, args: argparse.Namespace) -> CellSelection:
    if args.phi_stride >= field.shape[0]:
        raise ValueError("--phi-stride must be smaller than the phi grid size.")
    if args.theta_stride >= field.shape[1]:
        raise ValueError("--theta-stride must be smaller than the theta grid size.")
    if args.rho_stride >= field.shape[2]:
        raise ValueError("--rho-stride must be smaller than the rho grid size.")
    phi_lower_1d, phi_upper_1d = periodic_cell_indices(
        field.shape[0], args.phi_stride
    )
    theta_lower_1d, theta_upper_1d = periodic_cell_indices(
        field.shape[1], args.theta_stride
    )
    rho_lower_1d, rho_upper_1d = radial_cell_indices(
        field.shape[2], args.rho_stride
    )
    phi_lower, theta_lower, rho_lower = np.meshgrid(
        phi_lower_1d,
        theta_lower_1d,
        rho_lower_1d,
        indexing="ij",
    )
    phi_upper, theta_upper, rho_upper = np.meshgrid(
        phi_upper_1d,
        theta_upper_1d,
        rho_upper_1d,
        indexing="ij",
    )
    arrays = [
        array.ravel()
        for array in (
            phi_lower,
            phi_upper,
            theta_lower,
            theta_upper,
            rho_lower,
            rho_upper,
        )
    ]
    coarse_cell_count = arrays[0].size
    corners = cell_corner_values(field, *arrays)
    valid = np.all(np.isfinite(corners), axis=1)
    if args.color_scale == "log":
        valid &= np.all(corners > 0.0, axis=1)
    arrays = [array[valid] for array in arrays]
    corners = corners[valid]
    valid_cell_count = corners.shape[0]
    if not valid_cell_count:
        raise ValueError("No cells have eight valid field corners.")

    if args.color_scale == "log":
        values = np.exp(np.mean(np.log(corners), axis=1))
    else:
        values = np.mean(corners, axis=1)
    keep = np.ones(values.size, dtype=bool)
    if args.value_threshold is not None:
        keep &= values >= float(args.value_threshold)
    if not np.any(keep):
        raise ValueError("No cells remain after applying --value-threshold.")
    percentile_threshold = None
    if args.value_percentile > 0.0:
        percentile_threshold = float(np.percentile(values[keep], args.value_percentile))
        keep &= values >= percentile_threshold
    arrays = [array[keep] for array in arrays]
    values = values[keep]
    if not values.size:
        raise ValueError("No cells remain after applying the value filters.")

    if values.size > args.max_cells:
        selection = np.sort(
            np.random.default_rng(args.sample_seed).choice(
                values.size,
                size=args.max_cells,
                replace=False,
            )
        )
        arrays = [array[selection] for array in arrays]
        values = values[selection]
    return CellSelection(
        *arrays,
        values,
        coarse_cell_count,
        valid_cell_count,
        percentile_threshold,
    )


def unwrap_upper(
    lower: np.ndarray,
    upper: np.ndarray,
    period: float,
) -> np.ndarray:
    return np.where(upper <= lower, upper + float(period), upper)


def toroidal_xyz(
    phi: np.ndarray,
    theta: np.ndarray,
    rho: np.ndarray,
    R0: float,
) -> np.ndarray:
    radius = float(R0) + rho * np.cos(theta)
    return np.stack(
        (radius * np.cos(phi), -radius * np.sin(phi), rho * np.sin(theta)),
        axis=-1,
    )


def cell_base_vertices(
    cells: CellSelection,
    rho_grid: np.ndarray,
    theta_grid: np.ndarray,
    phi_grid_deg: np.ndarray,
    R0: float,
) -> np.ndarray:
    phi_lower = np.radians(phi_grid_deg[cells.phi_lower])
    phi_upper = unwrap_upper(
        phi_lower,
        np.radians(phi_grid_deg[cells.phi_upper]),
        2.0 * np.pi,
    )
    theta_lower = theta_grid[cells.theta_lower]
    theta_upper = unwrap_upper(
        theta_lower,
        theta_grid[cells.theta_upper],
        2.0 * np.pi,
    )
    rho_lower = rho_grid[cells.rho_lower]
    rho_upper = rho_grid[cells.rho_upper]

    coordinates = (
        (phi_lower, theta_lower, rho_lower),
        (phi_upper, theta_lower, rho_lower),
        (phi_upper, theta_upper, rho_lower),
        (phi_lower, theta_upper, rho_lower),
        (phi_lower, theta_lower, rho_upper),
        (phi_upper, theta_lower, rho_upper),
        (phi_upper, theta_upper, rho_upper),
        (phi_lower, theta_upper, rho_upper),
    )
    return np.stack(
        [toroidal_xyz(phi, theta, rho, R0) for phi, theta, rho in coordinates],
        axis=1,
    ).astype(np.float32)


def normalized_values(
    values: np.ndarray,
    args: argparse.Namespace,
    color_min: float,
    color_max: float,
) -> np.ndarray:
    if args.color_scale == "log":
        values = np.log10(values)
        color_min = np.log10(color_min)
        color_max = np.log10(color_max)
    return np.clip(
        (values - color_min) / (color_max - color_min),
        0.0,
        1.0,
    ).astype(np.float32)


def scale_cell_vertices(
    vertices: np.ndarray,
    normalized: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    if args.opacity_mode == "shrink":
        scales = args.shrink_min + (
            args.shrink_max - args.shrink_min
        ) * normalized ** float(args.shrink_power)
    else:
        scales = np.full(normalized.shape, args.alpha_cell_scale, dtype=np.float32)
    centers = np.mean(vertices, axis=1, keepdims=True)
    scaled = centers + (vertices - centers) * scales[:, None, None]
    return scaled.astype(np.float32), scales.astype(np.float32)


def cell_triangle_indices(cell_count: int) -> np.ndarray:
    # The toroidal mapping is left-handed in (phi, theta, rho), so these
    # windings are reversed from the standard right-handed hexahedron faces.
    triangles = np.asarray(
        [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
            (0, 5, 1), (0, 4, 5),
            (3, 6, 7), (3, 2, 6),
            (0, 7, 4), (0, 3, 7),
            (1, 6, 2), (1, 5, 6),
        ],
        dtype=np.uint32,
    )
    offsets = (np.arange(cell_count, dtype=np.uint32) * 8)[:, None, None]
    return (triangles[None, :, :] + offsets).reshape(-1, 3)


def build_cell_geometry(
    cells: CellSelection,
    rho: np.ndarray,
    theta: np.ndarray,
    phi_deg: np.ndarray,
    args: argparse.Namespace,
    data_min: float,
    data_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    colors, color_min, color_max = value_colors(
        cells.values,
        args,
        data_min,
        data_max,
    )
    normalized = normalized_values(cells.values, args, color_min, color_max)
    vertices = cell_base_vertices(cells, rho, theta, phi_deg, args.R0)
    vertices, scales = scale_cell_vertices(vertices, normalized, args)
    if args.opacity_mode == "shrink":
        colors[:, 3] = 1.0
    vertex_colors = np.repeat(colors, 8, axis=0).astype(np.float32)
    indices = cell_triangle_indices(cells.values.size)
    return (
        vertices.reshape(-1, 3),
        indices,
        vertex_colors,
        scales,
        color_min,
        color_max,
    )


def add_cell_mesh(
    subplot,
    positions: np.ndarray,
    indices: np.ndarray,
    colors: np.ndarray,
    args: argparse.Namespace,
):
    import pygfx as gfx

    geometry = gfx.Geometry(
        positions=positions,
        indices=indices,
        colors=colors,
    )
    meshes = []
    if args.cell_style in ("faces", "both"):
        alpha_mode = "solid" if args.opacity_mode == "shrink" else "weighted_blend"
        material = gfx.MeshBasicMaterial(
            color_mode="vertex",
            side="front" if args.opacity_mode == "shrink" else "both",
            alpha_mode=alpha_mode,
            depth_write=(args.opacity_mode == "shrink"),
        )
        mesh = gfx.Mesh(geometry, material)
        subplot.scene.add(mesh)
        meshes.append(mesh)
    if args.cell_style in ("wireframe", "both"):
        cell_vertices = positions.reshape(-1, 8, 3)
        edge_indices = np.asarray(
            [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ],
            dtype=np.int64,
        )
        edge_positions = cell_vertices[:, edge_indices, :].reshape(-1, 3)
        if args.cell_style == "wireframe":
            cell_colors = colors.reshape(-1, 8, 4)[:, 0, :]
            edge_colors = np.repeat(cell_colors, 24, axis=0)
            edge_geometry = gfx.Geometry(
                positions=edge_positions,
                colors=edge_colors,
            )
            edge_material = gfx.LineSegmentMaterial(
                thickness=float(args.edge_thickness),
                color_mode="vertex",
                alpha_mode=(
                    "solid" if args.opacity_mode == "shrink" else "weighted_blend"
                ),
                depth_write=(args.opacity_mode == "shrink"),
            )
        else:
            edge_geometry = gfx.Geometry(positions=edge_positions)
            edge_material = gfx.LineSegmentMaterial(
                thickness=float(args.edge_thickness),
                color=args.edge_color,
                color_mode="uniform",
                alpha_mode="solid",
                depth_compare="<=",
                depth_write=False,
            )
        edge_lines = gfx.Line(
            edge_geometry,
            edge_material,
            render_order=1,
        )
        subplot.scene.add(edge_lines)
        meshes.append(edge_lines)
    return meshes


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    validate_args(args)
    field, rho, theta, phi_deg, field_dir = load_regular_field(args)
    cells = select_cells(field, args)
    data_min, data_max = finite_data_range(
        field,
        positive_only=(args.color_scale == "log"),
    )
    positions, indices, colors, scales, color_min, color_max = build_cell_geometry(
        cells,
        rho,
        theta,
        phi_deg,
        args,
        data_min,
        data_max,
    )

    scale_name = "log10" if args.color_scale == "log" else "linear"
    print(f"field: {field_dir / args.field_file}")
    print(f"field shape (phi, theta, rho): {field.shape}, dtype={field.dtype}")
    print(
        "coarse grid: "
        f"strides=({args.phi_stride}, {args.theta_stride}, {args.rho_stride}), "
        f"cells={cells.coarse_cell_count:,}"
    )
    print(f"cells with eight valid corners: {cells.valid_cell_count:,}")
    print(f"cells displayed: {cells.values.size:,}")
    print(f"geometry: {positions.shape[0]:,} vertices, {indices.shape[0]:,} triangles")
    print(f"field range: {data_min:.6g}..{data_max:.6g}")
    print(f"color range ({scale_name}): {color_min:.6g}..{color_max:.6g}")
    print(
        f"opacity mode: {args.opacity_mode}; cell scale: "
        f"{float(scales.min()):.3f}..{float(scales.max()):.3f}"
    )
    if cells.percentile_threshold is not None:
        print(
            f"percentile threshold ({args.value_percentile:g}%): "
            f"{cells.percentile_threshold:.6g}"
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
        title="Toroidal scalar mesh cells",
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
    cell_meshes = add_cell_mesh(subplot, positions, indices, colors, args)
    del cell_meshes

    setup_camera(subplot, args.R0, args.a)
    figure.show(axes_visible=args.axes)
    status = (
        f"{args.value_label}: {color_min:.3g} to {color_max:.3g} ({scale_name}); "
        f"{cells.values.size:,} linear conformal cells; {args.opacity_mode} mode"
    )
    window = make_window(QtWidgets, canvas, args, status)
    window.setWindowTitle("Toroidal scalar mesh cells")
    window.show()
    if __name__ == "__main__":
        app.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())
