"""Export scientific wall maps as a textured OBJ/MTL/PNG in CAD coordinates.

Examples (from the repository root)::

    python misc_scripts/inventor_output.py --demo
    python misc_scripts/inventor_output.py Wallpt_OUTPUT.npy --kind deposition
    python misc_scripts/inventor_output.py magnetic_footprint.npz --kind footprint --wall-radius 0.19
    python misc_scripts/inventor_output.py map.npy --kind map --map-coordinates boris

Generic maps have shape (N_theta, N_phi), with no duplicate periodic endpoints.
Each value describes a pixel centered in a uniform angular bin: computational
bins span theta/phi [0, 360] degrees; Boris bins span theta [-180, 180] and
phi_wall [0, 360]. Footprint samples retain their saved computational locations
as pixel centers. Deposition reproduces boris_plotWallHist's angular probability
density (per degree squared), not particle flux per square meter.

Geometry is an ideal torus, in millimeters. Keep the three output files together.
OBJ is unitless: select millimeters in the importing application. The texture
contains only the scalar map, without plot labels or port overlays.
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from illiad.utilities.coordtrans import RTP_to_XYZ_many


# EDITABLE DEFAULTS
OUTPUT_DIR = PROJECT_ROOT / "output/hidra_inventor_wall"
RMAJOR_M = 0.72
# Preserved from the generated script. Choose the surface matching your data;
# magnetic_footprint.py uses RMINOR=0.19 m, with launches just inside that wall.
WALL_RADIUS_M = 0.19 - 0.001
PHI_WALL_OFFSET_DEG = 18.0
COLORMAP = "inferno"
COLOR_SCALE = "linear"
VMIN = None
VMAX = None
MM_PER_M = 1000.0

# Same transform and row-vector multiplication as boxMagneticField.py.
# To transform from X Y Z coordinate system set up in the code (+x at 18 degrees CW from the South Side and +y at 18 degrees CW from
# the East Side +Z towards the roof forming a right handed system)
# to the XYZ coordinates according to tokamak energ (+x at North Side Split, +y at the East Side and +z towards the floor - also right handed)
RAD18 = np.deg2rad(18.0)
CODE_TO_INVENTOR = np.array([
    [-np.cos(RAD18), -np.sin(RAD18), 0.0],
    [-np.sin(RAD18), np.cos(RAD18), 0.0],
    [0.0, 0.0, -1.0],
])
# T.T @ T = I and det(T) = +1: rotate both vertices and normals, preserving
# handedness and winding. This Cartesian rotation is separate from Boris's
# display-angle convention phi_wall = (198 - phi_comp_deg) % 360.


def validate_map(data):
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or min(data.shape) < 3:
        raise ValueError("Expected a (theta, phi) map with at least 3 pixels per axis.")
    if not np.any(np.isfinite(data)):
        raise ValueError("Map has no finite values.")
    return data


def periodic_centers(angles, name):
    """Sort a complete uniform periodic sample axis, excluding duplicate ends."""
    angles = np.mod(angles, 2 * np.pi)
    angles[np.isclose(angles, 2 * np.pi, rtol=0, atol=1e-10)] = 0.0
    order = np.argsort(angles)
    centers = angles[order]
    step = 2 * np.pi / centers.size
    if not np.allclose(np.diff(np.r_[centers, centers[0] + 2 * np.pi]),
                       step, rtol=0, atol=1e-8):
        raise ValueError(f"Expected a complete uniform periodic {name} axis.")
    return centers, order


def load_map(path, kind, map_coordinates="computational"):
    """Return data and pixel-center angles in computational radians."""
    if kind == "footprint":
        with np.load(path, allow_pickle=False) as saved:
            data = validate_map(saved["connection_length_m"])
            points = saved["initial_conditions_rtp"]
        if points.shape != (*data.shape, 3) or not np.all(np.isfinite(points)):
            raise ValueError("Expected finite initial_conditions_rtp of shape (*map_shape, 3).")
        if not (np.allclose(points[..., 1], points[:, :1, 1], rtol=0, atol=1e-10)
                and np.allclose(points[..., 2], points[:1, :, 2], rtol=0, atol=1e-10)):
            raise ValueError("Expected a separable (theta, phi) footprint grid.")
        theta, rows = periodic_centers(points[:, 0, 1], "theta")
        phi, columns = periodic_centers(points[0, :, 2], "phi")
        return data[np.ix_(rows, columns)], theta, phi

    data = np.load(path, allow_pickle=False)
    if kind == "deposition":
        # Wallpt_OUTPUT.npy: (7, N), first three rows are rho, theta, phi.
        if data.ndim != 2 or data.shape[0] not in (3, 7) or data.shape[1] == 0:
            raise ValueError("Expected nonempty Boris RTP output with shape (3, N) or (7, N).")
        if not np.all(np.isfinite(data[:3])):
            raise ValueError("Deposition coordinates must be finite.")
        theta = np.rad2deg(np.where(data[1] > np.pi, data[1] - 2 * np.pi, data[1]))
        phi = (-np.rad2deg(data[2]) + 180 + PHI_WALL_OFFSET_DEG) % 360
        if np.any((theta < -180) | (theta > 180)):
            raise ValueError("Deposition theta must lie in [-pi, 2*pi].")
        data = np.histogram2d(phi, theta, bins=[np.linspace(0, 360, 361),
                                              np.linspace(-180, 180, 181)], density=True)[0].T
        map_coordinates = "boris"
    elif kind != "map":
        raise ValueError("kind must be map, deposition, or footprint.")
    data = validate_map(data)
    theta = (np.arange(data.shape[0]) + 0.5) * 2 * np.pi / data.shape[0]
    phi = (np.arange(data.shape[1]) + 0.5) * 2 * np.pi / data.shape[1]
    if map_coordinates == "boris":
        theta -= np.pi
        phi = np.deg2rad(180 + PHI_WALL_OFFSET_DEG) - phi
    elif map_coordinates != "computational":
        raise ValueError("map_coordinates must be computational or boris.")
    theta, rows = periodic_centers(theta, "theta")
    phi, columns = periodic_centers(phi, "phi")
    return data[np.ix_(rows, columns)], theta, phi


def build_mesh(theta, phi, major_radius, wall_radius):
    """Use pixel edges for geometry; each map value lies at its cell center."""
    if not (np.isfinite(major_radius) and np.isfinite(wall_radius)
            and 0 < wall_radius < major_radius):
        raise ValueError("Radii must satisfy 0 < wall_radius < major_radius and be finite.")
    nt, np_ = len(theta), len(phi)
    for angles, name in ((theta, "theta"), (phi, "phi")):
        if len(angles) < 3 or not np.all(np.isfinite(angles)) or not np.allclose(
                np.diff(angles), 2 * np.pi / len(angles), rtol=0, atol=1e-8):
            raise ValueError(f"{name} centers must be increasing, uniform, and cover a full period.")
    th_edges = theta[0] + (np.arange(nt + 1) - 0.5) * 2 * np.pi / nt
    ph_edges = phi[0] + (np.arange(np_ + 1) - 0.5) * 2 * np.pi / np_
    th, ph = np.meshgrid(th_edges, ph_edges, indexing="ij")
    rtp = np.stack([np.full_like(th, wall_radius), th, ph], axis=-1)
    xyz = RTP_to_XYZ_many(rtp, Rmajor=major_radius)
    # Vacuum-facing normals are minus the unit displacement from tube center.
    radial = RTP_to_XYZ_many(np.stack([np.ones_like(th), th, ph], axis=-1), Rmajor=0.0)
    vertices = (xyz @ CODE_TO_INVENTOR).reshape(-1, 3) * MM_PER_M
    normals = (-radial @ CODE_TO_INVENTOR).reshape(-1, 3)
    v, u = np.meshgrid(np.linspace(0, 1, nt + 1), np.linspace(0, 1, np_ + 1), indexing="ij")
    uv = np.column_stack([u.ravel(), v.ravel()])
    # Duplicate seam vertices carry distinct UVs at physically identical points.
    base = (np.arange(nt)[:, None] * (np_ + 1) + np.arange(np_)[None, :]).ravel()
    # For clockwise phi, these cross products point into the vacuum volume.
    faces = np.stack([np.column_stack([base, base + np_ + 2, base + np_ + 1]),
                      np.column_stack([base, base + 1, base + np_ + 2])], axis=1).reshape(-1, 3)
    return vertices, uv, normals, faces


def export_map(data, theta, phi, output_dir=OUTPUT_DIR, *, major_radius=RMAJOR_M,
               wall_radius=WALL_RADIUS_M, cmap=COLORMAP, scale=COLOR_SCALE,
               vmin=VMIN, vmax=VMAX):
    """Write the texture and CAD-frame torus; return OBJ, MTL, PNG paths."""
    data = validate_map(data)
    if data.shape != (len(theta), len(phi)):
        raise ValueError("Map shape must match theta and phi center axes.")
    vertices, uv, normals, faces = build_mesh(theta, phi, major_radius, wall_radius)
    valid = np.isfinite(data)
    if scale == "log":
        valid &= data > 0
    elif scale != "linear":
        raise ValueError("scale must be linear or log.")
    if not np.any(valid):
        raise ValueError("No valid map values for the requested color scale.")
    low = float(data[valid].min()) if vmin is None else float(vmin)
    high = float(data[valid].max()) if vmax is None else float(vmax)
    if vmin is None and vmax is None and low == high:
        delta = abs(low) * 0.01 or 0.5
        low, high = low - delta, high + delta
    if not (np.isfinite(low) and np.isfinite(high) and low < high
            and (scale != "log" or low > 0)):
        raise ValueError("Require finite vmin < vmax (and vmin > 0 for log scale).")
    norm = (LogNorm if scale == "log" else Normalize)(vmin=low, vmax=high, clip=True)
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad("0.5")  # Missing/nonpositive log values are explicitly gray.
    rgb = color_map(norm(np.ma.masked_where(~valid, data)))[..., :3]
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    obj = output_dir / "HIDRA_inner_wall_textured.obj"
    mtl = output_dir / "HIDRA_inner_wall_textured.mtl"
    texture = output_dir / "HIDRA_wall_map.png"
    # Image row zero is the top; OBJ v=0 is the bottom.
    plt.imsave(texture, np.flipud(rgb))
    mtl.write_text("# HIDRA scientific wall-map material\nnewmtl HIDRA_Wall_Map\n"
                   "Ka 1 1 1\nKd 1 1 1\nKs 0 0 0\nNs 1\nd 1\nillum 1\n"
                   f"map_Kd {texture.name}\n", encoding="utf-8")
    with obj.open("w", encoding="utf-8") as stream:
        stream.write("# Ideal vacuum-facing torus; units: millimeters\n"
                     "# Tokamak Energy frame: +X North split, +Y East, +Z floor\n"
                     f"# Major radius: {major_radius * MM_PER_M:g}; wall radius: {wall_radius * MM_PER_M:g}\n"
                     f"# Texture: {cmap}, {scale}, vmin={low:g}, vmax={high:g}; missing=gray\n"
                     f"mtllib {mtl.name}\n")
        for prefix, values in (("v", vertices), ("vt", uv), ("vn", normals)):
            for row in values:
                stream.write(prefix + " " + " ".join(f"{value:.9f}" for value in row) + "\n")
        stream.write("usemtl HIDRA_Wall_Map\ns 1\n")
        for face in faces + 1:
            stream.write("f " + " ".join(f"{i}/{i}/{i}" for i in face) + "\n")
    return obj, mtl, texture


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_file", nargs="?", type=Path)
    parser.add_argument("--demo", action="store_true", help="Export explicit synthetic demonstration data.")
    parser.add_argument("--kind", choices=("map", "deposition", "footprint"), default="map")
    parser.add_argument("--map-coordinates", choices=("computational", "boris"), default="computational")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--major-radius", type=float, default=RMAJOR_M, help="Major radius in meters.")
    parser.add_argument("--wall-radius", type=float, default=WALL_RADIUS_M, help="Overlay minor radius in meters.")
    parser.add_argument("--cmap", default=COLORMAP)
    parser.add_argument("--scale", choices=("linear", "log"), default=COLOR_SCALE)
    parser.add_argument("--vmin", type=float, default=VMIN)
    parser.add_argument("--vmax", type=float, default=VMAX)
    args = parser.parse_args()
    if args.demo == (args.data_file is not None):
        parser.error("Supply either a data_file or --demo.")
    if args.demo:
        theta = (np.arange(180) + 0.5) * 2 * np.pi / 180
        phi = (np.arange(360) + 0.5) * 2 * np.pi / 360
        th, ph = np.meshgrid(theta, phi, indexing="ij")
        data = np.exp(-((th - np.pi) / 0.55)**2) * (1 + 0.7 * np.cos(3 * ph - 2 * th))
    else:
        data, theta, phi = load_map(args.data_file, args.kind, args.map_coordinates)
    paths = export_map(data, theta, phi, args.output_dir, major_radius=args.major_radius,
                       wall_radius=args.wall_radius, cmap=args.cmap, scale=args.scale,
                       vmin=args.vmin, vmax=args.vmax)
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
