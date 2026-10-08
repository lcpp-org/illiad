"""Export a saved Poincare surface as an untextured, closed binary STL.

From the repository root, for example::

    python misc_scripts/export_flux_surface.py \
        output/IOTA4_1000sp_atol1e-9/data/Poincare --surface-index 40 \
        --output output/lcfs_iota4.stl

Select millimeters when importing into Inventor. STL imports as a mesh, not
an editable STEP/B-rep solid. Defaults: 60 toroidal planes, 180 poloidal
vertices, 21,600 triangles, and the 18-degree CAD rotation used by
inventor_output.py. Use --cad-rotation-deg 0 for computational Cartesian axes.

Input: Poincare_DDD.npy arrays shaped (surfaces, 2, samples), storing theta
in radians and minor radius in meters. --surface-index is zero-based; choose
the LCFS index for your run or another nested surface. Planes must cover a
complete 360 degrees; no field tracing or field-file loading is performed.

The vectorized contour fit follows generateSeedShells at dr=0: bounding-box
center, angular sorting, pseudo-periodic cubic smoothing spline (s=8e-6).
It omits wall clipping, plots, normals, Torch, and simulation output files.
Each contour must be a single loop, star-shaped about its bounding-box center.
Island chains, stochastic layers, and self-intersecting contours are unsupported;
topological closure alone does not establish physical surface validity.
"""

import argparse
from pathlib import Path
import struct

import numpy as np
from scipy.interpolate import splev, splrep


def fit_contour(path, surface_index, ntheta, smoothing):
    """Return ordered (radial displacement, z) points in meters."""
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim != 3 or data.shape[1] != 2:
        raise ValueError(f"{path}: expected (surfaces, 2, samples), got {data.shape}")
    if not 0 <= surface_index < data.shape[0]:
        raise ValueError(f"{path}: surface index must be in [0, {data.shape[0] - 1}]")
    pairs = np.asarray(data[surface_index]).T
    pairs = np.unique(pairs[np.isfinite(pairs).all(axis=1)], axis=0)
    if len(pairs) < 8 or np.any(pairs[:, 1] < 0):
        raise ValueError(f"{path}: need at least 8 finite samples with nonnegative radii")
    theta, radius = pairs.T
    x, z = radius * np.cos(theta), radius * np.sin(theta)
    xc, zc = (x.max() + x.min()) / 2, (z.max() + z.min()) / 2
    angle = np.arctan2(z - zc, x - xc)
    angle = np.where(angle <= 0, angle + 2 * np.pi, angle)
    order = np.argsort(angle)
    angle, radius = angle[order], np.hypot(x - xc, z - zc)[order]
    if np.any(np.diff(angle) <= 0):
        raise ValueError(f"{path}: repeated contour angles; surface is not single-valued")
    gaps = np.diff(np.r_[angle, angle[0] + 2 * np.pi])
    if gaps.max() > np.pi / 2:
        raise ValueError(f"{path}: contour has an angular gap exceeding 90 degrees")
    half = len(angle) // 2
    extended_angle = np.r_[angle[half:-1] - 2 * np.pi, angle,
                           angle[1:half] + 2 * np.pi]
    extended_radius = np.r_[radius[half:-1], radius, radius[1:half]]
    spline, _, status, message = splrep(
        extended_angle, extended_radius, s=smoothing, k=3, full_output=True)
    if status > 0:
        raise ValueError(f"{path}: spline fit failed: {message}")
    theta = np.arange(ntheta) * (2 * np.pi / ntheta)
    radius = np.asarray(splev(theta, spline))
    if not np.isfinite(radius).all() or np.any(radius <= 0):
        raise ValueError(f"{path}: spline produced invalid radii")
    return np.column_stack((xc + radius * np.cos(theta), zc + radius * np.sin(theta)))


def build_mesh(poincare_dir, surface_index, nphi=60, ntheta=180,
               major_radius=0.72, smoothing=8e-6, cad_rotation_deg=18.0):
    """Fit uniform planes and connect both periodic axes without duplicate seams."""
    if nphi < 3 or 360 % nphi or ntheta < 3:
        raise ValueError("nphi must divide 360 and be >= 3; ntheta must be >= 3")
    if not np.isfinite(major_radius) or major_radius <= 0:
        raise ValueError("major-radius must be finite and positive")
    if not np.isfinite(smoothing) or smoothing < 0 or not np.isfinite(cad_rotation_deg):
        raise ValueError("smoothing must be finite and nonnegative; rotation must be finite")
    phi_degrees = np.arange(nphi) * (360 // nphi)
    paths = [Path(poincare_dir) / f"Poincare_{phi:03d}.npy" for phi in phi_degrees]
    # Saved runs commonly label their periodic endpoint 360 instead of 000.
    if not paths[0].is_file():
        paths[0] = Path(poincare_dir) / "Poincare_360.npy"
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} required planes; first: {missing[0]}")
    contours = np.stack([fit_contour(path, surface_index, ntheta, smoothing) for path in paths])
    cylindrical_radius = major_radius + contours[..., 0]
    if np.any(cylindrical_radius <= 0):
        raise ValueError("Surface reaches the cylindrical axis; expected a toroidal surface")
    # RTP_to_XYZ uses y = -R sin(phi). Row-vector CAD transform rotates -18 deg.
    phi = np.deg2rad(phi_degrees + cad_rotation_deg)[:, None]
    vertices = np.stack((cylindrical_radius * np.cos(phi),
                         -cylindrical_radius * np.sin(phi), contours[..., 1]), axis=-1)
    vertices = vertices.reshape(-1, 3) * 1000.0
    grid = np.arange(nphi * ntheta).reshape(nphi, ntheta)
    a = grid.ravel()
    b = np.roll(grid, -1, axis=0).ravel()
    c = np.roll(np.roll(grid, -1, axis=0), -1, axis=1).ravel()
    d = np.roll(grid, -1, axis=1).ravel()
    faces = np.stack((np.column_stack((a, c, b)), np.column_stack((a, d, c))), axis=1)
    return vertices, faces.reshape(-1, 3)


def write_stl(path, vertices, faces):
    """Write outward-oriented facets, validating at STL's float32 precision."""
    triangles = vertices.astype(np.float32)[faces].astype(np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    volume = np.einsum("ij,ij->i", triangles[:, 0],
                       np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
    if not np.isfinite(triangles).all() or np.any(lengths == 0) or not np.isfinite(volume) or volume == 0:
        raise ValueError("Mesh contains invalid/degenerate facets or has zero signed volume")
    if volume < 0:
        triangles = triangles[:, [0, 2, 1]]
        normals = -normals
    normals /= lengths[:, None]
    records = np.zeros(len(faces), dtype=[("normal", "<f4", (3,)),
                                        ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    records["normal"], records["vertices"] = normals, triangles
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"ILLIAD flux surface; coordinates in millimeters".ljust(80, b"\0"))
        stream.write(struct.pack("<I", len(faces)))
        records.tofile(stream)
    return abs(volume)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("poincare_dir", type=Path, help="Directory containing Poincare_DDD.npy")
    parser.add_argument("--surface-index", type=int, required=True, help="Zero-based surface index (LCFS or inner surface)")
    parser.add_argument("--output", type=Path, required=True, help="Destination .stl; existing file is replaced")
    parser.add_argument("--nphi", type=int, default=60, help="Toroidal plane count; divisor of 360 (default: 60)")
    parser.add_argument("--ntheta", type=int, default=180, help="Vertices per contour (default: 180)")
    parser.add_argument("--major-radius", type=float, default=0.72, help="Major radius in meters (default: 0.72)")
    parser.add_argument("--smoothing", type=float, default=8e-6, help="Spline smoothing factor (default: 8e-6)")
    parser.add_argument("--cad-rotation-deg", type=float, default=18.0, help="Clockwise Cartesian rotation (default: 18)")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".stl":
        parser.error("--output must end in .stl")
    try:
        vertices, faces = build_mesh(args.poincare_dir, args.surface_index, args.nphi,
                                    args.ntheta, args.major_radius, args.smoothing, args.cad_rotation_deg)
        volume = write_stl(args.output, vertices, faces)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Wrote {args.output}: {len(vertices):,} vertices, {len(faces):,} triangles")
    print(f"Bounds (mm): {vertices.min(axis=0)} to {vertices.max(axis=0)}")
    print(f"Enclosed mesh volume: {volume / 1e9:.8g} m^3. Import units: millimeters.")


if __name__ == "__main__":
    main()
