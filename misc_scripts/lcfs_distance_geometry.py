"""Closed LCFS triangle meshes and unsigned Cartesian point-to-surface distances."""

from pathlib import Path
import warnings

import numpy as np
from scipy.spatial import cKDTree

from illiad.sol.tracer import load_lcfs_boundary
from illiad.utilities.coordtrans import RTP_to_XYZ_many


def mesh_from_rings(rings):
    """Join matching poloidal samples, closing both toroidal and poloidal seams."""
    rings = np.asarray(rings, dtype=float)
    if rings.ndim != 3 or rings.shape[2] != 3 or min(rings.shape[:2]) < 3:
        raise ValueError("Need at least three rings and three points per ring.")
    ids = np.arange(rings.shape[0] * rings.shape[1]).reshape(rings.shape[:2])
    right = np.roll(ids, -1, axis=1)
    next_ring = np.roll(ids, -1, axis=0)
    diagonal = np.roll(next_ring, -1, axis=1)
    faces = np.concatenate((np.stack((ids, next_ring, diagonal), axis=-1).reshape(-1, 3),
                            np.stack((ids, diagonal, right), axis=-1).reshape(-1, 3)))
    return rings.reshape(-1, 3), faces


def resample_ring(contour, count):
    """Match rings by CCW perimeter fraction from the left mid-height crossing.

    Preserve the loader's fitted curve, including local angular backtracking.
    Each slice must represent a single closed surface, as in the seed loader.
    """
    if not np.isfinite(contour).all():
        raise ValueError("LCFS contour contains non-finite coordinates.")
    following = np.roll(contour, -1, axis=0)
    area = np.sum(contour[:, 0] * following[:, 1] - following[:, 0] * contour[:, 1])
    if area == 0:
        raise ValueError("LCFS contour has zero enclosed area.")
    if area < 0:
        contour = contour[::-1]
        following = np.roll(contour, -1, axis=0)
    mid_z = (contour[:, 1].min() + contour[:, 1].max()) / 2
    candidates = np.flatnonzero((contour[:, 1] > mid_z) != (following[:, 1] > mid_z))
    fraction = (mid_z - contour[candidates, 1]) / (following[candidates, 1] - contour[candidates, 1])
    crossings = contour[candidates] + fraction[:, None] * (following[candidates] - contour[candidates])
    chosen = np.argmin(crossings[:, 0])
    edge = candidates[chosen]
    seam = crossings[chosen]
    closed = np.vstack((seam, contour[edge + 1:], contour[:edge + 1], seam))
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    keep = np.r_[True, np.diff(lengths) > 0]
    lengths, closed = lengths[keep], closed[keep]
    targets = np.linspace(0, lengths[-1], count, endpoint=False)
    return np.column_stack([np.interp(targets, lengths, closed[:, j]) for j in range(2)])


def build_lcfs_mesh(run_dir, index, poloidal_points, smoothing, major_radius):
    """Fit every saved plane and match contours by anchored perimeter fraction."""
    if poloidal_points < 16:
        raise ValueError("--distance-poloidal-points must be at least 16.")
    planes = {}
    for path in sorted((Path(run_dir) / "data/Poincare").glob("Poincare_*.npy")):
        try:
            angle = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        planes.setdefault(angle % 360, angle)
    angles = np.array(sorted(planes))
    if len(angles) < 3:
        raise ValueError("3D LCFS reconstruction requires at least three saved toroidal planes.")
    gaps = np.diff(np.r_[angles, angles[0] + 360])
    if gaps.max() >= 180:
        raise ValueError("Saved LCFS planes do not cover the torus (gap >= 180 degrees).")
    if gaps.max() > 10:
        warnings.warn(f"LCFS mesh has a {gaps.max():g} degree toroidal gap; "
                      "check distance convergence with denser saved planes.", stacklevel=2)
    rings = []
    for phi in angles:
        contour, _ = load_lcfs_boundary(
            str(Path(run_dir).resolve()), planes[phi], index,
            spline_smoothing=smoothing, boundary_points=max(2048, poloidal_points * 4))
        curve = resample_ring(contour, poloidal_points)
        rtp = np.column_stack((np.linalg.norm(curve, axis=1), np.arctan2(curve[:, 1], curve[:, 0]),
                               np.full(poloidal_points, np.deg2rad(phi))))
        rings.append(RTP_to_XYZ_many(rtp, major_radius))
    vertices, faces = mesh_from_rings(rings)
    return vertices, faces, angles


def triangle_distance_squared(points, triangles):
    """Broadcast point/triangle pairs; include face interiors, edges and vertices."""
    a, b, c = np.moveaxis(triangles, -2, 0)
    ab, ac, ap = b - a, c - a, points - a
    normal = np.cross(ab, ac)
    normal2 = np.sum(normal * normal, axis=-1)
    safe = np.where(normal2 > 0, normal2, 1.0)
    v = np.sum(np.cross(ap, ac) * normal, axis=-1) / safe
    w = np.sum(np.cross(ab, ap) * normal, axis=-1) / safe
    inside = (normal2 > 0) & (v >= 0) & (w >= 0) & (v + w <= 1)
    best = np.where(inside, np.sum(ap * normal, axis=-1)**2 / safe, np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        edge2 = np.sum(edge * edge, axis=-1)
        t = np.clip(np.sum((points - start) * edge, axis=-1) /
                    np.where(edge2 > 0, edge2, 1.0), 0, 1)
        delta = points - start - t[..., None] * edge
        best = np.minimum(best, np.sum(delta * delta, axis=-1))
    return best


class SurfaceDistance:
    """Exact nearest-triangle queries using an axis-aligned bounding-box tree.

    Exactness is relative to the supplied mesh, not the unknown smooth LCFS.
    Query batches limit memory independently of the number of saved path points.
    """

    def __init__(self, vertices, faces):
        vertices = np.asarray(vertices, dtype=float)
        self.triangles = vertices[np.asarray(faces, dtype=int)]
        if not len(self.triangles) or not np.isfinite(self.triangles).all():
            raise ValueError("Distance surface must contain finite triangles.")
        self.vertex_tree = cKDTree(vertices)
        lower = self.triangles.min(axis=1)
        upper = self.triangles.max(axis=1)
        centers = (lower + upper) / 2

        def build(ids):
            lo, hi = lower[ids].min(axis=0), upper[ids].max(axis=0)
            if len(ids) <= 16:
                return lo, hi, ids, None
            axis = np.argmax(np.ptp(centers[ids], axis=0))
            mid = len(ids) // 2
            ordered = ids[np.argpartition(centers[ids, axis], mid)]
            return lo, hi, build(ordered[:mid]), build(ordered[mid:])

        self.root = build(np.arange(len(self.triangles)))

    def __call__(self, points, batch_size=256):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Query points must be finite with shape (N, 3).")
        distances = np.empty(len(points))
        for start in range(0, len(points), batch_size):
            batch = points[start:start + batch_size]
            best = self.vertex_tree.query(batch)[0]**2

            def visit(node, ids):
                lo, hi, left, right = node
                delta = np.maximum(np.maximum(lo - batch[ids], batch[ids] - hi), 0)
                ids = ids[np.sum(delta * delta, axis=1) <= best[ids] + 1e-24]
                if not len(ids):
                    return
                if right is None:
                    values = triangle_distance_squared(batch[ids, None, :], self.triangles[None, left])
                    best[ids] = np.minimum(best[ids], values.min(axis=1))
                else:
                    visit(left, ids)
                    visit(right, ids)

            visit(self.root, np.arange(len(batch)))
            distances[start:start + len(batch)] = np.sqrt(best)
        return distances
