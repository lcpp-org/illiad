"""Regression checks for reconstruction from saved Poincare crossings."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from illiad.sol.tracer import load_lcfs_boundary, minimum_boundary_distance


class LCFSBoundaryTests(unittest.TestCase):
    def test_periodic_fit_preserves_every_sparse_surface_point(self):
        # An irregularly sampled ellipse leaves an isolated final point in
        # angular order. Dropping that point causes a millimetre-scale error.
        angles = np.array([-2.9, -2.1, -1.2, -0.2, 0.6, 1.1, 1.8, 2.6])
        points = np.column_stack((0.14 * np.cos(angles), 0.08 * np.sin(angles)))
        theta = np.arctan2(points[:, 1], points[:, 0])
        rho = np.linalg.norm(points, axis=1)
        saved = np.full((2, 2, len(points) + 3), np.nan)
        saved[1, :, :len(points)] = np.stack((theta, rho))
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            directory = run / "data/Poincare"
            directory.mkdir(parents=True)
            source = directory / "Poincare_029.npy"
            np.save(source, saved)
            boundary, loaded = load_lcfs_boundary(
                str(run), 29, 1, spline_smoothing=0, boundary_points=8192)
        self.assertEqual(loaded, source)
        self.assertEqual(boundary.shape, (8192, 2))
        self.assertTrue(np.isfinite(boundary).all())
        # Allow discretization of the interpolating spline into line segments.
        self.assertLess(minimum_boundary_distance(points, boundary).max(), 1e-7)


if __name__ == "__main__":
    unittest.main()
