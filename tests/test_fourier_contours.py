"""Analytic geometry, regularization, and flux-calculator routing checks."""

import logging
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from illiad.cli.flux_calc import DEFAULT_INPUTS
from illiad.flux.calculator import FluxCalculator
from illiad.flux.fourier import FourierContour, fit_fourier_contour
from misc_scripts.compare_flux_surface_fits import trace_validation


class FourierContourTests(unittest.TestCase):
    def test_irregular_samples_recover_known_harmonics_and_derivatives(self):
        rng = np.random.default_rng(18)
        theta = np.sort(rng.uniform(0, 2*np.pi, 150))
        radius = .1 + .02*np.cos(2*theta) - .01*np.sin(3*theta)
        fit = fit_fourier_contour(theta, radius, modes=4, regularization=0)
        grid = np.linspace(-2*np.pi, 4*np.pi, 501)
        np.testing.assert_allclose(fit(grid), .1+.02*np.cos(2*grid)-.01*np.sin(3*grid), atol=1e-14)
        np.testing.assert_allclose(fit(grid, 1), -.04*np.sin(2*grid)-.03*np.cos(3*grid), atol=1e-14)
        np.testing.assert_allclose(fit(grid, 2), -.08*np.cos(2*grid)+.09*np.sin(3*grid), atol=1e-14)
        for derivative in (0, 1, 2):
            self.assertAlmostEqual(float(fit(0, derivative)), float(fit(2*np.pi, derivative)), places=14)

    def test_penalty_reduces_noise_on_unseen_angles(self):
        rng = np.random.default_rng(7)
        theta = np.linspace(0, 2*np.pi, 96, endpoint=False)
        radius = .1+.01*np.cos(2*theta)+rng.normal(0, .002, len(theta))
        noisy = fit_fourier_contour(theta, radius, modes=20, regularization=0)
        smooth = fit_fourier_contour(theta, radius, modes=20, regularization=1e-4)
        grid = theta + .021
        truth = .1+.01*np.cos(2*grid)
        self.assertLess(np.mean((smooth(grid)-truth)**2), np.mean((noisy(grid)-truth)**2)/2)
        self.assertAlmostEqual(smooth.coefficients[0], radius.mean(), places=14)

    def test_pair_filtering_periodic_duplicates_and_sample_count_invariance(self):
        theta = np.linspace(0, 2*np.pi, 64, endpoint=False)
        radius = .1+.01*np.sin(theta)
        fit = fit_fourier_contour(theta, radius, modes=4)
        repeated = fit_fourier_contour(np.r_[np.tile(theta, 3), 2*np.pi, np.nan, 1],
                                      np.r_[np.tile(radius, 3), radius[0], 1, np.nan], modes=4)
        np.testing.assert_allclose(fit.coefficients, repeated.coefficients, atol=1e-14)

    def test_sparse_and_bad_inputs_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'distinct angles'):
            fit_fourier_contour(np.arange(9), np.full(9, .1), modes=8)
        with self.assertRaisesRegex(ValueError, 'Angular gap'):
            fit_fourier_contour(np.linspace(0, 1, 40), np.full(40, .1), modes=4)
        for modes, reg in [(1.5, 0), (True, 0), (2, -1), (2, np.nan)]:
            with self.assertRaises(ValueError):
                fit_fourier_contour(np.arange(40), np.full(40, .1), modes=modes, regularization=reg)

    def test_calculator_opt_in_routing_and_failed_fit(self):
        params = dict(DEFAULT_INPUTS, PHI_GENs=[180])
        calc = FluxCalculator(SimpleNamespace(log=logging.getLogger('test')), None, params)
        theta = np.linspace(0, 2*np.pi, 80, endpoint=False)
        radius = .1+.01*np.cos(theta)
        baseline = calc.spline_data(theta, radius, calc.smooth_fctr)
        actual = calc.fit_surface_data(theta, radius)
        np.testing.assert_array_equal(baseline[0][1], actual[0][1])
        calc.surface_fit = 'fourier'
        fit, _, fail, _ = calc.fit_surface_data(theta, radius)
        self.assertFalse(fail)
        self.assertIsInstance(fit, FourierContour)
        self.assertFalse(isinstance(calc.fit_surface_data(theta, radius, num_subsets=3)[0], FourierContour))
        self.assertTrue(calc.fit_surface_data(theta[:8], radius[:8])[2])
        calc.integrate_flux = Mock(return_value=1.0)
        output = calc.fit_and_integrate_surface_subsets(
            [[np.column_stack((theta, radius))]], [np.array([[0., 0.]])],
            0, 0, 1, [False], np.array([np.pi, 0.]), 180)
        self.assertTrue(output[-1])
        self.assertIsInstance(calc.integrate_flux.call_args.args[0], FourierContour)
        np.testing.assert_allclose(output[0][0, :, 1], output[1][0, :, 1], atol=1e-14)
        output = calc.fit_and_integrate_surface_subsets(
            [[np.column_stack((theta[:8], radius[:8]))]], [np.array([[0., 0.]])],
            0, 0, 1, [False], np.array([np.pi, 0.]), 180)
        self.assertFalse(output[-1])
        self.assertTrue(np.isnan(output[0]).all())
        self.assertTrue(np.isnan(output[3]).all())

    def test_fresh_trace_events_on_analytic_torus(self):
        major_radius, radius, iota = .72, .08, .31

        def magnetic_field(xyz):
            x, y, z = xyz
            phi = -np.arctan2(y, x)
            R = np.hypot(x, y)
            theta = np.arctan2(z, R-major_radius)
            r = np.hypot(z, R-major_radius)
            dR, dZ = -r*np.sin(theta)*iota, r*np.cos(theta)*iota
            return np.array([[dR*np.cos(phi)-R*np.sin(phi),
                              -dR*np.sin(phi)-R*np.cos(phi), dZ]])

        field = SimpleNamespace(R0=major_radius, a=.19, interpField=magnetic_field)
        points, directions, records = trace_validation(
            field, [radius, .45, np.deg2rad(151)], spins=5, method='DOP853',
            rtol=1e-10, atol=1e-12, max_step=.05)
        self.assertGreaterEqual(len(points), 8)
        np.testing.assert_allclose(points[:, 0], radius, atol=1e-8)
        np.testing.assert_allclose(points[:, 2], np.deg2rad(151), atol=1e-9)
        self.assertEqual(set(directions), {-1, 1})
        self.assertFalse(any(record['wall_hit'] for record in records))

    def test_fourier_flux_bound_matches_analytic_area(self):
        theta = np.linspace(0, 2*np.pi, 80, endpoint=False)
        fit = fit_fourier_contour(theta, .1+.02*np.cos(2*theta), modes=2, regularization=0)
        calc = FluxCalculator(SimpleNamespace(log=logging.getLogger('test')), None,
                              dict(DEFAULT_INPUTS, PHI_GENs=[0]))
        # At phi=0 the repository's positive toroidal direction is -y.
        field = SimpleNamespace(interpField=lambda point, Cart: np.array([[0., -1., 0.]]))
        flux = calc.integrate_flux(fit, [0., 0.], 0., field, err_abs=1e-10, err_rel=1e-10)
        self.assertAlmostEqual(flux, np.pi*(.1**2+.02**2/2), places=11)


if __name__ == '__main__':
    unittest.main()
