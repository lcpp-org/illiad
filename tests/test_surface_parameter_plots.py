import unittest
import numpy as np
from illiad.flux.surface_parameter_plots import band_mesh, validate_options, normalize_iota, blend_profiles


def ring(radius, center=0):
    angle = np.linspace(0, 2*np.pi, 64, endpoint=False)
    return np.stack((radius*np.cos(angle)+center, radius*np.sin(angle)), axis=-1)[None]


class SurfaceParameterTests(unittest.TestCase):
    def test_linear_parameter_on_nested_bands(self):
        tri, z = band_mesh([ring(1), ring(.5)], [0, 1])
        self.assertEqual(len(tri.triangles), 128)
        # Along a shared radial edge, linear interpolation must place 0.5 midway.
        from matplotlib.tri import LinearTriInterpolator
        interp = LinearTriInterpolator(tri, z)
        self.assertAlmostEqual(float(interp(.75, 0)), .5)
        self.assertTrue(np.ma.is_masked(interp(0, 0)))
        self.assertTrue(np.ma.is_masked(interp(1.1, 0)))

    def test_missing_invalid_and_topology_changes_leave_gaps(self):
        for contours, values in [([ring(1), None, ring(.5)], [0, .5, 1]),
                                  ([ring(1), ring(.5)], [0, np.nan]),
                                  ([ring(1), ring(.5, 2)], [0, 1]),
                                  ([ring(1), np.concatenate([ring(.1), ring(.2)])], [0, 1])]:
            self.assertIsNone(band_mesh(contours, values)[0])

    def test_island_band_outside_lcfs_is_rejected(self):
        # Nested island rings can be valid locally while protruding outside LCFS.
        islands = np.concatenate([ring(.2, .95), ring(.1, -.5)])
        smaller = np.concatenate([ring(.15, .95), ring(.05, -.5)])
        tri, _ = band_mesh([ring(1), islands, smaller], [0, .5, 1])
        self.assertLessEqual(float(np.max(np.hypot(tri.x, tri.y))), 1.)

    def test_options(self):
        self.assertEqual(validate_options({}), ('flux', 'lines', 16))
        for key, value in [('SURFACE_PARAMETER', 'bad'), ('SURFACE_PARAMETER_STYLE', 'bad'),
                           ('SURFACE_PARAMETER_LEVELS', 1), ('SURFACE_PARAMETER_LEVELS', True)]:
            with self.assertRaises(ValueError):
                validate_options({key: value})


class ProfileTests(unittest.TestCase):
    def test_endpoint_normalization_ignores_axis_trace_and_handles_both_signs(self):
        iota = np.array([[.9, .9], [.2, .6], [.3, .4], [.4, .2], [123., -123.]])
        normalized, raw, refs, ids = normalize_iota(iota, 1)
        np.testing.assert_allclose(normalized, [[0, 0], [0, 0], [.5, .5], [1, 1], [1, 1]])
        np.testing.assert_array_equal(ids, [3, 3])
        np.testing.assert_allclose(refs, [.4, .2])

    def test_clipping_missing_and_degenerate_references(self):
        iota = np.array([[.2, .2, np.nan], [.5, .2, .3], [.4, .2, .4], [0, 0, 0]])
        norm, raw, _, _ = normalize_iota(iota, 0)
        self.assertGreater(raw[1, 0], 1)
        self.assertEqual(norm[1, 0], 1)
        self.assertTrue(np.isnan(norm[:, 1:]).all())
        norm, _, refs, ids = normalize_iota(iota[:, :1], 0, axis_value=.6)
        self.assertAlmostEqual(norm[2, 0], .5)
        self.assertEqual(norm[-1, 0], 1)
        self.assertEqual(ids[0], -1)

    def test_convex_blend_and_nan_endpoint_independence(self):
        flux = np.array([0, .25, 1])
        iota = np.array([0, .75, 1])
        np.testing.assert_allclose(blend_profiles(flux, iota, .25), [0, .375, 1])
        np.testing.assert_allclose(blend_profiles(flux, np.full(3, np.nan), 0), flux)
        np.testing.assert_allclose(blend_profiles(np.full(3, np.nan), iota, 1), iota)
        for weight in [-.1, 1.1, float('nan'), True]:
            with self.assertRaises(ValueError):
                validate_options({'IOTA_WEIGHT': weight})
