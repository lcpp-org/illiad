"""Straight-angle reconstruction must predict positions, not just assign phase."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use('Agg')
import numpy as np

from illiad.flux.straight_angle import basis, reconstruct, StraightAnglePlots


class StraightAngleTests(unittest.TestCase):
    def test_poincare_writer_preserves_crossing_order(self):
        from illiad.poincare import Poincare
        from illiad.utilities.coordtrans import RTP_to_XYZ_many
        theta = np.array([.6, 4.2, 1.7, 3.])
        rtp = np.column_stack((np.full(4, .08), theta, np.full(4, np.pi)))
        writer = Poincare.__new__(Poincare)
        writer.field = SimpleNamespace(R0=.72)
        writer.anlys_name = 'Poincare'
        writer.IO = Mock()
        writer.plotPoincareBW = Mock()
        writer.save_output((0, np.pi), [[RTP_to_XYZ_many(rtp, .72)]])
        saved = writer.IO.saveNumpyData.call_args.args[0]
        np.testing.assert_allclose(saved[0, 0], theta, atol=1e-12)

    def test_unreliable_return_estimate_is_diagnostic_only(self):
        phase = 2*np.pi*.3123456*np.arange(900)
        xy = np.column_stack((.1*np.cos(phase), .04*np.sin(phase)))
        with patch('illiad.flux.straight_angle.return_frequency', return_value=(.1, np.arange(8))):
            _, diag = reconstruct(xy)
        self.assertIn('bypassed', diag['return_frequency_note'])
        self.assertAlmostEqual(diag['frequency_per_return'], .3123456, places=7)
        with patch('illiad.flux.straight_angle.return_frequency', side_effect=ValueError('No minima')):
            _, diag = reconstruct(xy)
        self.assertIsNone(diag['coarse_eq4_frequency'])
        self.assertLess(diag['validation_rms_m'], 1e-6)

    def test_shared_frequency_does_not_require_local_distance_minima(self):
        frequency = .3123456
        phase = 2*np.pi*frequency*np.arange(900)
        xy = np.column_stack((.1*np.cos(phase), .04*np.sin(phase)))
        with patch('illiad.flux.straight_angle.return_frequency', side_effect=AssertionError('Must reuse frequency')):
            _, diag = reconstruct(xy, frequency=frequency)
        self.assertLess(diag['validation_rms_m'], 1e-6)
        self.assertIsNone(diag['coarse_eq4_frequency'])

    def test_known_nonuniform_geometric_angle_and_periodicity(self):
        frequency = .3123456
        phase = 2*np.pi*frequency*np.arange(900)
        xy = np.column_stack((.1*np.cos(phase)+.02*np.cos(2*phase), .06*np.sin(phase)))
        coeff, diag = reconstruct(xy, modes=4, regularization=1e-10)
        self.assertAlmostEqual(diag['frequency_per_return'], frequency, places=8)
        self.assertLess(diag['validation_rms_m'], 1e-6)
        angles = np.linspace(0, 2*np.pi, 128)
        expected = np.column_stack((.1*np.cos(angles)+.02*np.cos(2*angles), .06*np.sin(angles)))
        np.testing.assert_allclose(basis(angles, 4)@coeff, expected, atol=1e-6)
        np.testing.assert_allclose(basis([0, 2*np.pi], 4)@coeff, np.tile(expected[0], (2, 1)), atol=1e-6)

    def test_concave_island_uses_parametric_curve(self):
        phase = 2*np.pi*.043215*np.arange(900)
        # A crescent with two minima in its squared distance from a tip.
        xy = np.column_stack((.08*np.cos(phase)+.04*np.cos(2*phase), .02*np.sin(phase)))
        coeff, diag = reconstruct(xy, modes=4)
        self.assertAlmostEqual(diag['frequency_per_return'], .043215, places=7)
        self.assertLess(diag['validation_rms_m'], 1e-5)

    def test_unstable_and_sparse_maps_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Too few'):
            reconstruct(np.ones((9, 2)))
        rng = np.random.default_rng(123)
        phase = 2*np.pi*.12*np.arange(800)
        xy = np.column_stack((np.cos(phase), np.sin(phase)))
        xy[560:] += rng.normal(0, .15, (240, 2))
        with self.assertRaises(ValueError):
            reconstruct(xy)

    @patch('illiad.sol.tracer.load_poincare_settings')
    def test_input_defaults_and_invalid_caps(self, settings):
        settings.return_value = dict(DOUBLE_LINE=False, IC_PHI_DEG=0)
        params = dict(ANLYS_DIR='test', ANLYS_SUBDIR='test')
        plotter = StraightAnglePlots(None, params)
        self.assertEqual(plotter.tolerance, .03)
        self.assertIsNone(plotter.max_points)
        for cap in [0, -1, True, 12.5, '512']:
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, 'SL_MAX_POINTS'):
                StraightAnglePlots(None, dict(params, SL_MAX_POINTS=cap))

    @patch('illiad.sol.tracer.load_poincare_settings')
    def test_merged_directions_are_rejected(self, settings):
        settings.return_value = dict(DOUBLE_LINE=True, IC_PHI_DEG=0)
        with self.assertRaisesRegex(ValueError, 'DOUBLE_LINE'):
            StraightAnglePlots(None, dict(ANLYS_DIR='test', ANLYS_SUBDIR='test'))

    @patch('illiad.sol.tracer.load_poincare_settings')
    def test_three_lobe_plot_has_requested_angle_count_and_shared_frequency(self, settings):
        settings.return_value = dict(DOUBLE_LINE=False, IC_PHI_DEG=0)
        q, count, frequency = 3, 600, .043215
        surfaces = []
        for radius in [.025, .015]:
            xy = np.empty((q*count, 2))
            for offset in range(q):
                center = .1*np.array([np.cos(2*np.pi*offset/q), np.sin(2*np.pi*offset/q)])
                angle = 2*np.pi*frequency*(np.arange(count)+(offset+.5)/q)
                xy[offset::q] = center+radius*np.column_stack((np.cos(angle), np.sin(angle)))
            surfaces.append(np.array([np.arctan2(xy[:, 1], xy[:, 0]), np.linalg.norm(xy, axis=1)]))
        surfaces.append(np.zeros_like(surfaces[0]))  # Unused axis trace.
        with tempfile.TemporaryDirectory() as directory:
            io = SimpleNamespace(data_dir=directory, log=Mock(), saveFig=Mock(),
                                 _outputPath=lambda base, name: str(Path(base)/name))
            calc = SimpleNamespace(lcfs_index=0, simIO=io)
            def check_vessel(*args, **kwargs):
                import matplotlib.pyplot as plt
                ax = plt.gcf().axes[0]
                self.assertAlmostEqual(ax.patches[0].radius, .19)
                self.assertGreater(ax.get_xlim()[1], .19)
            io.saveFig.side_effect = check_vessel
            plotter = StraightAnglePlots(calc, dict(ANLYS_DIR='test', ANLYS_SUBDIR='run',
                                                    SL_NTHETA=12, SL_MODES=4,
                                                    SL_MAX_POINTS=901, SL_MAX_REL_ERROR=.04))
            plotter.plot_plane(180., np.array(surfaces), np.array([0., 0.]), [3, 3, 1])
            accepted = [r for r in plotter.report['planes'][0]['surfaces'] if r['status']=='accepted']
            self.assertEqual(len(accepted), 6)
            self.assertEqual(sorted(r["sample_count"] for r in accepted),
                             [300, 300, 300, 300, 301, 301])
            self.assertEqual(plotter.report["options"]["SL_MAX_POINTS"], 901)
            self.assertEqual(plotter.tolerance, .04)
            self.assertTrue(all(abs(r['iota_sl']-frequency/q) < 1e-8 for r in accepted))
            data = np.load(Path(directory)/'run/straight_angle/Coordinates_180.npz')
            self.assertEqual(data['coordinate_points'].shape, (6, 12, 2))
            self.assertEqual(io.saveFig.call_count, 2)
            origins = plotter.phase_origins.copy()
            for surface, radius in enumerate([.025, .015]):
                xy = np.empty((q*count, 2))
                for offset in range(q):
                    center = .1*np.array([np.cos(2*np.pi*offset/q), np.sin(2*np.pi*offset/q)])
                    angle = 2*np.pi*frequency*(np.arange(count)+(offset+.75)/q)
                    xy[offset::q] = center+radius*np.column_stack((np.cos(angle), np.sin(angle)))
                surfaces[surface] = np.array([np.arctan2(xy[:, 1], xy[:, 0]), np.linalg.norm(xy, axis=1)])
            plotter.plot_plane(270., np.array(surfaces), np.array([0., 0.]), [3, 3, 1])
            later = np.load(Path(directory)/'run/straight_angle/Coordinates_270.npz')
            self.assertEqual(plotter.phase_origins, origins)
            np.testing.assert_allclose(later['coordinate_points'], data['coordinate_points'], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
