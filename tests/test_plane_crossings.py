"""Streaming physical statistics, backend parity, legacy archives and replot CLI."""
import importlib.util
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import illiad.boris as bm
import illiad.mesh.torch_mesh as tm
import illiad.utilities.coordtrans as ct
from illiad.particle import Ion
from illiad.plane_crossings import PlaneCrossingHistogram
from illiad.plane_crossing_plots import histogram_values, load_histogram, plot_histogram
from misc_scripts.plot_plane_crossings import main


class Statistics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for module in (bm, tm, ct):
            patcher = patch.object(module, 'device', torch.device('cpu'))
            patcher.start()
            self.addCleanup(patcher.stop)
        np.save(self.root/'b.npy', np.zeros((3, 5, 8, 4)))
        self.field = tm.TorchMesh(R0=.72, a=.19)
        self.field.loadCartesianField(str(self.root/'b.npy'), att_mult=1.)

    def solver(self, method='torch', chunk=4, compact=4, wall=False):
        starts = [[.8, .00015, .01], [.8, -.00025, .01]]
        velocities = [[0, -1e4, 0], [0, 3e4, 0]]
        if wall:
            starts, velocities = [[.9095, .00002, 0]], [[1e5, -1e4, 0]]
        ions = [Ion(np.array(p), 6.941, 1) for p in starts]
        for ion, v in zip(ions, velocities):
            ion.initVelocity(np.asarray(v))
        h = PlaneCrossingHistogram(self.field, 0, min_normal_speed=2e4)
        solver = bm.Boris(None, method=method)
        solver.setConditions(ions, 'test', dt=1e-8, tmax=1e-7)
        solver.nsteps = 11
        solver.parallel_solver(ions, self.field, trace_IDs=[], trace_stride=7,
                               warp_step_chunk_size=chunk, warp_compaction_interval=compact,
                               plane_histogram=h)
        return h, ions

    def archive(self, h):
        import matplotlib.pyplot as plt
        io = SimpleNamespace(data_dir=str(self.root), log=logging.getLogger('test'),
                             _outputPath=lambda base, name: str(Path(base)/name),
                             saveFig=lambda name: plt.savefig(self.root/name))
        h.save(io, dt=1e-8, n_particles=2)
        return load_histogram(self.root/'Plane_crossings.npz')

    def test_analytic_energy_speed_and_normalizations(self):
        h, ions = self.solver()
        data = self.archive(h)
        self.assertEqual(data['counts'].sum(), 2)
        energies = .5*ions[0].mass*np.array([1e8, 9e8])/1.602176634e-19
        self.assertAlmostEqual(data['energy_sum_eV'].sum(), energies.sum())
        self.assertAlmostEqual(data['inverse_normal_speed_sum_s_m'].sum(), 1/3e4)
        self.assertEqual(data['density_excluded_counts'].sum(), 1)
        self.assertAlmostEqual(float(data['observation_duration_s']), 1e-7)
        expected = dict(count=4., area=4., flux=4/1e-7, density=2/3e4/1e-7)
        for mode in expected:
            values, _, excluded = histogram_values(data, mode, particle_weight=2.)
            integral = values.sum() if mode == 'count' else (values*data['cell_area_m2']).sum()
            self.assertAlmostEqual(integral, expected[mode])
            self.assertEqual(excluded, int(mode == 'density'))
        values, _, _ = histogram_values(data, 'energy', particle_weight=123.)
        self.assertAlmostEqual(values[np.isfinite(values)][0], energies.mean())
        self.assertTrue(np.isnan(values[data['counts'] == 0]).all())
        for direction, energy in zip(('positive', 'negative'), energies):
            values, _, _ = histogram_values(data, 'energy', direction)
            self.assertAlmostEqual(values[np.isfinite(values)][0], energy)

    def test_early_termination_duration(self):
        h, _ = self.solver(wall=True)
        self.assertAlmostEqual(h.duration_s, 1e-8)
        self.assertEqual(h.numpy_counts().sum(), 1)

    def test_legacy_replot_and_missing_statistics(self):
        h, _ = self.solver()
        data = self.archive(h)
        old = {key: data[key] for key in ('counts', 'counts_increasing_phi', 'counts_decreasing_phi',
               'rho_edges_m', 'theta_edges_rad', 'cell_area_m2', 'R0_m', 'vessel_radius_m', 'phi_deg')}
        path = self.root/'legacy.npz'
        np.savez(path, **old)
        before = path.read_bytes()
        for mode in ('count', 'area', 'flux'):
            main([str(path), '--histogram', mode, '--duration', '1e-7', '--log-hist',
                  '--output', str(self.root/'replot'), '--dpi', '50'])
        self.assertEqual(path.read_bytes(), before)
        for mode in ('energy', 'density', 'flux'):
            with self.assertRaises(ValueError):
                histogram_values(old, mode)
        self.assertTrue((self.root/'replot/Plane_crossings_flux_both.png').exists())

    def test_energy_cli_and_cutoff(self):
        h, _ = self.solver()
        self.archive(h)
        path = self.root/'Plane_crossings.npz'
        main([str(path), '--histogram', 'energy', '--cmap', 'magma', '--vmin', '0',
              '--vmax', '40', '--output', str(self.root/'plots'), '--formats', 'png', 'pdf'])
        with np.load(self.root/'plots/Plane_crossings_energy_both.npz') as result:
            self.assertTrue(np.isnan(result['values']).any())
        with self.assertRaises(SystemExit):
            main([str(path), '--histogram', 'density', '--min-normal-speed', '0'])

    def test_empty_log_and_invalid_limits(self):
        import matplotlib.pyplot as plt
        h = PlaneCrossingHistogram(self.field, 0)
        h.prepare_torch('cpu', statistics=True)
        h.duration_s = 0.
        data = self.archive(h)
        for mode in ('count', 'energy'):
            fig = plot_histogram(data, mode, log=True)
            fig.canvas.draw()
            plt.close(fig)
        for options in (dict(log=True, vmin=0), dict(vmin=5, vmax=1)):
            with self.assertRaises(ValueError):
                plot_histogram(data, **options)
        with self.assertRaises(ValueError):
            histogram_values(data, 'flux')

    @unittest.skipUnless(importlib.util.find_spec('warp'), 'Warp unavailable')
    def test_warp_statistics_chunk_compaction_parity(self):
        ref, _ = self.solver()
        for chunk, compact in ((1, 0), (4, 4), (16, 256)):
            h, _ = self.solver('warp', chunk, compact)
            np.testing.assert_array_equal(h.numpy_counts(), ref.numpy_counts())
            for field in ('energy_sum', 'inverse_speed_sum', 'density_excluded'):
                np.testing.assert_allclose(getattr(h, field), getattr(ref, field).numpy(), rtol=1e-12)
            self.assertAlmostEqual(h.duration_s, ref.duration_s)
        h, _ = self.solver('warp', wall=True)
        self.assertAlmostEqual(h.duration_s, 1e-8)


if __name__ == '__main__':
    unittest.main()
