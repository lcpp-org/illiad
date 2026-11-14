"""Profile convention, periodic flux interpolation and core/SOL join checks."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from misc_scripts import stitch_sol_profiles as s
from misc_scripts import sol_model_b as b
from test_sol_model_a import fixture


class StitchTests(unittest.TestCase):
    def test_profile_endpoints_and_alpha_edge_slope(self):
        q = np.linspace(0, 1, 101)
        for alpha in (.5, 1., 2.):
            y = s.core_profile(q, 10, 100, alpha)
            assert_allclose(y[[0, -1]], [10, 100])
            self.assertTrue(np.all(np.diff(y) > 0))
            h = 1e-7
            slope = (s.core_profile(h, 10, 100, alpha)-10)/h
            assert_allclose(slope, (100-10)*alpha, rtol=1e-6)
        for cfg in (s.ProfileConfig(density_alpha=0), s.ProfileConfig(temperature_peak_eV=np.inf),
                    s.ProfileConfig(density_peak_m3=1), s.ProfileConfig(temperature_field='bad')):
            with self.assertRaises(ValueError):
                cfg.validate(1e18, 10)

    def test_native_flux_axes_seams_and_phi_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'flux.npy'
            phi = np.arange(1, 5)*90.
            theta = np.arange(1, 9)*2*np.pi/8
            rho = np.linspace(0, .19, 6)
            field = np.cos(np.deg2rad(phi))[:,None,None]+np.cos(theta)[None,:,None]+rho[None,None,:]
            np.save(path, field)
            flux = s.FluxField(path, .19)
            f, diag = flux.plane(0)
            assert_allclose(f([0, .19], [0, 2*np.pi]), [2, 2.19])
            assert_allclose(flux.plane(360)[0](rho, 0), f(rho, 0))
            self.assertEqual(diag['phi_plane_indices'][0], 3)
            f45, _ = flux.plane(45)
            assert_allclose(f45(.1, np.pi), -.4)
            with self.assertRaises(ValueError):
                f(.2, 0)

    def test_join_preserves_sol_and_has_continuous_boundary_with_gaps(self):
        data, settings = fixture()
        result, _ = b.solve_plane(data, settings, 12)
        metadata = dict(model='B', config=dict(n_lcfs_m3=1e18, T_lcfs_eV=10,
                                             parallel_temperature_model='isothermal'))
        # Deliberately nonzero psi at the LCFS: the explicit anchor must still
        # enforce the boundary value, without rescaling or modifying SOL data.
        flux = lambda r, t: 1-(r/.09)**2
        fields, diag = s.stitch_plane(result, metadata, flux, s.ProfileConfig())
        ir, it = result['rho_index'], result['theta_index']
        assert_allclose(fields['n_e_m3_grid'][ir,it], result['n_e_m3'], rtol=0, atol=0)
        assert_allclose(fields['temperature_eV_grid'][ir,it], result['Tu_eV'], rtol=0, atol=0)
        self.assertEqual(diag['core_points']+diag['solved_sol_points']+diag['unavailable_points'], fields['region'].size)
        rb = fields['boundary_rho_m'][0]
        for name, edge in (('density', 1e18), ('temperature', 10)):
            y = s.radial_profile(fields, name, 0, [rb-1e-12, rb, rb+1e-12])
            assert_allclose(y, edge, rtol=2e-8)
        # A missing exterior point remains a real interpolation gap.
        first = np.flatnonzero(fields['region'][:,0] == 2)[0]
        fields['n_e_m3_grid'][first,0] = np.nan
        midpoint = (rb+fields['grid_rho_m'][first])/2
        self.assertTrue(np.isnan(s.radial_profile(fields, 'density', 0, midpoint)))
        self.assertTrue(np.isfinite(fields['n_e_m3_grid'][fields['region'] == 1]).all())
        with self.assertRaisesRegex(ValueError, 'overlap'):
            bad = dict(result, rho_index=np.zeros_like(ir))
            s.stitch_plane(bad, metadata, flux, s.ProfileConfig())
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/'solution.npz').touch()
            with self.assertRaisesRegex(ValueError, 'separate output'):
                s.save_output(out, fields, {})

    def test_cli_batch_exports_and_selection(self):
        data, trace = fixture(nrho=11, ntheta=12)
        result, _ = b.solve_plane(data, trace, 12)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traced = root/'trace'; traced.mkdir()
            (traced/'settings.json').write_text(json.dumps(trace))
            for angle in (0, 90):
                folder = root/'batch'/f'phi{angle:03d}'; folder.mkdir(parents=True)
                values = dict(result, phi_comp_deg=np.array(angle))
                np.savez(folder/'solution.npz', **values)
                (folder/'run_settings.json').write_text(json.dumps(dict(model='B', trace_dir=str(traced),
                     config=dict(n_lcfs_m3=1e18, T_lcfs_eV=10, parallel_temperature_model='isothermal'))))
            rho = np.linspace(0, .19, 11)
            np.save(root/'flux.npy', np.broadcast_to(1-rho/.06, (4,12,11)))
            cfg = dict(flux_file=str(root/'flux.npy'), flux_lcfs_index=21, flux_vessel_radius_m=.19)
            (root/'input.json').write_text(json.dumps(cfg))
            with redirect_stdout(io.StringIO()):
                s.main([str(root/'batch'), '--config', str(root/'input.json'), '--output', str(root/'out'),
                        '--temperature-peak-ev', '80', '--density-alpha', '2', '--individual'])
            for angle in (0, 90):
                out = root/'out'/f'phi{angle:03d}'
                saved = json.loads((out/'run_settings.json').read_text())
                self.assertEqual(saved['profile']['temperature_peak_eV'], 80)
                self.assertEqual(saved['profile']['density_alpha'], 2)
                self.assertEqual(len(list(out.glob('*.png'))), 3)
                with np.load(out/'stitched_profiles.npz') as values:
                    self.assertEqual(values['region'].shape, (11,12))
            self.assertEqual(s.select_solutions(root/'batch', [360])[0][0], 0)
            for angles in ([0,360], [12], [np.nan]):
                with self.assertRaises(ValueError):
                    s.select_solutions(root/'batch', angles)


if __name__ == '__main__':
    unittest.main()
