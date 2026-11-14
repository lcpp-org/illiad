"""Numerical/trace-contract checks for the standalone Model A prototype."""
from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from misc_scripts.sol_model_a import (
    SOLConfig, build_geometry, exponential_Tu, local_Te, parallel_geometry,
    main, select_planes, solve_density, solve_plane, wedge_volume,
)


def fixture(nrho=21, ntheta=24, shaped=False):
    rho = np.linspace(0, 0.19, nrho)
    theta = np.arange(ntheta)*2*np.pi/ntheta
    angle = np.linspace(0, 2*np.pi, 720, endpoint=False)
    radius = 0.06 + (0.015*np.cos(3*angle) if shaped else np.zeros_like(angle))
    boundary = np.column_stack([radius*np.cos(angle), radius*np.sin(angle)])
    local_radius = 0.06 + (0.015*np.cos(3*theta) if shaped else np.zeros_like(theta))
    keep = (rho[:, None] > local_radius) & (rho[:, None] < 0.19)
    ir, it = np.nonzero(keep)
    count = len(ir)
    data = dict(grid_rho_m=rho, grid_theta_rad=theta, grid_keep_mask=keep[None],
                boundaries_xz_m=boundary[None], plane_id=np.zeros(count, dtype=int),
                seed_rho_index=ir, seed_theta_index=it,
                length_m=np.tile([12., 8.], (count, 1)), hit_wall=np.ones((count, 2), dtype=bool),
                lcfs_distance_min_m=np.full(count, 0.01),
                lcfs_min_signed_s_m=np.full(count, 2.), lcfs_min_direction=np.ones(count, dtype=int))
    settings = dict(phi_deg=[12], surface_index=21, distance_surface_index=21,
                    offset_mm=0, major_radius_m=0.72, vessel_radius_m=0.19)
    return data, settings


class ModelATests(unittest.TestCase):
    def test_plane_selection(self):
        settings = {'phi_deg': [0, 1, 2, 3, 4, 5]}
        self.assertEqual(select_planes(settings, all_phi=True), settings['phi_deg'])
        self.assertEqual(select_planes(settings, [5, 360, 2]), [5, 0, 2])
        self.assertEqual(select_planes({'phi_deg': [12]}), [12])
        for requested in (None, [6], [0, 360], [np.nan]):
            with self.assertRaises(ValueError):
                select_planes(settings, requested)

    def test_batch_outputs_match_single_plane(self):
        data, settings = fixture()
        count = len(data['plane_id'])
        for key in data:
            if key not in ('grid_rho_m', 'grid_theta_rad'):
                data[key] = np.concatenate([data[key], data[key]], axis=0)
        data['plane_id'][count:] = 1
        data['length_m'][count:] *= 2
        settings['phi_deg'] = [12, 24]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez_compressed(root/'summary.npz', **data)
            (root/'settings.json').write_text(json.dumps(settings))
            output = root/'batch'
            with redirect_stdout(io.StringIO()):
                main([str(root), '--all-phi', '--output', str(output)])
                main([str(root), '--phi-deg', '24', '--output', str(root/'single')])
            for plane, angle in enumerate((12, 24)):
                folder = output/f'phi{angle:03d}'
                with np.load(folder/'solution.npz') as result:
                    self.assertEqual(result['phi_comp_deg'], angle)
                    self.assertEqual(result['plane_id'], plane)
                    assert_allclose(result['seed_id'], np.arange(count)+plane*count)
                    self.assertEqual(len(result['n_e_m3']), count)
                metadata = json.loads((folder/'run_settings.json').read_text())
                self.assertEqual(metadata['phi_deg'], angle)
                self.assertTrue((folder/'profiles.csv').is_file())
            with np.load(output/'phi024'/'solution.npz') as batch, np.load(root/'single'/'solution.npz') as single:
                assert_allclose(batch['n_e_m3'], single['n_e_m3'], rtol=0, atol=0)
            self.assertFalse((root/'single'/'phi024').exists())

    def test_profiles_and_both_parallel_branches(self):
        cfg = SOLConfig()
        assert_allclose(exponential_Tu(np.array([0, cfg.lambda_T_u_m]), 10, cfg), [10, 10/np.e])
        # Upstream is 4 m from the minus wall; grid lies on either branch.
        xi, minus, plus = parallel_geometry([[8, 2], [4, 6], [6, 4], [10, 0], [0, 10]],
                                            [2, -2, 0, 4, -6], [1, -1, 1, 1, -1])
        assert_allclose(xi, [.5, 1/3, 0, 1, 1])
        assert_allclose(minus, 4)
        assert_allclose(plus, 6)
        Tu = np.full(5, 10.)
        Te = local_Te(Tu, xi, cfg)
        assert_allclose(Te, Tu*(1-xi)**(2/7))
        assert_allclose(local_Te(Tu, xi, replace(cfg, target_T_fraction=1)), Tu)
        self.assertTrue(np.all((Te >= 0) & (Te <= Tu)))
        assert_allclose(local_Te(Tu, np.ones(5), replace(cfg, target_T_fraction=.2)), 2)
        for delta, direction in (([2], [0]), ([2], [-1]), ([20], [1])):
            with self.assertRaises(ValueError):
                parallel_geometry([[8, 2]], delta, direction)

    def test_switch_custom_profile_and_source_indices(self):
        data, settings = fixture()
        calls = []
        def custom(distance, T0, cfg):
            calls.append(distance.copy())
            return np.full_like(distance, 7.)
        sh, sh_diag = solve_plane(data, settings, 12, profile=custom)
        # Isothermal must not even read the signed upstream coordinates.
        del data['lcfs_min_signed_s_m'], data['lcfs_min_direction']
        iso, _ = solve_plane(data, settings, 12, replace(SOLConfig(), parallel_temperature_model='isothermal'),
                             profile=custom)
        self.assertEqual(len(calls), 2)
        assert_allclose(iso['Te_eV'], iso['Tu_eV'])
        assert_allclose(iso['n_e_m3'], sh['n_e_m3'], rtol=0, atol=0)
        assert_allclose(sh['Tu_eV'], 7.)
        self.assertLess(abs(sh_diag['relative_particle_balance']), 1e-12)
        self.assertLessEqual(sh['n_e_m3'].max(), SOLConfig().n_lcfs_m3)
        assert_allclose(sh['n_e_m3_grid'][sh['rho_index'], sh['theta_index']], sh['n_e_m3'])
        # Archive row order must not determine the grid mapping.
        n = len(data['plane_id'])
        for key in ('plane_id', 'seed_rho_index', 'seed_theta_index', 'length_m', 'hit_wall', 'lcfs_distance_min_m'):
            data[key] = data[key][::-1]
        shuffled, _ = solve_plane(data, settings, 12, replace(SOLConfig(), parallel_temperature_model='isothermal'),
                                  profile=custom)
        assert_allclose(shuffled['n_e_m3_grid'], iso['n_e_m3_grid'])
        assert_allclose(shuffled['seed_id'], np.arange(n)[::-1])

    def test_incomplete_trace_policy(self):
        data, settings = fixture()
        data['hit_wall'][7, 0] = False
        with self.assertRaisesRegex(ValueError, 'incomplete/invalid'):
            solve_plane(data, settings, 12)
        with warnings.catch_warnings(record=True) as caught:
            result, diag = solve_plane(data, settings, 12, incomplete='no_flux')
        self.assertEqual(len(caught), 1)
        assert_allclose(result['rejected_seed_id'], [7])
        self.assertEqual(diag['rejected_cells'], 1)
        self.assertTrue(np.isnan(result['n_e_m3_grid'][data['seed_rho_index'][7], data['seed_theta_index'][7]]))
        settings['offset_mm'] = 5
        with self.assertRaisesRegex(ValueError, 'zero offset'):
            solve_plane(data, settings, 12)

    def test_wedge_formula_and_periodic_clipped_geometry(self):
        # Independent tensor quadrature of R*rho d(rho)d(theta)d(phi).
        x, w = np.polynomial.legendre.leggauss(32)
        lo, hi, a, b, R0, dphi = .06, .15, -.4, .5, .72, .2
        r = (lo+hi)/2+(hi-lo)/2*x
        t = (a+b)/2+(b-a)/2*x
        integral = dphi*(hi-lo)*(b-a)/4*np.sum(w[:, None]*w*(R0+r[:, None]*np.cos(t))*r[:, None])
        assert_allclose(wedge_volume(lo, hi, a, b, R0, dphi), integral, rtol=1e-13)
        for shaped in (False, True):
            data, settings = fixture(shaped=shaped)
            args = (data['grid_rho_m'], data['grid_theta_rad'], data['grid_keep_mask'][0],
                    data['boundaries_xz_m'][0], .72, .19)
            mesh = build_geometry(*args, dphi=.2, quadrature=4)
            refined = build_geometry(*args, dphi=.2, quadrature=12)
            assert_allclose(mesh['volume_m3'], refined['volume_m3'], rtol=1e-10, atol=1e-18)
            assert_allclose(mesh['boundary_area_over_distance_m'], refined['boundary_area_over_distance_m'], rtol=1e-8)
            self.assertTrue(np.all(mesh['edge_area_over_distance_m'] > 0))
            # All circle/threefold LCFS cos(theta) contributions integrate to zero.
            expected = .2*np.pi*.72*(.19**2-.06**2-(.015**2/2 if shaped else 0))
            assert_allclose(np.sum(mesh['volume_m3']), expected, rtol=2e-6)
            ti = mesh['theta_index']
            seam = (ti[mesh['edge_i']] == 23) & (ti[mesh['edge_j']] == 0)
            self.assertTrue(np.any(seam))
            pair = np.column_stack([mesh['edge_i'], mesh['edge_j']])
            self.assertEqual(len(np.unique(np.sort(pair, axis=1), axis=0)), len(pair))
        # Density is independent of the arbitrary wedge thickness.
        small, _ = solve_plane(data, settings, 12, replace(SOLConfig(), wedge_angle_rad=.1))
        large, _ = solve_plane(data, settings, 12, replace(SOLConfig(), wedge_angle_rad=1))
        assert_allclose(small['n_e_m3'], large['n_e_m3'], rtol=1e-12)

    def test_long_leg_keeps_one_hit_and_preserves_trace_data(self):
        data, settings = fixture()
        baseline, _ = solve_plane(data, settings, 12)
        complete, diag = solve_plane(data, settings, 12, incomplete='long_leg')
        assert_allclose(complete['n_e_m3'], baseline['n_e_m3'], rtol=0, atol=0)
        self.assertEqual(diag['approximated_cells'], 0)
        data['hit_wall'][7] = [False, True]
        data['hit_wall'][8] = [True, False]
        data['hit_wall'][9] = [False, True]
        data['hit_wall'][10] = [False, False]
        data['hit_wall'][11] = [True, False]
        data['length_m'][7] = [100., 8.]
        data['length_m'][8] = [12., 150.]
        data['length_m'][9] = [100., 8.]
        data['length_m'][11, 1] = np.nan
        data['lcfs_min_signed_s_m'][7] = -2.
        data['lcfs_min_direction'][7] = -1
        original_lengths = data['length_m'].copy()
        original_hits = data['hit_wall'].copy()
        with warnings.catch_warnings(record=True) as caught:
            result, diag = solve_plane(data, settings, 12, incomplete='long_leg')
        self.assertEqual(len(caught), 2)
        assert_allclose(result['rejected_seed_id'], [10, 11])
        assert_allclose(result['seed_id'][result['connection_approximated']], [7, 8, 9])
        self.assertEqual(diag['approximated_cells'], 3)
        ids = result['seed_id']
        assert_allclose(result['traced_length_m'], original_lengths[ids])
        assert_allclose(result['hit_wall'], original_hits[ids])
        assert_allclose(data['length_m'], original_lengths)
        assert_allclose(data['hit_wall'], original_hits)
        # Default 100 m never shortens the already traced 150 m leg.
        assert_allclose(result['effective_length_m'][7:10], [[100, 8], [12, 150], [100, 8]])
        assert_allclose(result['xi'][7:10], [2/102, 2/152, 2/10])
        assert_allclose(result['Lc_m'], result['effective_length_m'].sum(axis=1))
        self.assertLess(abs(diag['relative_particle_balance']), 1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            longer, _ = solve_plane(data, settings, 12, replace(SOLConfig(), missing_leg_length_m=1000),
                                     incomplete='long_leg')
            # Isothermal needs no upstream coordinates, including on approximate cells.
            del data['lcfs_min_signed_s_m'], data['lcfs_min_direction']
            iso, _ = solve_plane(data, settings, 12, replace(SOLConfig(), parallel_temperature_model='isothermal'),
                                 incomplete='long_leg')
        assert_allclose(iso['n_e_m3'], result['n_e_m3'], rtol=0, atol=0)
        assert_allclose(iso['Te_eV'], iso['Tu_eV'])
        assert_allclose(longer['effective_length_m'][7:10], [[1000, 8], [12, 1000], [1000, 8]])
        assert_allclose(longer['xi'][7:10], [2/1002, 2/1002, 2/10])
        # Only losses weaken; density cannot decrease on this fixed connected domain.
        self.assertTrue(np.all(longer['n_e_m3'] >= result['n_e_m3']*(1-1e-12)))

    def test_long_leg_rejects_invalid_configuration_and_upstream_geometry(self):
        for length in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                replace(SOLConfig(), missing_leg_length_m=length).validate()
        data, settings = fixture()
        data['hit_wall'][7, 0] = False
        # Within the surrogate leg but outside the actual saved trace: invalid.
        data['lcfs_min_signed_s_m'][7] = 50.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with self.assertRaisesRegex(ValueError, 'outside the wall-to-wall line'):
                solve_plane(data, settings, 12, incomplete='long_leg')
            data['hit_wall'][:] = False
            with self.assertRaisesRegex(ValueError, 'No valid wall-connected cells'):
                solve_plane(data, settings, 12, incomplete='long_leg')

    def test_finite_slab_cosh_convergence_and_source_balance(self):
        D, Lc, cs, length = 1.3, 20., 15., 2.
        sink = 2*cs/Lc
        decay = np.sqrt(D/sink)
        errors = []
        for count in (20, 40, 80):
            dx = length/count
            x = (np.arange(count)+.5)*dx
            bc = np.zeros(count)
            bc[0] = 2*D/dx
            n, _, diag = solve_density(np.full(count, dx), np.arange(count-1), np.arange(1, count),
                                       np.full(count-1, D/dx), bc, np.full(count, sink), 1., 0.)
            exact = np.cosh((length-x)/decay)/np.cosh(length/decay)
            errors.append(np.max(np.abs(n-exact)))
            self.assertLess(abs(diag['relative_particle_balance']), 1e-11)
        self.assertLess(errors[1], errors[0]/3.8)
        self.assertLess(errors[2], errors[1]/3.8)
        data, settings = fixture()
        _, diag = solve_plane(data, settings, 12, replace(SOLConfig(), source_m3_s=1e20))
        self.assertGreater(diag['volume_source_s_inv'], 0)
        self.assertLess(abs(diag['relative_particle_balance']), 1e-12)


if __name__ == '__main__':
    unittest.main()
