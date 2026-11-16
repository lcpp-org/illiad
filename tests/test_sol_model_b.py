"""Independent balance, root, grid-convergence and saved-trace tests for Model B."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import atomic_mass, elementary_charge as EV_J
from scipy.integrate import solve_bvp
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from misc_scripts import sol_model_a as a
from misc_scripts import sol_model_b as b
from test_sol_model_a import fixture


def slab_mesh(count, length=.1):
    dx = length/count
    boundary = np.zeros(count)
    boundary[0] = 2/dx
    return dict(volume_m3=np.full(count, dx), edge_i=np.arange(count-1), edge_j=np.arange(1, count),
                edge_area_over_distance_m=np.full(count-1, 1/dx), boundary_area_over_distance_m=boundary)


class ModelBTests(unittest.TestCase):
    def test_all_valid_keeps_neither_hit_and_conserves(self):
        data, settings = fixture()
        data['hit_wall'][7] = False
        data['hit_wall'][8, 0] = False
        cfg = b.PowerSOLConfig(parallel_temperature_model='spitzer_harm', source_power_wedge_W=.1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result, diag = b.solve_plane(data, settings, 12, cfg, incomplete='all_valid')
        self.assertEqual(len(result['seed_id']), len(data['plane_id']))
        self.assertEqual(len(result['rejected_seed_id']), 0)
        mask = result['no_parallel_loss']
        assert_allclose(result['seed_id'][mask], [7])
        assert_allclose(result['parallel_loss_rate_s_inv'][mask], 0)
        assert_allclose(result['sheath_loss_W_m3'][mask], 0)
        assert_allclose(result['Te_eV'][mask], result['Tu_eV'][mask], rtol=0, atol=0)
        self.assertEqual(diag['no_parallel_loss_cells'], 1)
        self.assertLess(abs(diag['relative_particle_balance']), cfg.residual_tolerance)
        self.assertLess(abs(diag['relative_energy_balance']), cfg.residual_tolerance)
        data['length_m'][7, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'valid trace records'):
            b.solve_plane(data, settings, 12, cfg, incomplete='all_valid')

    def test_source_normalization_custom_kernel_and_total_power(self):
        volume, distance = np.array([.1, .3, .2]), np.array([0., .02, .1])
        cfg = b.PowerSOLConfig(source_power_wedge_W=17)
        source = b.normalized_source(distance, volume, cfg)
        assert_allclose(volume @ source, 17, rtol=2e-16)
        assert_allclose(source[1]/source[0], np.exp(-1))
        called = []
        def custom(d, config):
            called.append(d.copy())
            return np.array([1, 0, 2])*1e-300
        source = b.normalized_source(distance, volume, cfg, custom)
        assert_allclose(volume @ source, 17, rtol=2e-16)
        self.assertEqual(len(called), 1)
        self.assertEqual(source[1], 0)
        total = replace(cfg, source_power_wedge_W=None, source_power_total_W=360, wedge_angle_rad=np.pi/180)
        total.validate()
        assert_allclose(volume @ b.normalized_source(distance, volume, total), 1)
        for weights in ([0, 0, 0], [1, -1, 2], [1, np.nan, 2]):
            with self.assertRaises(ValueError):
                b.normalized_source(distance, volume, cfg, lambda d, c: weights)
        with self.assertRaises(ValueError):
            replace(cfg, source_power_total_W=100).validate()
        with self.assertRaises(ValueError):
            replace(cfg, source_power_wedge_W=-1).validate()

    def test_one_cell_against_independent_root_and_negative_lcfs_heat(self):
        cfg = b.PowerSOLConfig(source_power_wedge_W=1, state_tolerance=1e-9, residual_tolerance=1e-10)
        volume, face, Lc = 1e-4, .01, 1e6
        mesh = dict(volume_m3=np.array([volume]), edge_i=np.array([], dtype=int), edge_j=np.array([], dtype=int),
                    edge_area_over_distance_m=np.array([]), boundary_area_over_distance_m=np.array([face]))
        def analytical(T):
            nu = 2*np.sqrt(EV_J*(T+cfg.gamma_i*cfg.Ti_eV)/(cfg.ion_mass_u*atomic_mass))/Lc
            n = cfg.D_perp_m2_s*face*cfg.n_lcfs_m3/(cfg.D_perp_m2_s*face+volume*nu)
            bT = EV_J*cfg.chi_perp_m2_s*(n+cfg.n_lcfs_m3)/2*face
            residual = bT*(T-cfg.T_lcfs_eV)+volume*cfg.gamma_e*EV_J*n*T*nu-cfg.wedge_power_W
            return n, residual
        Troot = brentq(lambda T: analytical(T)[1], 1e-8, 1e6, xtol=1e-10)
        fields, diag = b.solve_coupled(mesh, np.array([Lc]), np.zeros(1), cfg)
        assert_allclose(fields['Tu_eV'], Troot, rtol=1e-9)
        assert_allclose(fields['n_e_m3'], analytical(Troot)[0], rtol=1e-9)
        self.assertLess(diag['lcfs_heat_influx_W'], 0)
        # Here most of the 1 W returns to the LCFS. Scale the absolute budget
        # error by the incoming/outgoing powers, not only their small difference.
        error = abs(diag['lcfs_heat_influx_W']+1-diag['sheath_power_W'])
        self.assertLess(error, cfg.residual_tolerance*(abs(diag['lcfs_heat_influx_W'])+1))
        self.assertLess(abs(diag['relative_particle_balance']), cfg.residual_tolerance)
        self.assertLess(abs(diag['relative_energy_balance']), cfg.residual_tolerance)
        with self.assertRaisesRegex(RuntimeError, 'did not converge'):
            b.solve_coupled(mesh, np.array([Lc]), np.zeros(1), replace(cfg, max_iterations=1))
        mesh['boundary_area_over_distance_m'][:] = 0
        with self.assertRaisesRegex(ValueError, 'Nonpositive/nonfinite density'):
            b.solve_coupled(mesh, np.array([Lc]), np.zeros(1), cfg)

    def test_coupled_slab_grid_convergence_against_ode(self):
        cfg = b.PowerSOLConfig(chi_perp_m2_s=5, source_power_wedge_W=0,
                               state_tolerance=1e-9, residual_tolerance=1e-10)
        length, Lc = .1, 100.
        # Independent continuum ODE: U=n/n_lcfs, F=particle_flux/n_lcfs,
        # T in eV, Q=heat_flux/(e*n_lcfs). No FV matrix is used by this reference.
        def ode(x, state):
            U, F, T, Q = state
            nu = 2*np.sqrt(EV_J*(T+cfg.gamma_i*cfg.Ti_eV)/(cfg.ion_mass_u*atomic_mass))/Lc
            return np.vstack([-F/cfg.D_perp_m2_s, -nu*U, -Q/(U*cfg.chi_perp_m2_s), -cfg.gamma_e*U*T*nu])
        def bc(left, right):
            return np.array([left[0]-1, left[2]-cfg.T_lcfs_eV, right[1], right[3]])
        x = np.linspace(0, length, 100)
        initial = np.vstack([np.ones_like(x), np.zeros_like(x), np.full_like(x, cfg.T_lcfs_eV), np.zeros_like(x)])
        reference = solve_bvp(ode, bc, x, initial, tol=1e-9, max_nodes=10000)
        self.assertTrue(reference.success, reference.message)
        errors = []
        for count in (20, 40, 80):
            centers = (np.arange(count)+.5)*length/count
            fields, diag = b.solve_coupled(slab_mesh(count, length), np.full(count, Lc), centers, cfg)
            exact = reference.sol(centers)
            errors.append([np.max(np.abs(fields['n_e_m3']/cfg.n_lcfs_m3-exact[0])),
                           np.max(np.abs(fields['Tu_eV']-exact[2]))/cfg.T_lcfs_eV])
            self.assertLess(abs(diag['relative_energy_balance']), cfg.residual_tolerance)
        errors = np.asarray(errors)
        self.assertTrue(np.all(errors[1:] < errors[:-1]/3.5), errors)

    def test_switch_conservation_initial_guess_and_heating_response(self):
        data, settings = fixture()
        cfg = b.PowerSOLConfig(source_power_wedge_W=.1)
        iso, diag = b.solve_plane(data, settings, 12, cfg)
        sh, _ = b.solve_plane(data, settings, 12, replace(cfg, parallel_temperature_model='spitzer_harm'))
        for key in ('n_e_m3', 'Tu_eV', 'source_W_m3', 'sheath_loss_W_m3'):
            assert_allclose(iso[key], sh[key], rtol=0, atol=0)
        assert_allclose(iso['Te_eV'], iso['Tu_eV'], rtol=0, atol=0)
        self.assertTrue(np.all(sh['Te_eV'] < sh['Tu_eV']))
        assert_allclose(diag['lcfs_heat_influx_W']+diag['source_power_wedge_W'], diag['sheath_power_W'], rtol=1e-8)
        different_guess, _ = b.solve_plane(data, settings, 12, replace(cfg, lambda_T_u_m=.1))
        assert_allclose(different_guess['Tu_eV'], iso['Tu_eV'], rtol=2e-7)
        heated, _ = b.solve_plane(data, settings, 12, replace(cfg, source_power_wedge_W=.2))
        self.assertGreater(float(np.mean(heated['Tu_eV'])), float(np.mean(iso['Tu_eV'])))
        # A uniformly scaled wedge and power must leave both solved fields invariant.
        scaled, _ = b.solve_plane(data, settings, 12, replace(cfg, wedge_angle_rad=.2, source_power_wedge_W=.02))
        assert_allclose(scaled['Tu_eV'], iso['Tu_eV'], rtol=2e-7)
        assert_allclose(scaled['n_e_m3'], iso['n_e_m3'], rtol=2e-7)

    def test_trace_policy_and_explicit_disconnected_exclusion(self):
        data, settings = fixture()
        cfg = b.PowerSOLConfig()
        data['hit_wall'][7, 0] = False
        data['hit_wall'][8] = False
        with self.assertRaisesRegex(ValueError, 'incomplete/invalid'):
            b.solve_plane(data, settings, 12, cfg)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            long, diag = b.solve_plane(data, settings, 12, cfg, incomplete='long_leg')
            excluded, _ = b.solve_plane(data, settings, 12, cfg, incomplete='no_flux')
        assert_allclose(long['seed_id'][long['connection_approximated']], [7])
        assert_allclose(long['rejected_seed_id'], [8])
        assert_allclose(excluded['rejected_seed_id'], [7, 8])
        self.assertEqual(diag['approximated_cells'], 1)
        data, settings = fixture()
        result, mesh, _ = a.prepare_plane(data, settings, 12, cfg)
        # Artificially disconnect cell zero to check exclusion rather than flooring.
        edge = (mesh['edge_i'] != 0) & (mesh['edge_j'] != 0)
        for key in ('edge_i', 'edge_j', 'edge_area_over_distance_m'):
            mesh[key] = mesh[key][edge]
        mesh['boundary_area_over_distance_m'][0] = 0
        original_volume = mesh['volume_m3'].sum()
        with warnings.catch_warnings(record=True) as caught:
            result, mesh, removed = b.exclude_unfed_components(result, mesh, cfg)
        self.assertEqual(len(caught), 1)
        assert_allclose(result['disconnected_seed_id'], [0])
        assert_allclose(result['rejected_seed_id'], [0])
        self.assertNotIn(0, result['seed_id'])
        assert_allclose(mesh['volume_m3'].sum()+removed, original_volume)


if __name__ == '__main__':
    unittest.main()
