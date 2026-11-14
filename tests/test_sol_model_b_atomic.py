"""Atomic mixture, independent nonlinear balance, and no-neutral regressions."""
from dataclasses import asdict, replace
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import atomic_mass, elementary_charge as e
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from misc_scripts import atomic_rates as rates
from misc_scripts import sol_model_b as b
from test_sol_model_a import fixture


class ModelBAtomicTests(unittest.TestCase):
    def test_batch_cli_mixture_override_and_disable(self):
        data, settings = fixture()
        count = len(data['plane_id'])
        for key in data:
            if key not in ('grid_rho_m', 'grid_theta_rad'):
                data[key] = np.concatenate([data[key], data[key]])
        data['plane_id'][count:] = 1
        data['length_m'][count:] *= 2
        settings['phi_deg'] = [12, 24]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root/'summary.npz', **data)
            (root/'settings.json').write_text(json.dumps(settings))
            config = root/'input.json'
            config.write_text(json.dumps(dict(T_lcfs_eV=30, neutral_densities_m3={'He': 1e19})))
            common = [str(root), '--config', str(config)]
            with redirect_stdout(io.StringIO()):
                b.main(common+['--all-phi', '--output', str(root/'batch'),
                               '--neutral', 'he=1e16', '--neutral', 'Li=1e14'])
                b.main(common+['--phi-deg', '24', '--output', str(root/'off'), '--no-neutrals'])
            for angle in (12, 24):
                run = json.loads((root/'batch'/f'phi{angle:03d}'/'run_settings.json').read_text())
                self.assertEqual(run['config']['neutral_densities_m3'], {'He': 1e16, 'Li': 1e14})
                self.assertEqual(run['variant'], 'extended')
                self.assertGreater(run['diagnostics']['atomic_power_W'], 0)
            off = json.loads((root/'off'/'run_settings.json').read_text())
            self.assertEqual(off['config']['neutral_densities_m3'], {})
            self.assertEqual(off['variant'], 'baseline')
            self.assertEqual(off['diagnostics']['atomic_power_W'], 0)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                b.main(common+['--neutral', 'He=1', '--neutral', 'he=2'])

    def test_mixture_additivity_units_and_table_coverage(self):
        te, ne = np.array([.01, 10., 50.]), np.full(3, 1e18)
        he = b.atomic_terms(te, ne, {'He': 1e16})
        li = b.atomic_terms(te, ne, {'Li': 1e14})
        mix = b.atomic_terms(te, ne, {'He': 1e16, 'Li': 1e14})
        for name in ('ionization_source_m3_s', 'ionization_loss_W_m3',
                     'excitation_loss_W_m3', 'atomic_loss_W_m3'):
            assert_allclose(mix[name], he[name]+li[name], rtol=2e-16)
        for species, n0 in (('He', 1e16), ('Li', 1e14)):
            energy, p, a, x, k, _, _ = rates.load_voronov_table()[species, 0]
            u = energy/te[1:]
            expected_rate = a*(1+p*np.sqrt(u))*u**k*np.exp(-u)/(x+u)
            assert_allclose(mix[f'ionization_source_{species}_m3_s'][1:], ne[1:]*n0*expected_rate)
            assert_allclose(mix[f'ionization_loss_{species}_W_m3'],
                            mix[f'ionization_source_{species}_m3_s']*energy*e)
            # At a tabulated grid node use the stored log-power, independently
            # of the coefficient evaluator (and without an extra e or energy).
            data = rates.load_adas_plt(species)
            tn, nn = 10**data['log_temperature_ev'][8], 10**data['log_density_m3'][4]
            node = b.atomic_terms(np.array([tn]), np.array([nn]), {species: n0}, 'raise')
            assert_allclose(node['excitation_loss_W_m3'], nn*n0*10**data['log_power_w_m3'][0][8, 4])
        coverage = b.atomic_diagnostics(mix, te, ne, np.ones(3), {'He': 1e16, 'Li': 1e14})
        self.assertGreater(coverage['He']['plt_temperature_outside_cells'], 0)
        self.assertGreater(coverage['He']['voronov_outside_fit_cells'], 0)
        with self.assertRaisesRegex(ValueError, 'outside'):
            b.atomic_terms(te, ne, {'He': 1e16}, 'raise')
        for mixture in ({'He': -1}, {'He': np.nan}, {'He': 1, 'he': 1}, {'Ar': 1e16}):
            with self.assertRaises(ValueError):
                replace(b.PowerSOLConfig(), neutral_densities_m3=mixture).validate()

    def test_two_species_one_cell_against_independent_root(self):
        cfg = b.PowerSOLConfig(T_lcfs_eV=30, D_perp_m2_s=10, chi_perp_m2_s=10,
                source_power_wedge_W=.1, neutral_densities_m3={'He': 1e16, 'Li': 1e14},
                residual_tolerance=1e-10, state_tolerance=1e-9, max_iterations=500)
        V, area, Lc = 1e-4, .1, 100.
        mesh = dict(volume_m3=np.array([V]), edge_i=np.array([], int), edge_j=np.array([], int),
                    edge_area_over_distance_m=np.array([]), boundary_area_over_distance_m=np.array([area]))
        # Eliminate n analytically from the particle equation at each T, then
        # independently bracket the full nonlinear scalar energy equation.
        def exact(T):
            nu = 2*np.sqrt(e*(T+cfg.gamma_i*cfg.Ti_eV)/(cfg.ion_mass_u*atomic_mass))/Lc
            alpha = sum(n0*rates.ionization_rate_m3_s(s, [0], T, check_fit_range=False)[0]
                        for s, n0 in cfg.neutral_densities_m3.items())
            n = cfg.D_perp_m2_s*area*cfg.n_lcfs_m3/(cfg.D_perp_m2_s*area+V*(nu-alpha))
            loss = sum(rates.ionization_loss_w_m3(s, [0], T, n, n0, check_fit_range=False)
                       + rates.excitation_loss_w_m3(s, [0], T, n, n0, bounds='clip')
                       for s, n0 in cfg.neutral_densities_m3.items())
            boundary = e*cfg.chi_perp_m2_s*(n+cfg.n_lcfs_m3)/2*area
            return n, boundary*(T-cfg.T_lcfs_eV)+V*(cfg.gamma_e*e*n*T*nu+loss)-cfg.wedge_power_W
        T = brentq(lambda T: exact(T)[1], .1, 100, xtol=1e-11)
        fields, diag = b.solve_coupled(mesh, np.array([Lc]), np.zeros(1), cfg)
        assert_allclose(fields['Tu_eV'], T, rtol=2e-9)
        assert_allclose(fields['n_e_m3'], exact(T)[0], rtol=2e-9)
        assert_allclose(diag['lcfs_heat_influx_W']+diag['source_power_wedge_W'],
                        diag['sheath_power_W']+diag['atomic_power_W'], rtol=2e-10)
        assert_allclose(diag['lcfs_influx_s_inv']+diag['ionization_source_s_inv'],
                        diag['parallel_loss_s_inv'], rtol=2e-10)
        assert_allclose(diag['atomic_power_W'], diag['ionization_power_W']+diag['excitation_power_W'])
        self.assertGreater(diag['atomic_power_W'], 0)
        with self.assertRaisesRegex(RuntimeError, 'did not converge'):
            b.solve_coupled(mesh, np.array([Lc]), np.zeros(1), replace(cfg, max_iterations=1))

    def test_local_temperature_feedback_baseline_and_saved_outputs(self):
        data, settings = fixture()
        cfg = b.PowerSOLConfig(T_lcfs_eV=30, source_power_wedge_W=.1)
        baseline, _ = b.solve_plane(data, settings, 12, cfg)
        zeros, diag = b.solve_plane(data, settings, 12, replace(cfg, neutral_densities_m3={'He': 0, 'Li': 0}))
        for key in ('n_e_m3', 'Tu_eV', 'iteration_history'):
            assert_allclose(baseline[key], zeros[key], rtol=0, atol=0)
        self.assertEqual(diag['atomic_power_W'], 0)
        cfg = replace(cfg, neutral_densities_m3={'He': 1e16, 'Li': 1e14})
        iso, _ = b.solve_plane(data, settings, 12, cfg)
        sh, diag = b.solve_plane(data, settings, 12, replace(cfg, parallel_temperature_model='spitzer_harm'))
        assert_allclose(sh['atomic_temperature_eV'], sh['Te_eV'], rtol=0, atol=0)
        self.assertFalse(np.allclose(iso['Tu_eV'], sh['Tu_eV'], rtol=1e-5))
        expected = b.atomic_terms(sh['Te_eV'], sh['n_e_m3'], cfg.neutral_densities_m3)
        assert_allclose(sh['atomic_loss_W_m3'], expected['atomic_loss_W_m3'])
        self.assertLess(diag['max_scaled_particle_residual'], cfg.residual_tolerance)
        self.assertLess(diag['max_scaled_energy_residual'], cfg.residual_tolerance)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            b.save_solution(path, sh, dict(model='B', config=asdict(cfg), diagnostics=diag))
            saved = np.genfromtxt(path/'atomic_processes.csv', delimiter=',', names=True)
            assert_allclose(saved['ionization_source_He_m3_s'], sh['ionization_source_He_m3_s'], rtol=1e-10)
            with np.load(path/'solution.npz', allow_pickle=False) as npz:
                assert_allclose(npz['atomic_loss_W_m3'], sh['atomic_loss_W_m3'])
            self.assertGreater(json.loads((path/'run_settings.json').read_text())['diagnostics']['atomic_power_W'], 0)


if __name__ == '__main__':
    unittest.main()
