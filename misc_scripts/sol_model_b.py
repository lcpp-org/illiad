"""Model B: coupled particle/upstream electron power balance on saved trace planes.

Optional homogeneous neutral mixtures add ionization and excitation (B extended).
See docs/sol_model_b.md. Uses Model A's geometry and trace/index conventions.
"""

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.constants import atomic_mass, elementary_charge as EV_J
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

if __package__:
    from . import sol_model_a as model_a
    from . import atomic_rates
else:
    import sol_model_a as model_a
    import atomic_rates


@dataclass(frozen=True)
class PowerSOLConfig(model_a.SOLConfig):
    # lambda_T_u_m inherited from A controls ONLY the initial guess.
    parallel_temperature_model: str = "isothermal"
    chi_perp_m2_s: float = 1.0
    gamma_e: float = 5.5
    source_lambda_m: float = 0.02
    source_power_wedge_W: float | None = 0.0
    source_power_total_W: float | None = None
    relaxation: float = 0.5
    max_iterations: int = 200
    state_tolerance: float = 1e-7
    residual_tolerance: float = 1e-8
    neutral_densities_m3: dict[str, float] = field(default_factory=dict)
    plt_bounds: str = "clip"

    def validate(self):
        model_a.SOLConfig(**{key: getattr(self, key) for key in model_a.SOLConfig.__dataclass_fields__}).validate()
        for name in ("chi_perp_m2_s", "gamma_e", "source_lambda_m", "relaxation",
                     "state_tolerance", "residual_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.relaxation > 1:
            raise ValueError("relaxation must lie in (0, 1]")
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if (self.source_power_wedge_W is None) == (self.source_power_total_W is None):
            raise ValueError("Set exactly one of source_power_wedge_W and source_power_total_W; set the other to null")
        value = self.source_power_wedge_W if self.source_power_total_W is None else self.source_power_total_W
        if not np.isfinite(value) or value < 0:
            raise ValueError("Source power must be finite and nonnegative")
        if not isinstance(self.neutral_densities_m3, dict):
            raise ValueError("neutral_densities_m3 must be a species-to-density dictionary")
        atomic_rates.validate_neutral_densities(self.neutral_densities_m3)
        if self.plt_bounds not in ('clip', 'raise'):
            raise ValueError("plt_bounds must be 'clip' or 'raise'")

    @property
    def wedge_power_W(self):
        if self.source_power_total_W is not None:
            return self.source_power_total_W * self.wedge_angle_rad / (2*np.pi)
        return self.source_power_wedge_W


def source_weight_exp(d_min_m, cfg):
    """Replaceable effective nonlocal heating kernel; not a heat flux."""
    if not np.isfinite(cfg.source_lambda_m) or cfg.source_lambda_m <= 0:
        raise ValueError("source_lambda_m must be finite and positive")
    return np.exp(-np.maximum(d_min_m, 0.0)/cfg.source_lambda_m)


def normalized_source(d_min_m, cell_volume_m3, cfg, weight_func=source_weight_exp):
    volume = np.asarray(cell_volume_m3, dtype=float)
    weights = np.broadcast_to(np.asarray(weight_func(d_min_m, cfg), dtype=float), volume.shape).copy()
    if (not np.isfinite(volume).all() or np.any(volume <= 0)
            or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError("Source normalization needs positive finite volumes and nonnegative finite weights")
    maximum = np.max(weights)
    if maximum <= 0:
        raise ValueError("Source normalization failed: all kernel weights are zero")
    weights /= maximum  # A common factor cancels; avoid underflow in volume*weight.
    denominator = np.sum(volume*weights)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Source normalization failed")
    source = cfg.wedge_power_W * weights/denominator
    if not np.isfinite(source).all():
        raise ValueError("Source normalization overflow")
    return source


def diffusion_matrix(size, edge_i, edge_j, conductance):
    i, j, g = edge_i, edge_j, conductance
    return coo_matrix((np.r_[g, g, -g, -g],
                       (np.r_[i, j, i, j], np.r_[i, j, j, i])), shape=(size, size)).tocsr()


def positive_state(values, name):
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"Nonpositive/nonfinite {name} (min={np.min(values):.6g}, max={np.max(values):.6g}); "
                         "no numerical floor is applied")


def balance(mesh, values, conductance, boundary, boundary_value, sink, source):
    """Original nonlinear cell residual and integrated budget from shared fluxes."""
    i, j = mesh['edge_i'], mesh['edge_j']
    flux = conductance*(values[i]-values[j])
    boundary_flux = boundary*(values-boundary_value)  # Positive out of the SOL.
    residual = (np.bincount(i, weights=flux, minlength=len(values))
                - np.bincount(j, weights=flux, minlength=len(values)) + boundary_flux + sink - source)
    scale = (np.bincount(i, weights=np.abs(flux), minlength=len(values))
             + np.bincount(j, weights=np.abs(flux), minlength=len(values))
             + np.abs(boundary_flux) + np.abs(sink) + np.abs(source))
    relative_residual = float(np.max(np.abs(residual)/np.maximum(scale, np.finfo(float).tiny)))
    influx, loss, injected = -float(np.sum(boundary_flux)), float(np.sum(sink)), float(np.sum(source))
    relative_balance = (influx+injected-loss)/max(abs(influx)+injected, loss, np.finfo(float).tiny)
    return residual, relative_residual, influx, loss, injected, relative_balance


def atomic_terms(temperature_eV, density_m3, neutrals, bounds='clip'):
    """Neutral -> singly ionized particles and electron energy loss, no recycling.

    Homogeneous cold targets, single Maxwellian electrons. PLT already includes
    radiated energy: multiply by densities only, never by another energy or e.
    The same ionization event rate supplies particles and threshold energy loss.
    """
    te, ne = np.broadcast_arrays(temperature_eV, density_m3)
    fields = {name: np.zeros_like(te, dtype=float) for name in
              ('ionization_source_m3_s', 'ionization_loss_W_m3', 'excitation_loss_W_m3')}
    for species, neutral_density in neutrals.items():
        events, ionization, excitation = (np.zeros_like(te, dtype=float) for _ in range(3))
        if neutral_density > 0:
            rate = atomic_rates.ionization_rate_m3_s(species, [0], te, check_fit_range=False)[0]
            events = ne*neutral_density*rate
            ionization = events*atomic_rates.load_voronov_table()[species, 0][0]*EV_J
            excitation = ne*neutral_density*atomic_rates.excitation_coefficient_w_m3(
                species, [0], te, ne, bounds=bounds)[0]
        for prefix, unit, quantity in (('ionization_source', 'm3_s', events),
                                       ('ionization_loss', 'W_m3', ionization),
                                       ('excitation_loss', 'W_m3', excitation)):
            fields[f'{prefix}_{unit}'] += quantity
            fields[f'{prefix}_{species}_{unit}'] = quantity
    fields['atomic_loss_W_m3'] = fields['ionization_loss_W_m3']+fields['excitation_loss_W_m3']
    return fields


def atomic_diagnostics(fields, temperature, density, volume, neutrals):
    """Final-state budgets and explicit table-coverage diagnostics per species."""
    diagnostics = {}
    for species, neutral_density in neutrals.items():
        enabled = neutral_density > 0
        coverage = dict(plt_temperature_outside_cells=0, plt_density_outside_cells=0,
                        voronov_outside_fit_cells=0)
        if enabled:
            data = atomic_rates.load_adas_plt(species)
            for name, values, grid in (('temperature', temperature, data['log_temperature_ev']),
                                       ('density', density, data['log_density_m3'])):
                outside = ((temperature > 0) & (density > 0)
                           & ((values < 10**grid[0]) | (values > 10**grid[-1])))
                coverage[f'plt_{name}_outside_cells'] = int(np.count_nonzero(outside))
            low, high = atomic_rates.load_voronov_table()[species, 0][-2:]
            coverage['voronov_outside_fit_cells'] = int(np.count_nonzero(
                (temperature > 0) & ((temperature < low) | (temperature > high))))
        diagnostics[species] = dict(neutral_density_m3=neutral_density, **coverage,
            ionization_source_s_inv=float(volume @ fields[f'ionization_source_{species}_m3_s']),
            ionization_power_W=float(volume @ fields[f'ionization_loss_{species}_W_m3']),
            excitation_power_W=float(volume @ fields[f'excitation_loss_{species}_W_m3']))
    return diagnostics


def solve_coupled(mesh, Lc_m, d_min_m, cfg, weight_func=source_weight_exp, progress=None, *, xi=None):
    """Alternating sparse solves; also relax density when neutrals are enabled.

    Chi is defined by q = -n*e*chi*grad(Tu); no additional 3/2 factor is used.
    Shared and LCFS heat faces use arithmetic averages of their two densities.
    """
    cfg.validate()
    neutrals = atomic_rates.validate_neutral_densities(cfg.neutral_densities_m3)
    atoms_enabled = any(value > 0 for value in neutrals.values())
    volume = np.asarray(mesh['volume_m3'])
    Lc = np.asarray(Lc_m)
    if (Lc.shape != volume.shape or not np.isfinite(Lc).all() or np.any(Lc <= 0)
            or not np.isfinite(volume).all() or np.any(volume <= 0)):
        raise ValueError("Require positive finite cell volumes and aligned wall-to-wall lengths")
    if xi is None:
        if atoms_enabled and cfg.parallel_temperature_model == 'spitzer_harm':
            raise ValueError("Supply xi for local Spitzer-Harm atomic temperatures")
        xi = np.zeros_like(volume)
    xi = np.asarray(xi)
    if cfg.parallel_temperature_model == 'isothermal':
        xi = np.zeros_like(volume)  # Isothermal trace preparation deliberately stores NaN xi.
    if xi.shape != volume.shape or not np.isfinite(xi).all() or np.any((xi < 0) | (xi > 1)):
        raise ValueError("xi must be aligned with cells and lie in [0, 1]")
    i, j = mesh['edge_i'], mesh['edge_j']
    area = mesh['edge_area_over_distance_m']
    boundary_area = mesh['boundary_area_over_distance_m']
    if (not np.isfinite(area).all() or np.any(area <= 0)
            or not np.isfinite(boundary_area).all() or np.any(boundary_area < 0)):
        raise ValueError("Invalid face geometry")
    source = normalized_source(d_min_m, volume, cfg, weight_func)
    Gn, bn = cfg.D_perp_m2_s*area, cfg.D_perp_m2_s*boundary_area
    Kn = diffusion_matrix(len(volume), i, j, Gn) + diags(bn)
    rhs_n = bn*cfg.n_lcfs_m3 + volume*cfg.source_m3_s
    Tu = model_a.exponential_Tu(np.asarray(d_min_m), cfg.T_lcfs_eV, cfg)
    positive_state(Tu, "initial temperature")
    density = np.full(len(volume), cfg.n_lcfs_m3)
    history = []
    for iteration in range(1, cfg.max_iterations+1):
        previous_n, previous_T = density, Tu
        cs_old = np.sqrt(EV_J*(previous_T+cfg.gamma_i*cfg.Ti_eV)/(cfg.ion_mass_u*atomic_mass))
        nu_old = 2*cs_old/Lc
        old_atoms = atomic_terms(model_a.local_Te(previous_T, xi, cfg), previous_n, neutrals, cfg.plt_bounds)
        density = spsolve(Kn+diags(volume*nu_old), rhs_n+volume*old_atoms['ionization_source_m3_s'])
        positive_state(density, "density trial (check components disconnected from the LCFS)")
        if atoms_enabled:
            density = (1-cfg.relaxation)*previous_n + cfg.relaxation*density
        positive_state(density, "density (check components disconnected from the LCFS)")
        GT = EV_J*cfg.chi_perp_m2_s*(density[i]+density[j])/2*area
        bT = EV_J*cfg.chi_perp_m2_s*(density+cfg.n_lcfs_m3)/2*boundary_area
        KT = diffusion_matrix(len(volume), i, j, GT) + diags(bT)
        coefficient = cfg.gamma_e*EV_J*density*nu_old
        # Positive sink linearization Q(T_old,n_new)*Tu/Tu_old. At convergence
        # this is the original atomic loss, including PLT's density dependence.
        frozen_atoms = atomic_terms(model_a.local_Te(previous_T, xi, cfg), density, neutrals, cfg.plt_bounds)
        coefficient += frozen_atoms['atomic_loss_W_m3']/previous_T
        trial = spsolve(KT+diags(volume*coefficient), bT*cfg.T_lcfs_eV+volume*source)
        positive_state(trial, "temperature trial")
        Tu = (1-cfg.relaxation)*previous_T + cfg.relaxation*trial
        positive_state(Tu, "temperature")
        cs = np.sqrt(EV_J*(Tu+cfg.gamma_i*cfg.Ti_eV)/(cfg.ion_mass_u*atomic_mass))
        nu = 2*cs/Lc
        sheath = cfg.gamma_e*EV_J*density*Tu*nu
        Te = model_a.local_Te(Tu, xi, cfg)
        atoms = atomic_terms(Te, density, neutrals, cfg.plt_bounds)
        particles = balance(mesh, density, Gn, bn, cfg.n_lcfs_m3,
                            volume*nu*density, volume*(cfg.source_m3_s+atoms['ionization_source_m3_s']))
        energy = balance(mesh, Tu, GT, bT, cfg.T_lcfs_eV,
                         volume*(sheath+atoms['atomic_loss_W_m3']), volume*source)
        change_n = float(np.max(np.abs(density-previous_n)/np.maximum(density, previous_n)))
        change_T = float(np.max(np.abs(Tu-previous_T)/np.maximum(Tu, previous_T)))
        history.append([iteration, change_n, change_T, particles[1], energy[1], particles[5], energy[5]])
        converged = (max(change_n, change_T) <= cfg.state_tolerance
                     and max(particles[1], energy[1], abs(particles[5]), abs(energy[5])) <= cfg.residual_tolerance)
        if progress is not None and (iteration == 1 or iteration % 10 == 0 or converged):
            progress(f"Iteration {iteration}: updates n={change_n:.3g}, Tu={change_T:.3g}; "
                     f"nonlinear residuals n={particles[1]:.3g}, energy={energy[1]:.3g}")
        if converged:
            break
    else:
        raise RuntimeError(f"Model B did not converge in {cfg.max_iterations} iterations; "
                           f"last [iteration, dn, dT, Rn, RE, particle balance, energy balance] = {history[-1]}")
    result = dict(n_e_m3=density, Tu_eV=Tu, sound_speed_m_s=cs, source_W_m3=source,
                  sheath_loss_W_m3=sheath, particle_residual_s_inv=particles[0],
                  energy_residual_W=energy[0], iteration_history=np.asarray(history))
    result.update(atoms, atomic_temperature_eV=Te)
    diagnostics = dict(converged=True, iterations=iteration, density_relative_update=change_n,
                       temperature_relative_update=change_T,
                       max_scaled_particle_residual=particles[1], max_scaled_energy_residual=energy[1],
                       relative_particle_balance=particles[5], relative_energy_balance=energy[5],
                       lcfs_influx_s_inv=particles[2], parallel_loss_s_inv=particles[3],
                       volume_source_s_inv=particles[4], lcfs_heat_influx_W=energy[2],
                       sheath_power_W=float(volume @ sheath), source_power_wedge_W=energy[4],
                       requested_source_power_wedge_W=cfg.wedge_power_W,
                       source_normalization_error_W=energy[4]-cfg.wedge_power_W,
                       atomic_power_W=float(volume @ atoms['atomic_loss_W_m3']),
                       ionization_source_s_inv=float(volume @ atoms['ionization_source_m3_s']),
                       prescribed_particle_source_s_inv=float(volume.sum()*cfg.source_m3_s),
                       ionization_power_W=float(volume @ atoms['ionization_loss_W_m3']),
                       excitation_power_W=float(volume @ atoms['excitation_loss_W_m3']),
                       atomic_species=atomic_diagnostics(atoms, Te, density, volume, neutrals),
                       min_density_m3=float(density.min()),
                       min_Tu_eV=float(Tu.min()), max_Tu_eV=float(Tu.max()))
    return result, diagnostics


def exclude_unfed_components(result, mesh, cfg):
    """Exclude components with no prescribed particle supply (n=0 is a solution).

    Remove these components explicitly before normalizing the heating budget.
    This is domain exclusion, not a positive floor or a zero-density solution.
    Electron-impact ionization needs existing electrons, not an independent
    supply; isolated self-sustained discharge branches are outside this model.
    """
    size = len(result['seed_id'])
    graph = diffusion_matrix(size, mesh['edge_i'], mesh['edge_j'], np.ones(len(mesh['edge_i'])))
    _, labels = connected_components(graph, directed=False)
    anchored = np.unique(labels[mesh['boundary_area_over_distance_m'] > 0])
    keep = np.isin(labels, anchored) if cfg.source_m3_s == 0 else np.ones(size, dtype=bool)
    rejected = result['seed_id'][~keep].copy()
    removed_volume = float(mesh['volume_m3'][~keep].sum())
    if not np.all(keep):
        if not np.any(keep):
            raise ValueError("No retained component has a particle supply from the LCFS")
        warnings.warn(f"Excluding {len(rejected)} wall-connected cells disconnected from the LCFS "
                      "with no particle source; heating is normalized on the remaining domain", stacklevel=2)
        for name in ('seed_id', 'rho_index', 'theta_index'):
            result['rejected_'+name] = np.r_[result['rejected_'+name], result[name][~keep]]
        cell_fields = ('seed_id', 'rho_index', 'theta_index', 'xi', 'leg_minus_m', 'leg_plus_m', 'Lc_m',
                       'traced_length_m', 'effective_length_m', 'hit_wall', 'Lc_traced_m',
                       'connection_approximated', 'd_min_m', 'volume_m3')
        for name in cell_fields:
            result[name] = result[name][keep]
        remap = np.full(size, -1, dtype=int)
        remap[keep] = np.arange(np.sum(keep))
        edge_keep = keep[mesh['edge_i']] & keep[mesh['edge_j']]
        mesh = dict(volume_m3=mesh['volume_m3'][keep],
                    boundary_area_over_distance_m=mesh['boundary_area_over_distance_m'][keep],
                    edge_i=remap[mesh['edge_i'][edge_keep]], edge_j=remap[mesh['edge_j'][edge_keep]],
                    edge_area_over_distance_m=mesh['edge_area_over_distance_m'][edge_keep])
    result['disconnected_seed_id'] = rejected
    return result, mesh, removed_volume


def solve_plane(data, settings, phi_deg, cfg=PowerSOLConfig(), incomplete='error', quadrature=8,
                weight_func=source_weight_exp, progress=None):
    # long_leg is an explicit user-requested extension of the PDF's strict policy.
    # Shared preparation preserves original lengths and flags every approximation.
    result, mesh, diagnostics = model_a.prepare_plane(data, settings, phi_deg, cfg, incomplete, quadrature)
    result, mesh, removed_volume = exclude_unfed_components(result, mesh, cfg)
    fields, balances = solve_coupled(mesh, result['Lc_m'], result['d_min_m'], cfg, weight_func, progress,
                                    xi=result['xi'])
    result.update(fields)
    result['Te_eV'] = model_a.local_Te(result['Tu_eV'], result['xi'], cfg)
    model_a.add_grid_fields(result)
    diagnostics.update(balances, valid_cells=len(result['seed_id']), rejected_cells=len(result['rejected_seed_id']),
                       complete_cells=int(np.count_nonzero(~result['connection_approximated'])),
                       approximated_cells=int(np.count_nonzero(result['connection_approximated'])), zero_density_cells=0,
                       disconnected_cells=len(result['disconnected_seed_id']), disconnected_volume_m3=removed_volume,
                       excluded_volume_m3=diagnostics['excluded_volume_m3']+removed_volume,
                       retained_volume_m3=float(mesh['volume_m3'].sum()))
    return result, diagnostics


def save_solution(output, result, metadata):
    model_a.save_solution(output, result, metadata)
    columns = ('seed_id', 'source_W_m3', 'sheath_loss_W_m3', 'ionization_loss_W_m3',
               'excitation_loss_W_m3', 'atomic_loss_W_m3', 'energy_residual_W')
    np.savetxt(output/'power_balance.csv', np.column_stack([result[k] for k in columns]),
               delimiter=',', header=','.join(columns), comments='', fmt=['%d']+['%.12g']*(len(columns)-1))
    columns = ['seed_id', 'atomic_temperature_eV', 'ionization_source_m3_s', 'atomic_loss_W_m3']
    columns += [name for name in result if name.startswith(('ionization_source_', 'ionization_loss_',
                                                           'excitation_loss_')) and name not in columns]
    np.savetxt(output/'atomic_processes.csv', np.column_stack([result[k] for k in columns]),
               delimiter=',', header=','.join(columns), comments='', fmt=['%d']+['%.12g']*(len(columns)-1))
    np.savetxt(output/'iteration_history.csv', result['iteration_history'], delimiter=',', comments='',
               header='iteration,density_relative_update,temperature_relative_update,particle_residual,energy_residual,particle_balance,energy_balance',
               fmt=['%d']+['%.12g']*6)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace_dir', type=Path)
    planes = parser.add_mutually_exclusive_group()
    planes.add_argument('--phi-deg', type=float, nargs='+')
    planes.add_argument('--all-phi', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--incomplete', choices=('error', 'no_flux', 'long_leg'), default='error')
    parser.add_argument('--quadrature', type=int, default=8)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot-individual', action='store_true')
    parser.add_argument('--marker-size', type=float, default=35)
    parser.add_argument('--density-scale', choices=('linear', 'log'), default='log')
    neutral_args = parser.add_mutually_exclusive_group()
    neutral_args.add_argument('--neutral', action='append', metavar='SPECIES=DENSITY',
                             help='Repeat for a homogeneous mixture; replaces the JSON mixture')
    neutral_args.add_argument('--no-neutrals', action='store_true', help='Disable all atomic processes')
    parser.add_argument('--plt-bounds', choices=('clip', 'raise'), help='Override PLT table boundary policy')
    args = parser.parse_args(argv)
    parameters = json.loads(args.config.read_text()) if args.config else {}
    if args.neutral is not None:
        mixture = {}
        for item in args.neutral:
            try:
                species, value = item.split('=')
                species = species.strip().capitalize()
                if species in mixture:
                    raise ValueError(f'duplicate species {species}')
                mixture[species] = float(value)
            except ValueError as exc:
                parser.error(f'Invalid --neutral {item!r}: {exc}; use e.g. He=1e16')
        parameters['neutral_densities_m3'] = mixture
    elif args.no_neutrals:
        parameters['neutral_densities_m3'] = {}
    if args.plt_bounds is not None:
        parameters['plt_bounds'] = args.plt_bounds
    cfg = PowerSOLConfig(**parameters)
    cfg.validate()
    if not np.isfinite(args.marker_size) or args.marker_size <= 0:
        parser.error('--marker-size must be finite and positive')
    settings = json.loads((args.trace_dir/'settings.json').read_text())
    angles = model_a.select_planes(settings, args.phi_deg, args.all_phi)
    batch = args.all_phi or len(angles) > 1
    root = args.output or args.trace_dir/('model_b_batch' if batch else f'model_b_phi{angles[0]:g}')
    if args.plot or args.plot_individual:
        if __package__:
            from .plot_sol_model_a import save_plots
        else:
            from plot_sol_model_a import save_plots
    with np.load(args.trace_dir/'summary.npz', allow_pickle=False) as saved:
        for number, phi in enumerate(angles, 1):
            print(f'[{number}/{len(angles)}] Model B, computational phi={phi:g} degrees', flush=True)
            result, diagnostics = solve_plane(saved, settings, phi, cfg, args.incomplete, args.quadrature,
                                               progress=lambda text: print(text, flush=True))
            metadata = dict(model='B', config=asdict(cfg), trace_dir=str(args.trace_dir.resolve()),
                            variant='extended' if any(float(n) > 0 for n in cfg.neutral_densities_m3.values()) else 'baseline',
                            atomic_temperature='Reconstructed local Te; single Maxwellian, cold homogeneous neutrals',
                            recycling=False,
                            phi_deg=float(result['phi_comp_deg']), incomplete=args.incomplete,
                            quadrature=args.quadrature, source_kernel='source_weight_exp',
                            power_convention='Additional volumetric power; LCFS heat influx computed separately',
                            geometry='LCFS-clipped polar cells; first/last cells extended to boundaries',
                            diagnostics=diagnostics)
            output = root/f'phi{phi:03g}' if batch else root
            save_solution(output, result, metadata)
            print(json.dumps(diagnostics, indent=2))
            print(f'Saved {output}')
            if args.plot or args.plot_individual:
                save_plots(result, metadata, output/'plots', args.density_scale, args.marker_size, args.plot_individual)


if __name__ == '__main__':
    main()
