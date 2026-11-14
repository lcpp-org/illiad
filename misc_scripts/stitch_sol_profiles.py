"""Plot prescribed flux-based core profiles together with saved Model A/B SOL fields.

See docs/stitch_sol_profiles.md. No transport solve or field-line tracing.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

if __package__:
    from .sol_model_a import radial_boundary
    from .plot_sol_model_a import load_solution, positive_norm
else:
    from sol_model_a import radial_boundary
    from plot_sol_model_a import load_solution, positive_norm


@dataclass(frozen=True)
class ProfileConfig:
    density_peak_m3: float = 2e18
    temperature_peak_eV: float = 100.
    density_alpha: float = 1.
    temperature_alpha: float = 1.
    temperature_field: str = 'Tu'

    def validate(self, n_edge, T_edge):
        for name, value in asdict(self).items():
            if name != 'temperature_field' and (not np.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be finite and positive')
        if self.temperature_field not in ('Tu', 'Te'):
            raise ValueError('temperature_field must be Tu or Te')
        if not 0 < n_edge <= self.density_peak_m3 or not 0 < T_edge <= self.temperature_peak_eV:
            raise ValueError('Require 0 < LCFS value <= peak for density and temperature')


def core_profile(psi, edge, peak, alpha):
    """Existing ILLIAD convention: psi=1-axis/0-edge, shape=1-(1-psi)**alpha."""
    q = np.clip(np.asarray(psi, dtype=float), 0, 1)
    return edge + (peak-edge)*(1-(1-q)**alpha)


def radial_profile(fields, quantity, theta_index, rho_m):
    """Evaluate the C0 join on one saved ray; NaN gaps are never bridged.

    quantity is 'density' or 'temperature'. This interpolates physical values
    between samples, including an explicit LCFS anchor, not a transport solve.
    """
    key, boundary_key = {'density': ('n_e_m3_grid', 'boundary_density_m3'),
                         'temperature': ('temperature_eV_grid', 'boundary_temperature_eV')}[quantity]
    rho = fields['grid_rho_m']
    rb = fields['boundary_rho_m'][theta_index]
    keep = ~np.isclose(rho, rb, rtol=0, atol=1e-12)
    r = np.r_[rho[keep], rb]
    values = np.r_[fields[key][keep, theta_index], fields[boundary_key][theta_index]]
    order = np.argsort(r)
    return np.interp(rho_m, r[order], values[order], left=np.nan, right=np.nan)


class FluxField:
    """Read native linear nField (phi,theta,rho), with periodic phi/theta seams.

    Native coordinates: phi=360/N..360 degrees, theta=2*pi/N..2*pi,
    rho=0..vessel_radius. Interpolate only the two needed phi planes.
    """
    def __init__(self, path, vessel_radius):
        self.path = Path(path)
        self.field = np.load(path, mmap_mode='r', allow_pickle=False)
        if self.field.ndim != 3 or min(self.field.shape) < 2:
            raise ValueError('Expected a linear nField array with shape (phi, theta, rho), each axis >= 2')
        if not np.isfinite(vessel_radius) or vessel_radius <= 0:
            raise ValueError('flux_vessel_radius_m must be finite and positive')
        self.radius = vessel_radius

    def plane(self, phi_deg):
        if not np.isfinite(phi_deg):
            raise ValueError('phi must be finite')
        nphi, ntheta, nr = self.field.shape
        index = (phi_deg % 360)*nphi/360-1
        left = int(np.floor(index))
        weight = index-left
        a = np.asarray(self.field[left % nphi], dtype=float)
        values = a if weight == 0 else (1-weight)*a + weight*self.field[(left+1) % nphi]
        if not np.isfinite(values).all():
            raise ValueError('Flux planes contain nonfinite values')
        # Add a theta=0 copy of the last (2*pi) row.
        theta = np.linspace(0, 2*np.pi, ntheta+1)
        rho = np.linspace(0, self.radius, nr)
        interpolator = RegularGridInterpolator((theta, rho), np.vstack([values[-1:], values]))
        def evaluate(r, t):
            r, t = np.broadcast_arrays(r, t)
            return interpolator(np.column_stack([np.mod(t.ravel(), 2*np.pi), r.ravel()])).reshape(r.shape)
        return evaluate, dict(phi_plane_indices=[left % nphi, (left+1) % nphi], phi_upper_weight=weight)


def stitch_plane(data, metadata, flux, cfg):
    """Keep SOL samples exact; prescribe every interior sample and add LCFS anchors.

    Continuous radial reconstruction passes through grid samples and the LCFS
    Dirichlet anchor. It is C0 at that anchor, not a matched-gradient solve.
    Missing exterior cells stay NaN, including the wall if it was never solved.
    """
    n_edge, T_edge = (metadata['config'][key] for key in ('n_lcfs_m3', 'T_lcfs_eV'))
    cfg.validate(n_edge, T_edge)
    rho, theta = (np.asarray(data[key]) for key in ('grid_rho_m', 'grid_theta_rad'))
    if (rho.ndim != 1 or theta.ndim != 1 or len(rho) < 2 or len(theta) < 2
            or not np.isfinite(rho).all() or not np.isfinite(theta).all()
            or np.any(np.diff(rho) <= 0) or np.any(np.diff(theta) <= 0) or rho[0] != 0):
        raise ValueError('Expected increasing finite rho/theta vectors, rho starting at zero')
    angles, radii = radial_boundary(data['boundary_xz_m'])
    boundary_rho = np.interp(theta, angles, radii)
    if np.any(boundary_rho >= rho[-1]):
        raise ValueError('LCFS must lie inside the grid vessel radius')
    R, Theta = np.meshgrid(rho, theta, indexing='ij')
    # Cells exactly on the boundary receive the exact Dirichlet value.
    on_boundary = np.isclose(R, boundary_rho[None, :], rtol=0, atol=1e-12)
    core = (R < boundary_rho[None, :]) | on_boundary
    raw = np.full(R.shape, np.nan)
    raw[core] = flux(R[core], Theta[core])
    psi = np.clip(raw, 0, 1)
    psi[on_boundary] = 0
    n = np.full(R.shape, np.nan)
    T = np.full(R.shape, np.nan)
    n[core] = core_profile(psi[core], n_edge, cfg.density_peak_m3, cfg.density_alpha)
    T[core] = core_profile(psi[core], T_edge, cfg.temperature_peak_eV, cfg.temperature_alpha)
    # A single physical point at rho=0 must not depend on its duplicated theta index.
    origin_psi = float(np.mean(psi[0]))
    psi[0] = origin_psi
    n[0] = core_profile(origin_psi, n_edge, cfg.density_peak_m3, cfg.density_alpha)
    T[0] = core_profile(origin_psi, T_edge, cfg.temperature_peak_eV, cfg.temperature_alpha)
    ir, it = data['rho_index'], data['theta_index']
    if len(set(zip(ir.tolist(), it.tolist()))) != len(ir) or np.any(core[ir, it]):
        raise ValueError('Duplicate SOL indices or SOL samples overlap the core; check boundary provenance')
    ns, ts = np.asarray(data['n_e_m3']), np.asarray(data[cfg.temperature_field+'_eV'])
    if not np.isfinite(ns).all() or not np.isfinite(ts).all() or np.any(ns < 0) or np.any(ts < 0):
        raise ValueError('SOL density/temperature must be finite and nonnegative')
    n[ir, it], T[ir, it] = ns, ts
    region = np.zeros(R.shape, dtype=np.uint8)  # 0 unavailable, 1 prescribed core, 2 solved SOL
    region[core], region[ir, it] = 1, 2
    # Boundary gradients of the explicit, piecewise-linear radial reconstruction.
    # Only use the immediately adjacent exterior grid node; never jump over a hole.
    gradients = {}
    for name, values, edge in (('density', n, n_edge), ('temperature', T, T_edge)):
        gin, gout = np.full(len(theta), np.nan), np.full(len(theta), np.nan)
        for j, rb in enumerate(boundary_rho):
            inside = np.flatnonzero(rho < rb-1e-12)
            outside = np.flatnonzero(rho > rb+1e-12)
            if len(inside):
                k = inside[-1]
                gin[j] = (edge-values[k, j])/(rb-rho[k])
            if len(outside) and region[outside[0], j] == 2:
                k = outside[0]
                gout[j] = (values[k, j]-edge)/(rho[k]-rb)
        gradients[name+'_radial_gradient_core'] = gin
        gradients[name+'_radial_gradient_sol'] = gout
    raw_boundary = flux(boundary_rho, theta)
    local_temperature_c0 = (cfg.temperature_field == 'Tu'
                            or metadata['config']['parallel_temperature_model'] == 'isothermal')
    if not local_temperature_c0:
        warnings.warn('Te with Spitzer-Harm is a visualization join to T_lcfs, not the SOL model\'s Tu boundary condition')
    diag = dict(core_points=int(core.sum()), solved_sol_points=len(ir), unavailable_points=int((region == 0).sum()),
                clipped_low_core_points=int(np.count_nonzero(raw[core] < 0)),
                clipped_high_core_points=int(np.count_nonzero(raw[core] > 1)),
                raw_flux_boundary_min=float(raw_boundary.min()), raw_flux_boundary_max=float(raw_boundary.max()),
                raw_flux_origin_spread=float(np.ptp(raw[0])), origin_psi=origin_psi,
                prescribed_density_max_m3=float(n[core].max()), prescribed_temperature_max_eV=float(T[core].max()),
                boundary_temperature_matches_solver=local_temperature_c0,
                continuity='C0 radial reconstruction with explicit LCFS anchors; gradients are not matched')
    fields = dict(grid_rho_m=rho, grid_theta_rad=theta, x_m=R*np.cos(Theta), z_m=R*np.sin(Theta),
                  phi_comp_deg=data['phi_comp_deg'], boundary_xz_m=data['boundary_xz_m'], region=region,
                  psi_raw_grid=raw, psi_grid=psi, n_e_m3_grid=n, temperature_eV_grid=T,
                  boundary_rho_m=boundary_rho, boundary_density_m3=np.full(len(theta), n_edge),
                  boundary_temperature_eV=np.full(len(theta), T_edge), raw_flux_boundary=raw_boundary,
                  sol_seed_id=data['seed_id'], sol_rho_index=ir, sol_theta_index=it, **gradients)
    return fields, diag


def save_output(output, fields, settings, marker_size=35., density_scale='log', individual=False):
    if (output/'solution.npz').exists():
        raise ValueError('Choose a separate output directory; do not overwrite SOL run settings')
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'stitched_profiles.npz', **fields)
    (output/'run_settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    names = ('x_m', 'z_m', 'region', 'psi_grid', 'n_e_m3_grid', 'temperature_eV_grid')
    ir, it = np.indices(fields['region'].shape)
    np.savetxt(output/'profiles.csv', np.column_stack([ir.ravel(), it.ravel()]+[fields[k].ravel() for k in names]),
               delimiter=',', header='rho_index,theta_index,'+','.join(names), comments='', fmt='%.12g')
    names = ('grid_theta_rad', 'boundary_rho_m', 'boundary_density_m3', 'boundary_temperature_eV',
             'density_radial_gradient_core', 'density_radial_gradient_sol',
             'temperature_radial_gradient_core', 'temperature_radial_gradient_sol')
    np.savetxt(output/'boundary_diagnostics.csv', np.column_stack([fields[k] for k in names]),
               delimiter=',', header=','.join(names), comments='', fmt='%.12g')
    panels = [('density', fields['n_e_m3_grid'], 'Electron density', r'$n_e$ [m$^{-3}$]', 'viridis'),
              ('temperature', fields['temperature_eV_grid'], 'Temperature ('+settings['profile']['temperature_field']+' in SOL)',
               'Temperature [eV]', 'inferno')]
    for selection in ([panels]+[[p] for p in panels] if individual else [panels]):
        fig, axes = plt.subplots(1, len(selection), figsize=(7*len(selection), 6), layout='constrained', squeeze=False)
        for ax, (name, values, title, label, cmap) in zip(axes.flat, selection):
            valid = np.isfinite(values)
            norm = positive_norm(values[valid], density_scale) if name == 'density' else Normalize(0, max(values[valid].max(), 1e-12))
            positive = valid & (values > 0) if name == 'density' and density_scale == 'log' else valid
            artist = ax.scatter(fields['x_m'][positive], fields['z_m'][positive], c=values[positive],
                                s=marker_size, linewidths=0, cmap=cmap, norm=norm)
            zero = valid & ~positive
            if np.any(zero):
                ax.scatter(fields['x_m'][zero], fields['z_m'][zero], s=marker_size, color='black', label='Zero', linewidths=0)
            missing = ~valid
            ax.scatter(fields['x_m'][missing], fields['z_m'][missing], s=marker_size, color='lightgray',
                       linewidths=0, label='Unavailable SOL / wall')
            boundary = fields['boundary_xz_m']
            ax.plot(*np.vstack([boundary, boundary[0]]).T, color='black', lw=.9)
            angle = np.linspace(0, 2*np.pi, 361); radius = fields['grid_rho_m'][-1]
            ax.plot(radius*np.cos(angle), radius*np.sin(angle), color='black', lw=.9)
            ax.set(title=title, xlabel=r'$R-R_0$ [m]', ylabel='Z [m]', aspect='equal')
            ax.legend(loc='upper right', fontsize=8)
            fig.colorbar(artist, ax=ax, label=label, shrink=.85)
        p = settings['profile']
        fig.suptitle(f"Core + SOL | computational φ={float(fields['phi_comp_deg']):g}°\n"
                     f"Peaks: n={p['density_peak_m3']:.3g} m⁻³, T={p['temperature_peak_eV']:g} eV | "
                     f"αn={p['density_alpha']:g}, αT={p['temperature_alpha']:g}")
        filename = 'overview' if len(selection) > 1 else selection[0][0]
        try:
            fig.savefig(output/f'stitched_{filename}.png', dpi=180)
        finally:
            plt.close(fig)


def select_solutions(path, angles=None):
    folders = [path] if (path/'solution.npz').is_file() else list(path.glob('phi*/'))
    available = []
    for folder in folders:
        if not (folder/'solution.npz').is_file():
            continue
        with np.load(folder/'solution.npz', allow_pickle=False) as saved:
            available.append((float(saved['phi_comp_deg']) % 360, folder))
    available.sort(key=lambda item: item[0])
    if not available or len({a for a, _ in available}) != len(available):
        raise ValueError('No saved solutions found, or duplicate phi angles')
    if angles is None:
        return available
    selected = []
    for angle in angles:
        found = [(a, f) for a, f in available if np.isfinite(angle) and np.isclose(a, angle % 360, rtol=0, atol=1e-8)]
        if len(found) != 1 or found[0] in selected:
            raise ValueError(f'Phi {angle:g} is missing or requested more than once')
        selected.extend(found)
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('solution_dir', type=Path, help='One solution or batch root; defaults to all contained planes')
    parser.add_argument('--config', type=Path, help='JSON input; CLI options override its values')
    parser.add_argument('--flux-file', type=Path, help='Native linear normalized nField, NOT an alpha-scaled field')
    parser.add_argument('--flux-lcfs-index', type=int, help='LCFS used to construct the flux file (provenance)')
    parser.add_argument('--flux-vessel-radius-m', type=float)
    parser.add_argument('--phi-deg', type=float, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--density-peak-m3', type=float)
    parser.add_argument('--temperature-peak-ev', dest='temperature_peak_eV', type=float)
    parser.add_argument('--density-alpha', type=float)
    parser.add_argument('--temperature-alpha', type=float)
    parser.add_argument('--temperature-field', choices=('Tu', 'Te'))
    parser.add_argument('--marker-size', type=float, default=35)
    parser.add_argument('--density-scale', choices=('log', 'linear'), default='log')
    parser.add_argument('--individual', action='store_true')
    args = parser.parse_args(argv)
    inputs = json.loads(args.config.read_text()) if args.config else {}
    allowed = set(ProfileConfig.__dataclass_fields__) | {'flux_file', 'flux_lcfs_index', 'flux_vessel_radius_m'}
    if set(inputs)-allowed:
        parser.error(f'Unknown config keys: {sorted(set(inputs)-allowed)}')
    for name in allowed:
        if getattr(args, name, None) is not None:
            inputs[name] = getattr(args, name)
    if not all(name in inputs for name in ('flux_file', 'flux_lcfs_index', 'flux_vessel_radius_m')):
        parser.error('Provide flux_file, flux_lcfs_index and flux_vessel_radius_m via config or CLI')
    index = inputs['flux_lcfs_index']
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        parser.error('flux_lcfs_index must be a nonnegative integer')
    if not np.isfinite(args.marker_size) or args.marker_size <= 0:
        parser.error('marker-size must be positive and finite')
    cfg = ProfileConfig(**{k: inputs[k] for k in ProfileConfig.__dataclass_fields__ if k in inputs})
    flux = FluxField(inputs['flux_file'], inputs['flux_vessel_radius_m'])
    selected = select_solutions(args.solution_dir, args.phi_deg)
    batch = not (args.solution_dir/'solution.npz').is_file()
    root = args.output or args.solution_dir/'stitched_profiles'
    for phi, folder in selected:
        data, metadata = load_solution(folder)
        if not np.isclose(data['grid_rho_m'][-1], flux.radius, rtol=0, atol=1e-12):
            raise ValueError('Flux and SOL vessel radii differ')
        trace_path = Path(metadata['trace_dir'])/'settings.json'
        trace = json.loads(trace_path.read_text())
        sol_lcfs = trace['surface_index']
        if inputs['flux_lcfs_index'] != sol_lcfs:
            warnings.warn(f'Flux LCFS {inputs["flux_lcfs_index"]} differs from SOL LCFS {sol_lcfs}; '
                          'preserving the supplied flux map and anchoring the join to the SOL boundary. See diagnostics.')
        evaluate, interpolation = flux.plane(phi)
        fields, diagnostics = stitch_plane(data, metadata, evaluate, cfg)
        settings = dict(profile=asdict(cfg), source_solution=str(folder.resolve()),
                        flux_file=str(flux.path.resolve()), flux_lcfs_index=inputs['flux_lcfs_index'],
                        flux_vessel_radius_m=flux.radius, flux_interpolation=interpolation,
                        sol_lcfs_index=sol_lcfs, source_model=metadata['model'], sol_config=metadata['config'],
                        diagnostics=diagnostics,
                        region_codes={'unavailable': 0, 'prescribed_core': 1, 'solved_SOL': 2})
        output = root/f'phi{phi:03g}' if batch else root
        save_output(output, fields, settings, args.marker_size, args.density_scale, args.individual)
        print(f'Saved {output}: {diagnostics["core_points"]} core, {diagnostics["solved_sol_points"]} SOL, '
              f'{diagnostics["unavailable_points"]} unavailable grid points', flush=True)


if __name__ == '__main__':
    main()
