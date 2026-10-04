"""Casual Boris benchmark using collision models and fields from an input JSON.

Run from any directory with --help. Times include particle setup and wall-result
download, but exclude field loading, field upload, and compilation warmup.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import warp as wp

import illiad.boris as boris_module
import illiad.mesh.torch_mesh as mesh_module
import illiad.utilities.coordtrans as coordtrans
import illiad.utilities.point_generators as point_generators
from illiad.cli.boris import DEFAULT_INPUTS, resolve_plasma_potential
from illiad.collisions import Collisions
from illiad.utilities import physical_constants as const
from illiad.boris_warp import (
    make_grid, make_density_grid, make_collision_config, run_wall_only,
)


def save_comparison_plots(results, directory, *, steps, dt, R0):
    """Plot final-repeat wall results outside timing, retaining data for replotting."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    phi_edges = np.linspace(0, 360, 361)
    theta_edges = np.linspace(-180, 180, 181)
    time_steps = np.arange(steps + 1)
    time_ms = time_steps * dt * 1000
    saved = {'time_ms': time_ms, 'dt': dt, 'steps': steps, 'R0': R0,
             'phi_edges_deg': phi_edges, 'theta_edges_deg': theta_edges}
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, name in zip(axes, ('torch', 'warp')):
        result = results[name]
        hit_steps = np.asarray(result['hit_step'])
        hit = hit_steps >= 0
        wall = result['wall_position_xyz'][hit]
        # Same angular conventions as boris_plotWallHist, including +pi endpoint.
        theta = np.arctan2(wall[:, 2], np.hypot(wall[:, 0], wall[:, 1]) - R0)
        theta = np.where(theta < 0, theta + 2*np.pi, theta)
        theta = np.where(theta > np.pi, theta - 2*np.pi, theta)
        phi_comp = np.mod(-np.arctan2(wall[:, 1], wall[:, 0]), 2*np.pi)
        phi = (198.0 - np.degrees(phi_comp)) % 360.0
        hist, _, _ = np.histogram2d(phi, np.degrees(theta), bins=(phi_edges, theta_edges))
        # Match production density=True normalization; avoid NaNs for no hits.
        density = hist.T / (max(int(hit.sum()), 1) * 2.0)  # bins: 1 deg x 2 deg
        mesh = ax.pcolormesh(phi_edges, theta_edges, np.ma.masked_equal(density, 0),
                             cmap=plt.get_cmap('Blues', 6),
                             norm=LogNorm(vmin=1e-6, vmax=1e-3), shading='flat')
        ax.set_title(f'{name.capitalize()}: {hit.sum():,} / {len(hit):,} wall hits')
        ax.set_yticks([-180, -90, 0, 90, 180], ['', 'Bottom', 'Outer', 'Top', ''])
        ax.set_ylabel('Poloidal location')
        ax.set_xlim(0, 360)
        ax.set_ylim(-180, 180)
        ax.set_xticks(np.arange(0, 361, 36))
        if not hit.any():
            ax.text(0.5, 0.5, 'No wall hits', ha='center', transform=ax.transAxes)
        active = len(hit_steps) - np.searchsorted(np.sort(hit_steps[hit]), time_steps, side='right')
        saved[f'{name}_active_count'] = active
        saved[f'{name}_wall_density_per_deg2'] = density
        for key, value in result.items():
            saved[f'{name}_{key}'] = value
    axes[-1].set_xlabel('Toroidal location (degrees CCW from South-Split)')
    fig.colorbar(mesh, ax=axes, label='Wall-hit probability density (deg$^{-2}$)', extend='both')
    wall_path = directory / 'Wall_comparison.png'
    fig.savefig(wall_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for name, color, style in [('torch', '#13294B', '-'), ('warp', '#FF5F05', '--')]:
        active = saved[f'{name}_active_count']
        total = len(results[name]['hit_step'])
        ax.step(time_ms, active / total, where='post', color=color, linestyle=style,
                label=f'{name.capitalize()} ({active[-1]:,} / {total:,} remain)')
    ax.set(xlabel='Time (ms)', ylabel='Active ion fraction', xlim=(0, steps*dt*1000),
           ylim=(0, 1.05), title='Ions remaining in the vessel')
    ax.grid(alpha=0.3)
    ax.legend()
    time_path = directory / 'Ions_vs_time_comparison.png'
    fig.savefig(time_path, dpi=200)
    plt.close(fig)
    data_path = directory / 'wall_results.npz'
    np.savez_compressed(data_path, **saved)
    return {'wall_plot': str(wall_path), 'ions_vs_time_plot': str(time_path),
            'wall_data': str(data_path)}


class QuietProgress:
    """Remove progress rendering from the existing Torch solver during timing."""
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass


class InitializerIO:
    """Read production Poincare data without overwriting initialization outputs."""
    def __init__(self, run_name):
        self.data_dir = ROOT / 'output' / run_name / 'data'
        self.log = logging.getLogger('benchmark.initializer')

    def loadNumpyData(self, name, subdir=None, mmap_mode=None):
        return np.load(self.data_dir / (subdir or '') / name, mmap_mode=mmap_mode)

    def createSubDir(self, *args, **kwargs):
        pass

    def saveFig(self, *args, **kwargs):
        pass

    def saveNumpyData(self, *args, **kwargs):
        pass


def initialize_particles(params, count, b, e):
    """Use production LCFS positions and normal-directed emission, once."""
    emitters = len(params['DELTRS']) * params['NPHI'] * params['NTHETA']
    if emitters < 1:
        raise ValueError('LCFS emitter grid must be nonempty')
    # Keep every configured emitter; --particles controls the copy count.
    copies = (count + emitters - 1) // emitters
    conditions = [params['LCFS_INDEX'], params['NPHI'], params['NTHETA'],
                  params['DELTRS'], copies]
    properties = [params['ION_MASS'], params['CHARGE_NUM'], params['ION_TEMP']]
    print(f'Initializing production LCFS {params["LCFS_INDEX"]}: {emitters} emitters, '
          f'{copies} copies per emitter before selecting {count} particles...', flush=True)
    ions, velocity_position, _ = point_generators.ionInitializer(
        conditions, properties, b, e, outputHandler=InitializerIO(params['OUTPUT_DIRECTORY_NAME']))
    # Uniform sampling avoids bias toward the first toroidal planes when count
    # is not a multiple of the emitter count. Keep the selected emitter order.
    if count < len(ions):
        selected = np.sort(np.random.default_rng().choice(len(ions), count, replace=False))
        ions = [ions[i] for i in selected]
        velocity_position = velocity_position[selected]
    v0 = np.ascontiguousarray(velocity_position[:, :3])
    x0 = np.ascontiguousarray(velocity_position[:, 3:])
    metadata = {'lcfs_index': params['LCFS_INDEX'], 'nphi': params['NPHI'],
                'ntheta': params['NTHETA'], 'deltrs': params['DELTRS'],
                'generated_copies_per_emitter': copies,
                'poincare_directory': str(InitializerIO(params['OUTPUT_DIRECTORY_NAME']).data_dir / 'Poincare')}
    return ions, x0, v0, metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path,
                        default=ROOT / 'input_files/boris_iota3_newsoltrace500_25v.json',
                        help='Existing Boris JSON; defaults to IOTA3 NewSOLTrace500 25 V')
    parser.add_argument('--particles', type=int, default=40960,
                        help='LCFS particle count (default: 40960); overrides JSON copies per emitter')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of position updates; overrides JSON TMAX')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--dt', type=float, help='Override JSON DT')
    parser.add_argument('--warp-step-chunk-size', type=int,
                        help='Physical steps per Warp launch (JSON WARP_STEP_CHUNK_SIZE, default 16; 1 disables)')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 (default), cuda:N, or cpu')
    parser.add_argument('--output', type=Path, help='Optional JSON timing/comparison report')
    parser.add_argument('--plot-dir', type=Path,
                        help='Plot/data directory (default: report path without suffix, '
                             'or output/warp_boris_benchmark)')
    args = parser.parse_args()
    if min(args.particles, args.steps, args.repeats) < 1 or args.steps >= 2**31:
        parser.error('particles, steps, and repeats must be positive; steps must fit int32')
    if args.dt is not None and (not np.isfinite(args.dt) or args.dt <= 0):
        parser.error('dt must be finite and positive')
    return args


def load_fields(params):
    """Use the same solved B components, scaling, and error field as the CLI."""
    b = mesh_module.TorchMesh(R0=0.72, a=0.19)
    b.setErrorField()
    b.loadCartesianField(
        str(ROOT / 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy'),
        coilCurrent=params['TOROIDAL_CURRENT'], att_mult=params['CONFIG_TOR'],
        errField=params['ENABLE_ERRFIELD'])
    b.addFieldPerturbation(
        str(ROOT / 'input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy'),
        coilCurrent=params['HELICAL_CURRENT'], att_mult=params['CONFIG_HEL'])
    e_path = Path(params['FIELD_FILE_ELECTRIC'])
    if not e_path.is_absolute():
        e_path = ROOT / e_path
    e = mesh_module.TorchMesh(R0=b.R0, a=b.a)
    e.loadCartesianField(str(e_path), period_=np.array([0, 1, 1]),
                         att_mult=float(params['PLASMA_POTENTIAL']))
    n, n_path = None, None
    if params['ION_ION_COLLISIONS']:
        n_path = Path(params['FIELD_FILE_DENSITY'])
        if not n_path.is_absolute():
            n_path = ROOT / n_path
        n = mesh_module.TorchMesh(R0=b.R0, a=b.a)
        n.loadScalarField(str(n_path), period_=np.array([0, 1, 1]),
                          att_mult=float(params['PLASMA_DENSITY']))
    return b, e, n, e_path, n_path


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type not in ('cuda', 'cpu'):
        raise SystemExit('Use --device cpu or cuda:N')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('PyTorch CUDA is unavailable in this environment. '
                         'Use --device cpu for a CPU comparison, or a CUDA-enabled '
                         'PyTorch environment for GPU timings.')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        device = torch.device('cuda', torch.cuda.current_device())
    # Existing modules select their device at import time; pin them here.
    for module in (boris_module, mesh_module, coordtrans, point_generators):
        module.device = device
    wp.init()
    warp_device = wp.get_device(str(device))
    logging.basicConfig(level=logging.WARNING)
    boris_module.tqdm = QuietProgress

    supplied = json.loads(args.inputs.read_text())
    # Accept current-name aliases used by some older Boris JSONs.
    for old, new in [('CURRENT_TOR', 'TOROIDAL_CURRENT'), ('CURRENT_HEL', 'HELICAL_CURRENT')]:
        if old in supplied and new not in supplied:
            supplied[new] = supplied[old]
    params = {**DEFAULT_INPUTS, **supplied}
    chunk_size = boris_module.Boris.validate_warp_step_chunk_size(
        args.warp_step_chunk_size if args.warp_step_chunk_size is not None
        else params.get('WARP_STEP_CHUNK_SIZE', 16))
    resolver = Collisions()
    neutral_model = resolver._resolve_ion_neutral_collision_model(params['ION_NEUTRAL_COLLISIONS'])
    ion_model = resolver._resolve_ion_ion_collision_model(params['ION_ION_COLLISIONS'])
    params['ION_NEUTRAL_COLLISIONS'] = neutral_model
    params['ION_ION_COLLISIONS'] = ion_model
    params['M_GAS_AMU'] = const.get_species_mass_amu(params['BACKGROUND_GAS_SPECIES'])
    params['PLASMA_POTENTIAL'], _ = resolve_plasma_potential(params)
    plasma_settings = dict(T_gas_eV=params['NEUTRAL_GAS_TEMP_EV'], Ti_eV=params['ION_TEMP'],
                           n_gas=params['NEUTRAL_GAS_DENSITY'], n_e=params['PLASMA_DENSITY'],
                           m_gas_amu=params['M_GAS_AMU'])
    collisions = make_collision_config(ion_neutral_collisions=neutral_model,
                                       ion_ion_collisions=ion_model, **plasma_settings)
    stochastic = neutral_model == 'langevin' or ion_model is not None
    dt = args.dt if args.dt is not None else float(params['DT'])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('DT must be finite and positive')
    print(f'Loading solved fields from {args.inputs}', flush=True)
    b, e, n, e_path, n_path = load_fields(params)
    error = b.err_adder.cpu().numpy() if bool(b.errField) else (0., 0., 0.)
    bw = make_grid(b.B.cpu().numpy(), R0=b.R0, a=b.a, periods=int(b.periodicity[2]),
                   device=warp_device, addend=error)
    ew = make_grid(e.B.cpu().numpy(), R0=e.R0, a=e.a, periods=1, device=warp_device)
    # Torch has already applied PLASMA_DENSITY and clamped negative values.
    # Convert back to the saved scalar layout; do not multiply a second time.
    nw = (make_density_grid(n.value.cpu().numpy().transpose(2, 1, 0), bw)
          if n is not None else None)

    # Generate once, outside timing; both backends and all repeats share it.
    ions, x0, v0, initial_conditions = initialize_particles(params, args.particles, b, e)
    q_over_m = np.array([ion.charge_mass_ratio for ion in ions])
    solver = boris_module.Boris(None)
    solver.setConditions(ions, 'Warp benchmark', dt=dt, tmax=args.steps*dt, **plasma_settings)

    def run_torch(steps):
        solver.nsteps = steps + 1  # Avoid float floor-division changing the workload.
        for ion in ions:
            ion.pos_XYZ.clear()
        wall_x, wall_v, last, _ = solver.parallel_solver(
            ions, b, Efield=e, nfield=n, trace_IDs=[], trace_stride=steps,
            ion_neutral_collisions=neutral_model, ion_ion_collisions=ion_model)
        wall_x, wall_v, last = wall_x.cpu().numpy(), wall_v.cpu().numpy(), last.cpu().numpy()
        hit = np.any(wall_x != 0, axis=1)
        return {'wall_position_xyz': wall_x, 'wall_velocity_xyz': wall_v,
                'last_inside_step': last, 'hit_step': np.where(hit, last+1, -1)}

    def run_warp(steps):
        return run_wall_only(x0, v0, q_over_m, bw, e_grid=ew, dt=dt, steps=steps,
                             collisions=collisions, density_grid=nw, step_chunk_size=chunk_size)

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        wp.synchronize_device(warp_device)

    print(f'{device}: {args.particles} particles, {args.steps} steps, dt={dt:g} s, '
          f'{args.repeats} repeats; no traces', flush=True)
    print(f'Collisions: neutral={neutral_model}, ion-ion={ion_model}', flush=True)
    print(f'Warp physical timesteps per launch: {chunk_size}', flush=True)
    if n_path is not None:
        print(f'Density: {n_path}\nDensity multiplier: {params["PLASMA_DENSITY"]:g} m^-3', flush=True)
    if stochastic:
        print('Stochastic models use independent random draws across backends and repeats; '
              'particle-by-particle differences are not a numerical parity test.', flush=True)
    print(f'E: {e_path}\nPotential multiplier: {params["PLASMA_POTENTIAL"]:g} V', flush=True)
    print('Warming both backends (including Warp compilation)...', flush=True)
    for run in (run_torch, run_warp):
        run(min(args.steps, 10))
    synchronize()

    times = {'torch': [], 'warp': []}
    results = {}
    runs = {'torch': run_torch, 'warp': run_warp}
    for repeat in range(args.repeats):
        # Alternate order to reduce a consistent first/second-run bias.
        for name in (('torch', 'warp') if repeat % 2 == 0 else ('warp', 'torch')):
            synchronize()
            start = perf_counter()
            result = runs[name](args.steps)
            synchronize()
            seconds = perf_counter() - start
            if not all(np.isfinite(value).all() for value in result.values()):
                raise RuntimeError(f'{name} returned nonfinite wall results')
            times[name].append(seconds)
            results[name] = result
            hits = int(np.count_nonzero(result['hit_step'] >= 0))
            print(f'{name:5s} repeat {repeat+1}: {seconds:.6f} s; '
                  f'{hits}/{args.particles} wall hits', flush=True)

    t, w = results['torch'], results['warp']
    thit, whit = t['hit_step'] >= 0, w['hit_step'] >= 0
    common = thit & whit
    comparison = {'torch_hits': int(thit.sum()), 'warp_hits': int(whit.sum()),
                  'hit_mask_disagreements': int(np.count_nonzero(thit != whit)),
                  'last_inside_step_disagreements': int(np.count_nonzero(
                      t['last_inside_step'] != w['last_inside_step'])),
                  'max_wall_position_difference_m': None,
                  'max_wall_velocity_difference_m_per_s': None}
    if common.any():
        for key, field in [('max_wall_position_difference_m', 'wall_position_xyz'),
                           ('max_wall_velocity_difference_m_per_s', 'wall_velocity_xyz')]:
            comparison[key] = float(np.linalg.norm(t[field][common] - w[field][common], axis=1).max())
    medians = {name: float(np.median(values)) for name, values in times.items()}
    ratio = medians['torch'] / medians['warp']
    report = {'inputs': str(args.inputs.resolve()), 'electric_field': str(e_path),
              'density_field': str(n_path) if n_path is not None else None,
              'collisions': {'ion_neutral': neutral_model, 'ion_ion': ion_model,
                             'stochastic': stochastic, **plasma_settings},
              'device': str(device), 'torch_version': torch.__version__, 'warp_version': wp.__version__,
              'particles': args.particles, 'steps': args.steps, 'dt': dt,
              'warp_step_chunk_size': chunk_size,
              'initial_conditions': initial_conditions, 'plasma_potential': params['PLASMA_POTENTIAL'],
              'seconds': times, 'median_seconds': medians, 'torch_over_warp': ratio,
              'comparison': comparison}
    print(f'Median: Torch {medians["torch"]:.6f} s; Warp {medians["warp"]:.6f} s; '
          f'Torch/Warp = {ratio:.2f}x')
    print('Wall comparison:', json.dumps(comparison, indent=2))
    print('Times include particle setup and wall download; exclude field loading/upload and warmup.\n'
          'Torch compacts active particles; Warp chunks timesteps and periodically checks for complete termination.\n'
          'This compares the eager Torch and package Warp solvers; no torch.compile or CUDA graphs.')
    plot_dir = args.plot_dir or (args.output.with_suffix('') if args.output else
                                ROOT / 'output/warp_boris_benchmark')
    report['artifacts'] = save_comparison_plots(results, plot_dir, steps=args.steps, dt=dt, R0=b.R0)
    for path in report['artifacts'].values():
        print(f'Saved {path}')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
