"""Plot all resolved Boris crossings of a computational toroidal half-plane.

See docs/boris_plane_plots.md for estimators, timing and sampling limitations.
Only NumPy and Matplotlib are needed; traces are read in time chunks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import warnings

import numpy as np

AMU_KG = 1.66053906892e-27
EV_J = 1.602176634e-19
DEFAULT_ANALYSIS_DIR = Path(__file__).resolve().parents[1] / 'output/IOTA3_1000sp_atol1e-9'
DEFAULT_SUBANALYSIS = '0mm_LCFS78_2eV_25V_Z1_HeEvap2_NoIslands2'
FIELDS = ('R_m', 'Z_m', 'time_s', 'energy_eV', 'v_normal_m_s', 'speed_m_s',
          'trace_column', 'segment_index')


def read_run_log(trace_path):
    """Read only the matching run's log, never unrelated current input JSONs."""
    path = trace_path.parent.parent.parent / 'logs' / trace_path.parent.name / 'boris.log'
    values = {}
    if path.is_file():
        content = path.read_text()
        for key, label in [('dt', 'TIME STEP'), ('stride', 'TRACE OUTPUT STRIDE'),
                           ('mass_amu', 'ION MASS'), ('nsteps', r'--> # OF TIME STEPS')]:
            match = re.search(r'\|\s*' + label + r':\s*([\d.eE+\-]+)', content)
            if match:
                values[key] = float(match.group(1))
        terminated = re.search(r'All particles terminated at step (\d+)', content)
        if terminated:
            values['final_step'] = int(terminated[1])
        elif 'nsteps' in values and '## SIM FINISHED! ##' in content and 'Boris method: warp' not in content:
            # Torch logs early termination explicitly; both backends advance at
            # most nsteps-1 times. Warp does not currently log its early stop.
            values['final_step'] = int(values['nsteps']) - 1
        values['log_path'] = str(path.resolve())
    return values


def logged_final_interval(log, n):
    """Use logged step counts only when they match the saved array length."""
    if not all(key in log for key in ('final_step', 'stride', 'dt')):
        return None
    stride, final = int(log['stride']), log['final_step']
    if stride < 1 or final < 1:
        return None
    expected = 1 + final // stride + int(final % stride != 0)
    if expected != n:
        return None
    return (final % stride or stride) * log['dt']


def sample_times(n, sample_dt, times_path=None, final_interval=None):
    if times_path:
        times = np.asarray(np.load(times_path, allow_pickle=False), dtype=float)
        if times.shape != (n,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('--times must contain one finite, strictly increasing time per saved row.')
        return times
    if sample_dt is None or not np.isfinite(sample_dt) or sample_dt <= 0:
        raise ValueError('Supply --sample-dt (DT * trace stride), --times, or a matching Boris log.')
    times = np.arange(n, dtype=float) * sample_dt
    # Boris may append a shortened final interval. Do not invent its duration.
    times[-1] = np.nan if final_interval is None else times[-2] + final_interval
    return times


def extract_crossings(traces, times, phi_deg, mass_amu, chunk_steps=128, progress=False):
    """Intersect straight saved segments with the positive-R toroidal half-plane.

    Positive normal velocity is increasing clockwise computational phi. Initial
    points on the plane are excluded. Exact sampled hits require continuation
    across the plane, avoiding double counts and tangential touches. Plane-dwell
    endpoints are excluded. Zero padding and nonfinite samples cannot bridge gaps.
    """
    if traces.ndim != 3 or traces.shape[-1] != 3 or traces.shape[0] < 2:
        raise ValueError('Expected (time >= 2, particle, xyz=3) position array.')
    if len(times) != len(traces) or chunk_steps < 1 or not np.isfinite(mass_amu) or mass_amu <= 0:
        raise ValueError('Invalid times, chunk size, or ion mass.')
    phi = np.deg2rad(phi_deg % 360)
    radial = np.array([np.cos(phi), -np.sin(phi), 0.])
    normal = np.array([-np.sin(phi), -np.cos(phi), 0.])
    pieces = {key: [] for key in FIELDS}
    for start in range(0, len(traces) - 1, chunk_steps):
        stop = min(start + chunk_steps, len(traces) - 1)
        xyz = np.asarray(traces[start:min(stop + 2, len(traces))], dtype=float)
        a, b = xyz[:stop-start], xyz[1:stop-start+1]
        d0, d1 = a @ normal, b @ normal
        valid = (np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
                 & np.any(a != 0, axis=-1) & np.any(b != 0, axis=-1))
        cross = ((d0 < 0) & (d1 > 0)) | ((d0 > 0) & (d1 < 0))
        # Use one lookahead row for an exact endpoint; shared endpoints count once.
        follow = np.full_like(d1, np.nan)
        nf = len(xyz) - 2
        if nf:
            next_xyz = xyz[2:]
            follow[:nf] = next_xyz @ normal
            follow[:nf][~(np.isfinite(next_xyz).all(axis=-1) & np.any(next_xyz != 0, axis=-1))] = np.nan
        cross |= (d1 == 0) & (((d0 < 0) & (follow > 0)) | ((d0 > 0) & (follow < 0)))
        intervals = np.diff(times[start:stop+1])
        rows, cols = np.nonzero(cross & valid & np.isfinite(intervals[:, None])
                                & (intervals[:, None] > 0))
        if len(rows):
            fraction = d0[rows, cols] / (d0[rows, cols] - d1[rows, cols])
            delta = b[rows, cols] - a[rows, cols]
            hit = a[rows, cols] + fraction[:, None] * delta
            radius = hit @ radial
            keep = radius > 0  # reject phi + 180 degrees
            velocity = delta / intervals[rows, None]
            speed = np.linalg.norm(velocity, axis=1)
            data = (radius, hit[:, 2], times[start + rows] + fraction * intervals[rows],
                    0.5 * mass_amu * AMU_KG * speed**2 / EV_J,
                    velocity @ normal, speed, cols, start + rows)
            for key, value in zip(FIELDS, data):
                pieces[key].append(value[keep])
        if progress and (start == 0 or stop == len(traces)-1 or start // chunk_steps % 20 == 0):
            print(f'  scanned {stop:,}/{len(traces)-1:,} saved intervals', flush=True)
    return {key: np.concatenate(value) if value else np.empty(0, dtype=(int if key in FIELDS[-2:] else float))
            for key, value in pieces.items()}


def histogram(crossings, r_edges, z_edges, mode='area', duration=None, particle_weight=1., min_normal_speed=0.):
    """Return map, label, and number excluded by the density grazing cutoff."""
    r, z = crossings['R_m'], crossings['Z_m']
    weights = np.full(len(r), particle_weight)
    excluded = 0
    if mode == 'density':
        speed = np.abs(crossings['v_normal_m_s'])
        keep = speed > min_normal_speed
        excluded = int((~keep).sum())
        r, z, weights = r[keep], z[keep], weights[keep] / speed[keep]
    values = np.histogram2d(r, z, bins=(r_edges, z_edges), weights=weights)[0]
    if mode == 'energy':
        total = np.histogram2d(r, z, bins=(r_edges, z_edges),
                               weights=weights * crossings['energy_eV'])[0]
        return np.divide(total, values, out=np.full_like(total, np.nan), where=values > 0), 'Mean crossing energy [eV]', 0
    labels = {'count': 'Weighted crossings / bin', 'area': 'Weighted crossings / m²',
              'flux': 'Mean unsigned crossing flux [m⁻² s⁻¹]',
              'density': 'Estimated mean traced-particle density [m⁻³]'}
    if mode not in labels:
        raise ValueError(f'Unknown histogram mode: {mode}')
    if mode != 'count':
        values /= np.diff(r_edges)[:, None] * np.diff(z_edges)[None, :]
    if mode in ('flux', 'density'):
        if duration is None or not np.isfinite(duration) or duration <= 0:
            raise ValueError('Flux/density requires a positive observation duration.')
        values /= duration
    return values, labels[mode], excluded


def node_histogram(crossings, field_path, major_radius, minor_radius, mode,
                   duration, particle_weight=1., min_normal_speed=0.):
    """Tally in midpoint-bounded polar cells using native magnetic mesh nodes.

    Native periodic theta nodes are dtheta,...,2pi. The coincident axis
    nodes are represented by one disk. All areas are clipped at the vessel.
    """
    shape = np.load(field_path, mmap_mode='r', allow_pickle=False).shape
    if len(shape) != 4 or shape[0] != 3 or shape[1] < 2 or shape[2] < 3:
        raise ValueError('Magnetic field must have shape (3, nrho>=2, ntheta>=3, nphi).')
    nr, nt = shape[1:3]
    rho = np.linspace(0, minor_radius, nr)
    theta = np.arange(1, nt+1) * (2*np.pi/nt)
    redges = np.r_[0., (rho[:-1]+rho[1:])/2, minor_radius]
    rr, tt = np.meshgrid(rho, theta, indexing='ij')
    area = np.broadcast_to((np.diff(redges**2)*np.pi/nt)[:, None], (nr, nt)).copy()
    area[0, 0] *= nt
    unique = np.ones((nr, nt), dtype=bool)
    unique[0, 1:] = False
    x = crossings['R_m'] - major_radius
    z = crossings['Z_m']
    radius = np.hypot(x, z)
    angle = np.mod(np.arctan2(z, x), 2*np.pi)
    ir = np.clip(np.searchsorted(redges, radius, side='right')-1, 0, nr-1)
    it = (np.floor(angle/(2*np.pi/nt) + .5).astype(int)-1) % nt
    it[ir == 0] = 0
    inside = radius <= minor_radius
    keep = inside.copy()
    weights = np.full(len(radius), particle_weight)
    excluded = 0
    if mode == 'density':
        speed = np.abs(crossings['v_normal_m_s'])
        keep &= speed > min_normal_speed
        excluded = int(np.count_nonzero(inside & ~keep))
        weights = np.divide(weights, speed, out=np.zeros_like(weights), where=speed > 0)
    indices = ir[keep]*nt + it[keep]
    count = np.bincount(indices, minlength=nr*nt).reshape(nr, nt)
    values = np.bincount(indices, weights=weights[keep], minlength=nr*nt).reshape(nr, nt)
    labels = {'count': 'Weighted crossings / node cell', 'area': 'Weighted crossings / m²',
              'flux': 'Mean unsigned crossing flux [m⁻² s⁻¹]',
              'density': 'Estimated mean traced-particle density [m⁻³]',
              'energy': 'Mean crossing energy [eV]'}
    if mode == 'energy':
        total = np.bincount(indices, weights=weights[keep]*crossings['energy_eV'][keep],
                            minlength=nr*nt).reshape(nr, nt)
        values = np.divide(total, values, out=np.full_like(total, np.nan), where=values > 0)
    elif mode != 'count':
        values /= area
    if mode in ('flux', 'density'):
        if duration is None or not np.isfinite(duration) or duration <= 0:
            raise ValueError('Flux/density requires a positive observation duration.')
        values /= duration
    return dict(R_m=(major_radius+rr*np.cos(tt))[unique], Z_m=(rr*np.sin(tt))[unique],
                rho_m=rr[unique], theta_rad=tt[unique], area_m2=area[unique],
                values=values[unique], crossing_count=count[unique], label=labels[mode],
                outside_vessel=int((~inside).sum()), speed_cutoff_excluded=excluded)


def load_poincare(path, phi_deg, major_radius):
    """Load native (surface, [theta,rho], point) data at the exact saved phi."""
    if path.is_dir():
        matches = []
        for candidate in path.glob('Poincare_*.npy'):
            try:
                angle = float(candidate.stem.split('_')[-1])
            except ValueError:
                continue
            if abs((angle - phi_deg + 180) % 360 - 180) < 1e-7:
                matches.append(candidate)
        if not matches:
            raise ValueError(f'No saved Poincare plane at phi={phi_deg:g} in {path}; no nearest-plane substitution.')
        path = sorted(matches)[0]
    else:
        match = re.fullmatch(r'Poincare_([\d.]+)', path.stem)
        if not match or abs((float(match[1]) - phi_deg + 180) % 360 - 180) > 1e-7:
            raise ValueError('Overlay file must be named Poincare_<matching phi>.npy.')
    data = np.load(path, allow_pickle=False)
    if data.ndim != 3 or data.shape[1] != 2:
        raise ValueError('Expected native Poincare shape (surface, 2, point).')
    theta, rho = data[:, 0].ravel(), data[:, 1].ravel()
    keep = np.isfinite(theta) & np.isfinite(rho) & (rho >= 0)
    return major_radius + rho[keep] * np.cos(theta[keep]), rho[keep] * np.sin(theta[keep])


def make_plot(crossings, values, r_edges, z_edges, label, args, overlay, nodes=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), layout='constrained', sharex=True, sharey=True)
    points = axes[0].scatter(crossings['R_m'], crossings['Z_m'], c=crossings['energy_eV'],
                             s=args.point_size, cmap='plasma', linewidths=0, rasterized=True)
    fig.colorbar(points, ax=axes[0], label='Segment-estimated kinetic energy [eV]')
    display_values = values if nodes is None else nodes['values']
    positive = display_values[np.isfinite(display_values) & (display_values > 0)]
    norm = None
    if args.log_hist and len(positive):
        lo, hi = positive.min(), positive.max()
        # Clip zero bins to the lowest display color; preserve their saved values.
        norm = LogNorm(vmin=lo, vmax=hi if hi > lo else lo * 10, clip=True)
    # Empty energy cells remain NaN in saved data; only their display uses zero.
    colors = np.nan_to_num(display_values, nan=0.)
    if nodes is None:
        grid = axes[1].pcolormesh(r_edges, z_edges, colors.T,
                                  cmap='viridis', norm=norm, vmin=0 if norm is None else None,
                                  shading='flat', rasterized=True)
    else:
        grid = axes[1].scatter(nodes['R_m'], nodes['Z_m'], c=colors,
                                s=args.node_size, cmap='viridis', norm=norm,
                                vmin=0 if norm is None else None, linewidths=0, rasterized=True)
    # Clip boundary bins exactly at the vessel wall, rather than by bin center.
    grid.set_clip_path(Circle((args.major_radius, 0), args.minor_radius,
                              transform=axes[1].transData))
    fig.colorbar(grid, ax=axes[1], label=label)
    axes[0].set_title(f'All {len(crossings["R_m"]):,} resolved crossings')
    kind = 'Poloidal histogram' if nodes is None else 'Magnetic mesh nodes'
    axes[1].set_title(f'{kind}: {args.histogram}')
    theta = np.linspace(0, 2*np.pi, 361)
    for ax in axes:
        if overlay is not None:
            ax.scatter(*overlay, s=args.poincare_size, c=args.poincare_color,
                       alpha=args.poincare_alpha, linewidths=0, rasterized=True, zorder=3)
        ax.plot(args.major_radius + args.minor_radius*np.cos(theta),
                args.minor_radius*np.sin(theta), color='0.3', lw=0.7)
        ax.set(xlabel='R [m]', ylabel='Z [m]', aspect='equal',
               xlim=(r_edges[0], r_edges[-1]), ylim=(z_edges[0], z_edges[-1]))
    fig.suptitle(f'Boris crossings — computational φ = {args.phi % 360:g}° ({args.direction})')
    for extension in args.formats:
        stem = 'plane' if nodes is None else 'plane_nodes'
        fig.savefig(args.output / f'{stem}.{extension}', dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('subanalysis', nargs='?', default=DEFAULT_SUBANALYSIS,
                   help='Subanalysis directory name beneath the analysis data/ directory.')
    p.add_argument('--analysis-dir', type=Path, default=DEFAULT_ANALYSIS_DIR,
                   help='Analysis root containing data/, logs/, and plots/ (default: output/IOTA3_1000sp_atol1e-9).')
    p.add_argument('--phi', type=float, default=0., help='Clockwise computational phi in degrees, as in Poincare filenames.')
    p.add_argument('--sample-dt', type=float, help='Seconds between regular saved rows; overrides log DT * stride.')
    p.add_argument('--mass-amu', type=float, help='Ion mass; defaults to matching run log.')
    p.add_argument('--times', type=Path, help='NPY of exact saved-row times in seconds.')
    p.add_argument('--final-interval', type=float, help='Seconds in final interval; inferred from matching log when possible, otherwise omitted.')
    p.add_argument('--histogram', choices=['count', 'area', 'flux', 'density', 'energy'], default='area')
    p.add_argument('--plot-type', choices=['histogram', 'nodes', 'both'], default='histogram')
    p.add_argument('--node-size', type=float, default=35., help='Mesh marker area in points squared.')
    p.add_argument('--magnetic-field', type=Path,
                   default=Path(__file__).resolve().parents[1] / 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy',
                   help='Native magnetic field NPY; its shape defines rho/theta nodes.')
    p.add_argument('--duration', type=float, help='Observation duration in seconds; default is included trace time span.')
    p.add_argument('--particle-weight', type=float, default=1., help='Physical particles represented by each traced particle.')
    p.add_argument('--min-normal-speed', type=float, default=0., help='For density, exclude crossings at or below this |v_phi| [m/s]; exclusions reported.')
    p.add_argument('--direction', choices=['both', 'positive', 'negative'], default='both')
    p.add_argument('--bins', type=int, nargs=2, default=[120, 120], metavar=('NR', 'NZ'))
    p.add_argument('--r-range', type=float, nargs=2)
    p.add_argument('--z-range', type=float, nargs=2)
    p.add_argument('--major-radius', type=float, default=0.72)
    p.add_argument('--minor-radius', type=float, default=0.19, help='Radius of circular vessel outline [m].')
    p.add_argument('--poincare', nargs='?', const='auto', help='Overlay matching plane; optionally give a Poincare directory or file.')
    p.add_argument('--poincare-size', type=float, default=0.15)
    p.add_argument('--poincare-alpha', type=float, default=0.7)
    p.add_argument('--poincare-color', default='black')
    p.add_argument('--point-size', type=float, default=1.)
    p.add_argument('--chunk-steps', type=int, default=128, help='Read this many time intervals per chunk; does not subsample.')
    p.add_argument('--log-hist', action='store_true')
    p.add_argument('--output', type=Path, help='Override output directory; default: <analysis-dir>/plots/<subanalysis>/plane_phi<angle>.')
    p.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'], default=['png'])
    p.add_argument('--dpi', type=int, default=200)
    p.add_argument('--show', action='store_true')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if not args.subanalysis or Path(args.subanalysis).name != args.subanalysis or args.subanalysis in ('.', '..'):
            raise ValueError('Supply a subanalysis directory name, not a trace path; use --analysis-dir for its analysis root.')
        args.analysis_dir = args.analysis_dir.expanduser()
        args.traces = args.analysis_dir / 'data' / args.subanalysis / 'Ion_traces.npy'
        for name in ('phi', 'major_radius', 'minor_radius', 'particle_weight', 'min_normal_speed'):
            if not np.isfinite(getattr(args, name)):
                raise ValueError(f'--{name.replace("_", "-")} must be finite.')
        if min(args.bins) < 1 or args.chunk_steps < 1 or args.minor_radius <= 0 or args.major_radius <= args.minor_radius or args.particle_weight <= 0 or args.min_normal_speed < 0:
            raise ValueError('Invalid bins, chunk size, radii, particle weight or speed cutoff.')
        if args.final_interval is not None and (not np.isfinite(args.final_interval) or args.final_interval <= 0):
            raise ValueError('--final-interval must be finite and positive.')
        if args.times and args.final_interval is not None:
            raise ValueError('--times and --final-interval are mutually exclusive.')
        if not np.isfinite(args.node_size) or args.node_size <= 0:
            raise ValueError('--node-size must be positive and finite.')
        traces = np.load(args.traces, mmap_mode='r', allow_pickle=False)
        if traces.ndim != 3 or traces.shape[-1] != 3 or len(traces) < 2:
            raise ValueError('Expected (time >= 2, particle, xyz=3) position array.')
        log = read_run_log(args.traces)
        mass = args.mass_amu if args.mass_amu is not None else log.get('mass_amu')
        dt = args.sample_dt if args.sample_dt is not None else (log['dt'] * log['stride'] if 'dt' in log and 'stride' in log else None)
        if mass is None:
            raise ValueError('Supply --mass-amu; no ion mass found in matching run log.')
        final_interval = args.final_interval
        if final_interval is None and args.sample_dt is None and not args.times:
            final_interval = logged_final_interval(log, len(traces))
        times = sample_times(len(traces), dt, args.times, final_interval)
        if not np.isfinite(times[-1]):
            warnings.warn('Omitting the last saved interval: its duration may be shorter than the stride. Supply --times or --final-interval to include it.')
        duration = args.duration if args.duration is not None else float(np.nanmax(times) - times[0])
        overlay_path = (args.traces.parent.parent / 'Poincare') if args.poincare == 'auto' else (Path(args.poincare) if args.poincare else None)
        overlay = load_poincare(overlay_path, args.phi, args.major_radius) if overlay_path else None
        print(f'Traces: {traces.shape}, {traces.dtype}; mass={mass:g} amu; regular interval={dt} s', flush=True)
        crossings = extract_crossings(traces, times, args.phi, mass, args.chunk_steps, progress=True)
        if args.direction != 'both':
            keep = crossings['v_normal_m_s'] > 0 if args.direction == 'positive' else crossings['v_normal_m_s'] < 0
            crossings = {key: value[keep] for key, value in crossings.items()}
        if not len(crossings['R_m']):
            raise ValueError('No resolved crossings at this phi/direction; check trace coverage and timing.')
        # Automatic bounds include every crossing, the vessel, and the optional overlay.
        r = np.r_[crossings['R_m'], args.major_radius-args.minor_radius, args.major_radius+args.minor_radius]
        z = np.r_[crossings['Z_m'], -args.minor_radius, args.minor_radius]
        if overlay is not None:
            r, z = np.r_[r, overlay[0]], np.r_[z, overlay[1]]
        edges = []
        for data, bounds, bins in zip((r, z), (args.r_range, args.z_range), args.bins):
            bounds = bounds if bounds is not None else [data.min(), data.max()]
            if not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError('Plot bounds must be finite and increasing.')
            edges.append(np.linspace(*bounds, bins+1))
        r_edges, z_edges = edges
        values, label, excluded = histogram(crossings, *edges, args.histogram, duration,
                                            args.particle_weight, args.min_normal_speed)
        outside = int(np.count_nonzero((crossings['R_m'] < r_edges[0]) | (crossings['R_m'] > r_edges[-1]) |
                                      (crossings['Z_m'] < z_edges[0]) | (crossings['Z_m'] > z_edges[-1])))
        args.output = args.output or args.analysis_dir / 'plots' / args.subanalysis / f'plane_phi{args.phi % 360:g}'
        args.output.mkdir(parents=True, exist_ok=True)
        metadata = dict(trace_path=str(args.traces.resolve()), phi_deg=args.phi % 360, mass_amu=mass,
                        sample_dt_s=dt, final_interval_s=final_interval,
                        observation_duration_s=duration, histogram=args.histogram,
                        particle_weight=args.particle_weight, direction=args.direction,
                        crossing_count=len(crossings['R_m']), outside_plot_bounds=outside,
                        density_speed_cutoff_excluded=excluded, final_interval_included=bool(np.isfinite(times[-1])),
                        energy_method='saved-segment displacement / elapsed time', log=log,
                        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        np.savez_compressed(args.output / 'crossings.npz', **crossings)
        np.savez_compressed(args.output / 'histogram.npz', values=values, R_edges_m=r_edges,
                            Z_edges_m=z_edges, label=label)
        (args.output / 'settings.json').write_text(json.dumps(metadata, indent=2) + '\n')
        if args.plot_type in ('histogram', 'both'):
            make_plot(crossings, values, *edges, label, args, overlay)
        if args.plot_type in ('nodes', 'both'):
            nodes = node_histogram(crossings, args.magnetic_field, args.major_radius,
                                   args.minor_radius, args.histogram, duration,
                                   args.particle_weight, args.min_normal_speed)
            np.savez_compressed(args.output / 'node_histogram.npz', **nodes)
            make_plot(crossings, values, *edges, nodes['label'], args, overlay, nodes=nodes)
            print(f"Saved {len(nodes['values']):,} mesh nodes; "
                  f"{nodes['outside_vessel']} crossings outside vessel, "
                  f"{nodes['speed_cutoff_excluded']} excluded by speed cutoff.")
        print(f'Saved {len(crossings["R_m"]):,} crossings and plots to {args.output}; '
              f'{outside} outside bounds, {excluded} excluded by density speed cutoff.')
    except (ValueError, OSError) as error:
        p.error(str(error))


if __name__ == '__main__':
    main()
