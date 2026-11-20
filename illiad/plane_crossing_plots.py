"""Read and plot streaming plane summaries without loading fields or trajectories."""
import numpy as np

MODES = ('count', 'area', 'flux', 'net-flux', 'density', 'energy')
LABELS = {'net-flux': 'Net crossing flux toward increasing phi [m⁻² s⁻¹]', 'count': 'Weighted crossing events / bin', 'area': 'Weighted crossings / m²',
          'flux': 'Mean unsigned crossing flux [m⁻² s⁻¹]',
          'density': 'Estimated mean particle density [m⁻³]', 'energy': 'Mean crossing energy [eV]'}


def load_histogram(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    rho, theta = data['rho_edges_m'], data['theta_edges_rad']
    for edges in (rho, theta):
        if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
            raise ValueError('Invalid histogram edges')
    shape = (len(rho)-1, len(theta)-1)
    if data['counts'].shape != shape or data['cell_area_m2'].shape != shape:
        raise ValueError('Counts and cell areas must match the saved mesh edges')
    if np.any(data['counts'] < 0) or not np.isfinite(data['counts']).all():
        raise ValueError('Invalid crossing counts')
    if np.any(data['cell_area_m2'] <= 0) or not np.isfinite(data['cell_area_m2']).all():
        raise ValueError('Cell areas must be finite and positive')
    return data


def histogram_values(data, mode='count', direction='both', duration=None, particle_weight=1.):
    if mode not in MODES or direction not in ('both', 'positive', 'negative'):
        raise ValueError('Invalid histogram mode or crossing direction')
    if not np.isfinite(particle_weight) or particle_weight <= 0:
        raise ValueError('Particle weight must be finite and positive')
    suffix = {'both': '', 'positive': '_increasing_phi', 'negative': '_decreasing_phi'}[direction]

    def field(name):
        key = name+suffix
        if key not in data:
            raise ValueError(f'{key} is absent: this plot needs a new run with streaming statistics enabled')
        value = np.asarray(data[key], dtype=float)
        if value.shape != data['counts'].shape or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f'Invalid saved statistic: {key}')
        return value

    counts = field('counts')
    if mode == 'net-flux':
        if direction != 'both':
            raise ValueError('net-flux requires --direction both; use flux for one-way flux')
        counts = field('counts_increasing_phi') - field('counts_decreasing_phi')
    excluded = 0
    if mode == 'energy':
        values = np.divide(field('energy_sum_eV'), counts,
                           out=np.full_like(counts, np.nan), where=counts > 0)
    else:
        values = field('inverse_normal_speed_sum_s_m') if mode == 'density' else counts.copy()
        if mode == 'density':
            excluded = int(field('density_excluded_counts').sum())
        values = values*particle_weight
        if mode != 'count':
            values /= data['cell_area_m2']
        if mode in ('flux', 'net-flux', 'density'):
            if duration is None:
                duration = data.get('observation_duration_s')
            if duration is None or not np.isfinite(duration) or duration <= 0:
                raise ValueError('Flux/density needs a positive --duration (seconds); old files lack saved duration')
            values /= float(duration)
    return values, LABELS[mode], excluded


def plot_histogram(data, mode='count', direction='both', duration=None, particle_weight=1.,
                   cmap=None, log=False, vmin=None, vmax=None, figsize=(6, 6),
                   r_range=None, z_range=None, overlay=None, poincare_size=.15,
                   poincare_alpha=.7, poincare_color='black'):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    values, label, _ = histogram_values(data, mode, direction, duration, particle_weight)
    if ((vmin is not None and not np.isfinite(vmin))
            or (vmax is not None and not np.isfinite(vmax))):
        raise ValueError('Color limits must be finite')
    finite = values[np.isfinite(values)]
    if cmap is None:
        cmap = 'RdBu_r' if mode == 'net-flux' else 'viridis'
    if mode == 'net-flux':
        if log:
            raise ValueError('Signed net flux cannot use --log-hist; use a linear scale')
        from matplotlib.colors import TwoSlopeNorm
        bound = max(float(np.max(np.abs(finite))) if finite.size else 0., 1e-30)
        low = -bound if vmin is None else vmin
        high = bound if vmax is None else vmax
        if not low < 0 < high:
            raise ValueError('Net flux color limits must straddle zero: vmin < 0 < vmax')
        norm = TwoSlopeNorm(vmin=low, vcenter=0., vmax=high)
    elif log:
        positive = finite[finite > 0]
        low = float(positive.min()) if positive.size else 1.
        high = float(positive.max()) if positive.size else 10.
        low = low if vmin is None else vmin
        high = max(high, low*10) if vmax is None and high <= low else (high if vmax is None else vmax)
        if low <= 0 or high <= low:
            raise ValueError('Log color limits require 0 < vmin < vmax')
        norm = LogNorm(low, high)
    else:
        low = 0. if vmin is None else vmin
        high = max(float(finite.max()) if finite.size else 0., low+1.) if vmax is None else vmax
        if high <= low:
            raise ValueError('Color limits require vmin < vmax')
        norm = Normalize(low, high)
    # Display empty energy cells at the floor; retain NaNs in numerical outputs.
    display = np.where(np.isfinite(values) & (values > norm.vmin), values, norm.vmin)
    theta = data['theta_edges_rad']
    subdivisions = max(1, int(np.ceil(720/(len(theta)-1))))
    theta = np.concatenate([np.linspace(a, b, subdivisions, endpoint=False)
                            for a, b in zip(theta[:-1], theta[1:])] + [theta[-1:]])
    rho, theta = np.meshgrid(data['rho_edges_m'], theta, indexing='ij')
    fig, ax = plt.subplots(figsize=figsize)
    R0, a = float(data['R0_m']), float(data['vessel_radius_m'])
    artist = ax.pcolormesh(R0+rho*np.cos(theta), rho*np.sin(theta),
                           np.repeat(display, subdivisions, axis=1),
                           shading='flat', cmap=cmap, norm=norm, rasterized=True)
    angle = np.linspace(0, 2*np.pi, 721)
    ax.plot(R0+a*np.cos(angle), a*np.sin(angle), color='black', lw=.8)
    if overlay is not None:
        ax.scatter(overlay[0], overlay[1], s=poincare_size, alpha=poincare_alpha,
                   c=poincare_color, linewidths=0)
    ax.set(aspect='equal', xlabel='R (m)', ylabel='Z (m)',
           title=rf'$\phi_{{comp}}={float(data["phi_deg"]):g}^\circ$ — {mode}, {direction}')
    if r_range is not None:
        ax.set_xlim(r_range)
    if z_range is not None:
        ax.set_ylim(z_range)
    fig.colorbar(artist, ax=ax, label=label)
    fig.tight_layout()
    return fig
