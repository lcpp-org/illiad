"""Replot saved Plane_crossings.npz without running Boris or loading full paths."""
import argparse
from pathlib import Path

import numpy as np

from illiad.plane_crossing_plots import MODES, histogram_values, load_histogram, plot_histogram

DEFAULT_ANALYSIS_DIR = Path(__file__).resolve().parents[1]/'output/IOTA3_1000sp_atol1e-9'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', help='Subanalysis name, or path to Plane_crossings.npz')
    p.add_argument('--analysis-dir', type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument('--histogram', choices=MODES, default='count')
    p.add_argument('--direction', choices=['both', 'positive', 'negative'], default='both',
                   help='Positive/negative means increasing/decreasing computational phi')
    p.add_argument('--duration', type=float, help='Observation duration in seconds; default: saved actual run span')
    p.add_argument('--particle-weight', type=float, default=1.)
    p.add_argument('--min-normal-speed', type=float, help='Assert the saved density cutoff; cannot change it after tracing')
    p.add_argument('--cmap', help='Default: RdBu_r for net-flux, viridis otherwise')
    p.add_argument('--log-hist', action='store_true')
    p.add_argument('--vmin', type=float)
    p.add_argument('--vmax', type=float)
    p.add_argument('--figsize', type=float, nargs=2, default=[6, 6])
    p.add_argument('--r-range', type=float, nargs=2)
    p.add_argument('--z-range', type=float, nargs=2)
    p.add_argument('--poincare', nargs='?', const='auto', help='Overlay matching phi; optional directory or file')
    p.add_argument('--poincare-size', type=float, default=.15)
    p.add_argument('--poincare-alpha', type=float, default=.7)
    p.add_argument('--poincare-color', default='black')
    p.add_argument('--output', type=Path, help='Output directory; default follows data/<subanalysis> to plots/<subanalysis>')
    p.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'], default=['png'])
    p.add_argument('--dpi', type=int, default=200)
    p.add_argument('--show', action='store_true')
    args = p.parse_args(argv)
    source = Path(args.source)
    if source.suffix != '.npz':
        source = args.analysis_dir/'data'/source/'Plane_crossings.npz'
    try:
        data = load_histogram(source)
        if args.min_normal_speed is not None:
            saved = data.get('min_normal_speed_m_s')
            if saved is None or not np.isclose(args.min_normal_speed, saved, rtol=0, atol=0):
                raise ValueError('The density speed cutoff is fixed during tracing; requested cutoff does not match the saved value')
        if args.dpi <= 0 or min(args.figsize) <= 0 or not np.isfinite(args.figsize).all():
            raise ValueError('DPI and figure dimensions must be positive')
        values, label, excluded = histogram_values(data, args.histogram, args.direction,
                                                    args.duration, args.particle_weight)
        overlay = None
        if args.poincare:
            try:
                from .plot_boris_plane import load_poincare
            except ImportError:
                from plot_boris_plane import load_poincare
            data_parents = [parent for parent in source.resolve().parents if parent.name == 'data']
            data_root = data_parents[0] if data_parents else args.analysis_dir/'data'
            path = data_root/'Poincare' if args.poincare == 'auto' else Path(args.poincare)
            overlay = load_poincare(path, float(data['phi_deg']), float(data['R0_m']))
        fig = plot_histogram(data, mode=args.histogram, direction=args.direction,
                             duration=args.duration, particle_weight=args.particle_weight,
                             cmap=args.cmap, log=args.log_hist, vmin=args.vmin, vmax=args.vmax,
                             figsize=args.figsize, r_range=args.r_range, z_range=args.z_range,
                             overlay=overlay, poincare_size=args.poincare_size,
                             poincare_alpha=args.poincare_alpha, poincare_color=args.poincare_color)
    except (ValueError, KeyError, OSError) as exc:
        p.error(str(exc))
    output = args.output
    if output is None:
        # Preserve nested subanalysis directories when given a direct archive path.
        parents = [parent for parent in source.resolve().parents if parent.name == 'data']
        output = (parents[0].parent/'plots'/source.resolve().parent.relative_to(parents[0])
                  if parents else source.parent/'plots')
    output.mkdir(parents=True, exist_ok=True)
    stem = f'Plane_crossings_{args.histogram}_{args.direction}'
    for fmt in args.formats:
        fig.savefig(output/f'{stem}.{fmt}', dpi=args.dpi)
    np.savez_compressed(output/f'{stem}.npz', values=values, label=label,
                        rho_edges_m=data['rho_edges_m'], theta_edges_rad=data['theta_edges_rad'],
                        mode=args.histogram, direction=args.direction, particle_weight=args.particle_weight,
                        duration_s=args.duration if args.duration is not None else data.get('observation_duration_s', np.nan),
                        min_normal_speed_m_s=data.get('min_normal_speed_m_s', np.nan),
                        density_excluded_counts=excluded, source=str(source.resolve()))
    print(f'Saved {stem} to {output}; density cutoff excluded {excluded} crossings.')
    import matplotlib.pyplot as plt
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
