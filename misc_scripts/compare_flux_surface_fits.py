"""Compare regular-surface Fourier/spline fits with fresh CPU field-line traces.

Run from the project root with ``python -m misc_scripts.compare_flux_surface_fits``.
Inputs are never overwritten. Output follows data/<subanalysis>, plots/<subanalysis>.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import splev

from illiad.cli.flux_calc import DEFAULT_INPUTS
from illiad.flux.calculator import FluxCalculator
from illiad.flux.fourier import fit_fourier_contour
from illiad.mesh import Mesh
from illiad.particle import FieldLine
from illiad.poincare import _extract_plane_crossings
from illiad.sol.tracer import load_poincare_settings, minimum_boundary_distance
from illiad.utilities.coordtrans import axisShift, RTP_to_XYZ, XYZ_to_RTP_many
from illiad.utilities.phi_events import inVV


def trace_validation(field, seed_rtp, *, spins, method, rtol, atol, max_step):
    """Retrace a measured crossing in +/-B, excluding the launch point.

    Use the production Cartesian Mesh/FieldLine RHS, CPU solver, and dense
    output plane-crossing reconstruction. Neither fitted curve sets the seed.
    """
    seed = RTP_to_XYZ(seed_rtp, field.R0)
    if seed_rtp[0] >= field.a:
        raise ValueError('Validation seed is outside the vessel')
    degree = np.rad2deg(seed_rtp[2]) % 360
    if not np.isclose(degree, round(degree), rtol=0, atol=1e-8):
        raise ValueError('Validation seed must lie on an integer-degree saved plane')
    plane_index = (int(round(degree))-1) % 360
    points, records, directions = [], [], []
    length = 2*np.pi*field.R0*spins
    for direction in (1, -1):
        particle = FieldLine(seed, length, direction=direction)
        started = perf_counter()
        sol = solve_ivp(particle.pushXYZ, (0, length), seed, args=(field,),
                        events=(inVV,), method=method, rtol=rtol,
                        atol=atol, max_step=max_step, dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        xyz = _extract_plane_crossings(sol.sol, sol.t, sol.y.T,
                                       np.deg2rad(np.arange(1, 361)))[plane_index]
        # A roundoff-level launch crossing can occur for an exactly planar seed.
        if len(xyz) and np.linalg.norm(xyz[0]-seed) < 1e-7:
            xyz = xyz[1:]
        hits = XYZ_to_RTP_many(xyz, field.R0)
        points.append(hits)
        directions.extend([direction]*len(hits))
        records.append(dict(direction=direction, count=len(hits),
                            length_m=float(sol.t[-1]), wall_hit=sol.status == 1,
                            seconds=perf_counter()-started))
    return np.concatenate(points), np.asarray(directions), records


def residual_metrics(evaluate, theta, radius, grid):
    if not len(theta):
        return {"count": 0, "radial_rms_mm": None, "radial_max_mm": None,
                "distance_rms_mm": None, "distance_max_mm": None}
    residual = evaluate(theta) - radius
    curve = np.column_stack((evaluate(grid)*np.cos(grid), evaluate(grid)*np.sin(grid)))
    points = np.column_stack((radius*np.cos(theta), radius*np.sin(theta)))
    distances = minimum_boundary_distance(points, curve)
    return dict(count=len(theta), radial_rms_mm=float(np.sqrt(np.mean(residual**2))*1000),
                radial_max_mm=float(np.max(np.abs(residual))*1000),
                distance_rms_mm=float(np.sqrt(np.mean(distances**2))*1000),
                distance_max_mm=float(distances.max()*1000))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', default='IOTA3_1000sp_atol1e-9')
    parser.add_argument('--subanalysis', default='fourier_regular_test')
    parser.add_argument('--surfaces', type=int, nargs='+', default=[36, 48, 80])
    parser.add_argument('--planes', type=int, nargs='+', default=[60, 180, 300])
    parser.add_argument('--modes', type=int, default=8)
    parser.add_argument('--regularization', type=float, default=1e-6)
    parser.add_argument('--max-gap-deg', type=float, default=90)
    parser.add_argument('--smoothing', type=float, default=6e-6)
    parser.add_argument('--max-subsets', type=int, default=4)
    parser.add_argument('--spins', type=float, default=32,
                        help='Trace length per direction in 2*pi*R0 units')
    parser.add_argument('--rtol', type=float, help='Default: source log RTOL')
    parser.add_argument('--atol', type=float, help='Default: source log ATOL')
    parser.add_argument('--method', choices=['LSODA', 'DOP853', 'RK45'],
                        help='Default: source log SOLVER')
    parser.add_argument('--max-step', type=float, default=0.05)
    parser.add_argument('--no-trace', action='store_true', help='Only fit saved data; no independent validation')
    args = parser.parse_args()
    if not np.isfinite(args.spins) or args.spins <= 0 or not np.isfinite(args.max_step) or args.max_step <= 0:
        parser.error('spins and max-step must be finite and positive')
    if any(p < 1 or p > 360 for p in args.planes):
        parser.error('planes must be saved integer degrees in 1..360')
    if Path(args.subanalysis).is_absolute() or '..' in Path(args.subanalysis).parts:
        parser.error('subanalysis must be relative and cannot contain ..')
    root = Path('output') / args.analysis
    settings = load_poincare_settings(args.analysis)
    args.method = args.method or settings['SOLVER']
    args.rtol = args.rtol if args.rtol is not None else settings['RTOL']
    args.atol = args.atol if args.atol is not None else settings['ATOL']
    for key in ('rtol', 'atol'):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f'{key} must be finite and positive')
    data_dir, plot_dir = root/'data'/args.subanalysis, root/'plots'/args.subanalysis
    # Every run is a separate, inspectable experiment.
    data_dir.mkdir(parents=True, exist_ok=False)
    plot_dir.mkdir(parents=True, exist_ok=False)
    field = Mesh(R0=0.72, a=0.19)
    field_paths = [Path('input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy'),
                   Path('input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy')]
    if not args.no_trace:
        field.loadCartesianField(file_path=str(field_paths[0]), coilCurrent=settings['CURRENT_TOR'],
                                 errField=settings['ENABLE_ERRFIELD'], att_mult=settings['CONFIG_TOR'])
        field.set_nonPer_errField()
        field.addFieldPerturbation(file_path=str(field_paths[1]), coilCurrent=settings['CURRENT_HEL'],
                                   att_mult=settings['CONFIG_HEL'])
    params = dict(DEFAULT_INPUTS, PHI_GENs=np.asarray(args.planes), MAX_SUBSETS=args.max_subsets)
    calc = FluxCalculator(SimpleNamespace(log=logging.getLogger(__name__)), field, params)
    grid = np.linspace(0, 2*np.pi, 4096, endpoint=False)
    report = dict(options=vars(args), source_settings=settings, results=[],
                  scope='Geometric contour test, not straight-field-line coordinates or flux validation',
                  field_files=[dict(path=str(p.resolve()), size=p.stat().st_size,
                                    mtime_ns=p.stat().st_mtime_ns) for p in field_paths],
                  trace_provenance='Fresh bidirectional integrations from a measured crossing; same field interpolator. Historical field files are not hash-verified.')
    for phi in args.planes:
        source = root/'data/Poincare'/f'Poincare_{phi:03d}.npy'
        surfaces = np.load(source)
        axis = calc.find_axis(*surfaces[-1].copy(), field)
        for surface in args.surfaces:
            if not 0 <= surface < len(surfaces):
                raise ValueError(f'Surface {surface} outside 0..{len(surfaces)-1}')
            entry = dict(phi_deg=phi, surface=surface, source=str(source.resolve()),
                         source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                         axis_theta_r=axis.tolist())
            report['results'].append(entry)
            theta, radius = surfaces[surface]
            finite = np.isfinite(theta) & np.isfinite(radius)
            theta, radius = theta[finite], radius[finite]
            shifted = axisShift(theta, radius, *axis).T
            subsets, _, iota, rational, _ = calc.find_subsets(shifted, axis)
            entry.update(input_points=len(theta), subsets=len(subsets),
                         iota=float(iota) if np.isfinite(iota) else None)
            try:
                if len(subsets) != 1:
                    raise ValueError(f'Island chain ({len(subsets)} subsets); regular-surface test skipped')
                train_theta, train_radius = subsets[0].T
                fourier = fit_fourier_contour(train_theta, train_radius, modes=args.modes,
                                              regularization=args.regularization,
                                              max_gap=np.deg2rad(args.max_gap_deg))
                spline, _, ier, message = calc.spline_data(train_theta, train_radius, args.smoothing)
                if ier > 0:
                    raise ValueError(f'Spline failed: {message}')
            except ValueError as exc:
                entry.update(status='rejected', reason=str(exc))
                print(f'phi={phi} surface={surface}: {exc}', flush=True)
                (data_dir/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                continue
            evaluate_spline = lambda t: splev(t, spline)
            validation, directions, trace_records = np.empty((0, 3)), np.empty(0, dtype=int), []
            if not args.no_trace:
                entry['status'] = 'tracing'
                (data_dir/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                print(f'Tracing phi={phi} surface={surface} in both directions...', flush=True)
                validation, directions, trace_records = trace_validation(
                    field, [radius[0], theta[0], np.deg2rad(phi)], spins=args.spins,
                    method=args.method, rtol=args.rtol, atol=args.atol, max_step=args.max_step)
            val_theta, val_radius = axisShift(validation[:, 1], validation[:, 0], *axis)
            entry.update(status='compared', fourier_diagnostics=fourier.diagnostics,
                         trace_records=trace_records, metrics={})
            for name, evaluate in [('fourier', fourier), ('spline', evaluate_spline)]:
                entry['metrics'][name] = {
                    'training': residual_metrics(evaluate, train_theta, train_radius, grid),
                    'independent': residual_metrics(evaluate, val_theta, val_radius, grid),
                    'seam_value_jump_m': float(evaluate(2*np.pi)-evaluate(0)),
                    'seam_derivative_jump_m_per_rad': float(
                        fourier(2*np.pi, 1)-fourier(0, 1) if name == 'fourier'
                        else splev(2*np.pi, spline, der=1)-splev(0, spline, der=1))}
                for sign in (1, -1):
                    mask = directions == sign
                    entry['metrics'][name][f'direction_{sign}'] = residual_metrics(
                        evaluate, val_theta[mask], val_radius[mask], grid)
            entry['fit_difference_rms_mm'] = float(np.sqrt(np.mean((fourier(grid)-evaluate_spline(grid))**2))*1000)
            # Fixed sensitivity grid; independent data are never used to fit/select coefficients.
            entry['sensitivity'] = []
            for modes in sorted(set([4, args.modes, 12])):
                for reg in sorted(set([1e-8, args.regularization, 1e-4])):
                    row = dict(modes=modes, regularization=reg)
                    try:
                        trial = fit_fourier_contour(train_theta, train_radius, modes=modes,
                                                    regularization=reg, max_gap=np.deg2rad(args.max_gap_deg))
                        row['independent_radial_rms_mm'] = (float(np.sqrt(np.mean((trial(val_theta)-val_radius)**2))*1000)
                                                           if len(val_theta) else None)
                    except ValueError as exc:
                        row['rejected'] = str(exc)
                    entry['sensitivity'].append(row)
            stem = f'phi{phi:03d}_surface{surface:03d}'
            np.savez(data_dir/f'{stem}.npz', theta_grid=grid, fourier_radius=fourier(grid),
                     spline_radius=evaluate_spline(grid), fourier_coefficients=fourier.coefficients,
                     training_theta=train_theta, training_radius=train_radius,
                     validation_rtp=validation, validation_direction=directions, axis_theta_r=axis,
                     spline_knots=spline[0], spline_coefficients=spline[1], spline_degree=spline[2])
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout='constrained')
            closed = np.r_[grid, 2*np.pi]
            axes[0].scatter(train_radius*np.cos(train_theta), train_radius*np.sin(train_theta), s=3, c='0.7', label='Saved crossings')
            for name, evaluate in [('Spline', evaluate_spline), ('Fourier', fourier)]:
                axes[0].plot(evaluate(closed)*np.cos(closed), evaluate(closed)*np.sin(closed), label=name, lw=1)
                axes[1].plot(np.rad2deg(grid), evaluate(grid)*1000, label=name)
                axes[2].scatter(np.rad2deg(val_theta), (evaluate(val_theta)-val_radius)*1000, s=9, label=name)
            axes[0].scatter(val_radius*np.cos(val_theta), val_radius*np.sin(val_theta), marker='x', s=12, c='k', label='Fresh traces')
            axes[0].set(aspect='equal', xlabel='R - R_axis [m]', ylabel='Z - Z_axis [m]')
            axes[0].legend(fontsize=7)
            axes[1].scatter(np.rad2deg(train_theta), train_radius*1000, s=2, c='0.6')
            axes[1].set(xlabel='Geometric angle about magnetic axis [deg]', ylabel='Radius [mm]')
            axes[2].axhline(0, color='0.5', lw=.5)
            axes[2].set(xlabel='Geometric angle [deg]', ylabel='Fit minus fresh trace radius [mm]')
            axes[2].legend(fontsize=8)
            fig.suptitle(f'{args.analysis}: surface {surface}, phi={phi} deg; M={args.modes}, lambda={args.regularization:g}')
            fig.savefig(plot_dir/f'{stem}.png', dpi=180)
            fig.savefig(plot_dir/f'{stem}.pdf')
            plt.close(fig)
            print(f'{stem}: '+json.dumps({k:v['independent'] for k,v in entry['metrics'].items()}), flush=True)
            (data_dir/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (data_dir/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Report: {data_dir / "metrics.json"}', flush=True)


if __name__ == '__main__':
    main()
