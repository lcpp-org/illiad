"""Predebon-inspired return-map coordinates for diagnostic cross-section plots.

Equations (4) and (6), PPCF 60 (2018) 045003, with a Fourier phase-coherence
refinement of the return frequency and a q-return extension for island chains.
No flux integration, metric calculation, or separatrix interpolation is done.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

TAU = 2*np.pi


def basis(angle, modes):
    angle = np.asarray(angle)
    phase = angle[..., None]*np.arange(1, modes+1)
    return np.concatenate((np.ones(angle.shape+(1,)), np.cos(phase), np.sin(phase)), axis=-1)


def fit_xy(angle, xy, modes, regularization):
    design = basis(angle, modes)
    penalty = np.sqrt(len(angle)*regularization)*np.r_[0., np.arange(1, modes+1)**2,
                                                     np.arange(1, modes+1)**2]
    return np.linalg.lstsq(np.vstack((design, np.diag(penalty))),
                          np.vstack((xy, np.zeros((2*modes+1, 2)))), rcond=None)[0]


def return_frequency(xy):
    """Eq. (4): average inverse intervals between near-reference minima.

    Work in return-index units (one index is q toroidal turns for a q-lobe
    chain). The reference is a sampled outboard extremum. Distance minima
    far from that reference, including concave-island inboard minima, are
    excluded rather than blindly taking every other minimum.
    """
    reference = xy[np.argmax(xy[:, 0])]
    distance2 = np.sum((xy-reference)**2, axis=1)
    interpolant = CubicSpline(np.arange(len(xy)), distance2)
    roots = interpolant.derivative().roots(extrapolate=False)
    minima = roots[(interpolant(roots, 2) > 0) & (interpolant(roots) < .15*distance2.max())]
    # Cubic interpolation of a flat distance minimum can introduce a pair of
    # extrema within one sample interval. They are not two resolved returns.
    clusters = np.split(minima, np.flatnonzero(np.diff(minima) >= 1.)+1)
    minima = np.array([group[np.argmin(interpolant(group))] for group in clusters if len(group)])
    if len(minima) < 5:
        raise ValueError('Fewer than five resolved near-reference returns')
    intervals = np.diff(minima)
    return float(np.mean(1/intervals)), minima


def reconstruct(xy, *, modes=8, regularization=1e-8, tolerance=.03,
                phase_offset=0., frequency=None):
    """Fit a periodic XY curve against an affine return-index phase.

    Requires ordered, consecutive, single-direction crossings. Frequencies
    above the return-map Nyquist limit cannot be inferred from these data.
    The last 30% of crossings validate a fit learned on the first 70%.
    tolerance is an RMS position error divided by the cross-section diameter.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('Expected finite ordered XY crossings')
    count = len(xy)
    if count < max(80, 6*(2*modes+1)):
        raise ValueError('Too few ordered crossings for frequency estimation and validation')
    diameter = float(np.linalg.norm(np.ptp(xy, axis=0)))
    if diameter < 1e-6:
        raise ValueError('Axis or degenerate contour has no resolved poloidal angle')
    coarse, minima, return_note = None, np.empty(0), 'Shared frequency; return estimate not needed'
    if frequency is None:
        try:
            coarse, minima = return_frequency(xy)
            return_note = 'Return estimate is diagnostic only'
        except ValueError as exc:
            return_note = f'Return estimate unavailable: {exc}'
    # Eq. (3) supplies only the initial frequency basin, never the final angle.
    centered = xy-(xy.min(axis=0)+xy.max(axis=0))/2
    geometric = np.unwrap(np.arctan2(centered[:, 1], centered[:, 0]))
    initial = abs(np.polyfit(np.arange(count), geometric, 1)[0])/TAU
    if frequency is None and (not 0 < initial < .48 or initial*count < 4):
        raise ValueError('Unresolved, aliased, or insufficient local poloidal turns')
    if coarse is not None and abs(coarse-initial) > max(.15*initial, 2/count):
        return_note = 'Return minima and winding estimates disagree; diagnostic gate bypassed'
    split = int(.7*count)
    indices = np.arange(count, dtype=float)

    def refine(n):
        t, data = indices[:n], xy[:n]
        def objective(f):
            angle = TAU*f*t
            coeff = fit_xy(angle, data, modes, regularization)
            return float(np.mean((basis(angle, modes)@coeff-data)**2))
        candidates = np.linspace(max(1e-8, initial-1/count), min(.499, initial+1/count), 17)
        values = [objective(f) for f in candidates]
        best = int(np.argmin(values))
        if best in (0, len(candidates)-1):
            raise ValueError('Frequency refinement reached search boundary')
        fit = minimize_scalar(objective, bounds=(candidates[best-1], candidates[best+1]),
                              method='bounded', options={'xatol': 1e-12})
        return float(fit.x)

    train_frequency = refine(split) if frequency is None else frequency
    train_coeff = fit_xy(TAU*train_frequency*indices[:split], xy[:split], modes, regularization)
    prediction = basis(TAU*train_frequency*indices[split:], modes)@train_coeff
    validation_rms = float(np.sqrt(np.mean(np.sum((prediction-xy[split:])**2, axis=1))))
    if validation_rms/diameter > tolerance:
        raise ValueError(f'Held-out curve error {100*validation_rms/diameter:.2f}% exceeds {100*tolerance:g}%')
    frequency = refine(count) if frequency is None else frequency
    phase_drift = TAU*abs(frequency-train_frequency)*count
    if phase_drift > .15:
        raise ValueError(f'Frequency unstable: accumulated phase drift {phase_drift:.3f} rad')
    angles = (TAU*frequency*indices+phase_offset) % TAU
    gaps = np.diff(np.r_[np.sort(angles), np.min(angles)+TAU])
    if np.max(gaps) > np.pi/2:
        raise ValueError('Straight-angle coverage has a gap larger than 90 degrees')
    coeff = fit_xy(angles, xy, modes, regularization)
    residual = basis(angles, modes)@coeff-xy
    diagnostics = dict(frequency_per_return=frequency, coarse_eq4_frequency=coarse,
                       return_frequency_note=return_note,
                       return_minima=minima.tolist(), sample_count=count,
                       validation_rms_m=validation_rms, diameter_m=diameter,
                       phase_drift_rad=phase_drift, max_gap_deg=float(np.rad2deg(gaps.max())),
                       fit_rms_m=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))
    return coeff, diagnostics


class StraightAnglePlots:
    """One shared frequency per surface/return multiplicity across phi planes."""

    def __init__(self, calculator, params):
        from illiad.sol.tracer import load_poincare_settings
        self.calc = calculator
        self.n_angles = params.get('SL_NTHETA', 16)
        self.modes = params.get('SL_MODES', 8)
        self.regularization = params.get('SL_REGULARIZATION', 1e-8)
        self.tolerance = params.get('SL_MAX_REL_ERROR', .03)
        self.max_points = params.get('SL_MAX_POINTS', None)
        if self.max_points is not None and (isinstance(self.max_points, bool)
                or not isinstance(self.max_points, int) or self.max_points < 1):
            raise ValueError('SL_MAX_POINTS must be null (uncapped) or a positive integer')
        self.direction = params.get('SL_TOROIDAL_DIRECTION', 1)
        for name, value in [('SL_NTHETA', self.n_angles), ('SL_MODES', self.modes)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.direction not in (-1, 1):
            raise ValueError('SL_TOROIDAL_DIRECTION must be +1 or -1')
        if not np.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError('SL_REGULARIZATION must be finite and nonnegative')
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError('SL_MAX_REL_ERROR must be finite and positive')
        settings = load_poincare_settings(params['ANLYS_DIR'])
        if settings.get('DOUBLE_LINE') is not False:
            raise ValueError('Straight-angle plots require logged DOUBLE_LINE=False; merged direction ordering is ambiguous')
        self.launch_phi = float(settings['IC_PHI_DEG'])
        self.frequencies = {}
        self.phase_origins = {}
        self.output = params['ANLYS_SUBDIR']+'/straight_angle'
        options = {k: v for k, v in params.items() if k.startswith('SL_')}
        options.update(SL_MAX_REL_ERROR=self.tolerance, SL_MAX_POINTS=self.max_points)
        self.report = dict(method='Predebon equations 4/6, Fourier refinement, q-return island extension',
                           options=options,
                           launch_phi_deg=self.launch_phi, planes=[])

    def plot_plane(self, phi, surfaces, mag_axis, subset_counts):
        calc = self.calc
        levels = np.linspace(0, TAU, self.n_angles, endpoint=False)
        grid = np.linspace(0, TAU, max(512, 32*self.modes))
        fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
        vessel_radius = getattr(getattr(calc, 'field', None), 'a', .19)
        ax.add_patch(Circle((0, 0), vessel_radius, fill=False, edgecolor='black', lw=1.3))
        # A geometric fit does not require a valid straight-angle map. Reuse
        # the production reconstruction, including its island-center shifts.
        geometric_count = 0
        if hasattr(calc, 'fit_and_integrate_surface_subsets'):
            smallest, counts, subsets, centers, iota = calc.identify_surface_subsets(surfaces, mag_axis)
            reverse_axis = np.array(mag_axis, copy=True)
            reverse_axis[0] += np.pi
            for surface in range(calc.lcfs_index, len(surfaces)):
                geometry = calc.fit_and_integrate_surface_subsets(
                    subsets, centers, surface, smallest, int(counts[surface]), iota[-1],
                    reverse_axis, phi, integrate=False)[0]
                for contour in geometry:
                    if not np.isfinite(contour).all() or np.any(contour[:, 1] < 0):
                        continue
                    xy = contour[:, 1, None]*np.column_stack((np.cos(contour[:, 0]), np.sin(contour[:, 0])))
                    xy = np.vstack((xy, xy[0]))
                    ax.plot(xy[:, 0], xy[:, 1], color='0.75', lw=.65, zorder=1)
                    geometric_count += 1
        report = dict(phi_deg=float(phi), surfaces=[])
        report['geometric_contours'] = geometric_count
        report['vessel_radius_m'] = float(vessel_radius)
        self.report['planes'].append(report)
        lines = {}
        records = []
        delta = (self.direction*(phi-self.launch_phi)) % 360 / 360
        # Production extraction excludes a launch-plane crossing at t=0.
        if np.isclose(delta, 0):
            delta = 1.
        for surface in range(calc.lcfs_index, len(surfaces)-1):
            theta, radius = surfaces[surface]
            finite = np.isfinite(theta) & np.isfinite(radius)
            ids = np.flatnonzero(finite)
            q = int(subset_counts[surface])
            if not len(ids):
                continue
            if not np.array_equal(ids, np.arange(len(ids))):
                report['surfaces'].append(dict(surface=surface, status='rejected', reason='Missing crossing indices'))
                continue
            xy = np.column_stack((radius[finite]*np.cos(theta[finite]), radius[finite]*np.sin(theta[finite])))
            # Keep consecutive crossings before splitting q-return lobes,
            # preserving original transit offsets while shortening history.
            xy = xy[:self.max_points]
            # Match lobes spatially across neighbouring surfaces within this section.
            groups = [xy[offset::q] for offset in range(q)]
            if any(len(g) == 0 for g in groups):
                report['surfaces'].append(dict(surface=surface, status='rejected',
                                              reason='Too few capped crossings to populate island lobes'))
                continue
            centers = np.array([np.mean(g, axis=0) for g in groups])
            axis_xy = mag_axis[1]*np.array([np.cos(mag_axis[0]), np.sin(mag_axis[0])])
            order = np.argsort(np.arctan2(centers[:, 1]-axis_xy[1], centers[:, 0]-axis_xy[0]))
            for lobe, offset in enumerate(order):
                data = groups[offset]
                row = dict(surface=surface, subsets=q, lobe=lobe, crossing_offset=int(offset))
                report['surfaces'].append(row)
                key = (surface, q)
                try:
                    # First accepted lobe establishes frequency, then all lobes/planes
                    # are tested against that same return frequency.
                    frequency = self.frequencies.get(key)
                    coeff, diag = reconstruct(data, modes=self.modes, regularization=self.regularization,
                                               tolerance=self.tolerance, frequency=frequency)
                    frequency = diag['frequency_per_return']
                    # Eq. (6), using actual original crossing offset and launch phase.
                    phase_offset = TAU*frequency*(offset+delta)/q
                    coeff = fit_xy((TAU*frequency*np.arange(len(data))+phase_offset) % TAU,
                                   data, self.modes, self.regularization)
                    gauge_key = (surface, q, int(offset))
                    if gauge_key not in self.phase_origins:
                        # Surface-dependent additive phase is permitted; a
                        # separately chosen phase on every phi plane is not.
                        sample_angles = np.linspace(0, TAU, 2048, endpoint=False)
                        maximum = sample_angles[np.argmax((basis(sample_angles, self.modes)@coeff)[:, 0])]
                        optimum = minimize_scalar(lambda a: -float((basis(a, self.modes)@coeff)[0]),
                                                  bounds=(maximum-TAU/2048, maximum+TAU/2048), method='bounded')
                        self.phase_origins[gauge_key] = (float(optimum.x % TAU), float(phi))
                    origin, reference_phi = self.phase_origins[gauge_key]
                    coeff = fit_xy((TAU*frequency*np.arange(len(data))+phase_offset-origin) % TAU,
                                   data, self.modes, self.regularization)
                    self.frequencies[key] = frequency
                except ValueError as exc:
                    row.update(status='rejected', reason=str(exc))
                    continue
                row.update(status='accepted', iota_sl=self.direction*frequency/q,
                           phase_origin_rad=origin, reference_phi_deg=reference_phi, **diag)
                curve = basis(grid, self.modes)@coeff
                points = basis(levels, self.modes)@coeff
                ax.plot(curve[:, 0], curve[:, 1], color='0.45', lw=.65)
                lines.setdefault((q, lobe), {})[surface] = (points, curve, int(offset))
                records.append((surface, q, lobe, coeff, points))
        # Missing surfaces and changes of island multiplicity break radial lines.
        for (q, lobe), rows in lines.items():
            for level in range(self.n_angles):
                # Explicit separator rows allow breaking a connection without
                # dropping either endpoint. Only adjacent nested sections join.
                points = np.full((2*len(surfaces), 2), np.nan)
                for surface, (values, curve, offset) in rows.items():
                    points[2*surface] = values[level]
                    if surface-1 in rows:
                        previous, previous_curve, previous_offset = rows[surface-1]
                        nested = (PolygonPath(curve).contains_points(previous_curve[:-1], radius=1e-8).all()
                                  or PolygonPath(previous_curve).contains_points(curve[:-1], radius=1e-8).all())
                        if nested and offset == previous_offset:
                            points[2*surface-1] = (previous[level]+values[level])/2
                ax.plot(points[:, 0], points[:, 1], color=plt.cm.hsv(level/self.n_angles),
                        lw=.9, marker='.', ms=2)
        accepted = len(records)
        rejected = sum(row['status']=='rejected' for row in report['surfaces'])
        ax.set(aspect='equal', xlabel=r'$R-R_0$ [m]', ylabel='$Z$ [m]',
               xlim=(-1.05*vessel_radius, 1.05*vessel_radius),
               ylim=(-1.05*vessel_radius, 1.05*vessel_radius),
               title=fr'Fourier contours and $\theta_{{sl}}$ coordinate lines, $\phi={phi:g}^\circ$'
                     f'\n{geometric_count} geometric contours; angle maps: {accepted} accepted / {rejected} rejected')
        ax.legend(handles=[Line2D([], [], color='0.75', label='Geometric fit'),
                           Line2D([], [], color='0.45', label='Accepted straight-angle fit'),
                           Line2D([], [], color='black', label='Vacuum vessel')], fontsize=8, loc='upper right')
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 360), cmap='hsv'), ax=ax, shrink=.7)
        colorbar.set_label(r'$\theta_{sl}$ [deg]')
        if not accepted:
            ax.text(.5, .5, 'No reliable straight-angle reconstruction', transform=ax.transAxes, ha='center')
        calc.simIO.saveFig(f'{self.output}/Coordinates_{phi:03.0f}.png', dpi=220)
        calc.simIO.saveFig(f'{self.output}/Coordinates_{phi:03.0f}.pdf')
        plt.close(fig)
        directory = Path(calc.simIO._outputPath(calc.simIO.data_dir, self.output))
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory/f'Coordinates_{phi:03.0f}.npz', theta_sl=levels,
                 surface_lobe=np.asarray([r[:3] for r in records], dtype=int).reshape(-1, 3),
                 coefficients=np.asarray([r[3] for r in records]).reshape(-1, 2*self.modes+1, 2),
                 coordinate_points=np.asarray([r[4] for r in records]).reshape(-1, self.n_angles, 2))
        (directory/'diagnostics.json').write_text(json.dumps(self.report, indent=2, allow_nan=False)+'\n')
        calc.simIO.log.info('Straight-angle phi=%g: %d accepted, %d rejected', phi, accepted, rejected)
