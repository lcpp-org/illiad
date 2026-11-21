"""Scalar contours on adjacent nested geometric surface bands."""
from pathlib import Path as FilePath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.patches import Circle


def validate_options(params):
    quantity = params.get('SURFACE_PARAMETER', 'flux')
    style = params.get('SURFACE_PARAMETER_STYLE', 'lines')
    levels = params.get('SURFACE_PARAMETER_LEVELS', 16)
    if quantity not in ('flux', 'iota', 'iota_normalized', 'blend', 'both', 'all'):
        raise ValueError('SURFACE_PARAMETER must be flux, iota, iota_normalized, blend, both, or all')
    if style not in ('lines', 'filled', 'both'):
        raise ValueError('SURFACE_PARAMETER_STYLE must be lines, filled, or both')
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 2:
        raise ValueError('SURFACE_PARAMETER_LEVELS must be an integer >= 2')
    weight = params.get('IOTA_WEIGHT', .5)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('IOTA_WEIGHT must be a finite number between 0 and 1')
    axis = params.get('IOTA_AXIS_VALUE')
    if axis is not None and (isinstance(axis, bool) or not isinstance(axis, (int, float)) or not np.isfinite(axis)):
        raise ValueError('IOTA_AXIS_VALUE must be null or a finite number')
    cmap = params.get('SURFACE_PARAMETER_CMAP', 'viridis')
    if not isinstance(cmap, str) or cmap not in plt.colormaps():
        raise ValueError('SURFACE_PARAMETER_CMAP must be a registered Matplotlib colormap name')
    return quantity, style, levels


def band_mesh(contours, values):
    """Triangulate only adjacent nested rings of equal island multiplicity.

    Rings are sampled at the same local geometric angles by the calculator.
    Match lobes by their centers, retaining cyclic point correspondence.
    Missing surfaces or topology changes leave holes, never a global Delaunay fill.
    """
    vertices, scalars, triangles = [], [], []
    boundary = next((c for c in contours if c is not None), None)
    if boundary is None or len(boundary) != 1 or not np.isfinite(boundary).all():
        return None, None
    lcfs = Path(np.vstack((boundary[0], boundary[0, 0])))
    for index in range(len(contours)-1):
        outer, inner = contours[index:index+2]
        if outer is None or inner is None or outer.shape != inner.shape:
            continue
        if not np.isfinite(values[index:index+2]).all():
            continue
        used = set()
        for a in outer:
            if not np.isfinite(a).all():
                continue
            distances = np.linalg.norm(inner.mean(axis=1)-a.mean(axis=0), axis=1)
            candidates = np.argsort(distances)
            match = next((int(j) for j in candidates if j not in used), None)
            if match is None:
                continue
            b = inner[match]
            used.add(match)
            if not np.isfinite(b).all():
                continue
            pa, pb = Path(np.vstack((a, a[0]))), Path(np.vstack((b, b[0])))
            if not (pa.contains_points(b, radius=1e-10).all()
                    or pb.contains_points(a, radius=1e-10).all()):
                continue
            # Island validity in the calculator does not test LCFS containment.
            # Reject the whole band if either fitted ring protrudes outside it.
            if not (lcfs.contains_points(a, radius=1e-9).all()
                    and lcfs.contains_points(b, radius=1e-9).all()):
                continue
            n, start = len(a), len(vertices)
            vertices.extend(a)
            vertices.extend(b)
            scalars.extend([values[index]]*n + [values[index+1]]*n)
            for k in range(n):
                nxt = (k+1) % n
                triangles.extend([(start+k, start+nxt, start+n+k),
                                  (start+nxt, start+n+nxt, start+n+k)])
    if not triangles:
        return None, None
    xy = np.asarray(vertices)
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles)
    # Exclude degenerate triangles (e.g. coincident surfaces).
    points = xy[tri.triangles]
    u, v = points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]
    tri.set_mask(np.abs(u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]) < 1e-16)
    if tri.mask.all():
        return None, None
    return tri, np.asarray(scalars)


def normalize_iota(iota, lcfs_index, axis_value=None):
    """Endpoint normalization per plane; retain raw overshoot and reference indices.

    The last row is the axis trace, not a usable winding measurement. Estimate
    its iota using the innermost finite non-axis surface unless explicitly given.
    Undefined (missing or equal) endpoint references leave the plane NaN.
    """
    iota = np.asarray(iota, dtype=float)
    raw = np.full_like(iota, np.nan)
    references = np.full(iota.shape[1], np.nan)
    indices = np.full(iota.shape[1], -1, dtype=int)
    for plane in range(iota.shape[1]):
        edge = iota[lcfs_index, plane]
        if axis_value is None:
            ids = np.flatnonzero(np.isfinite(iota[lcfs_index+1:-1, plane]))+lcfs_index+1
            if not len(ids):
                continue
            indices[plane] = ids[-1]
            axis = iota[ids[-1], plane]
        else:
            axis = axis_value
        references[plane] = axis
        if not np.isfinite(edge) or np.isclose(axis, edge, rtol=1e-10, atol=1e-12):
            continue
        raw[lcfs_index:, plane] = (iota[lcfs_index:, plane]-edge)/(axis-edge)
        raw[:lcfs_index+1, plane] = 0.
        raw[-1, plane] = 1.  # Axis boundary condition, not an inferred winding.
    return np.clip(raw, 0., 1.), raw, references, indices


def blend_profiles(flux, normalized_iota, weight):
    # Exact endpoints must not be contaminated by NaNs in the unused component.
    if weight == 0:
        return np.clip(flux, 0., 1.)
    if weight == 1:
        return normalized_iota.copy()
    return (1-weight)*np.clip(flux, 0., 1.) + weight*normalized_iota


def prepare_profiles(calc):
    options = calc.surface_parameter_options
    normalized, raw, references, indices = normalize_iota(
        calc.iota_array, calc.lcfs_index, options.get('IOTA_AXIS_VALUE'))
    weight = options.get('IOTA_WEIGHT', .5)
    blended = blend_profiles(calc.total_flux_norm, normalized, weight)
    for plane, phi in enumerate(calc.phi_gens):
        if not np.isfinite(normalized[:, plane]).any():
            calc.simIO.log.warning('Normalized iota undefined at phi=%g: missing or equal endpoint references', phi)
        overshoot = np.isfinite(raw[:, plane]) & ((raw[:, plane] < 0) | (raw[:, plane] > 1))
        if overshoot.any():
            calc.simIO.log.warning('Normalized iota at phi=%g: clipped %d out-of-range values', phi, overshoot.sum())
    directory = FilePath(calc.simIO._outputPath(calc.simIO.data_dir, calc.anlys_subdir+'/surface_parameter'))
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory/'profiles.npz', surface_index=np.arange(len(normalized)),
             phi_deg=calc.phi_gens, flux=calc.total_flux_norm, iota=calc.iota_array,
             iota_normalized=normalized, iota_normalized_unclipped=raw, blend=blended,
             iota_weight=weight, iota_axis_reference=references, iota_axis_surface_index=indices,
             lcfs_index=calc.lcfs_index)
    if options.get('PLOT_PARAMETER_PROFILES', False):
        ids = np.arange(calc.lcfs_index, len(normalized))
        for plane, phi in enumerate(calc.phi_gens):
            fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
            for values, label, style in [(calc.total_flux_norm, 'Flux surface parameter', '-'),
                                          (normalized, 'Normalized iota', '--'),
                                          (blended, f'Blend (iota weight={weight:g})', ':')]:
                ax.plot(ids, values[ids, plane], style, label=label)
            ax.set(xlabel='Surface index', ylabel='Normalized parameter', ylim=(-.03, 1.03),
                   title=fr'Surface profiles, $\phi={phi:g}^\circ$')
            ax.grid(alpha=.3)
            ax.legend()
            reference = (f'Axis iota estimated from surface {indices[plane]}' if indices[plane] >= 0
                         else 'Explicit axis iota reference')
            ax.text(.02, .02, reference, transform=ax.transAxes, fontsize=8)
            dest = f'{calc.anlys_subdir}/surface_parameter/Profiles_{phi:03.0f}'
            calc.simIO.saveFig(dest+'.png', dpi=220)
            calc.simIO.saveFig(dest+'.pdf')
            plt.close(fig)
    return normalized, blended


def plot_parameters(calc):
    quantity, style, nlevels = validate_options(calc.surface_parameter_options)
    quantities = ('flux', 'iota') if quantity == 'both' else (quantity,)
    if quantity == 'all':
        quantities = ('flux', 'iota', 'iota_normalized', 'blend')
    if not calc.surface_parameter_options.get('PLOT_SURFACE_PARAMETER', False):
        quantities = ()
    normalized, blended = prepare_profiles(calc)
    arrays = dict(flux=calc.total_flux_norm, iota=calc.iota_array,
                  iota_normalized=normalized, blend=blended)
    weight = calc.surface_parameter_options.get('IOTA_WEIGHT', .5)
    labels = dict(flux=r'$1-\Psi_s/\Psi_{\mathrm{LCFS}}$', iota=r'$\iota$',
                  iota_normalized=r'$\hat\iota$', blend=fr'$(1-w)\hat\Psi+w\hat\iota$, $w={weight:g}$')
    styles = ('lines', 'filled') if style == 'both' else (style,)
    for name in quantities:
        values = arrays[name]
        finite = values[calc.lcfs_index:][np.isfinite(values[calc.lcfs_index:])]
        if not len(finite):
            calc.simIO.log.warning('No finite values for %s surface-parameter plots', name)
            continue
        low, high = (0., 1.) if name != 'iota' else (float(finite.min()), float(finite.max()))
        if low == high:
            low, high = low-1e-6, high+1e-6
        levels = np.linspace(low, high, nlevels)
        label = labels[name]
        for plane, phi in enumerate(calc.phi_gens):
            tri, z = band_mesh(calc.parameter_contours[plane], values[:, plane])
            for mode in styles:
                fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
                if tri is not None:
                    draw = ax.tricontour if mode == 'lines' else ax.tricontourf
                    artist = draw(tri, z, levels=levels, cmap=calc.surface_parameter_options.get('SURFACE_PARAMETER_CMAP', 'viridis'), vmin=low, vmax=high)
                    fig.colorbar(artist, ax=ax, label=label, shrink=.75)
                    if mode == 'lines':
                        ax.clabel(artist, inline=True, fontsize=7, fmt='%.3g')
                else:
                    ax.text(.5, .5, 'No adjacent valid nested surface bands',
                            transform=ax.transAxes, ha='center')
                radius = calc.field.a
                ax.add_patch(Circle((0, 0), radius, fill=False, color='black', lw=1.2))
                ax.set(aspect='equal', xlim=(-1.05*radius, 1.05*radius),
                       ylim=(-1.05*radius, 1.05*radius), xlabel=r'$R-R_0$ [m]', ylabel='$Z$ [m]',
                       title=fr'{label}, $\phi={phi:g}^\circ$')
                dest = f'{calc.anlys_subdir}/surface_parameter/{name}_{mode}_{phi:03.0f}'
                calc.simIO.saveFig(dest+'.png', dpi=220)
                calc.simIO.saveFig(dest+'.pdf')
                plt.close(fig)
