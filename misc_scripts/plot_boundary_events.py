#!/usr/bin/env python3
"""Plot six face maps and a 3D geometry/event preview from boundary shards."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from illiad.boundaries import FACE_NAMES, iter_events, parse_boundaries


def box_corners(box):
    local = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])*box.size/2
    return local @ box.rotation.T + box.center


def north_up_xy(xyz):
    """Top-down projection with computational phi=162 degrees pointing up."""
    from illiad.utilities.coordtrans import RZphi_to_XYZ
    north = RZphi_to_XYZ([1., 0., 162.])[:2]
    right = np.array([north[1], -north[0]])
    return np.asarray(xyz)[..., :2] @ np.column_stack((right, north))


def face_net_coordinates(face, u, v, size):
    """Unfold outward-facing UV maps about +z, with -z beyond +y.

    +z keeps x=u, y=v. Each neighbor is rotated about its actual shared
    physical edge, so an edge's points coincide in both unfolded maps.
    """
    width, depth, height = size
    if face == 0:  # -x: u=y, v=-z
        return -width/2-height/2-v, u
    if face == 1:  # +x: u=y, v=z
        return width/2+height/2-v, u
    if face == 2:  # -y: u=z, v=-x
        return -v, -depth/2-height/2+u
    if face == 3:  # +y: u=z, v=x
        return v, depth/2+height/2-u
    if face == 4:  # -z: u=x, v=-y, attached to +y's outer edge
        return u, depth+height+v
    if face == 5:  # +z: u=x, v=y
        return u, v
    raise ValueError('face must be in 0..5')


def plot_boundary_preview(boxes, output, *, preview=None, vessel=None, preview_padding=1.25):
    """Box-focused XYZ view with poloidal and north-up toroidal locators."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import ConvexHull
    from illiad.utilities.coordtrans import RZphi_to_XYZ, XYZ_to_RZphi

    if not boxes:
        raise ValueError('At least one box is required for a geometry preview')
    if not np.isfinite(preview_padding) or preview_padding < 1.05:
        raise ValueError('preview_padding must be finite and at least 1.05')
    assumed = vessel is None
    vessel = dict(major_radius_m=.72, minor_radius_m=.19) if vessel is None else vessel
    R0, a = float(vessel['major_radius_m']), float(vessel['minor_radius_m'])
    if not np.isfinite([R0, a]).all() or not R0 > a > 0:
        raise ValueError('Vessel radii must satisfy major_radius > minor_radius > 0')
    corners = [box_corners(b) for b in boxes]
    xyz = np.concatenate(corners)
    extent = float(np.ptp(xyz, axis=0).max())
    focus = [xyz]
    for box in boxes:
        R, Z, phi = XYZ_to_RZphi(box.center)
        theta = np.arctan2(Z, R-R0)
        nearest_wall = RZphi_to_XYZ([R0+a*np.cos(theta), a*np.sin(theta), phi])
        # Include a nearby wall, but do not zoom out to a remote wall for a
        # tiny box in the core: the locator supplies that larger context.
        if np.linalg.norm(nearest_wall-box.center) <= .75*extent:
            focus.append(nearest_wall[None])
    focus = np.concatenate(focus)
    center = (focus.min(axis=0)+focus.max(axis=0))/2
    spans = np.maximum(np.ptp(focus, axis=0), .35*extent)*preview_padding
    lower, upper = center-spans/2, center+spans/2
    halfspan = spans.max()/2
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_axes([.02, .08, .79, .82], projection='3d')
    faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    for bi, (box, vertices) in enumerate(zip(boxes, corners)):
        # Draw only the nearby part of the torus. Sampling and explicit masking
        # avoid matplotlib versions that do not clip 3D paths to axis limits.
        R, Z, phi_deg = XYZ_to_RZphi(box.center)
        theta0 = np.arctan2(Z, R-R0)
        theta_half = min(np.pi, 2*halfspan/a)
        phi_half = np.rad2deg(min(np.pi/3, 2*halfspan/max(R, a)))
        theta = np.linspace(theta0-theta_half, theta0+theta_half, 121)
        phi = np.linspace(phi_deg-phi_half, phi_deg+phi_half, 121)
        def wall_curve(t, ph, **style):
            t, ph = np.broadcast_arrays(t, ph)
            line = RZphi_to_XYZ(np.stack((R0+a*np.cos(t), a*np.sin(t), ph), axis=-1))
            line[np.any((line < lower) | (line > upper), axis=1)] = np.nan
            ax.plot(*line.T, **style)
        for ph in np.linspace(phi[0], phi[-1], 7):
            wall_curve(theta, ph, color='#7696ab', alpha=.45, lw=.8)
        for t in np.linspace(theta[0], theta[-1], 7):
            wall_curve(t, phi, color='#7696ab', alpha=.45, lw=.8)
        wall_curve(theta, phi_deg, color='#315c79', alpha=.85, lw=1.7,
                   label='Vessel wall at box φ' if bi == 0 else None)
        ax.add_collection3d(Poly3DCollection([vertices[f] for f in faces],
                                             facecolors='#db9d4f', alpha=.14, edgecolors='#674a29', linewidths=1.1))
        ax.text(*box.center, box.name, fontsize=9)
        for axis, color in enumerate(('r', 'g', 'b')):
            vec = box.rotation[:, axis]*box.size[axis]/2
            ax.quiver(*box.center, *vec, color=color, linewidth=1.6)
        if preview is not None:
            hits = preview['xyz_m'][preview['boundary_id'] == bi]
            if len(hits):
                ax.scatter(*hits.T, s=3, color='#af471b', alpha=.65)
    # Box extent and only a nearby wall determine the main view, never a full torus.
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]),
           xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
    ax.set_box_aspect(spans)
    primary_phi = XYZ_to_RZphi(boxes[0].center)[2]
    ax.view_init(elev=25, azim=-float(primary_phi)-55)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    ax.legend(loc='lower left', fontsize=9, frameon=False)
    locator = fig.add_axes([.78, .70, .20, .24])
    theta = np.linspace(0, 2*np.pi, 241)
    locator.plot(R0+a*np.cos(theta), a*np.sin(theta), color='#315c79', lw=1.4)
    locator.plot(R0, 0., '+', color='#7696ab')
    ph = np.deg2rad(primary_phi)
    radial = np.array([np.cos(ph), -np.sin(ph), 0.])
    projection = np.column_stack((corners[0] @ radial, corners[0][:, 2]))
    hull = ConvexHull(projection)
    outline = projection[np.r_[hull.vertices, hull.vertices[0]]]
    locator.fill(*outline.T, color='#db9d4f', alpha=.3)
    locator.plot(*outline.T, color='#674a29', lw=1.2, label='Box projection')
    locator.set(xlabel='R (m)', ylabel='Z (m)', aspect='equal',
                title=f'R–Z locator · φ={primary_phi:g}°')
    locator.set_xlim(min(R0-a, projection[:, 0].min())-.025*a, max(R0+a, projection[:, 0].max())+.025*a)
    locator.set_ylim(min(-a, projection[:, 1].min())-.025*a, max(a, projection[:, 1].max())+.025*a)
    locator.tick_params(labelsize=8)
    locator.xaxis.set_major_locator(MaxNLocator(3))
    locator.yaxis.set_major_locator(MaxNLocator(3))
    locator.set_title(f'R–Z locator · φ={primary_phi:g}°', fontsize=10)
    locator.legend(loc='upper right', fontsize=7, frameon=False)
    top = fig.add_axes([.78, .18, .20, .24])
    outer = R0+a
    top.add_patch(Wedge((0, 0), outer, 0, 360, width=2*a,
                        facecolor='#7696ab', alpha=.12, edgecolor='none'))
    for radius in (R0-a, outer):
        top.plot(radius*np.cos(theta), radius*np.sin(theta), color='#315c79', lw=1.2)
    top.plot(R0*np.cos(theta), R0*np.sin(theta), '--', color='#7696ab', lw=.7)
    top.plot(0., 0., '+', color='#7696ab', markersize=5)
    top.plot([0., 0.], [R0-a, 1.08*outer], color='#315c79', lw=1.4)
    top.text(0., 1.13*outer, 'N · φ=162°', ha='center', va='bottom', fontsize=8)
    for box, vertices in zip(boxes, corners):
        footprint = north_up_xy(vertices)
        hull = ConvexHull(footprint)
        outline = footprint[np.r_[hull.vertices, hull.vertices[0]]]
        top.fill(*outline.T, color='#db9d4f', alpha=.5)
        top.plot(*outline.T, color='#674a29', lw=1.)
        xy = north_up_xy(box.center)
        top.plot([0., xy[0]], [0., xy[1]], ':', color='#674a29', lw=.8)
        top.plot(*xy, 'o', color='#af471b', markersize=3)
    xy = north_up_xy(boxes[0].center)
    top.annotate(f'φ={primary_phi:g}°', xy, xytext=(7, -10),
                 textcoords='offset points', fontsize=8, color='#674a29')
    limit = max(outer, np.abs(north_up_xy(xyz)).max())*1.3
    top.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal')
    top.set_title('Toroidal locator (top-down)', fontsize=10, pad=6)
    top.set_axis_off()
    count = 0 if preview is None else len(preview)
    fig.suptitle(f'Boundary geometry · {count} preview events', fontsize=15)
    fig.text(.5, .025, f'Local axes: RGB = XYZ   |   Vessel R₀={R0:g} m, a={a:g} m'
             + (' (default reference)' if assumed else ''), ha='center', fontsize=10)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output/'boundary_preview.png', dpi=180)
    plt.close(fig)



def plot_events(directory, output, *, bins=40, metric='count', kind='all', preview_limit=10000,
                vessel=None, preview_padding=1.25):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    directory, output = Path(directory), Path(output)
    if bins < 1 or preview_limit < 0:
        raise ValueError('bins must be positive; preview_limit must be nonnegative')
    meta = json.loads((directory/'manifest.json').read_text())
    boxes = parse_boundaries(meta['boxes'])
    output.mkdir(parents=True, exist_ok=True)
    shape = (len(boxes), 6, bins, bins)
    values, counts = np.zeros(shape), np.zeros(shape)
    seen = set()
    previews = []
    remaining = preview_limit
    for events in iter_events(directory):
        if kind != 'all':
            events = events[events['kind'] == {'entry': -1, 'exit': 1, 'contact': 0}[kind]]
        if remaining:
            previews.append(events[:remaining])
            remaining -= min(remaining, len(events))
        for bi, box in enumerate(boxes):
            for face in range(6):
                rows = events[(events['boundary_id'] == bi) & (events['face_id'] == face)]
                axis = face//2
                width, height = box.size[(axis+1) % 3], box.size[(axis+2) % 3]
                iu = np.clip(((rows['uv_m'][:, 0]/width+.5)*bins).astype(int), 0, bins-1)
                iv = np.clip(((rows['uv_m'][:, 1]/height+.5)*bins).astype(int), 0, bins-1)
                weight = np.ones(len(rows))
                if metric == 'energy':
                    weight = rows['energy_eV']
                elif metric == 'angle':
                    vector = rows['vector_xyz']
                    speed = np.linalg.norm(vector, axis=1)
                    cos_angle = np.abs(np.sum(vector*rows['normal_xyz'], axis=1))/np.maximum(speed, 1e-300)
                    weight = np.rad2deg(np.arccos(np.clip(cos_angle, 0, 1)))
                    weight[speed == 0] = np.nan
                elif metric == 'unique':
                    for j, row in enumerate(rows):
                        key = (bi, face, int(iu[j]), int(iv[j]), int(row['trajectory_id']))
                        weight[j] = key not in seen
                        seen.add(key)
                valid = np.isfinite(weight)
                np.add.at(values[bi, face], (iu[valid], iv[valid]), weight[valid])
                np.add.at(counts[bi, face], (iu[valid], iv[valid]), 1)
    if metric in ('energy', 'angle'):
        values = np.divide(values, counts, out=np.full_like(values, np.nan), where=counts > 0)
    elif metric == 'area':
        for bi, box in enumerate(boxes):
            for face in range(6):
                axis = face//2
                values[bi, face] /= box.size[(axis+1) % 3]*box.size[(axis+2) % 3]/bins**2
    preview = np.concatenate(previews) if previews else None
    label = {'count': 'Crossings', 'area': 'Crossings / m²', 'unique': 'Unique trajectories per bin',
             'energy': 'Mean kinetic energy (eV)', 'angle': 'Mean angle from normal (deg)'}[metric]
    for bi, box in enumerate(boxes):
        fig, ax = plt.subplots(figsize=(9, 10), constrained_layout=True)
        finite = values[bi][np.isfinite(values[bi])]
        vmax = max(float(finite.max()), 1e-30) if finite.size else 1.
        norm = Normalize(0, vmax)
        for face in range(6):
            axis = face//2
            width, height = box.size[(axis+1) % 3], box.size[(axis+2) % 3]
            u = np.linspace(-width/2, width/2, bins+1)
            v = np.linspace(-height/2, height/2, bins+1)
            U, V = np.meshgrid(u, v, indexing='ij')
            X, Y = face_net_coordinates(face, U, V, box.size)
            plot = ax.pcolormesh(X, Y, values[bi, face], norm=norm, cmap='viridis', shading='flat')
            edge_u = np.array([u[0], u[-1], u[-1], u[0], u[0]])
            edge_v = np.array([v[0], v[0], v[-1], v[-1], v[0]])
            edge_x, edge_y = face_net_coordinates(face, edge_u, edge_v, box.size)
            ax.plot(edge_x, edge_y, color='.65', lw=.9)
            inset = .06*min(width, height)
            ax.text(edge_x.min()+inset, edge_y.max()-inset, FACE_NAMES[face],
                    ha='left', va='top', color='white', weight='bold', fontsize=12,
                    bbox=dict(facecolor='black', alpha=.55, edgecolor='none', pad=2))
        width, depth, height = box.size
        fold_edges = [([-width/2, -width/2], [-depth/2, depth/2]),
                      ([width/2, width/2], [-depth/2, depth/2]),
                      ([-width/2, width/2], [-depth/2, -depth/2]),
                      ([-width/2, width/2], [depth/2, depth/2]),
                      ([-width/2, width/2], [depth/2+height, depth/2+height])]
        for i, (x, y) in enumerate(fold_edges):
            ax.plot(x, y, '--', color='white', lw=1.3, label='Fold edge' if i == 0 else None)
        ax.set(xlabel='Unfolded x (m)', ylabel='Unfolded y (m)', aspect='equal')
        ax.margins(.05)
        ax.legend(loc='upper left', facecolor='.25', labelcolor='white', fontsize=9)
        fig.colorbar(plot, ax=ax, label=label, shrink=.85)
        fig.suptitle(f'{box.name}: {box.mode}, {kind} events'+(' [incomplete run]' if not meta['complete'] else ''))
        stem = f'box_{bi:03d}_{metric}_{kind}'
        fig.savefig(output/(stem+'.png'), dpi=180)
        fig.savefig(output/(stem+'.pdf'))
        plt.close(fig)
        np.savez_compressed(output/(stem+'.npz'), values=values[bi], counts=counts[bi],
                            size_m=box.size, face_names=FACE_NAMES, metric=metric, kind=kind)
    reference = meta.get('vessel') if vessel is None else vessel
    plot_boundary_preview(boxes, output, preview=preview, vessel=reference, preview_padding=preview_padding)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?', help='Boundary_events directory containing manifest.json')
    parser.add_argument('--geometry', type=Path, help='Preview BOUNDARIES directly from a workflow JSON; no tracing needed')
    parser.add_argument('--major-radius', type=float, help='Reference vessel major radius (m)')
    parser.add_argument('--vessel-radius', type=float, help='Reference vessel minor radius (m)')
    parser.add_argument('--preview-padding', type=float, default=1.25, help='Padding around box and nearby wall; smaller values zoom in (minimum 1.05)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bins', type=int, default=40)
    parser.add_argument('--metric', choices=['count', 'area', 'unique', 'energy', 'angle'], default='count')
    parser.add_argument('--kind', choices=['all', 'entry', 'exit', 'contact'], default='all')
    parser.add_argument('--preview-limit', type=int, default=10000)
    args = parser.parse_args(argv)
    if (args.directory is None) == (args.geometry is None):
        parser.error('provide an event directory or --geometry INPUT_JSON, not both')
    reference = None
    if args.geometry is not None:
        config = json.loads(args.geometry.read_text())
        if 'MAJOR_RADIUS_M' in config or 'VESSEL_RADIUS_M' in config:
            reference = dict(major_radius_m=config.get('MAJOR_RADIUS_M', .72),
                             minor_radius_m=config.get('VESSEL_RADIUS_M', .19))
    else:
        reference = json.loads((args.directory/'manifest.json').read_text()).get('vessel')
    if args.major_radius is not None or args.vessel_radius is not None:
        reference = dict(reference or dict(major_radius_m=.72, minor_radius_m=.19))
        if args.major_radius is not None:
            reference['major_radius_m'] = args.major_radius
        if args.vessel_radius is not None:
            reference['minor_radius_m'] = args.vessel_radius
    if args.geometry is not None:
        plot_boundary_preview(parse_boundaries(config['BOUNDARIES']), args.output,
                              vessel=reference, preview_padding=args.preview_padding)
    else:
        plot_events(args.directory, args.output, bins=args.bins, metric=args.metric, kind=args.kind,
                    preview_limit=args.preview_limit, vessel=reference, preview_padding=args.preview_padding)


if __name__ == '__main__':
    main()
