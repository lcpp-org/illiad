"""Oriented-box geometry and versioned, streaming surface-crossing records.

Lengths are metres. R maps local column vectors to simulation XYZ. Intrinsic
XYZ rotations (about moving axes) give R = Rx @ Ry @ Rz. Face order is
-x,+x,-y,+y,-z,+z. Segment ownership is (start, end], within tolerance.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from .utilities.coordtrans import RZphi_to_XYZ, XYZ_to_RZphi

FACE_NAMES = ('-x', '+x', '-y', '+y', '-z', '+z')
EVENT_DTYPE = np.dtype([
    ('trajectory_id', '<i8'), ('source_id', '<i8'), ('direction', 'i1'),
    ('boundary_id', '<i4'), ('face_id', 'i1'), ('face_mask', 'u1'),
    ('kind', 'i1'),  # -1 entry, +1 exit, 0 tangent/contact
    ('step', '<i8'), ('fraction', '<f8'), ('coordinate', '<f8'),
    ('xyz_m', '<f8', (3,)), ('uv_m', '<f8', (2,)),
    ('normal_xyz', '<f8', (3,)), ('vector_xyz', '<f8', (3,)),
    ('energy_eV', '<f8'),
])


@dataclass(frozen=True)
class Box:
    name: str
    center: np.ndarray
    size: np.ndarray  # width, depth, height
    angles: np.ndarray
    rotation: np.ndarray
    mode: str

    @classmethod
    def from_config(cls, spec):
        allowed = {'name', 'shape', 'center_m', 'center_rzphi', 'width_m', 'depth_m', 'height_m',
                   'rotation_local_xyz_deg', 'mode'}
        if not isinstance(spec, dict) or set(spec) - allowed:
            raise ValueError(f'Invalid box keys; expected {sorted(allowed)}')
        name = spec.get('name')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Each boundary needs a nonempty name')
        if spec.get('shape', 'box') != 'box':
            raise ValueError('Only shape="box" is supported')
        mode = spec.get('mode', 'absorbing')
        if mode not in ('absorbing', 'transparent'):
            raise ValueError('Boundary mode must be absorbing or transparent')
        if ('center_rzphi' in spec) == ('center_m' in spec):
            raise ValueError('Specify center_rzphi [R_m, Z_m, phi_deg] (or legacy center_m), not both')
        if 'center_rzphi' in spec:
            rzphi = np.asarray(spec['center_rzphi'], dtype=float)
            if rzphi.shape != (3,) or not np.isfinite(rzphi).all() or rzphi[0] < 0:
                raise ValueError('center_rzphi must be finite [R_m >= 0, Z_m, phi_deg]')
            center = RZphi_to_XYZ(rzphi)
        else:
            center = np.asarray(spec['center_m'], dtype=float)
        size = np.asarray([spec.get(k) for k in ('width_m', 'depth_m', 'height_m')], dtype=float)
        angles = np.asarray(spec.get('rotation_local_xyz_deg', [0, 0, 0]), dtype=float)
        if any(a.shape != (3,) or not np.isfinite(a).all() for a in (center, size, angles)) or np.any(size <= 0):
            raise ValueError('Box center/rotation must be finite triples and dimensions must be positive')
        x, y, z = np.deg2rad(angles)
        cx, cy, cz = np.cos([x, y, z]); sx, sy, sz = np.sin([x, y, z])
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return cls(name, center, size, angles, rx @ ry @ rz, mode)

    def config(self):
        return dict(name=self.name, shape='box', center_rzphi=XYZ_to_RZphi(self.center).tolist(),
                    width_m=float(self.size[0]), depth_m=float(self.size[1]), height_m=float(self.size[2]),
                    rotation_local_xyz_deg=self.angles.tolist(), mode=self.mode)


def parse_boundaries(config):
    if config is None:
        return ()
    if not isinstance(config, (list, tuple)):
        raise ValueError('BOUNDARIES must be a list of boxes')
    boxes = tuple(s if isinstance(s, Box) else Box.from_config(s) for s in config)
    if len({b.name for b in boxes}) != len(boxes):
        raise ValueError('Boundary names must be unique')
    return boxes


def validate_initial_positions(boxes, xyz, tolerance=1e-10):
    xyz = np.asarray(xyz)
    for box in boxes:
        if box.mode == 'absorbing':
            inside = np.all(np.abs((xyz - box.center) @ box.rotation) <= box.size/2 + tolerance, axis=-1)
            if np.any(inside):
                raise ValueError(f'Initial positions inside/on absorbing boundary {box.name!r}: '
                                 f'{np.flatnonzero(inside)[:10].tolist()}')


def segment_hits(box, start, end, tolerance=1e-10):
    """Return sorted (fraction, kind) pairs for a single straight segment."""
    p = (np.asarray(start) - box.center) @ box.rotation
    d = (np.asarray(end) - np.asarray(start)) @ box.rotation
    half = box.size / 2
    length = np.linalg.norm(d)
    if length <= tolerance:
        return []
    parallel = np.abs(d) <= max(1e-30, 1e-14*length)
    if np.any(parallel & (np.abs(p) > half + tolerance)):
        return []
    safe = np.where(parallel, 1., d)
    a, b = (-half-p)/safe, (half-p)/safe
    lo = np.max(np.where(parallel, -np.inf, np.minimum(a, b)))
    hi = np.min(np.where(parallel, np.inf, np.maximum(a, b)))
    eps = tolerance / length
    if hi < lo - eps:
        return []
    touch = hi - lo <= eps or np.any(parallel & (np.abs(np.abs(p)-half) <= tolerance))
    result = []
    for alpha, kind in ((lo, -1), (hi, 1)):
        if eps < alpha <= 1 + eps and (not result or alpha-result[-1][0] > eps):
            result.append((float(np.clip(alpha, 0, 1)), 0 if touch else kind))
    return result


def torch_box_hits(box, start, end, tolerance=1e-10):
    """Batched slab intersections; two fractions per row, inf means no hit."""
    import torch
    center = torch.as_tensor(box.center, dtype=start.dtype, device=start.device)
    rotation = torch.as_tensor(box.rotation, dtype=start.dtype, device=start.device)
    half = torch.as_tensor(box.size/2, dtype=start.dtype, device=start.device)
    p, d = (start-center) @ rotation, (end-start) @ rotation
    length = torch.linalg.vector_norm(d, dim=-1)
    parallel = torch.abs(d) <= torch.clamp(1e-14*length[:, None], min=1e-30)
    safe = torch.where(parallel, 1., d)
    a, b = (-half-p)/safe, (half-p)/safe
    lo = torch.where(parallel, -torch.inf, torch.minimum(a, b)).amax(dim=-1)
    hi = torch.where(parallel, torch.inf, torch.maximum(a, b)).amin(dim=-1)
    eps = tolerance / torch.clamp(length, min=tolerance)
    valid = (length > tolerance) & ~torch.any(parallel & (torch.abs(p) > half+tolerance), dim=-1) & (hi >= lo-eps)
    touch = (hi-lo <= eps) | torch.any(parallel & (torch.abs(torch.abs(p)-half) <= tolerance), dim=-1)
    alphas = torch.stack((lo, hi), dim=-1)
    keep = valid[:, None] & (alphas > eps[:, None]) & (alphas <= 1+eps[:, None])
    keep[:, 1] &= ~((hi-lo <= eps) & keep[:, 0])
    alphas = torch.where(keep, torch.clamp(alphas, 0., 1.), torch.inf)
    kinds = torch.where(touch[:, None], 0, torch.tensor([-1, 1], device=start.device))
    return alphas, kinds


def torch_wall_fraction(start, end, major_radius, vessel_radius):
    """First vessel exit for an inside-to-outside short segment, else inf."""
    import torch
    def outside(x):
        return (torch.sqrt(x[:, 0]**2+x[:, 1]**2)-major_radius)**2+x[:, 2]**2 >= vessel_radius**2
    hit = outside(end)
    result = torch.full_like(start[:, 0], torch.inf)
    rows = torch.nonzero(hit, as_tuple=True)[0]
    if rows.numel() == 0:
        return result
    origin, delta = start[rows], end[rows]-start[rows]
    low, high = torch.zeros_like(origin[:, 0]), torch.ones_like(origin[:, 0])
    for _ in range(40):
        mid = (low+high)/2
        out = outside(origin+delta*mid[:, None])
        high = torch.where(out, mid, high)
        low = torch.where(out, low, mid)
    result[rows] = high
    return result


class BoundaryRecorder:
    """Bounded NPZ shards plus metadata. directory=None is for small API tests.

    Production runs must use a fresh output directory. An incomplete manifest
    remains readable if a run fails; only listed, completely written shards count.
    """
    def __init__(self, boxes, directory=None, *, coordinate='time_s', buffer_rows=65536, tolerance=1e-10):
        self.boxes = parse_boundaries(boxes)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('BOUNDARY_TOLERANCE_M must be finite and positive')
        if isinstance(buffer_rows, bool) or not isinstance(buffer_rows, int) or not 1 <= buffer_rows < 2**31:
            raise ValueError('BOUNDARY_BUFFER_ROWS must be a positive int32 integer')
        if coordinate not in ('time_s', 'arc_length_m'):
            raise ValueError('coordinate must be time_s or arc_length_m')
        self.tolerance = float(tolerance)
        self.buffer_rows = buffer_rows
        self.directory = Path(directory) if directory is not None else None
        self.pending = []
        self.pending_count = 0
        self.memory = []
        self.terminations = {}
        self.metadata = dict(version=2, coordinate=coordinate, boxes=[b.config() for b in self.boxes],
                             rotation='intrinsic XYZ; R=Rx@Ry@Rz; right handed',
                             faces=list(FACE_NAMES), tolerance_m=tolerance, shards=[], event_count=0,
                             vector='position-advance velocity (m/s)' if coordinate == 'time_s' else 'unit tracing tangent',
                             complete=False)
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=False)
            self._manifest()

    def set_vessel(self, major_radius, vessel_radius):
        """Persist the actual circular-torus wall used for intersection ordering."""
        vessel = dict(model='circular_torus', major_radius_m=float(major_radius),
                      minor_radius_m=float(vessel_radius))
        if self.metadata.get('vessel') != vessel:
            self.metadata['vessel'] = vessel
            self._manifest()

    def _manifest(self):
        if self.directory is not None:
            temp = self.directory/'manifest.tmp'
            temp.write_text(json.dumps(self.metadata, indent=2)+'\n')
            temp.replace(self.directory/'manifest.json')

    def record(self, boundary_id, ids, source_ids, direction, step, fraction, coordinate, xyz, vector, kind, mass=None):
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        if not len(xyz):
            return
        box = self.boxes[boundary_id]
        local = (xyz-box.center) @ box.rotation
        on_faces = np.abs(local[:, :, None] - box.size[None, :, None]/2*np.array([-1, 1])) <= self.tolerance*4
        on_faces = on_faces.reshape(-1, 6)
        # Floating point transforms may move the exact root slightly. Choose
        # nearest face if needed, while preserving all tied faces in the mask.
        nearest = np.argmin(np.abs(local[:, :, None]-box.size[None, :, None]/2*np.array([-1, 1])).reshape(-1, 6), axis=1)
        on_faces[np.arange(len(xyz)), nearest] = True
        face = on_faces.argmax(axis=1)
        axis, sign = face//2, np.where(face % 2, 1., -1.)
        data = np.zeros(len(xyz), dtype=EVENT_DTYPE)
        for name, value in [('trajectory_id', ids), ('source_id', source_ids), ('direction', direction),
                            ('step', step), ('fraction', fraction), ('coordinate', coordinate), ('kind', kind)]:
            data[name] = value
        data['boundary_id'], data['face_id'] = boundary_id, face
        data['face_mask'] = (on_faces * (1 << np.arange(6))).sum(axis=1)
        data['xyz_m'], data['vector_xyz'] = xyz, vector
        data['uv_m'][:, 0] = local[np.arange(len(xyz)), (axis+1) % 3]
        data['uv_m'][:, 1] = sign*local[np.arange(len(xyz)), (axis+2) % 3]
        data['normal_xyz'] = box.rotation[:, axis].T * sign[:, None]
        data['energy_eV'] = np.nan if mass is None else .5*np.asarray(mass)*np.sum(np.asarray(vector)**2, axis=-1)/1.602176634e-19
        if box.mode == 'absorbing':
            for row in data:
                self.terminations[(int(row['trajectory_id']), int(row['direction']))] = float(row['coordinate'])
        self.append(data)

    def append(self, data):
        for start in range(0, len(data), self.buffer_rows):
            part = data[start:start+self.buffer_rows]
            self.pending.append(part.copy())
            self.pending_count += len(part)
            if self.pending_count >= self.buffer_rows:
                self.flush()

    def flush(self):
        if not self.pending_count:
            return
        data = np.concatenate(self.pending)
        # Deterministic order within each shard; global ordering is by event
        # coordinate/trajectory, not GPU append order or shard number.
        data.sort(order=['trajectory_id', 'direction', 'coordinate', 'boundary_id'])
        if self.directory is None:
            self.memory.append(data)
        else:
            name = f'events_{len(self.metadata["shards"]):06d}.npz'
            np.savez_compressed(self.directory/name, events=data)
            self.metadata['shards'].append(name)
        self.metadata['event_count'] += len(data)
        self.pending.clear()
        self.pending_count = 0
        self._manifest()

    def finish(self):
        self.flush()
        self.metadata['complete'] = True
        self._manifest()

    def events(self):
        """Small in-memory runs only; production consumers iterate shards."""
        self.flush()
        if self.directory is not None:
            raise ValueError('Use iter_events for disk-backed runs')
        return np.concatenate(self.memory) if self.memory else np.empty(0, dtype=EVENT_DTYPE)

    def process_torch(self, start, end, ids, step, coordinate_start, coordinate_step,
                      vector, *, major_radius, vessel_radius, source_ids=None, direction=0, masses=None):
        """Record a batch; return clipped endpoints, absorbing mask and fraction.

        The vessel wins ties. It is used for ordering only; legacy vessel output
        retains its original convention. Only hit records move to the CPU.
        """
        import torch
        self.set_vessel(major_radius, vessel_radius)
        wall = torch_wall_fraction(start, end, major_radius, vessel_radius)
        limit = wall.clone()
        owner = torch.full_like(ids, -1)
        eps = self.tolerance / torch.clamp(torch.linalg.vector_norm(end-start, dim=1), min=self.tolerance)
        hits = []
        for i, box in enumerate(self.boxes):
            alpha, kind = torch_box_hits(box, start, end, self.tolerance)
            hits.append((alpha, kind))
            if box.mode == 'absorbing':
                first = alpha.amin(dim=-1)
                wins = first < limit-eps
                limit = torch.where(wins, first, limit)
                owner = torch.where(wins, i, owner)
        def cpu(a):
            return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a
        def select(a, rows):
            return cpu(a[rows]) if isinstance(a, torch.Tensor) and a.ndim else cpu(a)
        for i, (alpha, kind) in enumerate(hits):
            valid = torch.isfinite(alpha) & (alpha <= limit[:, None])
            if self.boxes[i].mode == 'absorbing':
                valid &= (owner == i)[:, None]
            rows, cols = torch.nonzero(valid, as_tuple=True)
            if rows.numel() == 0:
                continue
            f = alpha[rows, cols]
            xyz = start[rows] + f[:, None]*(end[rows]-start[rows])
            coords = select(coordinate_start, rows) + cpu(f)*select(coordinate_step, rows)
            self.record(i, cpu(ids[rows]), cpu(ids[rows]) if source_ids is None else select(source_ids, rows),
                        select(direction, rows), select(step, rows), cpu(f), coords,
                        cpu(xyz), cpu(vector[rows]), cpu(kind.expand_as(alpha)[rows, cols]),
                        None if masses is None else select(masses, rows))
        absorbed = owner >= 0
        fraction = torch.where(absorbed, limit, 1.)
        return torch.where(absorbed[:, None], start+(end-start)*fraction[:, None], end), absorbed, fraction


def iter_events(directory):
    directory = Path(directory)
    meta = json.loads((directory/'manifest.json').read_text())
    for name in meta['shards']:
        with np.load(directory/name, allow_pickle=False) as data:
            yield data['events']
