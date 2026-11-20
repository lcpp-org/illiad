"""Warp implementation of the oriented-box segment contract.

The event buffer is reset/drained after every kernel launch. Overflow fails the
run explicitly; it never produces an apparently complete, truncated archive.
"""
import numpy as np
import warp as wp


@wp.struct
class BoxEvents:
    count: int
    centers: wp.array(dtype=wp.vec3d)
    halves: wp.array(dtype=wp.vec3d)
    rotations: wp.array(dtype=wp.mat33d)
    absorbing: wp.array(dtype=wp.int32)
    tolerance: wp.float64
    capacity: int
    cursor: wp.array(dtype=wp.int64)
    particle: wp.array(dtype=wp.int32)
    boundary: wp.array(dtype=wp.int32)
    kind: wp.array(dtype=wp.int32)
    step: wp.array(dtype=wp.int32)
    fraction: wp.array(dtype=wp.float64)
    xyz: wp.array(dtype=wp.vec3d)
    vector: wp.array(dtype=wp.vec3d)

@wp.func
def box_roots(start: wp.vec3d, end: wp.vec3d, index: int, events: BoxEvents):
    rotation = events.rotations[index]
    p = wp.transpose(rotation) * (start-events.centers[index])
    d = wp.transpose(rotation) * (end-start)
    half = events.halves[index]
    lo = wp.float64(-1.0e100)
    hi = wp.float64(1.0e100)
    valid = int(1)
    touch = int(0)
    length = wp.length(d)
    eps = events.tolerance/wp.max(length, events.tolerance)
    if length <= events.tolerance:
        valid = 0
    for axis in range(3):
        if wp.abs(d[axis]) <= wp.max(wp.float64(1.e-30), wp.float64(1.e-14)*length):
            if wp.abs(p[axis]) > half[axis]+events.tolerance:
                valid = 0
            if wp.abs(wp.abs(p[axis])-half[axis]) <= events.tolerance:
                touch = 1
        else:
            a = (-half[axis]-p[axis])/d[axis]
            b = (half[axis]-p[axis])/d[axis]
            lo = wp.max(lo, wp.min(a, b))
            hi = wp.min(hi, wp.max(a, b))
    if hi < lo-eps:
        valid = 0
    if hi-lo <= eps:
        touch = 1
    first = wp.float64(2.0)
    last = wp.float64(2.0)
    if valid == 1:
        if lo > eps and lo <= wp.float64(1.0)+eps:
            first = wp.clamp(lo, wp.float64(0.0), wp.float64(1.0))
        if hi > eps and hi <= wp.float64(1.0)+eps:
            if first > wp.float64(1.0) or hi-lo > eps:
                last = wp.clamp(hi, wp.float64(0.0), wp.float64(1.0))
    return first, last, touch


@wp.func
def outside_wall(x: wp.vec3d, R0: wp.float64, radius: wp.float64):
    r = wp.sqrt(x[0]*x[0]+x[1]*x[1])-R0
    return r*r+x[2]*x[2] >= radius*radius


@wp.func
def surface_events(start: wp.vec3d, end: wp.vec3d, velocity: wp.vec3d, particle: int,
                   step: int, R0: wp.float64, radius: wp.float64, events: BoxEvents):
    limit = wp.float64(2.0)
    owner = int(-1)
    eps = events.tolerance/wp.max(wp.length(end-start), events.tolerance)
    if outside_wall(end, R0, radius):
        low = wp.float64(0.0)
        high = wp.float64(1.0)
        for iteration in range(40):
            mid = (low+high)*wp.float64(0.5)
            if outside_wall(start+(end-start)*mid, R0, radius):
                high = mid
            else:
                low = mid
        limit = high
    for box in range(events.count):
        if events.absorbing[box] == 1:
            a, b, touch = box_roots(start, end, box, events)
            first = wp.min(a, b)
            if first < limit-eps:
                owner = box
                limit = first
    for box in range(events.count):
        a, b, touch = box_roots(start, end, box, events)
        for root in range(2):
            f = a
            kind = int(-1)
            if root == 1:
                f = b
                kind = 1
            if touch == 1:
                kind = 0
            if f <= wp.float64(1.0) and f <= limit:
                if events.absorbing[box] == 0 or box == owner:
                    slot = wp.atomic_add(events.cursor, 0, wp.int64(1))
                    if slot < events.capacity:
                        events.particle[slot] = particle
                        events.boundary[slot] = box
                        events.kind[slot] = kind
                        events.step[slot] = step
                        events.fraction[slot] = f
                        events.xyz[slot] = start+(end-start)*f
                        events.vector[slot] = velocity
    return wp.min(limit, wp.float64(1.0)), owner


def make_buffer(recorder, device):
    events = BoxEvents()
    boxes = recorder.boxes if recorder is not None else ()
    events.count = len(boxes)
    events.tolerance = recorder.tolerance if recorder is not None else 1e-10
    events.centers = wp.array(np.asarray([b.center for b in boxes]).reshape(-1, 3), dtype=wp.vec3d, device=device)
    events.halves = wp.array(np.asarray([b.size/2 for b in boxes]).reshape(-1, 3), dtype=wp.vec3d, device=device)
    events.rotations = wp.array(np.asarray([b.rotation for b in boxes]).reshape(-1, 3, 3), dtype=wp.mat33d, device=device)
    events.absorbing = wp.array([int(b.mode == 'absorbing') for b in boxes], dtype=wp.int32, device=device)
    events.capacity = recorder.buffer_rows if recorder is not None else 0
    events.cursor = wp.zeros(1, dtype=wp.int64, device=device)
    for name in ('particle', 'boundary', 'kind', 'step'):
        setattr(events, name, wp.empty(events.capacity, dtype=wp.int32, device=device))
    events.fraction = wp.empty(events.capacity, dtype=wp.float64, device=device)
    events.xyz = wp.empty(events.capacity, dtype=wp.vec3d, device=device)
    events.vector = wp.empty(events.capacity, dtype=wp.vec3d, device=device)
    return events


def drain_buffer(events, recorder, dt, source_ids, masses):
    count = int(events.cursor.numpy()[0])
    if count > events.capacity:
        raise RuntimeError(f'Boundary event buffer overflow ({count} > {events.capacity}). '
                           'Run is incomplete. Increase BOUNDARY_BUFFER_ROWS or reduce WARP_STEP_CHUNK_SIZE.')
    if count:
        data = {name: getattr(events, name).numpy()[:count] for name in
                ('particle', 'boundary', 'kind', 'step', 'fraction', 'xyz', 'vector')}
        for box in range(events.count):
            rows = data['boundary'] == box
            ids = data['particle'][rows]
            fraction = data['fraction'][rows]
            step = data['step'][rows]
            recorder.record(box, ids, source_ids[ids], 0, step, fraction,
                            (step-1+fraction)*dt, data['xyz'][rows], data['vector'][rows],
                            data['kind'][rows], masses[ids])
    events.cursor.zero_()
