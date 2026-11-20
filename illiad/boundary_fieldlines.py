"""Adaptive Cartesian field-line tracing with finite surface events.

The solver is stepped explicitly so absorbing events stop further integration.
Dense-output subdivision bounds chord error before segment/face intersection.
"""
from types import SimpleNamespace

import numpy as np
from scipy.integrate import RK23, RK45, DOP853, Radau, BDF, LSODA, OdeSolution
from scipy.optimize import brentq

from .boundaries import segment_hits, validate_initial_positions


def trace_fieldline(fun, initial, length, recorder, *, trajectory_id=0, source_id=0,
                    direction=1, major_radius=.72, vessel_radius=.19,
                    method='LSODA', rtol=1e-8, atol=1e-10,
                    max_step=.005, curve_tolerance=1e-7):
    """Return solve_ivp-like XYZ data and one legacy vessel event array.

    coordinate/fraction refer to positive arc length and the accepted solver
    step. Forward/backward direction is separately recorded. Box termination
    sets status=1 but never populates the vessel event array.
    """
    if not np.isfinite(max_step) or max_step <= 0 or not np.isfinite(curve_tolerance) or curve_tolerance <= 0:
        raise ValueError('BOUNDARY_MAX_STEP_M and BOUNDARY_CURVE_TOLERANCE_M must be positive and finite')
    if not np.isfinite(length) or length <= 0:
        raise ValueError('Trace length must be positive and finite')
    if recorder.boxes:
        curve_tolerance = min(curve_tolerance, min(float(b.size.min()) for b in recorder.boxes)/20)
    recorder.metadata['integration'] = dict(method=method, max_step_m=max_step,
                                           curve_tolerance_m=curve_tolerance, rtol=rtol, atol=atol)
    recorder.set_vessel(major_radius, vessel_radius)
    initial = np.asarray(initial, dtype=float)
    validate_initial_positions(recorder.boxes, initial[None], recorder.tolerance)
    def wall(x):
        return vessel_radius**2 - ((np.hypot(x[0], x[1])-major_radius)**2+x[2]**2)
    if wall(initial) <= 0:
        raise ValueError('Field-line seeds must be strictly inside the vessel')
    methods = {c.__name__: c for c in (RK23, RK45, DOP853, Radau, BDF, LSODA)}
    if method not in methods:
        raise ValueError(f'Unsupported boundary field-line solver {method!r}')
    solver = methods[method](fun, 0., initial, length, rtol=rtol, atol=atol, max_step=max_step)
    ts, ys, interpolants = [0.], [initial.copy()], []
    wall_times, wall_points = [], []
    status, step = 0, 0
    last_events = {}
    while solver.status == 'running':
        message = solver.step()
        if solver.status == 'failed':
            raise RuntimeError(f'Boundary field-line integration failed: {message}')
        step += 1
        dense = solver.dense_output()
        t0, t1 = solver.t_old, solver.t
        segments = []
        def subdivide(a, b, p, q, depth=0):
            fractions = np.array([.25, .5, .75])
            middle = dense(a+(b-a)*fractions).T
            error = np.max(np.linalg.norm(middle-(p+(q-p)*fractions[:, None]), axis=1))
            if error > curve_tolerance:
                if depth >= 24:
                    raise RuntimeError('Boundary dense-output refinement did not converge')
                mid = (a+b)/2
                subdivide(a, mid, p, middle[1], depth+1)
                subdivide(mid, b, middle[1], q, depth+1)
            else:
                segments.append((a, b, p, q))
        subdivide(t0, t1, dense(t0), dense(t1))
        stop = None
        for a, b, p, q in segments:
            candidates = []
            wall_time = np.inf
            if wall(q) <= 0:
                wall_time = brentq(lambda t: wall(dense(t)), a, b, xtol=1e-14)
            for index, box in enumerate(recorder.boxes):
                for alpha, kind in segment_hits(box, p, q, recorder.tolerance):
                    estimate = p+alpha*(q-p)
                    local = (estimate-box.center) @ box.rotation
                    face = np.argmin(np.abs(local[:, None]-box.size[:, None]/2*np.array([-1, 1])))
                    axis, sign = face//2, (-1 if face % 2 == 0 else 1)
                    def residual(t):
                        return ((dense(t)-box.center) @ box.rotation)[axis]-sign*box.size[axis]/2
                    fa, fb = residual(a), residual(b)
                    if fa*fb <= 0:
                        t = brentq(residual, a, b, xtol=1e-14)
                    elif kind == 0:
                        t = a+alpha*(b-a)
                    else:
                        continue
                    xyz = dense(t)
                    local = (xyz-box.center) @ box.rotation
                    if np.any(np.abs(local) > box.size/2+recorder.tolerance*4):
                        continue
                    if t-last_events.get(index, -np.inf) <= recorder.tolerance:
                        continue
                    candidates.append((t, index, kind, xyz))
            candidates.sort(key=lambda item: (item[0], item[1]))
            limit, owner = wall_time, -1
            for t, index, kind, xyz in candidates:
                if recorder.boxes[index].mode == 'absorbing' and t < limit-recorder.tolerance:
                    limit, owner = t, index
            stop = limit if np.isfinite(limit) else None
            for t, index, kind, xyz in candidates:
                if t > limit:
                    continue
                if recorder.boxes[index].mode == 'absorbing' and index != owner:
                    continue
                recorder.record(index, trajectory_id, source_id, direction, step,
                                (t-t0)/(t1-t0), t, xyz[None], np.asarray(fun(t, xyz))[None], kind)
                last_events[index] = t
            if stop is not None:
                if stop == wall_time:
                    wall_times.append(stop); wall_points.append(dense(stop))
                break
        end = t1 if stop is None else stop
        interpolants.append(dense)
        ts.append(end); ys.append(dense(end))
        if stop is not None:
            status = 1
            break
    return SimpleNamespace(t=np.asarray(ts), y=np.asarray(ys).T,
                           sol=OdeSolution(ts, interpolants), status=status,
                           t_events=[np.asarray(wall_times)],
                           y_events=[np.asarray(wall_points).reshape(-1, 3)])
