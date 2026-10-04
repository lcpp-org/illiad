"""Standalone collisionless Warp prototype; see docs/warp_boris_prototype.md.

Requires NumPy and NVIDIA Warp to execute. No production ILLIAD imports, field
generation, collisions, trajectory storage, or frequency correction. This is
an inspectable prototype, not a production replacement for Boris.
"""

import numpy as np
import warp as wp


@wp.struct
class VectorGrid:
    # One Cartesian vector per (r, theta, phi) node, in float64.
    values: wp.array3d(dtype=wp.vec3d)
    nr: int
    ntheta: int
    nphi: int
    periods: int
    R0: wp.float64
    a: wp.float64
    dr: wp.float64
    dtheta: wp.float64
    dphi: wp.float64
    phi_max: wp.float64
    addend: wp.vec3d


@wp.func
def minor_radius(x: wp.vec3d, R0: wp.float64):
    xy2 = x[0] * x[0] + x[1] * x[1]
    r2 = xy2 + x[2] * x[2] + R0 * R0 - wp.float64(2.0) * R0 * wp.sqrt(xy2)
    # Same expression as TorchMesh, with a guard for cancellation at the axis.
    return wp.sqrt(wp.max(r2, wp.float64(0.0)))


@wp.func
def rotate_z(v: wp.vec3d, angle: wp.float64):
    # Match the actual TorchMesh matrix; phi-coordinate signs are handled below.
    c = wp.cos(angle)
    s = wp.sin(angle)
    return wp.vec3d(c * v[0] - s * v[1], s * v[0] + c * v[1], v[2])


@wp.func
def sample_vector(g: VectorGrid, x: wp.vec3d):
    """Port of TorchMesh's volume-weighted eight-corner interpolation.

    Caller must supply an interior point. Supported periodicity is [0, 1, P].
    Angular samples start at one grid spacing, not zero.
    """
    zero = wp.float64(0.0)
    half = wp.float64(0.5)
    tau = wp.float64(6.283185307179586)
    r = minor_radius(x, g.R0)
    R = wp.sqrt(x[0] * x[0] + x[1] * x[1])
    theta = wp.atan2(x[2], R - g.R0)
    phi = -wp.atan2(x[1], x[0])
    if theta < zero:
        theta = theta + tau
    if phi < zero:
        phi = phi + tau

    theta = theta - wp.floor(theta / tau) * tau
    sector = int(wp.floor(phi / g.phi_max))
    phi_local = phi - wp.float64(sector) * g.phi_max
    ir = wp.min(int(wp.floor(r / g.dr)), g.nr - 2)
    it = int(wp.floor(theta / g.dtheta))
    ip = int(wp.floor(phi_local / g.dphi))
    er = r - wp.floor(r / g.dr) * g.dr
    et = theta - wp.floor(theta / g.dtheta) * g.dtheta
    ep = phi_local - wp.floor(phi_local / g.dphi) * g.dphi
    r_low = wp.float64(ir) * g.dr
    theta_high = wp.float64(it + 1) * g.dtheta

    value = wp.vec3d(zero, zero, zero)
    total = wp.float64(0.0)
    # Order matches A1,...,A8 in get_weights(): radial index changes fastest.
    for p_side in range(2):
        for t_side in range(2):
            for r_side in range(2):
                ri = ir + 1
                radial_volume = (r_low + half * er) * er
                jacobian_r = r_low
                if r_side == 1:
                    ri = ir
                    inv_er = g.dr - er
                    radial_volume = (r + half * inv_er) * inv_er
                    jacobian_r = r

                theta_width = et
                jacobian_theta = theta_high
                if t_side == 1:
                    theta_width = g.dtheta - et
                    jacobian_theta = theta
                phi_width = ep
                if p_side == 1:
                    phi_width = g.dphi - ep

                weight = (g.R0 + jacobian_r * wp.cos(jacobian_theta))
                weight = weight * radial_volume * theta_width * phi_width
                # PyTorch accepts -1 here; Warp gets explicit positive indices.
                ti = (it - t_side + g.ntheta) % g.ntheta
                pi = (ip - p_side + g.nphi) % g.nphi
                corner = g.values[ri, ti, pi]
                if ip == 0 and p_side == 1:
                    corner = rotate_z(corner, g.phi_max)
                value = value + corner * weight
                total = total + weight

    value = value / total
    if g.periods > 1:
        value = rotate_z(value, -wp.float64(sector) * g.phi_max)
    return value + g.addend


@wp.kernel
def initialize_velocity(
    x: wp.array(dtype=wp.vec3d),
    v: wp.array(dtype=wp.vec3d),
    qdt2m: wp.array(dtype=wp.float64),
    b_grid: VectorGrid,
    e_grid: VectorGrid,
    has_e: int,
):
    """Reproduce parallel_solver's existing startup formula literally."""
    i = wp.tid()
    e = wp.vec3d(wp.float64(0.0))
    if has_e == 1:
        e = sample_vector(e_grid, x[i]) * qdt2m[i]
    t = sample_vector(b_grid, x[i]) * qdt2m[i]
    vm = v[i] + e
    vp = vm + wp.cross(vm, t)
    s = t * (wp.float64(2.0) / (wp.float64(1.0) + wp.dot(t, t)))
    v[i] = vm - wp.cross(vp, s) * wp.float64(0.5) + e


@wp.kernel
def push_and_record_wall(
    x: wp.array(dtype=wp.vec3d),
    v: wp.array(dtype=wp.vec3d),
    qdt2m: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
    wall_x: wp.array(dtype=wp.vec3d),
    wall_v: wp.array(dtype=wp.vec3d),
    hit_step: wp.array(dtype=wp.int32),
    last_inside_step: wp.array(dtype=wp.int32),
    b_grid: VectorGrid,
    e_grid: VectorGrid,
    has_e: int,
    dt: wp.float64,
    step: int,
):
    i = wp.tid()  # One logical GPU thread owns one particle.
    if active[i] == 0:
        return

    xi = x[i]
    vi = v[i]
    e = wp.vec3d(wp.float64(0.0))
    if has_e == 1:
        e = sample_vector(e_grid, xi) * qdt2m[i]
    t = sample_vector(b_grid, xi) * qdt2m[i]

    vm = vi + e
    vp = vm + wp.cross(vm, t)
    s = t * (wp.float64(2.0) / (wp.float64(1.0) + wp.dot(t, t)))
    vi = vm + wp.cross(vp, s) + e
    xi = xi + vi * dt
    x[i] = xi
    v[i] = vi

    if minor_radius(xi, b_grid.R0) >= b_grid.a:
        active[i] = 0
        wall_x[i] = xi  # First outside point, not a projected wall intersection.
        wall_v[i] = vi
        hit_step[i] = step
    else:
        last_inside_step[i] = step  # Existing Boris maxStep convention.


def make_grid(values, *, R0, a, periods, device, addend=(0.0, 0.0, 0.0)):
    """Upload already scaled/combined (nr, ntheta, nphi, 3) Cartesian vectors."""
    values = np.ascontiguousarray(values, dtype=np.float64)
    if values.ndim != 4 or values.shape[-1] != 3 or min(values.shape[:3]) < 2:
        raise ValueError('values must have shape (nr>=2, ntheta>=2, nphi>=2, 3)')
    if not np.isfinite(values).all():
        raise ValueError('field values must be finite')
    if not np.isfinite([R0, a]).all() or not R0 > a > 0:
        raise ValueError('geometry must satisfy R0 > a > 0')
    if isinstance(periods, bool) or int(periods) != periods or periods < 1:
        raise ValueError('periods must be a positive integer')
    addend = np.asarray(addend, dtype=np.float64)
    if addend.shape != (3,) or not np.isfinite(addend).all():
        raise ValueError('addend must be a finite Cartesian 3-vector')
    g = VectorGrid()
    g.nr, g.ntheta, g.nphi = values.shape[:3]
    g.periods = int(periods)
    g.R0, g.a = float(R0), float(a)
    g.dr = a / (g.nr - 1)
    g.dtheta = 2.0 * np.pi / g.ntheta
    g.phi_max = 2.0 * np.pi / g.periods
    g.dphi = g.phi_max / g.nphi
    g.addend = wp.vec3d(*addend)
    g.values = wp.array(values, dtype=wp.vec3d, device=device)
    return g


def run_wall_only(x0, v0, charge_mass_ratio, b_grid, *, dt, steps, e_grid=None):
    """Return NumPy wall arrays; preserve original particle row order.

    steps is the number of position updates (existing self.nsteps - 1).
    hit_step == -1 marks survivors; their wall position/velocity rows are zero.
    Initial positions must be strictly inside the wall. No trace is allocated.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    if isinstance(steps, bool) or int(steps) != steps or not 0 <= steps < 2**31:
        raise ValueError('steps must be a nonnegative int32 count')
    x0 = np.ascontiguousarray(x0, dtype=np.float64)
    v0 = np.ascontiguousarray(v0, dtype=np.float64)
    if x0.ndim != 2 or x0.shape[1] != 3 or len(x0) == 0 or v0.shape != x0.shape:
        raise ValueError('x0 and v0 must have matching nonempty (N, 3) shapes')
    q_over_m = np.broadcast_to(np.asarray(charge_mass_ratio, dtype=np.float64), (len(x0),))
    if not all(np.isfinite(arr).all() for arr in (x0, v0, q_over_m)):
        raise ValueError('particle data must be finite')
    xy2 = x0[:, 0]**2 + x0[:, 1]**2
    r2 = xy2 + x0[:, 2]**2 + b_grid.R0**2 - 2 * b_grid.R0 * np.sqrt(xy2)
    if np.any(np.sqrt(np.maximum(r2, 0.0)) >= b_grid.a):
        raise ValueError('initial particles must be strictly inside the wall')
    device = b_grid.values.device
    has_e = int(e_grid is not None)
    if e_grid is None:
        e_grid = b_grid  # Valid placeholder; has_e prevents E reads.
    elif (e_grid.values.device != device or e_grid.R0 != b_grid.R0
          or e_grid.a != b_grid.a or e_grid.periods != 1
          or e_grid.nr != b_grid.nr or e_grid.ntheta != b_grid.ntheta
          or e_grid.nphi != b_grid.nphi * b_grid.periods):
        raise ValueError('E must be a full-torus grid aligned with B on the same device')

    n = len(x0)
    x = wp.array(x0, dtype=wp.vec3d, device=device)
    v = wp.array(v0, dtype=wp.vec3d, device=device)
    qdt2m = wp.array(np.ascontiguousarray(q_over_m * (dt / 2)),
                    dtype=wp.float64, device=device)
    active = wp.ones(n, dtype=wp.int32, device=device)
    wall_x = wp.zeros(n, dtype=wp.vec3d, device=device)
    wall_v = wp.zeros(n, dtype=wp.vec3d, device=device)
    hit_step = wp.full(n, value=-1, dtype=wp.int32, device=device)
    last_inside_step = wp.zeros(n, dtype=wp.int32, device=device)

    wp.launch(initialize_velocity, dim=n,
              inputs=[x, v, qdt2m, b_grid, e_grid, has_e], device=device)
    for step in range(1, int(steps) + 1):
        wp.launch(push_and_record_wall, dim=n,
                  inputs=[x, v, qdt2m, active, wall_x, wall_v, hit_step,
                          last_inside_step, b_grid, e_grid, has_e, dt, step],
                  device=device)
    wp.synchronize_device(device)
    return {'wall_position_xyz': wall_x.numpy(), 'wall_velocity_xyz': wall_v.numpy(),
            'hit_step': hit_step.numpy(), 'last_inside_step': last_inside_step.numpy()}
