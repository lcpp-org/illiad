"""Optional Warp implementation of the Boris particle solver.

Imported lazily by Boris; importing the ordinary Torch solver does not need Warp.
"""

import numpy as np
import warp as wp

from .collisions_warp import (
    CollisionConfig, collision_hstep, initialize_random_states, make_collision_config,
)


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


@wp.struct
class DensityGrid:
    values: wp.array3d(dtype=wp.float64)  # full-torus (r, theta, phi), in m^-3


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


@wp.func
def sample_density(n: DensityGrid, g: VectorGrid, x: wp.vec3d):
    """B-grid volume weights with the full-torus density index offset.

    No vector rotation is applied to scalar data. The low phi index is offset
    BEFORE wrapping, matching Boris.parallel_solver's full_phi_corner_indices.
    """
    tau = wp.float64(6.283185307179586)
    r = minor_radius(x, g.R0)
    theta = wp.atan2(x[2], wp.sqrt(x[0]*x[0] + x[1]*x[1]) - g.R0)
    phi = -wp.atan2(x[1], x[0])
    theta = theta - wp.floor(theta/tau)*tau
    phi = phi - wp.floor(phi/tau)*tau
    sector = int(wp.floor(phi/g.phi_max))
    phi_local = phi - wp.float64(sector)*g.phi_max
    ir = wp.min(int(wp.floor(r/g.dr)), g.nr-2)
    it = int(wp.floor(theta/g.dtheta))
    ip = int(wp.floor(phi_local/g.dphi))
    er = r - wp.floor(r/g.dr)*g.dr
    et = theta - wp.floor(theta/g.dtheta)*g.dtheta
    ep = phi_local - wp.floor(phi_local/g.dphi)*g.dphi
    r_low = wp.float64(ir)*g.dr
    theta_high = wp.float64(it+1)*g.dtheta
    total = wp.float64(0.0)
    value = wp.float64(0.0)
    full_nphi = g.nphi*g.periods
    for p_side in range(2):
        for t_side in range(2):
            for r_side in range(2):
                ri = ir+1
                radial = (r_low + wp.float64(0.5)*er)*er
                jac_r = r_low
                if r_side == 1:
                    ri = ir
                    inv_er = g.dr-er
                    radial = (r + wp.float64(0.5)*inv_er)*inv_er
                    jac_r = r
                theta_width = et
                jac_theta = theta_high
                if t_side == 1:
                    theta_width = g.dtheta-et
                    jac_theta = theta
                phi_width = ep
                if p_side == 1:
                    phi_width = g.dphi-ep
                weight = (g.R0 + jac_r*wp.cos(jac_theta))*radial*theta_width*phi_width
                ti = (it-t_side+g.ntheta) % g.ntheta
                pi = (ip-p_side + (sector % g.periods)*g.nphi + full_nphi) % full_nphi
                value = value + n.values[ri, ti, pi]*weight
                total = total + weight
    return value/total


@wp.func
def rotation_vector(b: wp.vec3d, qdt2m: wp.float64, freq_corr: int):
    t = b * qdt2m
    if freq_corr == 1:
        magnitude = wp.length(b)
        if magnitude > wp.float64(0.0):
            t = b * (wp.tan(qdt2m*magnitude)/magnitude)
    return t


@wp.kernel
def record_trace(x: wp.array(dtype=wp.vec3d), ids: wp.array(dtype=wp.int32),
                 traces: wp.array2d(dtype=wp.vec3d), row: int):
    i = wp.tid()
    traces[row, i] = x[ids[i]]


@wp.kernel
def count_active(active: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if active[i] != 0:
        wp.atomic_add(count, 0, 1)


@wp.kernel
def initialize_velocity(
    x: wp.array(dtype=wp.vec3d),
    v: wp.array(dtype=wp.vec3d),
    qdt2m: wp.array(dtype=wp.float64),
    b_grid: VectorGrid,
    e_grid: VectorGrid,
    has_e: int,
    freq_corr: int,
):
    """Reproduce parallel_solver's existing startup formula literally."""
    i = wp.tid()
    e = wp.vec3d(wp.float64(0.0))
    if has_e == 1:
        e = sample_vector(e_grid, x[i]) * qdt2m[i]
    t = rotation_vector(sample_vector(b_grid, x[i]), qdt2m[i], freq_corr)
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
    collisions: CollisionConfig,
    density: DensityGrid,
    has_density: int,
    random_states: wp.array(dtype=wp.uint32),
    freq_corr: int,
):
    i = wp.tid()  # One logical GPU thread owns one particle.
    if active[i] == 0:
        return

    xi = x[i]
    vi = v[i]
    e = wp.vec3d(wp.float64(0.0))
    if has_e == 1:
        e = sample_vector(e_grid, xi) * qdt2m[i]
    t = rotation_vector(sample_vector(b_grid, xi), qdt2m[i], freq_corr)

    ne = collisions.n_e
    if collisions.ion != 0 and has_density == 1:
        ne = sample_density(density, b_grid, xi)
    state = wp.uint32(0)
    stochastic = collisions.neutral == 2 or collisions.ion != 0
    if stochastic:
        state = random_states[i]
    if collisions.neutral != 0 or collisions.ion != 0:
        vi, state = collision_hstep(vi, ne, collisions, dt, state)

    vm = vi + e
    vp = vm + wp.cross(vm, t)
    s = t * (wp.float64(2.0) / (wp.float64(1.0) + wp.dot(t, t)))
    vi = vm + wp.cross(vp, s) + e
    xi = xi + vi * dt
    # Match Torch: reuse pre-push density, and collide before recording wall v.
    if collisions.neutral != 0 or collisions.ion != 0:
        vi, state = collision_hstep(vi, ne, collisions, dt, state)
    if stochastic:
        random_states[i] = state
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


def make_density_grid(values, b_grid, *, scale=1.0):
    """Upload saved (phi, theta, r) scalar data, multiplying once into m^-3.

    Same transpose, scaling and nonnegative clamp as TorchMesh.loadScalarField.
    A pre-scaled field uses scale=1; a normalized profile uses PLASMA_DENSITY.
    """
    values = np.asarray(values, dtype=np.float64)
    expected = (b_grid.nphi*b_grid.periods, b_grid.ntheta, b_grid.nr)
    if values.shape != expected:
        raise ValueError(f'density must have aligned full-torus (phi, theta, r) shape {expected}')
    if not np.isfinite(scale) or scale < 0 or not np.isfinite(values).all():
        raise ValueError('density values must be finite and scale finite/nonnegative')
    internal = np.ascontiguousarray(np.maximum(values.transpose(2, 1, 0)*scale, 0.0))
    if not np.isfinite(internal).all():
        raise ValueError('scaled density must be finite')
    grid = DensityGrid()
    grid.values = wp.array(internal, dtype=wp.float64, device=b_grid.values.device)
    return grid


def _run_particles(x0, v0, charge_mass_ratio, b_grid, *, dt, steps, e_grid=None,
                   collisions=None, density_grid=None, seed=None, trace_ids=(),
                   trace_stride=1, freq_corr=False, show_progress=False):
    """Advance particles, returning device arrays and the valid trace length.

    steps is the number of position updates (existing self.nsteps - 1).
    hit_step == -1 marks survivors; their wall position/velocity rows are zero.
    Initial positions must be strictly inside the wall.
    collisions is a make_collision_config() result (default: all off).
    density_grid overrides the ion-ion constant density. seed controls Warp only.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    if isinstance(steps, bool) or int(steps) != steps or not 0 <= steps < 2**31:
        raise ValueError('steps must be a nonnegative int32 count')
    x0 = np.ascontiguousarray(x0, dtype=np.float64)
    v0 = np.ascontiguousarray(v0, dtype=np.float64)
    if x0.ndim != 2 or x0.shape[1] != 3 or len(x0) == 0 or v0.shape != x0.shape:
        raise ValueError('x0 and v0 must have matching nonempty (N, 3) shapes')
    trace_stride = int(trace_stride)
    if trace_stride < 1:
        raise ValueError('trace_stride must be a positive integer')
    ids = np.asarray(trace_ids)
    if ids.ndim != 1 or (ids.size and not np.issubdtype(ids.dtype, np.integer)):
        raise ValueError('trace_ids must be a one-dimensional sequence of particle indices')
    if np.any(ids < -len(x0)) or np.any(ids >= len(x0)):
        raise ValueError('trace index is outside the particle array')
    ids = np.asarray(ids, dtype=np.int32) % len(x0)
    q_over_m = np.broadcast_to(np.asarray(charge_mass_ratio, dtype=np.float64), (len(x0),))
    if not all(np.isfinite(arr).all() for arr in (x0, v0, q_over_m)):
        raise ValueError('particle data must be finite')
    xy2 = x0[:, 0]**2 + x0[:, 1]**2
    r2 = xy2 + x0[:, 2]**2 + b_grid.R0**2 - 2 * b_grid.R0 * np.sqrt(xy2)
    if np.any(np.sqrt(np.maximum(r2, 0.0)) >= b_grid.a):
        raise ValueError('initial particles must be strictly inside the wall')
    device = b_grid.values.device
    if collisions is None:
        collisions = make_collision_config()
    has_density = int(density_grid is not None)
    if density_grid is not None:
        expected = (b_grid.nr, b_grid.ntheta, b_grid.nphi*b_grid.periods)
        if density_grid.values.shape != expected or density_grid.values.device != device:
            raise ValueError('density must be aligned with B on the same device')
    else:
        density_grid = DensityGrid()
        density_grid.values = wp.zeros((1, 1, 1), dtype=wp.float64, device=device)
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
    trace_indices = wp.array(ids, dtype=wp.int32, device=device)
    traces = wp.empty((1 + int(steps)//trace_stride + 1, len(ids)), dtype=wp.vec3d, device=device)
    if len(ids):
        wp.launch(record_trace, dim=len(ids), inputs=[x, trace_indices, traces, 0], device=device)
    trace_row = 1
    remaining = wp.zeros(1, dtype=wp.int32, device=device)
    stochastic = collisions.neutral == 2 or collisions.ion != 0
    random_states = wp.empty(n if stochastic else 0, dtype=wp.uint32, device=device)
    if stochastic:
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31))
        if isinstance(seed, bool) or int(seed) != seed or not 0 <= seed < 2**31:
            raise ValueError('seed must be an integer in [0, 2**31)')
        wp.launch(initialize_random_states, dim=n, inputs=[random_states, int(seed)], device=device)

    wp.launch(initialize_velocity, dim=n,
              inputs=[x, v, qdt2m, b_grid, e_grid, has_e, int(freq_corr)], device=device)
    final_step = int(steps)
    iterator = range(1, int(steps) + 1)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, ncols=100, mininterval=2.0, desc='Boris (warp)')
    for step in iterator:
        wp.launch(push_and_record_wall, dim=n,
                  inputs=[x, v, qdt2m, active, wall_x, wall_v, hit_step,
                          last_inside_step, b_grid, e_grid, has_e, dt, step,
                          collisions, density_grid, has_density, random_states, int(freq_corr)],
                  device=device)
        if step % trace_stride == 0:
            if len(ids):
                wp.launch(record_trace, dim=len(ids), inputs=[x, trace_indices, traces, trace_row], device=device)
            trace_row += 1
        # Amortize host/device synchronization. Dead particles have frozen state,
        # so trim samples back to the exact last hit when all have terminated.
        if step % 256 == 0 or step == steps:
            remaining.zero_()
            wp.launch(count_active, dim=n, inputs=[active, remaining], device=device)
            n_remaining = int(remaining.numpy()[0])
            if show_progress:
                iterator.set_postfix({'active': n_remaining}, refresh=False)
            if n_remaining == 0:
                final_step = int(hit_step.numpy().max())
                break
    if show_progress:
        iterator.close()
    trace_count = 1 + final_step//trace_stride
    if final_step % trace_stride != 0:
        if len(ids):
            wp.launch(record_trace, dim=len(ids), inputs=[x, trace_indices, traces, trace_count], device=device)
        trace_count += 1
    wp.synchronize_device(device)
    return {'wall_position_xyz': wall_x, 'wall_velocity_xyz': wall_v,
            'hit_step': hit_step, 'last_inside_step': last_inside_step,
            'traces': traces, 'trace_count': trace_count}


def run_wall_only(x0, v0, charge_mass_ratio, b_grid, *, dt, steps, e_grid=None,
                  collisions=None, density_grid=None, seed=None):
    """Wall-only NumPy interface retained for the prototype benchmark."""
    result = _run_particles(x0, v0, charge_mass_ratio, b_grid, dt=dt, steps=steps,
                            e_grid=e_grid, collisions=collisions,
                            density_grid=density_grid, seed=seed)
    return {key: result[key].numpy() for key in
            ('wall_position_xyz', 'wall_velocity_xyz', 'hit_step', 'last_inside_step')}


def _vector_grid_from_torch(mesh):
    """Share a TorchMesh's float64 field without a host roundtrip or field copy."""
    import torch
    periods = np.asarray(mesh.periodicity)
    if periods.shape != (3,) or tuple(periods[:2]) != (0, 1) or periods[2] < 1:
        raise ValueError('Warp Boris requires mesh periodicity [0, 1, P] with P >= 1')
    expected = (mesh.nr, mesh.ntheta, mesh.nphi, 3)
    if mesh.B.dtype != torch.float64 or tuple(mesh.B.shape) != expected:
        raise ValueError('Warp Boris requires float64 Cartesian fields with shape (nr, ntheta, nphi, 3)')
    if min(expected[:3]) < 2 or not mesh.R0 > mesh.a > 0:
        raise ValueError('Warp Boris requires at least two nodes per axis and R0 > a > 0')
    g = VectorGrid()
    g.nr, g.ntheta, g.nphi = mesh.nr, mesh.ntheta, mesh.nphi
    g.periods = int(periods[2])
    g.R0, g.a = float(mesh.R0), float(mesh.a)
    g.dr, g.dtheta, g.dphi, g.phi_max = map(float, (mesh.dr, mesh.dtheta, mesh.dphi, mesh.phi_max))
    expected_spacing = (g.a/(g.nr-1), 2*np.pi/g.ntheta,
                        2*np.pi/(g.periods*g.nphi), 2*np.pi/g.periods)
    if not np.allclose((g.dr, g.dtheta, g.dphi, g.phi_max), expected_spacing, rtol=1e-12, atol=0):
        raise ValueError('Warp Boris requires uniform, aligned toroidal mesh spacing')
    addend = np.zeros(3)
    if bool(mesh.errField):
        addend = mesh.err_adder.detach().cpu().numpy() if torch.is_tensor(mesh.err_adder) else mesh.err_adder
    g.addend = wp.vec3d(*np.asarray(addend, dtype=np.float64))
    g.values = wp.from_torch(mesh.B, dtype=wp.vec3d)
    return g


def solve(solver, ions, Bfield, Efield=None, nfield=None, trace_IDs=(), trace_stride=1,
          freq_corr=False, ion_neutral_collisions=None, ion_ion_collisions=None):
    """Adapter returning the same four Torch tensors as Boris.parallel_solver."""
    from contextlib import nullcontext
    import logging
    import torch

    neutral = solver._resolve_ion_neutral_collision_model(ion_neutral_collisions)
    ion = solver._resolve_ion_ion_collision_model(ion_ion_collisions)
    c = make_collision_config(ion_neutral_collisions=neutral, ion_ion_collisions=ion,
                              n_gas=solver.n_gas, n_e=solver.n_e,
                              T_gas_eV=solver.T_gas_eV, Ti_eV=solver.Ti_eV,
                              m_gas_amu=solver.m_gas_amu)
    # Keep support for callers that explicitly override the linear-FP mass.
    c.thermal_ion = float(1.602176634e-19 * solver.Ti_eV / (solver.m_ion_amu * 1.660539068e-27))
    device = Bfield.B.device
    for field in (Efield, nfield if ion else None):
        if field is not None:
            values = field.B if hasattr(field, 'B') else field.value
            if values.device != device:
                raise ValueError('B, E and density fields must reside on the same device')
    wp.init()
    stream = wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device))) if device.type == 'cuda' else nullcontext()
    with stream:
        b = _vector_grid_from_torch(Bfield)
        e = _vector_grid_from_torch(Efield) if Efield is not None else None
        n = None
        if ion and nfield is not None:
            expected = (b.nr, b.ntheta, b.nphi*b.periods)
            if (tuple(nfield.value.shape) != expected or nfield.value.dtype != torch.float64
                    or tuple(nfield.periodicity) != (0, 1, 1)
                    or nfield.R0 != b.R0 or nfield.a != b.a
                    or not np.allclose([float(nfield.dr), float(nfield.dtheta), float(nfield.dphi)],
                                       [b.dr, b.dtheta, b.dphi], rtol=1e-12, atol=0)):
                raise ValueError('Warp density must be a float64 full-torus grid aligned with B')
            n = DensityGrid()
            n.values = wp.from_torch(nfield.value, dtype=wp.float64)
        logging.getLogger().info('Boris method: warp; ion-neutral=%s; ion-ion=%s', neutral, ion)
        result = _run_particles(
            np.asarray([particle.pos0_XYZ for particle in ions]),
            np.asarray([particle.vel0_XYZ for particle in ions]),
            np.asarray([particle.charge_mass_ratio for particle in ions]), b,
            dt=solver.dt, steps=solver.nsteps-1, e_grid=e,
            collisions=c, density_grid=n, trace_ids=trace_IDs,
            trace_stride=trace_stride, freq_corr=freq_corr, show_progress=True)
        for particle in ions:
            particle.setPosition(0, particle.pos0_XYZ)
        traces = (wp.to_torch(result['traces'])[:result['trace_count']] if len(trace_IDs)
                  else torch.empty((result['trace_count'], 0, 3), dtype=torch.float64, device=device))
        return (wp.to_torch(result['wall_position_xyz']), wp.to_torch(result['wall_velocity_xyz']),
                wp.to_torch(result['last_inside_step']), traces)
