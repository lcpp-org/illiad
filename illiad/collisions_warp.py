"""Warp ports of illiad.collisions, preserving the Boris driver's conventions."""

import numpy as np
import warp as wp

KB = wp.constant(wp.float64(1.602176634e-19))  # joules/eV
AMU = wp.constant(wp.float64(1.660539068e-27))
EPS0 = wp.constant(wp.float64(8.8541878128e-12))
SQRT_PI = wp.constant(wp.float64(np.sqrt(np.pi)))
LI_MASS = wp.constant(wp.float64(6.941 * 1.660539068e-27))
HE_MASS = wp.constant(wp.float64(4.002602 * 1.660539068e-27))


@wp.struct
class CollisionConfig:
    neutral: int  # 0=off, 1=viscous_drag, 2=langevin
    ion: int      # 0=off, 1=linear_fp, 2=fokker_planck
    n_gas: wp.float64
    thermal_neutral: wp.float64
    thermal_ion: wp.float64
    Ti_eV: wp.float64
    n_e: wp.float64  # fallback when no density grid is supplied


@wp.func
def neutral_hstep(v: wp.vec3d, eta: wp.vec3d, c: CollisionConfig, dt: wp.float64):
    nu = c.n_gas * wp.float64(1e-19) * wp.length(v)
    alpha = wp.exp(-nu * dt * wp.float64(0.5))
    updated = v * alpha
    if c.neutral == 2:
        sigma = wp.sqrt(c.thermal_neutral * (wp.float64(1.0) - alpha * alpha))
        updated = updated + sigma * eta
    return updated


@wp.func
def linear_fp_hstep(v: wp.vec3d, eta: wp.vec3d, ne: wp.float64,
                   c: CollisionConfig, dt: wp.float64):
    nu = wp.float64(3.61e-10) * ne * wp.pow(c.Ti_eV, wp.float64(-1.5))
    alpha = wp.exp(-nu * dt * wp.float64(0.5))
    sigma = wp.sqrt(c.thermal_ion * (wp.float64(1.0) - alpha * alpha))
    return v * alpha + sigma * eta


@wp.func
def chandrasekhar_psi(x: wp.float64):
    x = wp.max(x, wp.float64(0.0))
    sx = wp.sqrt(x)
    value = wp.float64(0.0)
    if x < wp.float64(1e-3):
        value = (wp.float64(4.0) / (wp.float64(3.0) * SQRT_PI) * x * sx
                 * (wp.float64(1.0) - wp.float64(0.6) * x
                    + wp.float64(3.0)/wp.float64(14.0) * x*x - x*x*x / wp.float64(18.0)))
    else:
        value = wp.erf(sx) - wp.float64(2.0) / SQRT_PI * sx * wp.exp(-x)
    return value


@wp.func
def coulomb_fp_rates(v: wp.vec3d, ne: wp.float64, Ti_eV: wp.float64):
    """Li-He, Z_a=Z_b=1, lnLambda=10, zero background drift, as in Boris."""
    w2 = wp.dot(v, v)
    wmag = wp.sqrt(wp.max(w2, wp.float64(1e-60)))
    x = wp.max(HE_MASS * w2 / (wp.float64(2.0) * KB * Ti_eV), wp.float64(1e-30))
    psi = chandrasekhar_psi(x)
    prime = wp.float64(2.0) / SQRT_PI * wp.sqrt(x) * wp.exp(-x)
    prefactor = ne * (KB*KB)*(KB*KB) * wp.float64(10.0)
    prefactor = prefactor / (wp.float64(12.566370614359172) * EPS0*EPS0 * LI_MASS*LI_MASS)
    nu0 = prefactor / (wmag*wmag*wmag)
    nu_s = (wp.float64(1.0) + LI_MASS/HE_MASS) * psi * nu0
    nu_perp = wp.float64(2.0) * ((wp.float64(1.0) - wp.float64(0.5)/x)*psi + prime) * nu0
    nu_parallel = psi / x * nu0
    return wp.vec3d(wp.max(nu_s, wp.float64(0.0)),
                    wp.max(nu_perp, wp.float64(0.0)),
                    wp.max(nu_parallel, wp.float64(0.0)))


@wp.func
def fokker_planck_hstep(v: wp.vec3d, eta: wp.vec3d, ne: wp.float64,
                      c: CollisionConfig, dt: wp.float64):
    w2 = wp.dot(v, v)
    what = v / wp.sqrt(wp.max(w2, wp.float64(1e-12)))
    rates = coulomb_fp_rates(v, ne, c.Ti_eV)
    eta_parallel = wp.dot(eta, what)
    eta_perp = eta - eta_parallel * what
    dt_h = dt * wp.float64(0.5)
    var_parallel = wp.max(rates[2] * w2 * dt_h, wp.float64(0.0))
    var_perp = wp.max(wp.float64(0.5) * rates[1] * w2 * dt_h, wp.float64(0.0))
    return (v - rates[0] * v * dt_h + wp.sqrt(var_parallel) * eta_parallel * what
            + wp.sqrt(var_perp) * eta_perp)


@wp.func
def collision_hstep(v: wp.vec3d, ne: wp.float64, c: CollisionConfig,
                    dt: wp.float64, state: wp.uint32):
    """Neutral then ion-ion; return the advanced RNG state explicitly."""
    eta = wp.vec3d(wp.float64(0.0))
    if c.neutral == 2:
        # Warp 1.17 randn returns float32; promote draws before float64 arithmetic.
        eta = wp.vec3d(wp.float64(wp.randn(state)), wp.float64(wp.randn(state)),
                       wp.float64(wp.randn(state)))
    if c.neutral != 0:
        v = neutral_hstep(v, eta, c, dt)
    if c.ion != 0:
        eta = wp.vec3d(wp.float64(wp.randn(state)), wp.float64(wp.randn(state)),
                       wp.float64(wp.randn(state)))
        if c.ion == 1:
            v = linear_fp_hstep(v, eta, ne, c, dt)
        else:
            v = fokker_planck_hstep(v, eta, ne, c, dt)
    return v, state


@wp.kernel
def initialize_random_states(states: wp.array(dtype=wp.uint32), seed: int):
    i = wp.tid()
    states[i] = wp.rand_init(seed, i)


def make_collision_config(*, ion_neutral_collisions=None, ion_ion_collisions=None,
                          n_gas=3e18, T_gas_eV=0.025, Ti_eV=2.0,
                          m_gas_amu=4.002603, n_e=1e18):
    """Mirror Boris.setConditions. None/False/'none'/'false'/'off' disable models.

    Compatibility: neutral thermal variance and linear_fp both use m_gas_amu,
    as in the current driver. The full FP model retains its fixed Li-He masses.
    """
    def resolve(value, models):
        if value is None or value is False:
            return 0
        if isinstance(value, str):
            key = value.strip().lower()
            if key in ('none', 'false', 'off'):
                return 0
            if key in models:
                return models[key]
        raise ValueError(f'Unknown collision model {value!r}; use {tuple(models)} or None')

    c = CollisionConfig()
    c.neutral = resolve(ion_neutral_collisions, {'viscous_drag': 1, 'langevin': 2})
    c.ion = resolve(ion_ion_collisions, {'linear_fp': 1, 'fokker_planck': 2})
    values = np.array([n_gas, T_gas_eV, Ti_eV, m_gas_amu, n_e], dtype=float)
    if not np.isfinite(values).all() or min(n_gas, T_gas_eV, n_e) < 0 or m_gas_amu <= 0 or Ti_eV <= 0:
        raise ValueError('Use finite nonnegative densities/neutral temperature and positive mass/ion temperature')
    c.n_gas, c.n_e, c.Ti_eV = float(n_gas), float(n_e), float(Ti_eV)
    c.thermal_neutral = float(1.602176634e-19 * T_gas_eV / (m_gas_amu * 1.660539068e-27))
    c.thermal_ion = float(1.602176634e-19 * Ti_eV / (m_gas_amu * 1.660539068e-27))
    return c
