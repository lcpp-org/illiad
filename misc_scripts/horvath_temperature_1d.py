"""Simple RLP-midplane version of Horvath et al., PPCF 48 (2006) 315.

Reference: input_files/Horvath_2006_WEGA_TemeraturesPlasma_Phys._Control._Fusion_48_315.pdf
Equations (3)-(6), (11)-(15): fast-to-slow electron heating, perpendicular
transport, and the SECOND (sheath-limited) SOL heat-loss model.

Run from anywhere: python misc_scripts/horvath_temperature_1d.py
Edit the settings and input functions below for a different discharge.

This is a slab approximation along z=0 at computational phi=306 degrees:
no equivalent-area r_eff, cylindrical Jacobian, or flux-surface averaging.
The inner symmetry point is the maximum of the saved flux on this midplane
(an axis proxy, not necessarily the true 3D magnetic axis). Only its outboard
branch is solved. Plot distance d=R0+a-R increases INWARD from the outer wall;
the solver coordinate x=R-R_axis increases OUTWARD. ILLIAD psi is 1 at the
axis and 0 at the LCFS, and supplies the example core density profile.

The input profiles are EXAMPLES, not measurements or a reproduction of the
paper's fit. Prescribed neutrals cause ionization and excitation losses;
recombination remains omitted. D is prescribed rather than
obtained from a particle-balance solve. chi is prescribed unless an axis
temperature is supplied, in which case it is fitted as in the paper.
Temperature and total perpendicular heat flux are continuous at the LCFS;
dT/dx=0 at the inner boundary and a prescribed gradient at the outer endpoint.
The endpoint is slightly inside the wall to avoid L=0. No field lines retrace.
Saved connection lengths retain any finite tracing cutoff; this script does
not recover uncensored wall distances from trace-limited values.
"""

import argparse
import json
from functools import lru_cache
import re
import warnings
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e, epsilon_0, m_e, atomic_mass
from scipy.integrate import solve_bvp
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from illiad.sol import load_lcfs_boundary


# Saved geometry: these three inputs must describe the same configuration/LCFS.
ANALYSIS = "IOTA3_1000sp_atol1e-9"
FLUX_FILE = "LCFS40/nField_LCFS40_linear.npy"
SOL_SUBDIR = "SOLtrace_500c_RegularGrid"
LCFS_INDEX = 40
PHI_DEG = 306.0                 # Computational angle, as in plotPsiProfile_NEW.py
VESSEL_RADIUS_M = 0.19
WALL_CLEARANCE_M = 0.002
SAMPLES_PER_REGION = 250

# Editable example plasma inputs; all temperatures are eV, densities m^-3.
N_AXIS = 1.0e18
N_LCFS = 1.0e17
N_FLOOR = 1.0e15
DENSITY_SOL_DECAY_M = 0.01
FAST_FRACTION = 0.05            # n_fast / n_slow
FAST_TEMPERATURE_EV = 120.0
COULOMB_LOG = 10.0
ION_MASS_U = 4.0026             # Argon 39.948; use  for helium, 1.0078 for H
SHEATH_DELTA = 5.5              # Horvath: Ar ~11, He ~5.5, H ~5
L_TO_WALL_FACTOR = 1.0          # Fraction of saved wall-to-wall L used in sheath loss
CHI_M2_S = 10.5                 # Constant perpendicular thermal diffusivity
PARTICLE_D_M2_S = 0.8            # Set >0 to include core particle convection
AXIS_TEMPERATURE_EV = None      # Measured value: fit chi if supplied
OUTER_GRADIENT_EV_M = 0.0       # dT/dx, x increasing outward (not plot distance)

# Uniform, prescribed NEUTRAL densities [m^-3]; examples, not measurements.
# One species: {"He": 1e16}; mixture: {"He": 1e16, "Li": 1e14}; {} disables.
# CLI --neutral He=1e16 --neutral Li=1e14 replaces this entire dictionary.
NEUTRAL_DENSITIES_M3 = {"He": 1e16, "Li": 1e14}
# The cold SOL extends below the ADAS Te grid: explicitly hold boundary PLT.
# Set "raise" to require all physical samples to lie within the ADAS grid.
PLT_BOUNDS = "clip"


def heating_w_m3(temperature_ev, n_slow, n_fast, t_fast):
    """Horvath (3)-(4), with every temperature converted from eV to joules."""
    tau = (4*np.pi*epsilon_0/e**2)**2 * 3*np.sqrt(m_e)*(e*t_fast)**1.5
    tau = tau / (8*np.sqrt(2*np.pi)*n_slow*COULOMB_LOG)
    return 1.5*n_fast*e*(t_fast-temperature_ev)/tau



def slow_density_m3(x, psi, lcfs_x):
    """INPUT HOOK: replace with measured density or a particle-balance model."""
    core = N_LCFS + (N_AXIS - N_LCFS) * np.clip(psi, 0.0, 1.0)
    sol = N_FLOOR + (N_LCFS - N_FLOOR) * np.exp(-np.maximum(x-lcfs_x, 0) / DENSITY_SOL_DECAY_M)
    return np.where(x <= lcfs_x, core, sol)


def fast_density_m3(x, psi, n_slow):
    """INPUT HOOK: replace the example constant fast/slow density ratio."""
    return FAST_FRACTION * n_slow


def fast_temperature_ev(x, psi):
    """INPUT HOOK: replace with a measured/calculated fast-electron profile."""
    return np.full_like(x, FAST_TEMPERATURE_EV)



# LOSS DUE TO ATOMIC PROCESSES
def validate_neutral_densities(neutrals):
    """Normalize species symbols and check prescribed uniform densities."""
    result = {}
    for species, density in neutrals.items():
        symbol = str(species).strip().capitalize()
        if symbol in result:
            raise ValueError(f"Duplicate neutral species: {symbol}.")
        value = float(density)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Neutral {symbol} density must be finite and nonnegative.")
        # Every enabled species needs both tables. Fail before running a BVP.
        if value > 0:
            load_adas_plt(symbol)
            if (symbol, 0) not in load_voronov_table():
                raise ValueError(f"No neutral ionization data for {symbol}.")
        result[symbol] = value
    return result


def atomic_loss_w_m3(x, temperature_ev, n_slow, n_fast, *,
                     neutral_densities=None, return_details=False):
    """Slow-electron ionization + PLT excitation losses from prescribed neutrals.

    Neutral densities are constant in space, in m^-3, and are not depleted or
    evolved. Species created by ionization are not tracked. Only energy lost
    by SLOW electrons is subtracted from this BVP; fast electrons remain a
    prescribed heating population. PLT uses total electron density ns+nf for
    its collisional-radiative density coordinate and slow Te for temperature.
    Neutrals are treated as stationary targets. Recombination is not included.
    x supplies the spatial shape, in outward midplane meters.
    """
    neutrals = validate_neutral_densities(
        NEUTRAL_DENSITIES_M3 if neutral_densities is None else neutral_densities)
    x, te, ns, nf = np.broadcast_arrays(x, temperature_ev, n_slow, n_fast)
    for value, name in ((te, "temperature_ev"), (ns, "n_slow"), (nf, "n_fast")):
        _nonnegative_finite(value, name)
    total = np.zeros_like(te, dtype=float)
    details = {}
    for symbol, density in neutrals.items():
        ionization = np.zeros_like(total)
        excitation = np.zeros_like(total)
        if density > 0:
            ionization = ionization_loss_w_m3(symbol, [0], te, ns, density,
                                              check_fit_range=False)
            coefficient = excitation_coefficient_w_m3(symbol, [0], te, ns+nf,
                                                       bounds=PLT_BOUNDS)[0]
            excitation = ns*density*coefficient
        details[symbol] = dict(ionization_w_m3=ionization, excitation_w_m3=excitation)
        total = total+ionization+excitation
    return (total, details) if return_details else total


@lru_cache(maxsize=4)
def load_voronov_table(table_path=ROOT / "input_files" / "voronov_table.dat"):
    """Read the fixed-width ion label followed by Z,N,E,P,A,X,K,Tmin,Tmax.

    Source/units: https://www.pa.uky.edu/~verner/dima/col/cfit.txt
    A is converted from cm^3/s to m^3/s; Tmax from keV to eV.
    Clear this function's cache after modifying the table in a live session.
    """
    records = {}
    for line in Path(table_path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        symbol = re.match(r"[A-Za-z]+", line).group().capitalize()
        z, electrons, energy, p, a, x, k, tmin, tmax = map(float, line[5:].split())
        charge = int(z - electrons)
        records[symbol, charge] = (energy, p, a*1e-6, x, k, tmin, tmax*1000)
    return records


def _ionization_states(species, charge_states, table):
    """Each listed initial charge q selects q->q+1; preserve input order."""
    symbol = str(species).strip().capitalize()
    if not any(key[0] == symbol for key in table):
        raise ValueError(f"Unknown species {species!r}; use an element symbol from H to Ni.")
    selection = np.asarray(charge_states)
    if (selection.ndim != 1 or selection.size == 0
            or selection.dtype.kind not in "iu"):
        raise ValueError("charge_states must be a nonempty 1D list or array of integers, e.g. [0, 1].")
    states = [int(q) for q in selection]
    if any(q < 0 for q in states) or len(set(states)) != len(states):
        raise ValueError("charge_states must contain unique, nonnegative integers.")
    if any((symbol, q) not in table for q in states):
        raise ValueError(f"No ionization fit for one or more selected {symbol} charge states.")
    return symbol, states


def _nonnegative_finite(value, name):
    value = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(value)) or np.any(value < 0):
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def ionization_rate_m3_s(species, charge_states, temperature_ev, *,
                         target_temperature_ev=0.0,
                         target_mass_u=None, check_fit_range=True,
                         table_path=ROOT / "input_files" / "voronov_table.dat"):
    """Return {initial_charge: rate_coefficient [m^3/s]} for Maxwellian electrons.

    'O', [0, 1] selects O->O+ and O+->O2+; [0] selects only O->O+.
    Lists or 1D integer arrays may select nonconsecutive states, e.g. [0, 2].
    All temperatures are eV. target_temperature_ev is a scalar/profile shared
    by all selected charge states, or {charge: temperature} for distinct
    neutral/ion temperatures. Dictionary values may also be spatial arrays.
    Array axes represent spatial samples, NOT charge states; use a dictionary
    for temperatures that differ by charge. Default zero means cold targets.
    Supply target_mass_u (atomic mass units) for nonzero target temperatures.
    For co-moving Maxwellians, use T_eff=Te+(m_e/m_target)*T_target in the
    stationary-target fit. This thermal-motion approximation excludes bulk
    drift and non-Maxwellian populations. U is calculated internally.

    Rates use U=E_ion/T_eff and A*(1+P*sqrt(U))*U**K*exp(-U)/(X+U).
    Source: https://www.pa.uky.edu/~verner/dima/col/cfit.f
    At zero T the rate is zero. Positive temperatures outside the tabulated
    validity interval warn (no clamping); check_fit_range=False suppresses it.
    """
    table = load_voronov_table(table_path)
    symbol, states = _ionization_states(species, charge_states, table)
    te = _nonnegative_finite(temperature_ev, "temperature_ev")
    if isinstance(target_temperature_ev, dict):
        if set(target_temperature_ev) != set(states):
            raise ValueError(f"target_temperature_ev keys must be exactly {states}.")
        temperatures = target_temperature_ev
    else:
        temperatures = dict.fromkeys(states, target_temperature_ev)
    temperatures = {q: _nonnegative_finite(t, f"target_temperature_ev[{q}]")
                    for q, t in temperatures.items()}
    if target_mass_u is not None and (not np.isfinite(target_mass_u) or target_mass_u <= 0):
        raise ValueError("target_mass_u must be a positive finite scalar.")
    if target_mass_u is None and any(np.any(t > 0) for t in temperatures.values()):
        raise ValueError("Supply target_mass_u for nonzero target temperatures.")
    mass_ratio = 0.0 if target_mass_u is None else m_e/(target_mass_u*atomic_mass)
    rates = {}
    for q in states:
        energy, p, a, x, k, tmin, tmax = table[symbol, q]
        effective = te + mass_ratio*temperatures[q]
        if check_fit_range and np.any((effective > 0) & ((effective < tmin) | (effective > tmax))):
            warnings.warn(f"{symbol} {q}->{q+1}: extrapolating Voronov fit outside "
                          f"{tmin:g}-{tmax:g} eV.", RuntimeWarning, stacklevel=2)
        # The published reference routine returns zero for U>80; this also
        # avoids division overflow at arbitrarily small positive temperatures.
        active = effective >= energy/80.0
        u = energy/np.where(active, effective, energy)
        rates[q] = np.where(active, a*(1+p*np.sqrt(u))*u**k*np.exp(-u)/(x+u), 0.0)
    return rates


def ionization_loss_w_m3(species, charge_states, temperature_ev, n_slow, n_target, *,
                         target_temperature_ev=0.0,
                         target_mass_u=None, fasttemp_ev=None, fast_frac=0.0,
                         return_details=False, check_fit_range=True,
                         table_path=ROOT / "input_files" / "voronov_table.dat"):
    """Sum n_target[q] * (n_slow*k_slow[q]+n_fast*k_fast[q]) * E_ion[q]*e.

    Returns total threshold-energy loss [W/m^3]. All densities are m^-3.
    n_target is {initial_charge: density}; a scalar/array is also accepted for
    a SINGLE transition. [0, 1] requires {0: n_neutral, 1: n_singly_ionized}.
    No charge-state abundances are inferred or evolved. Arrays broadcast.
    fast_frac means n_fast/n_slow (the same convention as FAST_FRACTION),
    not n_fast/(n_slow+n_fast). Supply fasttemp_ev when fast_frac is nonzero.
    Both electron populations encounter the same target_temperature_ev for
    each charge state. See ionization_rate_m3_s for temperature input formats.
    With return_details=True, return (total_loss, details_by_initial_charge),
    including coefficients, events per volume per second, and separate losses.
    The slow-electron BVP should use only loss_slow_w_m3, not the fast loss.

    Example: ionization_loss_w_m3('O', [0, 1], 20., 1e18,
                {0: 1e16, 1: 2e15}, target_temperature_ev={0: 0.03, 1: 1.},
                target_mass_u=15.999)
    """
    ns = _nonnegative_finite(n_slow, "n_slow")
    fraction = _nonnegative_finite(fast_frac, "fast_frac")
    te = _nonnegative_finite(temperature_ev, "temperature_ev")
    kwargs = dict(target_temperature_ev=target_temperature_ev, target_mass_u=target_mass_u,
                  check_fit_range=check_fit_range, table_path=table_path)
    slow = ionization_rate_m3_s(species, charge_states, te, **kwargs)
    if np.any(fraction > 0) and fasttemp_ev is None:
        raise ValueError("Supply fasttemp_ev when fast_frac is nonzero.")
    fast = (ionization_rate_m3_s(species, charge_states, fasttemp_ev, **kwargs)
            if fasttemp_ev is not None else {q: np.zeros_like(k) for q, k in slow.items()})
    if not isinstance(n_target, dict):
        if len(slow) != 1:
            raise ValueError("Multiple transitions require n_target={charge: density, ...}.")
        n_target = {next(iter(slow)): n_target}
    if set(n_target) != set(slow):
        raise ValueError(f"n_target keys must be exactly the initial charge states {list(slow)}.")
    table = load_voronov_table(table_path)
    symbol = str(species).strip().capitalize()
    details, total = {}, 0.0
    for q, kslow in slow.items():
        target = _nonnegative_finite(n_target[q], f"n_target[{q}]")
        energy = table[symbol, q][0]
        slow_events = target*ns*kslow
        fast_events = target*ns*fraction*fast[q]
        slow_loss, fast_loss = slow_events*energy*e, fast_events*energy*e
        details[q] = dict(rate_slow_m3_s=kslow, rate_fast_m3_s=fast[q],
                          events_m3_s=slow_events+fast_events,
                          loss_slow_w_m3=slow_loss, loss_fast_w_m3=fast_loss,
                          ionization_energy_ev=energy)
        total = total + slow_loss + fast_loss
    return (total, details) if return_details else total


@lru_cache(maxsize=4)
def load_adas_plt(species, table_path=None):
    """Load the uploaded unresolved ADF11 PLT tables (He or Li), independently.

    Format: https://www.adas.ac.uk/man/appxa-11.pdf
    Density is log10(cm^-3), temperature log10(eV), power log10(W cm^3).
    Convert to log10(m^-3) and log10(W m^3). Density varies fastest in each
    temperature row. For excitation, emitting charge q=Z1-1 (not Z1).
    Only one unresolved block per charge is supported; resolved/metastable
    files are rejected. Clear the cache after editing a file in a live session.
    """
    symbol = str(species).strip().capitalize()
    atomic_numbers = {"He": 2, "Li": 3}
    if symbol not in atomic_numbers:
        raise ValueError("Excitation PLT data are available for He and Li only.")
    path = (ROOT / "input_files" / "excitation_coefs" / f"plt96_{symbol.lower()}.dat"
            if table_path is None else Path(table_path))
    lines = path.read_text().splitlines()
    z, nd, nt, first, last = map(int, lines[0].split()[:5])
    if z != atomic_numbers[symbol] or nd < 2 or nt < 2:
        raise ValueError("PLT species/header does not match the requested element or grid.")
    headers = [(i, int(match.group(1))) for i, line in enumerate(lines)
               if (match := re.search(r"^[-]+/.*\bZ1\s*=\s*(\d+)", line))]
    if [stage for _, stage in headers] != list(range(first, last+1)):
        raise ValueError("Expected exactly one unresolved PLT block per charge stage.")

    def numbers(start, stop):
        values = []
        for line in lines[start:stop]:
            stripped = line.strip()
            if not stripped or stripped.startswith(("C", "c", "#", "--")):
                continue
            values.extend(float(token.replace("D", "E").replace("d", "e"))
                          for token in stripped.split())
        return np.asarray(values)

    if not headers:
        raise ValueError("No PLT charge blocks found.")
    grids = numbers(1, headers[0][0])
    if grids.size != nd+nt or not np.all(np.isfinite(grids)):
        raise ValueError("Invalid PLT grids; only unresolved ADF11 files are supported.")
    log_ne, log_te = grids[:nd]+6.0, grids[nd:]
    if np.any(np.diff(log_ne) <= 0) or np.any(np.diff(log_te) <= 0):
        raise ValueError("PLT density and temperature grids must be strictly increasing.")
    power, interpolators = {}, {}
    for index, (line_number, stage) in enumerate(headers):
        stop = headers[index+1][0] if index+1 < len(headers) else len(lines)
        block = numbers(line_number+1, stop)
        if block.size != nt*nd or not np.all(np.isfinite(block)):
            raise ValueError(f"Invalid PLT coefficient block Z1={stage}.")
        q = stage-1
        power[q] = block.reshape(nt, nd)-6.0
        interpolators[q] = RegularGridInterpolator((log_te, log_ne), power[q],
                                                   method="linear", bounds_error=True)
    return dict(log_temperature_ev=log_te, log_density_m3=log_ne,
                log_power_w_m3=power, interpolators=interpolators, path=path)


def excitation_coefficient_w_m3(species, charge_states, temperature_ev, n_e, *,
                                bounds="raise", table_path=None):
    """Return {emitting_charge: PLT coefficient [W m^3]}, NOT event rates.

    charge_states is a list/1D integer array: [0] for neutrals, [0, 1] for
    neutral and singly charged emitters. Excitation preserves charge.
    Te [eV] and total electron density n_e [m^-3] broadcast over spatial points.
    Bilinear interpolation uses log10(Te), log10(ne), and log10(PLT).
    bounds='raise' rejects points outside the table; 'clip' explicitly holds
    coefficients at the nearest temperature/density boundary. At Te=0 or ne=0
    return zero (a zero-loss convention, not a tabulated PLT value).

    PLT is a Maxwellian collisional-radiative power coefficient, with density
    dependence. It is not a single-transition <sigma*v>; do not multiply by
    an excitation energy or e. No target-temperature correction is applied.
    """
    data = load_adas_plt(species, table_path)
    states = np.asarray(charge_states)
    if (states.ndim != 1 or states.size == 0 or states.dtype.kind not in "iu"
            or len(set(states.tolist())) != len(states)
            or any(int(q) not in data["interpolators"] for q in states)):
        raise ValueError(f"charge_states must be unique integer emitting charges from {list(data['interpolators'])}.")
    if bounds not in ("raise", "clip"):
        raise ValueError("bounds must be 'raise' or 'clip'.")
    te, ne = np.broadcast_arrays(np.asarray(temperature_ev, dtype=float), np.asarray(n_e, dtype=float))
    if np.any(~np.isfinite(te)) or np.any(~np.isfinite(ne)) or np.any(te < 0) or np.any(ne < 0):
        raise ValueError("Electron temperature and density must be finite and nonnegative.")
    active = (te > 0) & (ne > 0)
    points = np.column_stack((np.log10(te[active]), np.log10(ne[active])))
    for axis, grid in enumerate((data["log_temperature_ev"], data["log_density_m3"])):
        if bounds == "raise" and np.any((points[:, axis] < grid[0]-1e-12)
                                         | (points[:, axis] > grid[-1]+1e-12)):
            name = "temperature [eV]" if axis == 0 else "density [m^-3]"
            raise ValueError(f"PLT {name} outside {10**grid[0]:g} to {10**grid[-1]:g}; "
                             "use bounds='clip' only if boundary holding is intended.")
        points[:, axis] = np.clip(points[:, axis], grid[0], grid[-1])
    result = {}
    for q in states:
        values = np.zeros(te.shape)
        values[active] = 10.0**data["interpolators"][int(q)](points)
        result[int(q)] = values
    return result


def excitation_loss_w_m3(species, charge_states, temperature_ev, n_slow, n_target, *,
                          fasttemp_ev=None, fast_frac=0.0, bounds="raise",
                          return_details=False, table_path=None):
    """Return sum_q n_target[q]*(n_slow*PLT_slow[q]+n_fast*PLT_fast[q]) [W/m^3].

    Densities are m^-3 and temperatures eV. n_target={charge: density/profile};
    for one selected charge a scalar/profile is also accepted. fast_frac is
    n_fast/n_slow. By default this is the single-Maxwellian ADAS PLT loss.
    A nonzero fast_frac is an APPROXIMATE superposition of Maxwellian PLT
    coefficients, both evaluated at total ne=n_slow*(1+fast_frac). It does not
    solve the collisional-radiative populations for a two-temperature EEDF.
    Both coefficients already contain radiated energy; no extra e or E_exc.
    This excludes recombination/bremsstrahlung (PRB) and ionization losses.
    return_details=True returns (total, {q: coefficients and separate losses}).
    Use only loss_slow_w_m3 when adding losses to the slow-electron BVP.

    Example: excitation_loss_w_m3('He', [0, 1], 10., 1e18,
                                  {0: 1e16, 1: 2e15})
    """
    ns, fraction = np.broadcast_arrays(np.asarray(n_slow, dtype=float),
                                       np.asarray(fast_frac, dtype=float))
    if np.any(~np.isfinite(ns)) or np.any(~np.isfinite(fraction)) or np.any(ns < 0) or np.any(fraction < 0):
        raise ValueError("n_slow and fast_frac must be finite and nonnegative.")
    if np.any(fraction > 0) and fasttemp_ev is None:
        raise ValueError("Supply fasttemp_ev when fast_frac is nonzero.")
    ne = ns*(1+fraction)
    kwargs = dict(bounds=bounds, table_path=table_path)
    slow = excitation_coefficient_w_m3(species, charge_states, temperature_ev, ne, **kwargs)
    fast = (excitation_coefficient_w_m3(species, charge_states, fasttemp_ev, ne, **kwargs)
            if fasttemp_ev is not None else {q: np.zeros_like(value) for q, value in slow.items()})
    if not isinstance(n_target, dict):
        if len(slow) != 1:
            raise ValueError("Multiple emitting charges require n_target={charge: density, ...}.")
        n_target = {next(iter(slow)): n_target}
    if set(n_target) != set(slow):
        raise ValueError(f"n_target keys must be exactly {list(slow)}.")
    details, total = {}, 0.0
    for q in slow:
        target = np.asarray(n_target[q], dtype=float)
        if np.any(~np.isfinite(target)) or np.any(target < 0):
            raise ValueError(f"n_target[{q}] must be finite and nonnegative.")
        slow_loss, fast_loss = target*ns*slow[q], target*ns*fraction*fast[q]
        details[q] = dict(coefficient_slow_w_m3=slow[q], coefficient_fast_w_m3=fast[q],
                          loss_slow_w_m3=slow_loss, loss_fast_w_m3=fast_loss)
        total = total+slow_loss+fast_loss
    return (total, details) if return_details else total


def recombination_loss_w_m3(x, temperature_ev, n_slow, n_fast, n_ion, recombination_energy_ev):
    """CALCULATION SKELETON: return recombination energy loss in W/m^3.

    TODO: add species-specific recombination rates and thresholds.
    """
    return np.zeros_like(temperature_ev)



# LOSS DUE TO TRANSPORT
def perpendicular_cond_loss_w_m3(temperature_ev, n_slow, length_to_wall):
    """Horvath (11)-(14); perpendicular conduction loss in W/m^3."""
    # Nonnegative trial values keep Newton iterations real; final T must be >0.
    energy = e*np.maximum(temperature_ev, 0.0)

    return n_slow * energy * np.sqrt(energy/(ION_MASS_U*atomic_mass)) / length_to_wall


def parallel_cond_loss_w_m3(temperature_ev, n_slow, length_to_wall):
    """Horvath (11)-(14) parallel conduction loss in W/m^3.; no X-point multiplier in this second SOL method."""
    # Nonnegative trial values keep Newton iterations real; final T must be >0.
    energy = e*np.maximum(temperature_ev, 0.0)
    return SHEATH_DELTA*n_slow*energy*np.sqrt(energy/(ION_MASS_U*atomic_mass))/length_to_wall


def perpendicular_conv_loss_w_m3(r, x, t, core):
    """Horvath perpendicular convective loss in W/m^3."""
    return -1.5*e*t*PARTICLE_D_M2_S*r["ns"].derivative()(x) if core else 0.0


def parallel_conv_loss_w_m3(temperature_ev, n_slow, length_to_wall):
    """Horvath parallel convective loss in W/m^3. (neglected for all regions!)"""
    pass


def midplane_line(field, rho, theta, phi):
    """Read only the requested plane/rows, preserving the saved grid contract."""
    if field.shape != (len(phi), len(theta), len(rho)):
        raise ValueError("Field shape does not match (phi, theta, rho) coordinates.")
    for coordinate in (rho, theta, phi):
        if not np.all(np.isfinite(coordinate)) or np.any(np.diff(coordinate) <= 0):
            raise ValueError("Grid coordinates must be finite and strictly increasing.")
    def index(values, target):
        matches = np.flatnonzero(np.isclose(values, target, rtol=0, atol=1e-8))
        if len(matches) != 1:
            raise ValueError(f"Grid must contain the requested coordinate {target:g} exactly.")
        return matches[0]
    ip = index(phi, PHI_DEG)
    inner = np.asarray(field[ip, index(theta, np.pi)], dtype=float)
    outer = np.asarray(field[ip, index(theta, 2*np.pi)], dtype=float)
    # x here is signed R-R0. Drop the duplicate rho=0 on the inboard branch.
    negative = rho > 0
    return np.r_[-rho[negative][::-1], rho], np.r_[inner[negative][::-1], outer]


def load_profiles():
    data = ROOT / "output" / ANALYSIS / "data"
    sol_dir = data / SOL_SUBDIR
    if int(np.load(sol_dir / "trace_lcfs_index.npy")) != LCFS_INDEX:
        raise ValueError("SOL trace LCFS does not match LCFS_INDEX.")
    if not np.isclose(float(np.load(sol_dir / "trace_vessel_radius_m.npy")), VESSEL_RADIUS_M):
        raise ValueError("SOL trace vessel radius does not match VESSEL_RADIUS_M.")
    flux = np.load(data / FLUX_FILE, mmap_mode="r")
    # Native nField coordinates, as in Mesh.loadScalarField(period=[0,1,1]).
    nphi, ntheta, nr = flux.shape
    r_flux, psi = midplane_line(flux, np.linspace(0, VESSEL_RADIUS_M, nr),
                              np.linspace(2*np.pi/ntheta, 2*np.pi, ntheta),
                              np.linspace(360/nphi, 360, nphi))
    # Saved interpolants may overshoot below zero near/outside the LCFS.
    # Preserve their core samples; only the example density hook clips psi.
    if not np.all(np.isfinite(psi)) or not 0 < np.max(psi) <= 1.001:
        raise ValueError("Expected finite ILLIAD normalized flux with a maximum near 1.")
    boundary, _ = load_lcfs_boundary(ANALYSIS, PHI_DEG, LCFS_INDEX)
    end = np.roll(boundary, -1, axis=0)
    crossing = (boundary[:, 1] > 0) != (end[:, 1] > 0)
    a, b = boundary[crossing], end[crossing]
    intersections = a[:, 0] - a[:, 1]*(b[:, 0]-a[:, 0])/(b[:, 1]-a[:, 1])
    if len(intersections) != 2:
        raise ValueError("This simple model requires two LCFS intersections with z=0.")
    r_lcfs = intersections.max()
    confined = (r_flux > intersections.min()) & (r_flux < r_lcfs)
    axis_index = np.flatnonzero(confined)[np.argmax(psi[confined])]
    r_axis = r_flux[axis_index]
    lcfs_x = r_lcfs - r_axis
    wall_x = VESSEL_RADIUS_M - WALL_CLEARANCE_M - r_axis
    if not 0 < lcfs_x < wall_x:
        raise ValueError("Require axis proxy < outboard LCFS < outer endpoint.")

    length = np.load(sol_dir / "connection_length_field_m.npy", mmap_mode="r")
    r_sol, lengths = midplane_line(length, *(np.load(sol_dir / name) for name in
                                          ("rho_grid_m.npy", "theta_grid_rad.npy", "phi_grid_deg.npy")))
    valid = (r_sol > r_lcfs) & np.isfinite(lengths) & (lengths > 0)
    if valid.sum() < 2 or r_sol[valid][-1] < r_axis+wall_x:
        raise ValueError("Insufficient positive SOL connection lengths to the outer endpoint.")
    first = np.flatnonzero(valid)[0]
    last = np.searchsorted(r_sol, r_axis+wall_x)
    if not np.all(valid[first:last+1]):
        raise ValueError("Missing connection lengths inside the SOL; regularize them first.")
    gap = r_sol[first] - r_lcfs
    if gap > 3*np.max(np.diff(r_sol)):
        raise ValueError("Connection-length gap at LCFS exceeds three grid spacings.")
    print(f"Holding first exterior L constant across {gap*1000:.2f} mm LCFS mask gap.")
    length_interp = PchipInterpolator(r_sol[valid]-r_axis, lengths[valid], extrapolate=False)

    regions = []
    for x in (np.linspace(0, lcfs_x, SAMPLES_PER_REGION),
              np.linspace(lcfs_x, wall_x, SAMPLES_PER_REGION)):
        core = len(regions) == 0
        p = np.interp(x+r_axis, r_flux, psi) if core else np.zeros_like(x)
        if core:
            p[-1] = 0.0  # exact LCFS value for the core hook
        ns = slow_density_m3(x, p, lcfs_x)
        nf, tf = fast_density_m3(x, p, ns), fast_temperature_ev(x, p)
        if not (np.all(np.isfinite([ns, nf, tf])) and np.all(ns > 0)
                and np.all(nf >= 0) and np.all(tf > 0)):
            raise ValueError("Input hooks must return finite positive densities/temperatures (nf may be zero).")
        L = np.full_like(x, np.inf) if core else L_TO_WALL_FACTOR*length_interp(np.maximum(x, r_sol[first]-r_axis))
        regions.append(dict(x=x, psi=p, ns=PchipInterpolator(x, ns),
                            nf=PchipInterpolator(x, nf), tf=PchipInterpolator(x, tf),
                            L=None if core else PchipInterpolator(x, L)))
    return regions, r_axis, lcfs_x


def solve_temperature(regions, chi, axis_temperature=None, *, neutral_densities=None):
    """Two-domain BVP in (T, q_total); explicit LCFS matching avoids a step RHS."""
    if not np.isfinite(chi) or chi <= 0 or (axis_temperature is not None and
            (not np.isfinite(axis_temperature) or axis_temperature <= 0)):
        raise ValueError("chi and an optional measured axis temperature must be positive.")

    neutrals = validate_neutral_densities(
        NEUTRAL_DENSITIES_M3 if neutral_densities is None else neutral_densities)
    lengths = [r["x"][-1]-r["x"][0] for r in regions]
    q_scale = 1000.0  # W/m^2, just numerical conditioning

    # perpendicular convective transport only in core
    # def convection(r, x, t, core):
    #     return -1.5*e*t*PARTICLE_D_M2_S*r["ns"].derivative()(x) if core else 0.0

    def rhs(u, y, parameters=()):
        diffusivity = np.exp(parameters[0]) if axis_temperature is not None else chi
        result = []
        for i, r in enumerate(regions):

            x = r["x"][0] + lengths[i]*u
            t, q = y[2*i], y[2*i+1]*q_scale
            ns, nf, tf = r["ns"](x), r["nf"](x), r["tf"](x)

            # perpendicular convective transport only in core
            dtransp = (perpendicular_conv_loss_w_m3(r, x, t, i == 0)-q)/(e*ns*diffusivity)

            # parallel conductive transport only outside core
            parallel = 0.0
            if i > 0:
                parallel = parallel_cond_loss_w_m3(t, ns, r["L"](x))

            # Newton trial temperatures may be negative; final temperatures must be positive.
            loss = atomic_loss_w_m3(x, np.maximum(t, 0.0), ns, nf,
                                    neutral_densities=neutrals)

            dq = heating_w_m3(t, ns, nf, tf)-loss-parallel
            result.extend((lengths[i]*dtransp, lengths[i]*dq/q_scale))
        return np.array(result)

    def bc(left, right, parameters=()):
        diffusivity = np.exp(parameters[0]) if axis_temperature is not None else chi
        r = regions[1]
        q_outer = -e*r["ns"](r["x"][-1])*diffusivity*OUTER_GRADIENT_EV_M
        residual = [left[1]-perpendicular_conv_loss_w_m3(regions[0], 0, left[0], True)/q_scale,
                    right[0]-left[2], right[1]-left[3], right[3]-q_outer/q_scale]
        if axis_temperature is not None:
            residual.append(left[0]-axis_temperature)
        return np.array(residual)

    u = np.linspace(0, 1, SAMPLES_PER_REGION)
    guess = np.zeros((4, len(u)))
    guess[[0, 2]] = 5.0
    if axis_temperature is not None:
        # Start the chi fit from a balanced profile, not a flat hot SOL.
        guess, _ = solve_temperature(regions, chi, neutral_densities=neutrals)
    parameters = [np.log(chi)] if axis_temperature is not None else []
    solution = solve_bvp(rhs, bc, u, guess, p=parameters, tol=1e-4, max_nodes=20000)
    if not solution.success:
        raise RuntimeError(f"Temperature solve failed: {solution.message}; check input profiles and boundary conditions.")
    values = solution.sol(u)
    if not np.all(np.isfinite(values)) or np.any(values[[0, 2]] <= 0):
        raise RuntimeError("The input model produced a nonpositive/nonfinite temperature.")
    fitted_chi = float(np.exp(solution.p[0])) if axis_temperature is not None else chi
    print(f"chi_perp={fitted_chi:.6g} m^2/s; max BVP residual={max(solution.rms_residuals):.3g}")
    return values, fitted_chi


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / ANALYSIS / "Horvath_temperature_1d_neutrals")
    parser.add_argument("--chi", type=float, default=CHI_M2_S)
    parser.add_argument("--axis-temperature", type=float, default=AXIS_TEMPERATURE_EV,
                        help="Measured inner T [eV]; if supplied, fit chi (using --chi as initial guess).")
    parser.add_argument("--neutral", action="append", metavar="SPECIES=DENSITY_M3",
                        help="Uniform neutral density; repeat for a mixture. Replaces script defaults.")
    parser.add_argument("--no-neutrals", action="store_true", help="Disable neutral atomic losses.")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.neutral and args.no_neutrals:
        parser.error("Use --neutral or --no-neutrals, not both.")
    neutrals = {} if args.no_neutrals else NEUTRAL_DENSITIES_M3
    if args.neutral:
        neutrals = {}
        for entry in args.neutral:
            try:
                species, value = entry.split("=", 1)
                species = species.strip().capitalize()
                if species in neutrals:
                    raise ValueError("duplicate species")
                neutrals[species] = float(value)
            except ValueError:
                parser.error(f"Invalid or duplicate --neutral {entry!r}; use e.g. He=1e16.")
    neutrals = validate_neutral_densities(neutrals)
    print(f"Prescribed neutral densities [m^-3]: {neutrals}; PLT bounds={PLT_BOUNDS}.")
    regions, r_axis, lcfs_x = load_profiles()
    values, chi = solve_temperature(regions, args.chi, args.axis_temperature,
                                    neutral_densities=neutrals)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, layout="constrained")
    power_fig, power_ax = plt.subplots(figsize=(8, 4), layout="constrained")
    rows = []
    species_rows = []
    for i, r in enumerate(regions):
        x, t, q = r["x"], values[2*i], 1000*values[2*i+1]
        d = VESSEL_RADIUS_M-r_axis-x
        ns, nf, tf = r["ns"](x), r["nf"](x), r["tf"](x)
        L = np.full_like(x, np.nan) if i == 0 else r["L"](x)
        heat = heating_w_m3(t, ns, nf, tf)
        atom, atom_details = atomic_loss_w_m3(x, t, ns, nf,
                                              neutral_densities=neutrals, return_details=True)
        ion_total = sum((item["ionization_w_m3"] for item in atom_details.values()), np.zeros_like(t))
        exc_total = sum((item["excitation_w_m3"] for item in atom_details.values()), np.zeros_like(t))
        for symbol, item in atom_details.items():
            for j in range(len(x)):
                species_rows.append((d[j], symbol, neutrals[symbol],
                                     item["ionization_w_m3"][j], item["excitation_w_m3"][j]))
        parallel = np.zeros_like(x) if i == 0 else parallel_cond_loss_w_m3(t, ns, L)
        axes[0].plot(100*d, t, label="Inside LCFS" if i == 0 else "Outside LCFS (method 2)")
        axes[1].plot(100*d, r["psi"], color="C2")
        axes[2].semilogy(100*d, L, color="C3")
        for power, label, color in ((ion_total, "Neutral ionization", "C1"), (exc_total, "Neutral excitation (PLT)", "C2"),
                                    (parallel, "Sheath loss", "C3"), (heat, "Fast-to-slow heating", "C0"),):
            power_ax.plot(100*d, power, color=color, label=label if i == 0 else None)
        rows.append(np.column_stack((d, x, r["psi"], L, ns, nf, tf, t, q, heat, atom, parallel, ion_total, exc_total)))
    for ax in (*axes, power_ax):
        ax.axvline(100*(VESSEL_RADIUS_M-r_axis-lcfs_x), color="k", ls="--", lw=1, label="LCFS")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Slow electron temperature [eV]")
    axes[0].legend()
    mixture = ", ".join(f"{name}={density:g} m⁻³" for name, density in neutrals.items()) or "no neutrals"
    fig.suptitle(f"Prescribed neutrals: {mixture}", fontsize=10)
    axes[0].set_title(f"Example inputs: RLP midplane, phi={PHI_DEG:g} deg; chi={chi:.3g} m²/s")
    axes[1].set_ylabel("ILLIAD normalized flux")
    axes[2].set_ylabel(f"L to wall [m]\n({L_TO_WALL_FACTOR:g} × wall-to-wall)")
    for ax in (axes[2], power_ax):
        ax.set_xlabel("Distance inward from outer wall along RLP midplane [cm]")
    power_ax.set_ylabel("Power density [W/m³]")
    power_ax.legend()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "temperature_1d.png", dpi=180)
    power_fig.savefig(args.output_dir / "power_balance_1d.png", dpi=180)
    table = np.vstack(rows)  # Duplicate LCFS rows intentionally preserve both one-sided inputs.
    table = table[np.argsort(table[:, 0], kind="stable")]
    np.savetxt(args.output_dir / "profiles_1d.csv", table, delimiter=",", comments="",
               header="distance_from_outer_wall_m,x_outward_m,psi,L_to_wall_m,n_slow_m3,n_fast_m3,T_fast_eV,T_slow_eV,q_perp_W_m2,Q_fast_slow_W_m3,Q_atomic_W_m3,Q_parallel_W_m3,Q_ionization_W_m3,Q_excitation_W_m3")
    import csv
    with (args.output_dir / "neutral_losses_1d.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["distance_from_outer_wall_m", "species", "neutral_density_m3",
                         "Q_ionization_W_m3", "Q_excitation_W_m3"])
        writer.writerows(sorted(species_rows))
    coverage = {}
    for symbol, density in neutrals.items():
        if density == 0:
            continue
        data = load_adas_plt(symbol)
        log_t, log_n = np.log10(table[:, 7]), np.log10(table[:, 4]+table[:, 5])
        tgrid, ngrid = data["log_temperature_ev"], data["log_density_m3"]
        outside = (log_t < tgrid[0]) | (log_t > tgrid[-1]) | (log_n < ngrid[0]) | (log_n > ngrid[-1])
        threshold = load_voronov_table()[symbol, 0]
        coverage[symbol] = dict(plt_boundary_held_samples=int(outside.sum()),
                               voronov_outside_fit_samples=int(((table[:, 7] < threshold[5]) |
                                                                (table[:, 7] > threshold[6])).sum()))
        print(f"{symbol} table coverage ({len(table)} samples): {coverage[symbol]}")
    settings = {name: value for name, value in globals().items()
                if name.isupper() and isinstance(value, (str, int, float, type(None)))}
    settings.update(neutral_densities_m3=neutrals, chi_m2_s=chi,
                    axis_temperature_ev=args.axis_temperature, table_coverage=coverage,
                    temperature_min_ev=float(table[:, 7].min()), temperature_max_ev=float(table[:, 7].max()))
    (args.output_dir / "run_settings.json").write_text(json.dumps(settings, indent=2)+"\n")
    print(f"T_s range: {table[:, 7].min():.4g} to {table[:, 7].max():.4g} eV")
    print(f"Saved plots, settings, and profiles to {args.output_dir}; prescribed-neutral example.")
    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
