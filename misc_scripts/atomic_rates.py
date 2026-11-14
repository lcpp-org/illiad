"""Shared Maxwellian atomic rates for the Horvath and Model B prototypes.

Voronov ionization and ADAS PLT excitation; temperatures eV, densities m^-3.
These functions were extracted unchanged from horvath_temperature_1d.py.
"""
from functools import lru_cache
from pathlib import Path
import re
import warnings

import numpy as np
from scipy.constants import e, m_e, atomic_mass
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]


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
    """Returns total threshold-energy loss [W/m^3]. All densities are m^-3.
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

    Sum n_target[q] * (n_slow*k_slow[q]+n_fast*k_fast[q]) * E_ion[q]*e.
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


