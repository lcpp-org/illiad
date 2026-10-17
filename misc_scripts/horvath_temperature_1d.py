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
paper's fit. Atomic losses default to zero; D is prescribed rather than
obtained from a particle-balance solve. chi is prescribed unless an axis
temperature is supplied, in which case it is fitted as in the paper.
Temperature and total perpendicular heat flux are continuous at the LCFS;
dT/dx=0 at the inner boundary and a prescribed gradient at the outer endpoint.
The endpoint is slightly inside the wall to avoid L=0. No field lines retrace.
Saved connection lengths retain any finite tracing cutoff; this script does
not recover uncensored wall distances from trace-limited values.
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e, epsilon_0, m_e, atomic_mass
from scipy.integrate import solve_bvp
from scipy.interpolate import PchipInterpolator

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
FAST_FRACTION = 0.001            # n_fast / n_slow
FAST_TEMPERATURE_EV = 80.0
COULOMB_LOG = 10.0
ION_MASS_U = 4.0026             # Argon 39.948; use  for helium, 1.0078 for H
SHEATH_DELTA = 5.5              # Horvath: Ar ~11, He ~5.5, H ~5
L_TO_WALL_FACTOR = 1.0          # ILLIAD L is wall-to-wall; equal-leg approximation
CHI_M2_S = 10.5                 # Constant perpendicular thermal diffusivity
PARTICLE_D_M2_S = 0.8            # Set >0 to include core particle convection
AXIS_TEMPERATURE_EV = None      # Measured value: fit chi if supplied
OUTER_GRADIENT_EV_M = 0.0       # dT/dx, x increasing outward (not plot distance)


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
def atomic_loss_w_m3(x, temperature_ev, n_slow, n_fast):
    """CALCULATION SKELETON: return total atomic energy loss in W/m^3.

    TODO: add ionization, neutral/ion excitation, recombination, and electron-
    ion exchange as needed. For a rate coefficient K(T) [m^3/s] and threshold
    E [eV], an example term is n_slow * n_neutral * K(T) * E * e.
    Supply species-specific rates and neutral density before enabling terms.
    Zero explicitly omits these processes; it is not an atomic-loss model.
    """
    return np.zeros_like(temperature_ev)


def ionization_loss_w_m3(x, temperature_ev, fasttemp_ev, fast_frac, n_target, prefactor, ionization_energy_ev):
    """CALCULATION SKELETON: return ionization energy loss in W/m^3.
   
    TODO: add species-specific ionization rates and thresholds.
    """
    # prefactor passed as an argument
    n_e = slow_density_m3
    Q_slow = n_e * (1 - fast_frac) * n_target * prefactor * ionization_energy_ev * np.exp(-ionization_energy_ev/temperature_ev)
    Q_fast = n_e * fast_frac * n_target * prefactor * ionization_energy_ev * np.exp(-ionization_energy_ev/fasttemp_ev)

    return Q_slow + Q_fast


def excitation_loss_w_m3(x, temperature_ev, n_slow, n_fast, n_neutral, excitation_energy_ev):
    """CALCULATION SKELETON: return excitation energy loss in W/m^3.

    TODO: add species-specific excitation rates and thresholds.
    """
    return np.zeros_like(temperature_ev)


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
    """Horvath (11)-(14); no X-point multiplier in this second SOL method."""
    # Nonnegative trial values keep Newton iterations real; final T must be >0.
    energy = e*np.maximum(temperature_ev, 0.0)
    return SHEATH_DELTA*n_slow*energy*np.sqrt(energy/(ION_MASS_U*atomic_mass))/length_to_wall


def parallel_conv_loss_w_m3(temperature_ev, n_slow, length_to_wall):
    pass


def perpendicular_conv_loss_w_m3(r, x, t, core):
        return -1.5*e*t*PARTICLE_D_M2_S*r["ns"].derivative()(x) if core else 0.0




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


def solve_temperature(regions, chi, axis_temperature=None):
    """Two-domain BVP in (T, q_total); explicit LCFS matching avoids a step RHS."""
    if not np.isfinite(chi) or chi <= 0 or (axis_temperature is not None and
            (not np.isfinite(axis_temperature) or axis_temperature <= 0)):
        raise ValueError("chi and an optional measured axis temperature must be positive.")

    
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

            loss = atomic_loss_w_m3(x, t, ns, nf)

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
        guess, _ = solve_temperature(regions, chi)
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / ANALYSIS / "Horvath_temperature_1d")
    parser.add_argument("--chi", type=float, default=CHI_M2_S)
    parser.add_argument("--axis-temperature", type=float, default=AXIS_TEMPERATURE_EV,
                        help="Measured inner T [eV]; if supplied, fit chi (using --chi as initial guess).")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    regions, r_axis, lcfs_x = load_profiles()
    values, chi = solve_temperature(regions, args.chi, args.axis_temperature)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, layout="constrained")
    power_fig, power_ax = plt.subplots(figsize=(8, 4), layout="constrained")
    rows = []
    for i, r in enumerate(regions):
        x, t, q = r["x"], values[2*i], 1000*values[2*i+1]
        d = VESSEL_RADIUS_M-r_axis-x
        ns, nf, tf = r["ns"](x), r["nf"](x), r["tf"](x)
        L = np.full_like(x, np.nan) if i == 0 else r["L"](x)
        heat = heating_w_m3(t, ns, nf, tf)
        atom = atomic_loss_w_m3(x, t, ns, nf)
        parallel = np.zeros_like(x) if i == 0 else parallel_cond_loss_w_m3(t, ns, L)
        axes[0].plot(100*d, t, label="Inside LCFS" if i == 0 else "Outside LCFS (method 2)")
        axes[1].plot(100*d, r["psi"], color="C2")
        axes[2].semilogy(100*d, L, color="C3")
        for power, label, color in ((atom, "Atomic losses", "C1"),
                                    (parallel, "Sheath loss", "C3"), (heat, "Fast-to-slow heating", "C0"),):
            power_ax.plot(100*d, power, color=color, label=label if i == 0 else None)
        rows.append(np.column_stack((d, x, r["psi"], L, ns, nf, tf, t, q, heat, atom, parallel)))
    for ax in (*axes, power_ax):
        ax.axvline(100*(VESSEL_RADIUS_M-r_axis-lcfs_x), color="k", ls="--", lw=1, label="LCFS")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Slow electron temperature [eV]")
    axes[0].legend()
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
               header="distance_from_outer_wall_m,x_outward_m,psi,L_to_wall_m,n_slow_m3,n_fast_m3,T_fast_eV,T_slow_eV,q_perp_W_m2,Q_fast_slow_W_m3,Q_atomic_W_m3,Q_parallel_W_m3")
    print(f"Saved plots and profiles to {args.output_dir}; example inputs, atomic losses omitted by default.")
    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
