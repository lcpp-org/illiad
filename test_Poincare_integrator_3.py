import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from time import perf_counter
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import os

from classes.mesh import Mesh

def wrap_angle(a): return np.mod(a, 2.0 * np.pi)

def poincare_intersections_multi_planes_single_line_local(
    rho0, theta0, phi0, phi_planes, rhs_func,
    num_turns=1000, n_steps_per_2pi=400,
    solver='LSODA', rtol=1e-7, atol=1e-9):
    """
    Trace one field line starting at (ρ0, θ0) on plane φ=phi0 and return
    intersections with multiple planes φ = phi_planes (mod 2π), for the
    specified number of toroNumber of points in polar plot:idal turns.

    Returns a dict mapping each plane value to a tuple (phi_cross, rho_cross, theta_cross).
    """
    phi_start = phi0
    phi_end   = phi0 + num_turns * 2.0 * np.pi
    max_step  = 2.0 * np.pi / n_steps_per_2pi

    def stop_integration_on_rho_limit(phi, y): return 0.19 - y[0]
    stop_integration_on_rho_limit.terminal = True
    stop_integration_on_rho_limit.direction = -1

    # Precompute all desired crossing angles across all planes, then evaluate only at those angles.
    two_pi = 2.0 * np.pi
    plane_to_phi_cross = {}
    all_phi = []
    for p in np.atleast_1d(phi_planes):
        p = float(p)
        k0 = int(np.ceil((phi_start - p) / two_pi))
        k1 = int(np.floor((phi_end   - p) / two_pi))
        if k1 < k0:
            plane_to_phi_cross[p] = np.array([])
            continue
        ks = np.arange(k0, k1 + 1, dtype=int)
        phi_cross = p + ks * two_pi
        plane_to_phi_cross[p] = phi_cross
        all_phi.append(phi_cross)

    if not all_phi:
        return {p: (np.array([]), np.array([]), np.array([])) for p in plane_to_phi_cross.keys()}

    phi_eval = np.unique(np.concatenate(all_phi))
    phi_eval.sort()

    sol = solve_ivp(
        fun=rhs_func, t_span=(phi_start, phi_end), y0=[rho0, theta0],
        method=solver, rtol=rtol, atol=atol,
        t_eval=phi_eval, dense_output=False,
        events=stop_integration_on_rho_limit,
    )
    # If integration failed, return empties for all planes
    if not sol.success or sol.t.size == 0:
        return {p: (np.array([]), np.array([]), np.array([])) for p in plane_to_phi_cross.keys()}
    t = sol.t
    rho, theta = sol.y[:2]
    #theta = sol.y[1]

    # Re-slice per plane using searchsorted (t is sorted)
    results = {}
    for p, phi_cross in plane_to_phi_cross.items():
        if phi_cross.size == 0:
            results[p] = (np.array([]), np.array([]), np.array([]))
            continue

        idx = np.searchsorted(t, phi_cross)
        valid = (idx < t.size)
        # Guard against early termination and any float mismatch
        valid &= np.isclose(t[idx.clip(max=t.size-1)], phi_cross, rtol=0.0, atol=1e-12)
        idx = idx[valid]
        if idx.size == 0:
            results[p] = (np.array([]), np.array([]), np.array([]))
            continue

        phi_out = t[idx]
        rho_out = rho[idx]
        theta_out = wrap_angle(theta[idx])
        results[p] = (phi_out, rho_out, theta_out)

    return results

def _trace_single_surface_multi(rho0, theta_start, phi0, phi_planes,
    num_turns, n_steps_per_2pi, solver, rtol, atol, R0):
    """Worker for one field line; returns intersections for many φ planes."""

    def fieldline_rhs_local(phi, y):
        eps_rho =  getattr(b_hidra_RTP, "dr") #5e-4[meter]
        eps_Bp = 1e-12
        delt_rho = 0.0
        delt_theta = 0.0
        
        rho, theta = float(y[0]), float(y[1])
        phi = float(phi)

        if rho < 0.0:
            rho_eval = -rho
            theta_eval = theta + np.pi
        else:
            rho_eval = rho
            theta_eval = theta
        
        # if rho_eval < eps_rho:
        #     delt_rho = eps_rho - rho_eval
        #     point_eval = np.array([ eps_rho, theta_eval, phi])

        #     Br, Bt, Bp = b_hidra_RTP.interpField(point_eval, Cart=False)[0]
        #     if abs(Bp) < eps_Bp: Bp = np.copysign(eps_Bp, Bp if Bp != 0.0 else 1.0)

        #     if Br < 0.0:
        #         Br = -Br
        #         delt_theta = np.pi
            
        # else:
        #     point_eval = np.array([ rho_eval, theta_eval, phi])
        #     Br, Bt, Bp = b_hidra_RTP.interpField(point_eval, Cart=False)[0]
        #     if abs(Bp) < eps_Bp: Bp = np.copysign(eps_Bp, Bp if Bp != 0.0 else 1.0)

        # return [Br/Bp + delt_rho, Bt/Bp + delt_theta]

        pt = np.array([ rho_eval, theta_eval, phi])

        Br, Bt, Bp = b_hidra_RTP.interpField(pt, Cart=False)[0]
        if abs(Bp) < eps_Bp: Bp = np.copysign(eps_Bp, Bp if Bp != 0.0 else 1.0)

        if abs(rho) < eps_rho:
            fBt *= 0.0
            #Bt = Bp * np.pi
            #Br = np.abs(Br)
        return [Br/Bp, Bt/Bp]
    
    print(f'Starting surface rho0={rho0:.4f}...')
    plane_results = poincare_intersections_multi_planes_single_line_local(
        rho0, theta_start, phi0, phi_planes, fieldline_rhs_local,
        num_turns=num_turns, n_steps_per_2pi=n_steps_per_2pi,
        solver=solver, rtol=rtol, atol=atol,
    )

    by_plane = {}
    for p, (phi_cross, rho_cross, theta_cross) in plane_results.items():
        if rho_cross.size == 0:
            by_plane[p] = {'rho': np.array([]), 'theta': np.array([])}
        else:
            by_plane[p] = {'rho': rho_cross, 'theta': theta_cross}
    print(f'...Finished surface rho0={rho0:.4f}!')
    return {'rho0': rho0, 'by_plane': by_plane}


def compute_poincare_surfaces_parallel_multi(num_surfaces=16, num_turns=300, phi_step_deg=72, 
                                            phi_start = 0.0, theta_start=np.pi, rho_min=0.02, rho_max=0.14,
                                            solver='LSODA', rtol=1e-7, atol=1e-9,
                                            n_steps_per_2pi=90, R0=0.72, n_jobs=-1 ):
    """
    Compute Poincaré intersections for many flux surfaces and multiple φ planes.
    Returns a list where each item is {'rho0': float, 'by_plane': {phi: {...}}}
    """
    
    phi_planes_deg = np.arange(0, 360, phi_step_deg)
    phi_planes = np.deg2rad(phi_planes_deg)
    rho_starts = np.linspace(rho_min, rho_max, num_surfaces)

    surfaces = Parallel(n_jobs=n_jobs)(
        delayed(_trace_single_surface_multi)(
            rho0=rho0, theta_start=theta_start, phi0=phi_start,
            phi_planes=phi_planes,
            num_turns=num_turns,
            n_steps_per_2pi=n_steps_per_2pi,
            solver=solver, rtol=rtol,atol=atol,
            R0=R0,
        )
        for rho0 in rho_starts
    )
    return surfaces


if __name__ == "__main__":
    # N,, N+N-1, N+N-1+(N+N-1)-1, ...
    # 11, 21, 41, 81, 161
    # 13, 25, 49, 97, 193
    NUM_SURFACES = 49 #97 #49 
    NUM_TURNS    = 500
    PHI_PLANE    = np.radians(324.) # starting phi for field line tracing
    PHI_STEP_DEG = 9 # Multi-plane sampling every nth-degree
    THETA_START  = np.pi #inner midplane
    RHO_MIN = 0.02
    RHO_MAX = 0.14
    SOLVER='LSODA'
    RTOL = 1e-12
    ATOL = 5e-7
    N_STEPS_PER_2PI = 90 #deprecated, remove!
    TAG = None #"i5-860"

    #OUT_DIR = f'Poincare_{SOLVER}_rtol{RTOL:.0e}_atol{ATOL:.0e}_n{NUM_SURFACES}x{NUM_TURNS}_{TAG}'
    #OUT_DIR = f'OG-negRhoTweak_i5-860_epRho-dr'
    OUT_DIR = f'OG-negRhoTweak_Ih860_copyAll-toRho0-2'

    # DEFINE FIELDS #
    # CURRENT_TOR = 0.486 #[kA]
    # CURRENT_HEL = 0.710 #[kA]
    # CONFIG_TOR = "default_toroidal"
    # CONFIG_HEL = "default_helical"

    ## SET UP RUN DIRECTORY 
    full_output_directory = os.path.join('output', OUT_DIR)
    os.makedirs(full_output_directory, exist_ok=True)

    ## DEFINE MESH AND LOAD MAGNETIC FIELD
    b_hidra_RTP = Mesh(R0=0.72, a=0.19)
    #b_hidra_RTP.loadCartesianField("Bfield_RTP_contravariant_i5.npy", period = np.array([0, 1, 1]), att_mult=1.0)
    #b_hidra_RTP.loadCartesianField("Bfield_RTP_contravariant_i5-e4avg.npy", period = np.array([0, 1, 1]), att_mult=1.0)
    #b_hidra_RTP.loadCartesianField("Bfield_RTP_contravariant_i5-dr.npy", period = np.array([0, 1, 1]), att_mult=1.0)
    #b_hidra_RTP.loadCartesianField("Bfield_RTP_contravariant_i5-820.npy", period = np.array([0, 1, 1]), att_mult=1.0)
    #b_hidra_RTP.loadCartesianField("Bfield_RTP_contravariant_i5-860.npy", period = np.array([0, 1, 1]), att_mult=1.0)

    b_hidra_RTP.loadCartesianField("Bfield_RTP_contravariant_Ih860_copyAll-toRho0.npy", period = np.array([0, 1, 1]), att_mult=1.0)

    # Compute Poincaré surfaces in parallel
    tic = perf_counter()
    surfaces_multi = compute_poincare_surfaces_parallel_multi(
        num_surfaces=NUM_SURFACES, num_turns=NUM_TURNS, phi_step_deg=PHI_STEP_DEG, 
        phi_start=PHI_PLANE, theta_start=THETA_START, rho_min=RHO_MIN, rho_max=RHO_MAX,
        solver=SOLVER, rtol=RTOL,atol=ATOL,
        n_steps_per_2pi=N_STEPS_PER_2PI, R0=0.72, n_jobs=-1,
    )
    toc = perf_counter()
    print(f"Computation took {toc - tic:.2f} seconds")

    # Pre-aggregate per plane across surfaces for faster plotting
    phi_planes_deg = np.arange(0, 360, PHI_STEP_DEG)
    phi_planes = np.deg2rad(phi_planes_deg)
    plane_to_points2 = {float(p): {"rho": [], "theta": []} for p in phi_planes}

    # Plot Poincaré surfaces in series
    for s in surfaces_multi:
        by_plane = s['by_plane']
        for p, data in by_plane.items():
            if data['rho'].size > 0:
                plane_to_points2[p]["rho"].append(data['rho'])
                plane_to_points2[p]["theta"].append(data['theta'])
    for p in phi_planes:
        # Polar plot
        RHO_list = plane_to_points2[float(p)]["rho"]
        THETA_list = plane_to_points2[float(p)]["theta"]

        fig_polar = plt.figure(figsize=(6, 6))
        ax_polar = fig_polar.add_subplot(111, polar=True)
        if RHO_list:
            RHO_all = np.concatenate(RHO_list)
            THETA_all = np.concatenate(THETA_list)
            THETA_all = wrap_angle(THETA_all)
            ax_polar.scatter(THETA_all, RHO_all, marker='.', s=1.0, c='k', linewidths=0)

        ax_polar.set_rmax(0.19)
        ax_polar.set_rticks(np.arange(0.0, 0.19, 0.02))
        ax_polar.set_ylim(0, 0.19)
        ax_polar.yaxis.set_tick_params(labelsize=5)
        ax_polar.grid(linewidth = 0.25, linestyle=':', c='k', alpha=0.7)

        phi_phys = (p + (198 * np.pi/180.)) % (2*np.pi)
        p_comp_degrees = np.degrees(p)
        phi_phys_deg = np.degrees(phi_phys)
        phi_phys_string = '$\phi_{{phys}}$={:02.0f}$\degree$ CW from North Split\n'.format(phi_phys_deg)
        phi_comp_string = '$\phi_c$={:02.0f}$\degree$'.format(p_comp_degrees)
        ax_polar.set_title(phi_phys_string + phi_comp_string, loc='left')

        print(f"Saving polar plot at phi={p_comp_degrees:.0f}° to: {full_output_directory}")
        if TAG: fname_polar = TAG+f'_phi{p_comp_degrees:03.0f}_deg.png'
        else:   fname_polar = f'phi{p_comp_degrees:03.0f}_deg.png'
        fig_polar.savefig(os.path.join(full_output_directory, fname_polar), dpi=400)
        plt.close(fig_polar)