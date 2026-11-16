"""Prototype prescribed-Tu SOL model; see docs/sol_model_a.md.

Only NumPy/SciPy are needed to solve saved CPU-v2 or Warp trace summaries.
Temperatures are in eV, geometry in meters, and particle rates in particles/s.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.constants import atomic_mass, elementary_charge
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class SOLConfig:
    lambda_T_u_m: float = 0.02
    T_lcfs_eV: float = 10.0
    n_lcfs_m3: float = 1e18
    parallel_temperature_model: str = "spitzer_harm"
    target_T_fraction: float = 0.0
    Ti_eV: float = 2.0
    gamma_i: float = 1.0
    ion_mass_u: float = 4.002602
    D_perp_m2_s: float = 1.0
    source_m3_s: float = 0.0
    wedge_angle_rad: float = 1.0
    missing_leg_length_m: float = 100.0

    def validate(self):
        for name, value in asdict(self).items():
            if name != "parallel_temperature_model" and not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("lambda_T_u_m", "T_lcfs_eV", "n_lcfs_m3", "ion_mass_u",
                     "D_perp_m2_s", "wedge_angle_rad", "missing_leg_length_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if min(self.Ti_eV, self.gamma_i, self.source_m3_s) < 0:
            raise ValueError("Ti_eV, gamma_i and source_m3_s must be nonnegative")
        if self.parallel_temperature_model not in ("isothermal", "spitzer_harm"):
            raise ValueError("Unknown parallel_temperature_model")
        if not 0 <= self.target_T_fraction <= 1:
            raise ValueError("target_T_fraction must lie in [0, 1]")


def exponential_Tu(d_min_m, T_lcfs_eV, cfg):
    """Replaceable profile: callable(d_min_m, T_lcfs_eV, cfg) -> Tu_eV."""
    if not np.isfinite(cfg.lambda_T_u_m) or cfg.lambda_T_u_m <= 0:
        raise ValueError("lambda_T_u_m must be finite and positive")
    return np.asarray(T_lcfs_eV) * np.exp(-np.maximum(d_min_m, 0) / cfg.lambda_T_u_m)


def parallel_geometry(length_m, signed_s_m, direction):
    """Columns are (+B, -B); signed_s points FROM the seed TOWARD upstream."""
    lengths = np.asarray(length_m, dtype=float)
    delta, direction = np.asarray(signed_s_m), np.asarray(direction)
    if lengths.ndim != 2 or lengths.shape[1] != 2 or delta.shape != (len(lengths),):
        raise ValueError("Expected lengths (N, 2) and signed upstream distances (N,)")
    if direction.shape != delta.shape or not np.isin(direction, [-1, 1]).all():
        raise ValueError("Closest-approach direction must be +1 or -1")
    if not np.isfinite(lengths).all() or np.any(lengths < 0) or not np.isfinite(delta).all():
        raise ValueError("Invalid parallel distances")
    if np.any(delta * direction < 0):
        raise ValueError("Signed upstream distance disagrees with direction")
    Lc = lengths.sum(axis=1)
    su = lengths[:, 1] + delta
    tolerance = 1e-10 * np.maximum(Lc, 1.0)
    if np.any(Lc <= 0) or np.any(su < -tolerance) or np.any(su > Lc + tolerance):
        raise ValueError("Closest approach lies outside the wall-to-wall line")
    su = np.clip(su, 0, Lc)
    leg = np.where(delta >= 0, su, Lc - su)
    xi = np.divide(np.abs(delta), leg, out=np.zeros_like(su), where=leg > 0)
    if np.any(xi > 1 + 1e-10):
        raise ValueError("Invalid upstream-to-wall fraction")
    return np.clip(xi, 0, 1), su, Lc - su


def local_Te(Tu_eV, xi, cfg):
    if cfg.parallel_temperature_model == "isothermal":
        return np.asarray(Tu_eV).copy()
    if cfg.parallel_temperature_model != "spitzer_harm":
        raise ValueError("Unknown parallel_temperature_model")
    xi = np.asarray(xi)
    if not np.isfinite(xi).all() or np.any((xi < 0) | (xi > 1)):
        raise ValueError("xi must lie in [0, 1]")
    # Factored form avoids large intermediate Tu**3.5 values.
    return Tu_eV * (1 - xi + cfg.target_T_fraction**3.5 * xi)**(2 / 7)


def radial_boundary(boundary_xz):
    """Periodic piecewise-linear rho(theta); reject non-star-shaped sections."""
    boundary = np.asarray(boundary_xz, dtype=float)
    if (boundary.ndim != 2 or boundary.shape[1] != 2 or len(boundary) < 3
            or not np.isfinite(boundary).all()):
        raise ValueError("Expected a finite LCFS polygon (N, 2)")
    if np.allclose(boundary[0], boundary[-1], atol=1e-12, rtol=0):
        boundary = boundary[:-1]
    angle = np.unwrap(np.arctan2(boundary[:, 1], boundary[:, 0]))
    if np.median(np.diff(angle)) < 0:
        boundary = boundary[::-1]
        angle = np.unwrap(np.arctan2(boundary[:, 1], boundary[:, 0]))
    if np.any(np.diff(angle) <= 0) or not 0 < angle[-1] - angle[0] < 2 * np.pi:
        raise ValueError("Prototype requires an LCFS star-shaped about the torus center")
    radius = np.linalg.norm(boundary, axis=1)
    if np.any(radius <= 0):
        raise ValueError("LCFS radius must be positive")
    angle = angle - 2 * np.pi * np.floor(angle[0] / (2 * np.pi))
    return (np.concatenate([angle - 2*np.pi, angle, angle + 2*np.pi]),
            np.tile(radius, 3))


def wedge_volume(lo, hi, theta_lo, theta_hi, R0, dphi):
    """Exact uncut toroidal-wedge volume, useful independently for validation."""
    return dphi * (R0 / 2 * (hi**2-lo**2) * (theta_hi-theta_lo)
                   + (hi**3-lo**3) / 3 * (np.sin(theta_hi)-np.sin(theta_lo)))


def build_geometry(rho, theta, keep, boundary, R0, wall, dphi=1.0, quadrature=8):
    """Clipped polar cells with one unknown at each retained trace seed.

    First/last radial cells extend to LCFS/vessel. LCFS rho(theta) is linear
    between saved vertices. Integrate cut volumes and faces by Gauss quadrature,
    split at LCFS knots and radial crossings. Shared theta faces use interval
    intersections. Boundary conductance uses nearest LCFS distance; this is a
    two-point approximation, without nonorthogonal gradient corrections.
    """
    rho, theta, keep = np.asarray(rho), np.asarray(theta), np.asarray(keep, dtype=bool)
    if (len(rho) < 3 or len(theta) < 3 or keep.shape != (len(rho), len(theta))
            or not np.isfinite(rho).all() or not np.isfinite(theta).all()
            or np.any(np.diff(rho) <= 0) or R0 <= wall or wall <= 0
            or rho[0] != 0 or not np.isclose(rho[-1], wall)):
        raise ValueError("Expected the tracer's regular polar grid inside a circular torus")
    dt = 2*np.pi / len(theta)
    if not np.allclose(theta, np.arange(len(theta))*dt, atol=1e-12, rtol=0):
        raise ValueError("theta must be uniform, periodic, start at zero, and omit 2*pi")
    angles, radii = radial_boundary(boundary)
    if np.max(radii) >= wall:
        raise ValueError("LCFS must lie strictly inside the vessel")
    xg, wg = np.polynomial.legendre.leggauss(quadrature)
    index = np.full(keep.shape, -1, dtype=int)
    ir, it = np.nonzero(keep)
    index[ir, it] = np.arange(len(ir))
    volumes, bc = np.zeros(len(ir)), np.zeros(len(ir))
    lower, upper, columns = {}, {}, []
    edges_i, edges_j, area_over_distance = [], [], []

    def add_edge(i, j, conductance):
        if conductance > 0:
            edges_i.append(i)
            edges_j.append(j)
            area_over_distance.append(conductance)

    for t, center_angle in enumerate(theta):
        rows = np.flatnonzero(keep[:, t])
        columns.append(rows)
        # A narrow SOL wedge may contain no radial seed at this resolution.
        # Preserve the empty column; do not create cells or bridge across it.
        if not len(rows):
            continue
        if (rows[0] == 0 or rows[-1] != len(rho)-2
                or np.any(np.diff(rows) != 1)):
            raise ValueError("Each nonempty theta column must contain contiguous SOL seeds through the last interior radius")
        a, b = center_angle-dt/2, center_angle+dt/2
        knots = np.r_[a, angles[(angles > a) & (angles < b)], b]
        for k, r in enumerate(rows):
            i = index[r, t]
            lo = 0.0 if k == 0 else (rho[r-1]+rho[r])/2
            hi = wall if k == len(rows)-1 else (rho[r]+rho[r+1])/2
            lower[r, t], upper[r, t] = lo, hi
            # Include exact crossings of the radial faces with linear rho(theta).
            rb = np.interp(knots, angles, radii)
            splits = list(knots)
            for level in (lo, hi):
                crossing = (rb[:-1]-level)*(rb[1:]-level) < 0
                for q in np.flatnonzero(crossing):
                    splits.append(knots[q] + (level-rb[q])/(rb[q+1]-rb[q])*(knots[q+1]-knots[q]))
            splits = np.sort(splits)
            qa, qb = splits[:-1], splits[1:]
            nodes = ((qa+qb)[:, None]/2 + (qb-qa)[:, None]/2*xg).ravel()
            weights = ((qb-qa)[:, None]/2*wg).ravel()
            rb = np.interp(nodes, angles, radii)
            low = np.minimum(np.maximum(rb, lo), hi)
            volumes[i] = dphi * np.sum(weights * (
                R0/2*(hi**2-low**2) + np.cos(nodes)/3*(hi**3-low**3)))
            on_boundary = (rb >= lo) & (rb < hi)
            if np.any(on_boundary):
                segments = np.searchsorted(angles, nodes)-1
                slope = np.diff(radii)[segments] / np.diff(angles)[segments]
                face = (R0+rb*np.cos(nodes))*np.hypot(rb, slope)
                # Distance from this seed to the saved LCFS polygon in the plane.
                p = rho[r]*np.array([np.cos(center_angle), np.sin(center_angle)])
                start = np.asarray(boundary)
                edge = np.roll(start, -1, axis=0)-start
                length2 = np.sum(edge*edge, axis=1)
                fraction = np.divide(np.sum((p-start)*edge, axis=1), length2,
                                     out=np.zeros_like(length2), where=length2 > 0)
                distance = np.min(np.linalg.norm(p-start-np.clip(fraction, 0, 1)[:, None]*edge, axis=1))
                if distance <= 1e-12:
                    raise ValueError("SOL seed lies on LCFS")
                bc[i] = dphi*np.sum(weights[on_boundary]*face[on_boundary])/distance
            if k < len(rows)-1:
                # Open portion of the shared radial face, after LCFS clipping.
                area = dphi*hi*np.sum(weights*(R0+hi*np.cos(nodes))*(rb < hi))
                add_edge(i, index[rows[k+1], t], area/(rho[rows[k+1]]-rho[r]))

    # Pair each theta face once, including the periodic seam. Boundary-adjacent
    # cells may span several radial cells in the neighboring column.
    for t, rows in enumerate(columns):
        next_t = (t+1) % len(theta)
        angle = theta[t]+dt/2
        rb = np.interp(angle, angles, radii)
        other = columns[next_t]
        k = 0
        for r in rows:
            lo, hi = max(lower[r, t], rb), upper[r, t]
            while k < len(other) and upper[other[k], next_t] <= lo:
                k += 1
            q = k
            while q < len(other) and lower[other[q], next_t] < hi:
                s = other[q]
                left, right = max(lo, lower[s, next_t]), min(hi, upper[s, next_t])
                if right > left:
                    area = dphi*(R0*(right-left)+np.cos(angle)/2*(right**2-left**2))
                    distance = (rho[r]+rho[s])/2*dt
                    add_edge(index[r, t], index[s, next_t], area/distance)
                q += 1
    if not len(volumes) or np.any(volumes <= 0) or not np.isfinite(volumes).all():
        raise ValueError("Mesh contains empty or invalid SOL cells")
    return dict(volume_m3=volumes, boundary_area_over_distance_m=bc,
                edge_i=np.asarray(edges_i, dtype=int), edge_j=np.asarray(edges_j, dtype=int),
                edge_area_over_distance_m=np.asarray(area_over_distance),
                rho_index=ir, theta_index=it)


def solve_density(volume, edge_i, edge_j, conductance, boundary_conductance,
                  sink_s_inv, n_lcfs, source):
    """Shared-face conservative FV solve; conductances already include D_perp."""
    i, j, g = edge_i, edge_j, conductance
    diffusion = coo_matrix((np.r_[g, g, -g, -g],
                           (np.r_[i, j, i, j], np.r_[i, j, j, i])),
                          shape=(len(volume), len(volume))).tocsr()
    sink = volume * sink_s_inv
    matrix = diffusion + diags(sink + boundary_conductance)
    rhs = boundary_conductance*n_lcfs + volume*source
    density = spsolve(matrix, rhs)
    residual = matrix @ density - rhs
    influx = float(np.sum(boundary_conductance*(n_lcfs-density)))
    losses = float(np.sum(sink*density))
    sources = float(np.sum(volume*source))
    scale = max(float(np.max(np.abs(rhs))), float(np.max(np.abs(matrix) @ np.abs(density))), 1.0)
    diagnostics = dict(lcfs_influx_s_inv=influx, parallel_loss_s_inv=losses,
                       volume_source_s_inv=sources,
                       relative_particle_balance=(influx+sources-losses)/max(abs(influx)+sources, losses, 1.0),
                       max_scaled_residual=float(np.max(np.abs(residual)))/scale,
                       min_density_m3=float(np.min(density)))
    if (not np.isfinite(density).all() or np.any(density < 0)
            or diagnostics["max_scaled_residual"] > 1e-8
            or abs(diagnostics["relative_particle_balance"]) > 1e-8):
        raise ValueError(f"Density solve failed numerical checks: {diagnostics}")
    return density, residual, diagnostics


def prepare_plane(data, settings, phi_deg, cfg, incomplete="error", quadrature=8):
    """Shared trace selection and geometry for the standalone Model A/B solvers."""
    cfg.validate()
    if incomplete not in ("error", "no_flux", "long_leg", "all_valid"):
        raise ValueError("incomplete must be error, no_flux, long_leg or all_valid")
    if quadrature < 2:
        raise ValueError("quadrature must be >= 2")
    if settings["offset_mm"] != 0 or settings["surface_index"] != settings["distance_surface_index"]:
        raise ValueError("SOL models need zero offset and the same exclusion/distance LCFS surface")
    planes = np.asarray(settings["phi_deg"])
    matches = np.flatnonzero(np.isclose((planes-phi_deg+180) % 360-180, 0, atol=1e-8))
    if len(matches) != 1:
        raise ValueError(f"Choose one saved plane from {planes.tolist()}")
    plane = int(matches[0])
    seed_ids = np.flatnonzero(data["plane_id"] == plane)
    keep = np.asarray(data["grid_keep_mask"][plane], dtype=bool)
    ir, it = np.nonzero(keep)
    # Do not rely on row ordering in the archive.
    lookup = np.full(keep.shape, -1, dtype=int)
    sr, st = data["seed_rho_index"][seed_ids], data["seed_theta_index"][seed_ids]
    if len(seed_ids) != len(ir) or len(set(zip(sr, st))) != len(seed_ids):
        raise ValueError("Seed indices do not match the saved SOL mask")
    lookup[sr, st] = seed_ids
    seed_ids = lookup[ir, it]
    if np.any(seed_ids < 0):
        raise ValueError("Missing trace records inside the SOL mask")
    lengths = np.asarray(data["length_m"][seed_ids], dtype=float)
    hits = np.asarray(data["hit_wall"][seed_ids])
    dmin = np.asarray(data["lcfs_distance_min_m"][seed_ids], dtype=float)
    clean = (np.isfinite(lengths).all(axis=1) & (lengths > 0).all(axis=1)
             & np.isfinite(dmin) & (dmin >= 0) & np.isin(hits, [0, 1]).all(axis=1))
    hit_count = (hits == 1).sum(axis=1)
    approximated = clean & (hit_count == 1) & (incomplete in ("long_leg", "all_valid"))
    no_parallel_loss = clean & (hit_count == 0) & (incomplete == "all_valid")
    valid = clean & ((hit_count == 2) | approximated | no_parallel_loss)
    if incomplete == "all_valid" and not np.all(clean):
        raise ValueError("all_valid requires valid trace records for every domain cell; corrupt data cannot be filled")
    rejected = seed_ids[~valid]
    if len(rejected):
        message = f"{len(rejected)}/{len(seed_ids)} seeds have incomplete/invalid wall traces or distances"
        if incomplete == "error":
            raise ValueError(message + "; retrace, or use --incomplete no_flux or long_leg")
        warnings.warn(message + "; excluded cells create artificial no-flux boundaries", stacklevel=2)
    if not np.any(valid):
        raise ValueError("No valid wall-connected cells remain")
    mesh = build_geometry(data["grid_rho_m"], data["grid_theta_rad"], keep,
                          data["boundaries_xz_m"][plane], settings["major_radius_m"],
                          settings["vessel_radius_m"], cfg.wedge_angle_rad, quadrature)
    remap = np.full(len(valid), -1, dtype=int)
    remap[valid] = np.arange(np.sum(valid))
    ei, ej = mesh["edge_i"], mesh["edge_j"]
    edge_valid = valid[ei] & valid[ej]
    lengths, dmin, seed_ids = lengths[valid], dmin[valid], seed_ids[valid]
    hits, approximated = hits[valid].astype(bool), approximated[valid]
    traced_lengths = lengths.copy()
    missing = approximated[:, None] & ~hits
    # A capped trace is a lower bound: never shorten it to the configured value.
    lengths[missing] = np.maximum(lengths[missing], cfg.missing_leg_length_m)
    if np.any(approximated):
        warnings.warn(f"{np.count_nonzero(approximated)} one-wall-hit cells use an effective missing-leg "
                      "length; closest approach and reconstructed temperature remain approximate", stacklevel=2)
    xi = np.full(len(lengths), np.nan)
    leg_minus, leg_plus = xi.copy(), xi.copy()
    if cfg.parallel_temperature_model == "spitzer_harm":
        signed_s = data["lcfs_min_signed_s_m"][seed_ids]
        direction = data["lcfs_min_direction"][seed_ids]
        # An extended endpoint must not legitimize a closest approach outside
        # the actually traced interval.
        if np.any(approximated):
            parallel_geometry(traced_lengths, signed_s, direction)
        xi, leg_minus, leg_plus = parallel_geometry(
            lengths, signed_s, direction)
    no_parallel_loss = no_parallel_loss[valid]
    xi[no_parallel_loss] = 0.0
    volume = mesh["volume_m3"][valid]
    result = dict(seed_id=seed_ids, rejected_seed_id=rejected,
                  rho_index=ir[valid], theta_index=it[valid],
                  rejected_rho_index=ir[~valid], rejected_theta_index=it[~valid],
                  boundary_xz_m=data["boundaries_xz_m"][plane],
                  grid_rho_m=data["grid_rho_m"], grid_theta_rad=data["grid_theta_rad"],
                  phi_comp_deg=np.asarray(planes[plane]), plane_id=np.asarray(plane),
                  xi=xi, no_parallel_loss=no_parallel_loss,
                  leg_minus_m=leg_minus, leg_plus_m=leg_plus, Lc_m=lengths.sum(axis=1),
                  traced_length_m=traced_lengths, effective_length_m=lengths, hit_wall=hits,
                  Lc_traced_m=traced_lengths.sum(axis=1), connection_approximated=approximated,
                  d_min_m=dmin, volume_m3=volume)
    diagnostics = dict(valid_cells=len(seed_ids), rejected_cells=len(rejected),
                       complete_cells=int(np.count_nonzero(hits.all(axis=1))),
                       no_parallel_loss_cells=int(no_parallel_loss.sum()),
                       approximated_cells=int(np.count_nonzero(approximated)),
                       excluded_volume_m3=float(np.sum(mesh["volume_m3"][~valid])),
                       retained_volume_m3=float(np.sum(volume)))
    geometry = dict(volume_m3=volume, edge_i=remap[ei[edge_valid]], edge_j=remap[ej[edge_valid]],
                    edge_area_over_distance_m=mesh["edge_area_over_distance_m"][edge_valid],
                    boundary_area_over_distance_m=mesh["boundary_area_over_distance_m"][valid])
    return result, geometry, diagnostics


def add_grid_fields(result):
    """Preserve the tracer's (rho, theta) layout and leave unsolved cells as NaN."""
    shape = (len(result["grid_rho_m"]), len(result["grid_theta_rad"]))
    for name in ("n_e_m3", "Tu_eV", "Te_eV"):
        field = np.full(shape, np.nan)
        field[result["rho_index"], result["theta_index"]] = result[name]
        result[name+"_grid"] = field


def solve_plane(data, settings, phi_deg, cfg=SOLConfig(), profile=exponential_Tu,
                incomplete="error", quadrature=8):
    """Solve one saved plane, retaining source seed IDs and (rho, theta) indices."""
    result, mesh, diagnostics = prepare_plane(data, settings, phi_deg, cfg, incomplete, quadrature)
    dmin = result["d_min_m"]
    Tu = np.broadcast_to(np.asarray(profile(dmin, cfg.T_lcfs_eV, cfg), dtype=float), dmin.shape).copy()
    if not np.isfinite(Tu).all() or np.any(Tu < 0):
        raise ValueError("Upstream profile must return finite nonnegative temperatures")
    Te = local_Te(Tu, result["xi"], cfg)
    cs = np.sqrt(elementary_charge*(Tu+cfg.gamma_i*cfg.Ti_eV)/(cfg.ion_mass_u*atomic_mass))
    if np.any(cs <= 0):
        raise ValueError("Sound speed must be positive; check profile underflow and Ti")
    density, residual, balance = solve_density(
        mesh["volume_m3"], mesh["edge_i"], mesh["edge_j"],
        cfg.D_perp_m2_s*mesh["edge_area_over_distance_m"],
        cfg.D_perp_m2_s*mesh["boundary_area_over_distance_m"],
        2*cs/result["Lc_m"], cfg.n_lcfs_m3, cfg.source_m3_s)
    result.update(n_e_m3=density, Tu_eV=Tu, Te_eV=Te, sound_speed_m_s=cs,
                  particle_residual_s_inv=residual)
    add_grid_fields(result)
    diagnostics.update(balance, zero_density_cells=int(np.count_nonzero(density == 0)))
    return result, diagnostics


def select_planes(settings, requested=None, all_phi=False):
    """Resolve angles to saved planes, rejecting missing/duplicate selections early."""
    saved = np.asarray(settings["phi_deg"], dtype=float)
    if saved.ndim != 1 or not len(saved) or not np.isfinite(saved).all():
        raise ValueError("Saved phi_deg must be a nonempty list of finite angles")
    if all_phi and requested is not None:
        raise ValueError("Use either --phi-deg or --all-phi")
    if all_phi:
        requested = saved
    elif requested is None:
        if len(saved) != 1:
            raise ValueError("Use --phi-deg ANGLE [ANGLE ...] or --all-phi for a multi-plane summary")
        requested = saved
    selected = []
    for phi in requested:
        if not np.isfinite(phi):
            raise ValueError("Requested angles must be finite")
        matches = np.flatnonzero(np.isclose((saved-phi+180) % 360-180, 0, atol=1e-8))
        if len(matches) != 1:
            raise ValueError(f"Angle {phi:g} must match exactly one saved plane from {saved.tolist()}")
        index = int(matches[0])
        if index in selected:
            raise ValueError(f"Duplicate selection of saved angle {saved[index]:g}")
        selected.append(index)
    return saved[selected].tolist()


def save_solution(output, result, metadata):
    """Write one plane in the same format for single and batch runs."""
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/"solution.npz", **result)
    columns = ["seed_id", "rho_index", "theta_index", "n_e_m3", "Tu_eV", "Te_eV", "xi",
               "Lc_m", "d_min_m", "volume_m3", "particle_residual_s_inv",
               "Lc_traced_m", "connection_approximated"]
    np.savetxt(output/"profiles.csv", np.column_stack([result[k] for k in columns]),
               delimiter=",", header=",".join(columns), comments="", fmt=["%d"]*3+["%.12g"]*9+["%d"])
    (output/"run_settings.json").write_text(json.dumps(metadata, indent=2)+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path, help="Directory containing summary.npz and settings.json")
    planes = parser.add_mutually_exclusive_group()
    planes.add_argument("--phi-deg", type=float, nargs="+", help="One or more saved angles, in the requested order")
    planes.add_argument("--all-phi", action="store_true", help="Solve every angle in the saved summary")
    parser.add_argument("--config", type=Path, help="JSON overrides for SOLConfig (see docs/sol_model_a.md)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--incomplete", choices=("error", "no_flux", "long_leg"), default="error",
                        help="long_leg retains one-wall-hit cells using missing_leg_length_m; excludes neither-hit cells")
    parser.add_argument("--quadrature", type=int, default=8, help="Gauss order for clipped LCFS cells")
    parser.add_argument("--plot", action="store_true", help="Save a six-panel PNG overview after solving")
    parser.add_argument("--plot-individual", action="store_true", help="Save overview and six individual PNGs (implies --plot)")
    parser.add_argument("--marker-size", type=float, default=35.0, help="Plot marker area in points squared (default: 35)")
    parser.add_argument("--density-scale", choices=("linear", "log"), default="log", help="Density plot color scale")
    args = parser.parse_args(argv)
    if not np.isfinite(args.marker_size) or args.marker_size <= 0:
        parser.error("--marker-size must be finite and positive")
    cfg = SOLConfig(**(json.loads(args.config.read_text()) if args.config else {}))
    settings = json.loads((args.trace_dir/"settings.json").read_text())
    try:
        cfg.validate()
        angles = select_planes(settings, args.phi_deg, args.all_phi)
    except ValueError as exc:
        parser.error(str(exc))
    batch = args.all_phi or len(angles) > 1
    root = args.output or args.trace_dir/("model_a_batch" if batch else f"model_a_phi{angles[0]:g}")
    if args.plot or args.plot_individual:
        if __package__:
            from .plot_sol_model_a import save_plots
        else:
            from plot_sol_model_a import save_plots
    with np.load(args.trace_dir/"summary.npz", allow_pickle=False) as saved:
        for number, phi in enumerate(angles, 1):
            print(f"[{number}/{len(angles)}] Solving computational phi={phi:g} degrees", flush=True)
            result, diagnostics = solve_plane(saved, settings, phi, cfg,
                                              incomplete=args.incomplete, quadrature=args.quadrature)
            output = root / f"phi{phi:03g}" if batch else root
            metadata = dict(config=asdict(cfg), trace_dir=str(args.trace_dir.resolve()),
                            phi_deg=float(result["phi_comp_deg"]), incomplete=args.incomplete,
                            quadrature=args.quadrature, upstream_profile="exponential_Tu",
                            geometry="LCFS-clipped polar cells; first/last cells extended to boundaries",
                            diagnostics=diagnostics)
            save_solution(output, result, metadata)
            print(json.dumps(diagnostics, indent=2))
            print(f"Saved {output}")
            if args.plot or args.plot_individual:
                save_plots(result, metadata, output / "plots", args.density_scale,
                           args.marker_size, args.plot_individual)


if __name__ == "__main__":
    main()
