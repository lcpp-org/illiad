"""Compare saved magnetic footprints: four angles, three iota curves per panel.

Run with no arguments; edit the settings below to select another band or period.
No magnetic field is loaded and no field lines are traced. Angles follow Boris.

The (toroidal=5, poloidal=2) Fourier phase selects a corresponding streak set.
Its center lies halfway between the broad low-length gaps flanking the set;
a local quadratic fit to these midpoints gives the cut center and tangent. This
is a geometric selection, not a topological identification of private flux or
a threshold-based width measurement; inspect the accompanying cut-location map.
Smoothing is used ONLY to locate the band, never for the plotted lengths.

Cuts are surface geodesics, perpendicular to the fitted band at their centers.
The distance axis uses the vessel surface metric
    ds**2 = (R + a*cos(theta))**2 * dphi**2 + a**2 * dtheta**2.
Lengths sampled 1 mm inside the wall are mapped to the same wall angles.
Bilinear interpolation is periodic on both axes and uses linear lengths.
Dense curve samples do not increase the original footprint's spatial resolution.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


# EDITABLE SETTINGS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = {
    "IOTA3": PROJECT_ROOT / "output/magnetic_footprint/iota3_1mm/data/magnetic_footprint.npz",
    "IOTA4": PROJECT_ROOT / "output/magnetic_footprint/IOTA4_footprint_1mm/data/magnetic_footprint.npz",
    "IOTA5": PROJECT_ROOT / "output/magnetic_footprint/IOTA5_footprint_1mm/data/magnetic_footprint.npz",
}
OUTPUT_DIR = PROJECT_ROOT / "output/magnetic_footprint/cross_sections"
START_PHI_DEG = 162.0
PHI_ANGLES_DEG = START_PHI_DEG + np.arange(4) * 18.0
TOROIDAL_MODE = 5
POLOIDAL_MODE = 2
HELICAL_SLOPE = TOROIDAL_MODE / POLOIDAL_MODE  # d(theta)/d(phi), initially 2.5.
BAND_BRANCH = 0  # 0 centers near theta=-90 at phi=162; 1 selects the set 180 deg away.
BAND_PHASE_DEG = None  # None estimates the set-center phase theta - (5/2)*phi.
CENTER_SEARCH_HALF_WIDTH_DEG = 60.0
FIT_HALF_WINDOW_DEG = 12.0  # Local toroidal window for center/tangent fitting.
SMOOTH_SIGMA_DEG = 4.5  # Periodic Gaussian smoothing of log(length) for fitting.
RMAJOR_M = 0.72  # Verified against the saved runs' logs.
WALL_RADIUS_M = 0.19
PHI_WALL_OFFSET_DEG = 18.0
CUT_HALF_LENGTH_M = 0.25
N_SAMPLES = 1001  # Odd, so the center is sampled exactly.
Y_SCALE = "log"  # "linear" also supported.
DPI = 200
COLORS = ("tab:blue", "tab:orange", "tab:green")


def load_footprint(path):
    """Return increasing full-period angular axes (radians) and matching data."""
    with np.load(path, allow_pickle=False) as data:
        points = data["initial_conditions_rtp"]
        lengths = data["connection_length_m"]
    if lengths.ndim != 2 or min(lengths.shape) < 3:
        raise ValueError(f"{path}: expected a 2D connection-length grid.")
    if points.shape != (*lengths.shape, 3) or not np.isfinite(points).all():
        raise ValueError(f"{path}: invalid computational launch coordinates.")
    if not np.all(np.isfinite(lengths) & (lengths > 0)):
        raise ValueError(f"{path}: expected positive finite lengths; do not fill missing traces.")
    if not np.allclose(points[..., 0], points[0, 0, 0], rtol=0, atol=1e-10):
        raise ValueError(f"{path}: launch surface must have constant minor radius.")
    radius = float(points[0, 0, 0])
    if not 0 < radius <= WALL_RADIUS_M:
        raise ValueError(f"{path}: launch radius must lie inside the configured vessel.")
    phi = (-points[..., 2] + np.deg2rad(180 + PHI_WALL_OFFSET_DEG)) % (2*np.pi)
    phi[np.isclose(phi, 2*np.pi, rtol=0, atol=1e-10)] = 0
    theta = (points[..., 1] + np.pi) % (2*np.pi) - np.pi
    if not (np.allclose(phi, phi[:1], rtol=0, atol=1e-10)
            and np.allclose(theta, theta[:, :1], rtol=0, atol=1e-10)):
        raise ValueError(f"{path}: expected a separable (theta, phi) grid.")
    rows, cols = np.argsort(theta[:, 0]), np.argsort(phi[0])
    theta, phi = theta[rows, 0], phi[0, cols]
    for axis in (theta, phi):
        if not np.allclose(np.diff(np.r_[axis, axis[0] + 2*np.pi]),
                           2*np.pi / axis.size, rtol=0, atol=1e-9):
            raise ValueError(f"{path}: expected full, uniform periodic angular grids.")
    return phi, theta, lengths[np.ix_(rows, cols)], radius


def periodic_sampler(phi, theta, values):
    """Bilinear sampler accepting unwrapped phi and theta in radians."""
    axes = (np.r_[theta, theta[0] + 2*np.pi], np.r_[phi, phi[0] + 2*np.pi])
    interp = RegularGridInterpolator(axes, np.pad(values, ((0, 1), (0, 1)), mode="wrap"))

    def sample(phi_query, theta_query):
        p, t = np.broadcast_arrays(phi_query, theta_query)
        query = np.stack(((t-theta[0]) % (2*np.pi) + theta[0],
                          (p-phi[0]) % (2*np.pi) + phi[0]), axis=-1)
        return interp(query)

    return sample


def fit_band(phi, theta, lengths, anchor):
    """Fit midpoints of flanking gaps; return center, slope, phase and residual."""
    log_lengths = np.log(lengths)
    if BAND_PHASE_DEG is None:
        phase_coordinate = POLOIDAL_MODE*theta[:, None] - TOROIDAL_MODE*phi[None, :]
        coefficient = np.sum((log_lengths-log_lengths.mean()) * np.exp(1j*phase_coordinate))
        if abs(coefficient) < 0.05 * np.sum(np.abs(log_lengths-log_lengths.mean())):
            raise ValueError("No clear helical band; set BAND_PHASE_DEG explicitly.")
        trough_phase = (np.angle(coefficient) + 2*np.pi) % (2*np.pi) - np.pi
        # Move half a band spacing from the broad gap into the streak set.
        phase = (trough_phase + 2*np.pi*BAND_BRANCH - np.pi) / POLOIDAL_MODE
    else:
        phase = np.deg2rad(BAND_PHASE_DEG)
    sigma = np.deg2rad(SMOOTH_SIGMA_DEG) / np.array([theta[1]-theta[0], phi[1]-phi[0]])
    sample = periodic_sampler(phi, theta, gaussian_filter(log_lengths, sigma, mode="wrap"))
    delta_phi = (phi-anchor+np.pi) % (2*np.pi) - np.pi
    local_phi = np.sort(delta_phi[np.abs(delta_phi) <= np.deg2rad(FIT_HALF_WINDOW_DEG)+1e-10])
    if local_phi.size < 5:
        raise ValueError("Increase FIT_HALF_WINDOW_DEG to include at least five toroidal samples.")
    half_width = np.deg2rad(CENTER_SEARCH_HALF_WIDTH_DEG)
    theta_step = theta[1] - theta[0]
    residuals = []
    for delta in local_phi:
        predicted = HELICAL_SLOPE*(anchor+delta) + phase
        gaps = []
        for side in (-1, 1):
            gap_prediction = predicted + side*np.pi/POLOIDAL_MODE
            first = int(np.ceil((gap_prediction-half_width-theta[0])/theta_step))
            last = int(np.floor((gap_prediction+half_width-theta[0])/theta_step))
            candidates = theta[0] + np.arange(first, last+1)*theta_step
            values = sample(anchor+delta, candidates)
            index = int(np.argmin(values))
            if index in (0, len(candidates)-1):
                raise ValueError("Flanking gap reached search edge; adjust BAND_PHASE_DEG or search width.")
            left, middle, right = values[index-1:index+2]
            curvature = left - 2*middle + right
            if curvature <= 1e-12:
                raise ValueError("Gap minimum is flat; choose a better defined set or fitting window.")
            # Parabolic sub-grid minimum from the three neighboring smoothed samples.
            correction = 0.5*(left-right)/curvature*theta_step
            gaps.append(candidates[index] + correction)
        residuals.append(np.mean(gaps) - predicted)
    coefficients = np.polynomial.polynomial.polyfit(local_phi, residuals, 2)
    rms = np.sqrt(np.mean((np.polynomial.polynomial.polyval(local_phi, coefficients)-residuals)**2))
    if rms > np.deg2rad(5):
        raise ValueError("Band fit RMS exceeds 5 degrees; inspect the branch and fitting window.")
    center = HELICAL_SLOPE*anchor + phase + coefficients[0]
    slope = HELICAL_SLOPE + coefficients[1]
    return center, slope, phase, rms


def normal_cut(phi_center, theta_center, slope, distances):
    """Geodesic parameterized by signed wall arclength, normal at s=0."""
    a, major = WALL_RADIUS_M, RMAJOR_M
    h = major + a*np.cos(theta_center)
    norm = np.hypot(h, a*slope)
    initial = [phi_center, theta_center, -a*slope/(h*norm), h/(a*norm)]

    def rhs(_, state):
        _, theta, p_dot, t_dot = state
        h = major + a*np.cos(theta)
        return [p_dot, t_dot, 2*a*np.sin(theta)/h*p_dot*t_dot,
                -h*np.sin(theta)/a*p_dot**2]

    states = np.empty((len(distances), 4))
    for sign in (-1, 1):
        indices = np.flatnonzero(distances*sign > 0)
        indices = indices[np.argsort(np.abs(distances[indices]))]
        if not indices.size:
            continue
        result = solve_ivp(rhs, (0, distances[indices[-1]]), initial,
                           t_eval=distances[indices], rtol=1e-10, atol=1e-12)
        if not result.success:
            raise RuntimeError(result.message)
        states[indices] = result.y.T
    states[distances == 0] = initial
    return states


def wrapped_cut_for_plot(phi, theta):
    """Insert breaks at periodic seams instead of drawing across the map."""
    p, t = np.rad2deg(phi) % 360, (np.rad2deg(theta)+180) % 360 - 180
    jumps = np.flatnonzero((np.abs(np.diff(p)) > 180) | (np.abs(np.diff(t)) > 180)) + 1
    return np.insert(p, jumps, np.nan), np.insert(t, jumps, np.nan)


def make_plots(output_dir=OUTPUT_DIR):
    if not (0 < WALL_RADIUS_M < RMAJOR_M and CUT_HALF_LENGTH_M > 0
            and N_SAMPLES >= 3 and N_SAMPLES % 2 == 1):
        raise ValueError("Require RMAJOR_M > WALL_RADIUS_M > 0, positive cut length and odd N_SAMPLES >= 3.")
    if len(PHI_ANGLES_DEG) != 4 or len(DATA_FILES) != 3 or Y_SCALE not in {"log", "linear"}:
        raise ValueError("Configure four angles, three datasets and a log or linear Y_SCALE.")
    if not np.allclose(np.diff(PHI_ANGLES_DEG), 18):
        raise ValueError("The four anchor angles must be spaced by 18 degrees.")
    if not (0 <= BAND_BRANCH < POLOIDAL_MODE and 0 < CENTER_SEARCH_HALF_WIDTH_DEG < 180/POLOIDAL_MODE):
        raise ValueError("Select a valid band branch and a search window narrower than half the band spacing.")
    output_dir = Path(output_dir).expanduser().resolve()
    distances = np.linspace(-CUT_HALF_LENGTH_M, CUT_HALF_LENGTH_M, N_SAMPLES)
    datasets, curves, records = [], [], []
    for label, path in DATA_FILES.items():
        phi, theta, lengths, radius = load_footprint(path)
        sample = periodic_sampler(phi, theta, lengths)
        cuts = []
        for angle in PHI_ANGLES_DEG:
            anchor = np.deg2rad(angle)
            center, slope, phase, rms = fit_band(phi, theta, lengths, anchor)
            states = normal_cut(anchor, center, slope, distances)
            sampled = sample(states[:, 0], states[:, 1])
            cuts.append((states, sampled))
            records.append([angle, (np.rad2deg(center)+180) % 360-180, slope,
                            np.rad2deg(phase), np.rad2deg(rms), radius])
            print(f"{label}, phi={angle:g} deg: theta={records[-1][1]:.2f} deg, "
                  f"dtheta/dphi={slope:.3f}, fit RMS={np.rad2deg(rms):.2f} deg")
        datasets.append((phi, theta, lengths))
        curves.append(cuts)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, layout="constrained")
    try:
        for panel, (axis, angle) in enumerate(zip(axes.flat, PHI_ANGLES_DEG)):
            for label, color, cuts in zip(DATA_FILES, COLORS, curves):
                axis.plot(100*distances, cuts[panel][1], color=color, label=label, linewidth=1.5)
            axis.axvline(0, color="0.6", linewidth=0.7, linestyle=":")
            axis.set(title=rf"Cut anchored at $\phi = {angle % 360:g}^\circ$", yscale=Y_SCALE,
                     xlim=100*np.array([-CUT_HALF_LENGTH_M, CUT_HALF_LENGTH_M]))
            axis.grid(alpha=0.25, which="both")
        axes[0, 0].legend()
        fig.supxlabel("Signed distance along vessel surface from fitted streak-set center [cm]")
        fig.supylabel("Connection length [m]")
        fig.suptitle("Magnetic-footprint cross-sections centered on a streak set")
        fig.savefig(output_dir / "magnetic_footprint_cross_sections.png", dpi=DPI)
    finally:
        plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, sharey=True, layout="constrained")
    try:
        vmin = min(data[2].min() for data in datasets)
        vmax = max(data[2].max() for data in datasets)
        for axis, label, data, cuts in zip(axes, DATA_FILES, datasets, curves):
            phi, theta, lengths = data
            # Close both seams for background rendering, including theta=+180.
            phi_pad = np.r_[phi[-1]-2*np.pi, phi, phi[0]+2*np.pi]
            theta_pad = np.r_[theta[-1]-2*np.pi, theta, theta[0]+2*np.pi]
            mesh = axis.pcolormesh(np.rad2deg(phi_pad), np.rad2deg(theta_pad),
                                   np.pad(lengths, 1, mode="wrap"), shading="nearest",
                                   norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis", rasterized=True)
            for panel, (angle, (states, _)) in enumerate(zip(PHI_ANGLES_DEG, cuts)):
                p, t = wrapped_cut_for_plot(states[:, 0], states[:, 1])
                color = plt.get_cmap("tab10")(panel)
                axis.plot(p, t, color="white", linewidth=3)
                axis.plot(p, t, color=color, linewidth=1.4, label=rf"$\phi_0={angle % 360:g}^\circ$")
                middle = states[N_SAMPLES//2]
                axis.plot(np.rad2deg(middle[0]) % 360, (np.rad2deg(middle[1])+180) % 360-180,
                          "o", color=color, markeredgecolor="white", markersize=5)
            axis.set(title=label, xlim=(0, 360), ylim=(-180, 180), ylabel=r"$\theta$ [deg]")
            axis.set_xticks(np.arange(0, 361, 36))
            axis.set_yticks(np.arange(-180, 181, 90))
        axes[0].legend(loc="upper right", ncols=4, fontsize=8)
        axes[-1].set_xlabel(r"Boris wall $\phi$ [deg CCW from South-side split]")
        fig.colorbar(mesh, ax=axes, shrink=0.75, label="Connection length [m]")
        fig.suptitle("Cross-section locations (angular map; perpendicularity uses the surface metric)")
        fig.savefig(output_dir / "magnetic_footprint_cut_locations.png", dpi=DPI)
    finally:
        plt.close(fig)

    np.savez_compressed(
        output_dir / "magnetic_footprint_cross_sections.npz",
        labels=np.array(list(DATA_FILES)), source_files=np.array([str(p) for p in DATA_FILES.values()]),
        phi_anchor_deg=PHI_ANGLES_DEG, distance_m=distances,
        connection_length_m=np.array([[cut[1] for cut in cuts] for cuts in curves]),
        cut_phi_rad=np.array([[cut[0][:, 0] for cut in cuts] for cuts in curves]),
        cut_theta_rad=np.array([[cut[0][:, 1] for cut in cuts] for cuts in curves]),
        fit_columns=np.array(["phi_anchor_deg", "theta_center_deg", "dtheta_dphi", "band_phase_deg",
                              "fit_rms_deg", "launch_radius_m"]),
        fit_parameters=np.array(records).reshape(3, 4, 6),
        rmajor_m=RMAJOR_M, wall_radius_m=WALL_RADIUS_M, phi_wall_offset_deg=PHI_WALL_OFFSET_DEG,
        helical_slope=HELICAL_SLOPE, toroidal_mode=TOROIDAL_MODE, poloidal_mode=POLOIDAL_MODE,
        band_branch=BAND_BRANCH, smooth_sigma_deg=SMOOTH_SIGMA_DEG,
        centering_method="midpoint_of_flanking_low_length_gaps",
        fit_half_window_deg=FIT_HALF_WINDOW_DEG, center_search_half_width_deg=CENTER_SEARCH_HALF_WIDTH_DEG,
    )
    print(f"Saved comparison, cut-location map and sampled NPZ to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    make_plots(parser.parse_args().output_dir)
