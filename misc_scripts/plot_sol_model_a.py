"""Plot saved Model A/B fields and trace status without solving or retracing.

Example: python misc_scripts/plot_sol_model_a.py output/sol_model_a_phi000
"""

import argparse
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np


def load_solution(folder):
    """New solutions are self-contained; older ones can reuse trace geometry."""
    folder = Path(folder)
    with np.load(folder / "solution.npz", allow_pickle=False) as saved:
        data = dict(saved)
    settings_file = folder / "run_settings.json"
    settings = json.loads(settings_file.read_text()) if settings_file.exists() else {}
    geometry_keys = ("boundary_xz_m", "rejected_rho_index", "rejected_theta_index")
    if any(key not in data for key in geometry_keys):
        trace_dir = settings.get("trace_dir")
        summary = Path(trace_dir) / "summary.npz" if trace_dir else None
        if summary is not None and summary.is_file():
            with np.load(summary, allow_pickle=False) as trace:
                data.setdefault("boundary_xz_m", trace["boundaries_xz_m"][int(data["plane_id"])])
                rejected = data["rejected_seed_id"]
                data.setdefault("rejected_rho_index", trace["seed_rho_index"][rejected])
                data.setdefault("rejected_theta_index", trace["seed_theta_index"][rejected])
        else:
            warnings.warn("Older solution lacks plot geometry and the trace summary is unavailable; "
                          "LCFS outline and excluded-cell positions may be omitted", stacklevel=2)
    return data, settings


def plane_points(data, prefix=""):
    """Match the tracer's poloidal display: x = R-R0 = rho*cos(theta)."""
    rho = data["grid_rho_m"][data[prefix + "rho_index"]]
    theta = data["grid_theta_rad"][data[prefix + "theta_index"]]
    return rho * np.cos(theta), rho * np.sin(theta)


def positive_norm(values, scale):
    if scale == "log" and np.any(values > 0):
        positive = values[values > 0]
        lo, hi = positive.min(), positive.max()
        if lo == hi:
            lo, hi = lo / 10, hi * 10
        return LogNorm(lo, hi)
    return Normalize(0, max(float(np.max(values)), 1.0))


PANEL_NAMES = ("density", "Tu", "Te", "connection_length", "lcfs_distance", "trace_status")
DEFAULT_MARKER_SIZE = 35.0


def plot_solution(data, settings, density_scale="log", marker_size=DEFAULT_MARKER_SIZE, panel=None):
    """Plot actual cell-center samples; do not interpolate across excluded cells."""
    if not np.isfinite(marker_size) or marker_size <= 0:
        raise ValueError("marker_size must be finite and positive")
    x, z = plane_points(data)
    excluded = (plane_points(data, "rejected_") if "rejected_rho_index" in data
                else (np.array([]), np.array([])))
    approximate = data.get("connection_approximated", np.zeros(len(x), dtype=bool))
    density, Tu, Te = (np.asarray(data[key]) for key in ("n_e_m3", "Tu_eV", "Te_eV"))
    for key in ("n_e_m3", "Tu_eV", "Te_eV", "Lc_m", "d_min_m"):
        values = data[key]
        if values.shape != x.shape or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{key} must be finite, nonnegative and aligned with the seed indices")
    temperature_norm = Normalize(0, max(float(np.max(Tu)), float(np.max(Te)), 1e-12))
    model = settings.get("model", "A")
    upstream_title = "Solved upstream temperature" if model == "B" else "Prescribed upstream temperature"
    panels = [
        ("Electron density", density, r"$n_e$ [m$^{-3}$]", "viridis", positive_norm(density, density_scale)),
        (upstream_title, Tu, r"$T_u$ [eV]", "inferno", temperature_norm),
        ("Local electron temperature", Te, r"$T_e$ [eV]", "inferno", temperature_norm),
        ("Effective connection length", data["Lc_m"], r"$L_{c,\mathrm{eff}}$ [m]", "viridis",
         positive_norm(data["Lc_m"], "log")),
        ("Sampled minimum LCFS distance", data["d_min_m"] * 1000, r"$d_{\min}$ [mm]", "cividis",
         positive_norm(data["d_min_m"] * 1000, "linear")),
    ]
    if panel is None:
        fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout="constrained")
        selected = range(6)
    else:
        selected = [PANEL_NAMES.index(panel)]
        fig, axes = plt.subplots(figsize=(7, 7), layout="constrained")
    axes = np.asarray(axes).reshape(-1)
    for ax, index in zip(axes, selected):
        if index == 5:
            for mask, color, label in ((~approximate, "#a8bac7", "Both wall hits"),
                                      (approximate, "#e69f00", "One hit: approximated")):
                ax.scatter(x[mask], z[mask], color=color, s=marker_size, linewidths=0,
                           label=f"{label} ({np.count_nonzero(mask):,})")
            ax.scatter(*excluded, color="#c62828", marker="x", s=marker_size, linewidths=.8,
                       label=f"Excluded ({len(data['rejected_seed_id']):,})")
            ax.set_title("Trace treatment", fontsize=11)
            ax.legend(loc="center", fontsize=8, framealpha=.95)
            continue
        title, values, label, cmap, norm = panels[index]
        shown = values > 0 if isinstance(norm, LogNorm) else np.ones(len(x), dtype=bool)
        artist = ax.scatter(x[shown], z[shown], c=values[shown], s=marker_size,
                            cmap=cmap, norm=norm, linewidths=0)
        if np.any(~shown):
            ax.scatter(x[~shown], z[~shown], c="black", marker="x", s=marker_size,
                       linewidths=.8, label=f"Zero ({np.count_nonzero(~shown)})")
            ax.legend(loc="center", fontsize=8)
        fig.colorbar(artist, ax=ax, label=label, shrink=.8, pad=.02)
        ax.set_title(title, fontsize=11)

    angle = np.linspace(0, 2*np.pi, 361)
    wall = float(data["grid_rho_m"][-1])
    for ax in axes.flat:
        ax.plot(wall*np.cos(angle), wall*np.sin(angle), color="black", linewidth=.9)
        if "boundary_xz_m" in data:
            boundary = data["boundary_xz_m"]
            closed = np.vstack([boundary, boundary[0]])
            ax.plot(*closed.T, color="black", linewidth=.9)
        ax.set(xlabel=r"$R-R_0$ [m]", ylabel="Z [m]", aspect="equal",
               xlim=(-1.06*wall, 1.06*wall), ylim=(-1.06*wall, 1.06*wall))
    config = settings.get("config", {})
    mode = config.get("parallel_temperature_model", "unspecified")
    policy = settings.get("incomplete", "unspecified")
    label = f"{model} extended" if settings.get('variant') == 'extended' else model
    context = f"Model {label} — computational φ = {float(data['phi_comp_deg']):g}°"
    if panel is None:
        context += (f" | {mode} | incomplete: {policy}\n"
                    "Cell-center samples; black outlines: LCFS / vessel; temperatures share a color scale")
    else:
        context += f"\n{mode} | incomplete: {policy}"
    fig.suptitle(context, fontsize=12)
    return fig


def save_plots(data, settings, output, density_scale="log", marker_size=DEFAULT_MARKER_SIZE,
               individual=False):
    """Save the overview and optionally all six individual panels as PNG files."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for panel in (None, *PANEL_NAMES) if individual else (None,):
        fig = plot_solution(data, settings, density_scale, marker_size, panel)
        model = settings.get("model", "A").lower()
        path = output / f"sol_model_{model}_{panel or 'overview'}.png"
        try:
            fig.savefig(path, dpi=220)
        finally:
            plt.close(fig)
        print(f"Saved {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solution_dir", type=Path, help="Directory containing solution.npz")
    parser.add_argument("--output", type=Path, help="Plot directory; default: <solution_dir>/plots")
    parser.add_argument("--density-scale", choices=("linear", "log"), default="log")
    parser.add_argument("--marker-size", type=float, default=DEFAULT_MARKER_SIZE,
                        help="Scatter marker area in points squared (default: 35)")
    parser.add_argument("--individual", action="store_true", help="Also save each of the six panels separately")
    args = parser.parse_args(argv)
    data, settings = load_solution(args.solution_dir)
    output = args.output or args.solution_dir / "plots"
    save_plots(data, settings, output, args.density_scale, args.marker_size, args.individual)


if __name__ == "__main__":
    main()
