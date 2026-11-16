"""Saved-path and per-seed LCFS figures, including shared scales across offsets.

Only summaries/settings and optional saved paths are needed; no field loading
or Warp initialization occurs here. Figure names and PNG format are preserved.
"""

import json
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np

from misc_scripts.lcfs_distance_utils import (
    ensure_path_metrics, load_summary, signed_boundary_distance,
)

# Preserve the existing distance and connection-length palettes independently.
COLORMAP = "Blues"
CONNECTION_COLORMAP = "Blues_r"
COLOR_SCALE = "log"
DISTANCE_STATISTIC = "min"
_PLOT_DISTANCE_CACHE = {}


def clear_plot_cache():
    """Release contour grids between launch-plane batches."""
    _PLOT_DISTANCE_CACHE.clear()

def save_figure(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def connection_color_scale(total, color_scale):
    """Use the same length normalization for individual and combined plots."""
    # Empty traces still produce a geometry-only initial-point plot. The dummy
    # normalization is never displayed when the entire dataset is empty.
    if total.size == 0:
        total = np.array([1.0])
    if color_scale == "log":
        if not np.all(np.isfinite(total) & (total > 0)):
            raise ValueError("Log color scale requires finite, positive total connection lengths.")
        vmin, vmax = float(total.min()), float(total.max())
        if vmin == vmax:
            # Keep a nonzero logarithmic range when all lines reach the same cap.
            vmin /= np.sqrt(10.0)
            vmax *= np.sqrt(10.0)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif color_scale == "linear":
        norm = Normalize(0.0, max(float(total.max()), 1e-12))
    else:
        raise ValueError("Color scale must be 'linear' or 'log'.")
    cmap = plt.get_cmap(CONNECTION_COLORMAP)
    label = "Traced length $L_+ + L_-$ [m]; lower bound if capped"
    if color_scale == "log":
        label += " (log scale)"
    return norm, cmap, label


def plot_paths(output, settings, data, color_scale):
    """Render retained trajectories; per-seed figures are handled by plot_results."""
    major_radius = settings["major_radius_m"]
    vessel_radius = settings["vessel_radius_m"]
    xyz = data["seed_xyz_m"]
    total = data["length_m"].sum(axis=1)
    norm, cmap, label = connection_color_scale(total, color_scale)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    surface_index = settings.get("surface_index", settings.get("lcfs_index"))
    title = f"Surface {surface_index}, offset {settings['offset_mm']:g} mm; {settings.get('solver', 'LSODA')}"
    if not len(xyz):
        return

    fig3 = plt.figure(figsize=(10, 6.5), layout="constrained")
    ax3 = fig3.add_subplot(projection="3d")
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for i in range(len(xyz)):
        with np.load(output / "paths" / f"seed_{i:05d}.npz") as paths:
            for name, style in (("forward", "-"), ("backward", "--")):
                p = paths[f"{name}_xyz_m"]
                color = cmap(norm(total[i]))
                ax3.plot(*p.T, style, color=color, linewidth=0.8, alpha=0.8)
                axes[0].plot(p[:, 0], p[:, 1], style, color=color, linewidth=0.5, alpha=0.7)
                axes[1].plot(np.hypot(p[:, 0], p[:, 1]) - major_radius, p[:, 2],
                             style, color=color, linewidth=0.5, alpha=0.7)
    ax3.scatter(*xyz.T, c=total, norm=norm, cmap=cmap, s=18, edgecolors="black")
    # Fixed vessel extents provide equal physical scaling across all runs.
    extent = major_radius + vessel_radius
    ax3.set(xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]",
            xlim=(-extent, extent), ylim=(-extent, extent),
            zlim=(-vessel_radius, vessel_radius),
            title=title + "\nSolid: +B; dashed: -B (computational coordinates)")
    ax3.set_box_aspect((2 * extent, 2 * extent, 2 * vessel_radius), zoom=0.85)
    ax3.tick_params(labelsize=8)
    fig3.colorbar(mappable, ax=ax3, shrink=0.65, pad=0.12, label=label)
    save_figure(fig3, plots, "paths_3d")
    axes[0].set(xlabel="X [m]", ylabel="Y [m]", title="Top view")
    axes[1].set(xlabel="$R-R_0$ [m]", ylabel="Z [m]", title="Poloidal projection (all toroidal angles)")
    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    fig2.suptitle(title + "\nSolid: +B; dashed: -B")
    fig2.colorbar(mappable, ax=axes, label=label, shrink=0.8)
    save_figure(fig2, plots, "paths_projections")


def exclusion_plot_grid(boundary, vessel_radius):
    """Resolve the displayed buffer independently of a deliberately coarse launch grid."""
    key = (boundary.tobytes(), vessel_radius)
    if key not in _PLOT_DISTANCE_CACHE:
        axis = np.linspace(-vessel_radius, vessel_radius, 201)
        x, z = np.meshgrid(axis, axis)
        distance = signed_boundary_distance(np.column_stack((x.ravel(), z.ravel())), boundary)
        _PLOT_DISTANCE_CACHE[key] = x, z, distance.reshape(x.shape)
    return _PLOT_DISTANCE_CACHE[key]


def distance_color_scale(values, color_scale, statistic):
    values = np.asarray(values)
    if not np.all(np.isfinite(values) & (values >= 0)):
        raise ValueError("LCFS distances must be finite and nonnegative.")
    maximum = max(float(values.max()) if values.size else 0.0, 1e-9)
    if color_scale == "linear":
        norm = Normalize(0, maximum)
    elif color_scale == "log":
        positive = values[values > 0]
        minimum = float(positive.min()) if positive.size else maximum / 10
        if minimum == maximum:
            minimum /= np.sqrt(10)
            maximum *= np.sqrt(10)
        # clip=True maps exact zeros to the low color instead of hiding seeds.
        norm = LogNorm(minimum, maximum, clip=True)
    else:
        raise ValueError("Color scale must be linear or log.")
    label = {"min": "Minimum sampled",
             "mean": "Arc-length-weighted mean", "max": "Sampled maximum"}[statistic]
    label += " distance\nto 3D LCFS [mm]"
    if color_scale == "log":
        label += "\n(log; zeros use lowest color)"
    return norm, plt.get_cmap(COLORMAP), label


def plot_results(output, color_scale=None, statistic=DISTANCE_STATISTIC, include_paths=True):
    settings = json.loads((output / "settings.json").read_text())
    data = load_summary(output)
    ensure_path_metrics(output, data, settings)
    if include_paths and all((output / "paths" / f"seed_{i:05d}.npz").is_file() for i in range(len(data["seed_xyz_m"]))):
        plot_paths(output, settings, data, color_scale or COLOR_SCALE)
    elif include_paths:
        print(f"Full paths unavailable; replotting saved per-point metrics: {output}")
    (output / "plots").mkdir(exist_ok=True)
    norm, cmap, label = distance_color_scale(data[f"lcfs_distance_{statistic}_m"] * 1000,
                                            color_scale or COLOR_SCALE, statistic)
    title = (f"Grid {settings['nrho']} x {settings['ntheta']}; exclusion offset {settings['offset_mm']:g} mm; "
             f"distance to LCFS {settings['distance_surface_index']}\n"
             "Distance statistic covers the traced portions only")
    lengths = data["length_m"].sum(axis=1)
    length_norm, length_cmap, length_label = connection_color_scale(lengths, color_scale or COLOR_SCALE)
    plot_distance_initial_points(output / "plots", settings, [data], length_norm, length_cmap,
                                 length_label, title, statistic="length")
    plot_distance_initial_points(output / "plots", settings, [data], norm, cmap, label,
                                 title, statistic=statistic)
    plot_extra_metrics(output / "plots", settings, [data], title, color_scale or COLOR_SCALE)


def metric_values(data, statistic):
    if statistic == "length":
        return data["length_m"].sum(axis=1)
    if statistic == "lcfs_min_s":
        return data["lcfs_min_s_m"]
    if statistic == "wall_connection":
        return data["wall_connection_length_m"]
    return data[f"lcfs_distance_{statistic}_m"] * 1000


def plot_extra_metrics(plots, settings, datasets, title, color_scale, suffix=""):
    for metric, label in (("lcfs_min_s", "Arc length from launch to\nclosest sampled LCFS approach [m]"),
                          ("wall_connection", "Shortest arc length to wall [m]\n(lower bound if capped)")):
        values = np.concatenate([metric_values(data, metric) for data in datasets])
        norm, cmap, _ = distance_color_scale(values, color_scale, "min")
        if color_scale == "log":
            label += "\n(log; zeros use lowest color)"
        plot_distance_initial_points(plots, settings, datasets, norm, cmap, label, title,
                                     suffix=suffix, statistic=metric)


def plot_distance_initial_points(plots, settings, datasets, norm, cmap, label, title, suffix="", statistic=DISTANCE_STATISTIC):
    """Overlay seed datasets in list order, retaining wall/capped marker meanings."""
    surface_index = settings.get("surface_index", settings.get("lcfs_index"))
    has_seeds = any(len(data["seed_xyz_m"]) for data in datasets)
    for plane_id, phi in enumerate(settings["phi_deg"]):
        boundary = datasets[0]["boundaries_xz_m"][plane_id]
        fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
        closed = np.vstack((boundary, boundary[0]))
        ax.plot(*closed.T, color="black", linewidth=1, label=f"Exclusion LCFS {surface_index}")
        reference = datasets[0]["lcfs_reference_boundaries_xz_m"][plane_id]
        ax.plot(*np.vstack((reference, reference[0])).T, color="tab:orange", linestyle="--",
                linewidth=1, label=f"Distance LCFS {settings['distance_surface_index']}")
        angle = np.linspace(0.0, 2 * np.pi, 361)
        ax.plot(settings["vessel_radius_m"] * np.cos(angle), settings["vessel_radius_m"] * np.sin(angle),
                color="gray", linewidth=1, label="Vessel wall")
        seen = set()
        for data in datasets:
            offset = data.get("offset_mm", settings["offset_mm"])
            if offset != 0:
                x, z, values = exclusion_plot_grid(boundary, settings["vessel_radius_m"])
                if values.min() < offset / 1000.0 < values.max():
                    ax.contour(x, z, values, levels=[offset / 1000.0],
                               colors=["tab:green"], linestyles=":", linewidths=1)
                    description = f"Exclusion margin {offset:g} mm"
                    if description not in seen:
                        ax.plot([], [], color="tab:green", linestyle=":", label=description)
                        seen.add(description)
            mask = data["plane_id"] == plane_id
            xyz = data["seed_xyz_m"][mask]
            points = np.column_stack((np.hypot(xyz[:, 0], xyz[:, 1]) - settings["major_radius_m"], xyz[:, 2]))
            total = metric_values(data, statistic)[mask]
            complete = (~data["wall_connection_is_lower_bound"][mask] if statistic == "wall_connection"
                        else data["hit_wall"][mask].all(axis=1))
            for is_complete, marker, description in ((True, "o", "Nearest wall reached" if statistic == "wall_connection" else "Both ends hit wall"),
                                                      (False, "^", "Length limit reached")):
                selected = complete == is_complete
                if np.any(selected):
                    ax.scatter(*points[selected].T, c=total[selected], norm=norm, cmap=cmap,
                               marker=marker, s=30, edgecolors="black",
                               linewidths=0.5 if len(xyz) < 500 else 0.1,
                               label=description if description not in seen else "_nolegend_")
                    seen.add(description)
        if not any(np.any(data["plane_id"] == plane_id) for data in datasets):
            ax.text(0.5, 0.5, "No grid seeds survive LCFS/vessel filtering", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
        ax.set(xlabel="$R-R_0$ [m]", ylabel="Z [m]", aspect="equal",
               xlim=(-1.1 * settings["vessel_radius_m"], 1.1 * settings["vessel_radius_m"]),
               ylim=(-1.1 * settings["vessel_radius_m"], 1.1 * settings["vessel_radius_m"]),
               title=title + f"\nLaunch plane $\\phi_{{comp}}={phi:g}^\\circ$")
        ax.grid(alpha=0.2)
        if has_seeds:
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=label)
        prefix = {"length": "initial_points", "lcfs_min_s": "initial_points_lcfs_min_s",
                  "wall_connection": "initial_points_wall_connection_length"}.get(statistic, f"initial_points_lcfs_distance_{statistic}")
        save_figure(fig, plots, f"{prefix}{suffix}_phi{phi:03d}")


def plot_combined_offsets(output, color_scale=None, statistic=DISTANCE_STATISTIC):
    """Plot all offsets using saved summaries only, with a global color range."""
    color_scale = COLOR_SCALE if color_scale is None else color_scale
    entries = json.loads((output / "offset_scan.json").read_text())["offsets"]
    if not entries:
        raise ValueError("Offset scan contains no offsets.")
    datasets, reference = [], None
    for entry in entries:
        folder = output / entry["directory"]
        settings = json.loads((folder / "settings.json").read_text())
        if reference is None:
            reference = settings
        for key in ("phi_deg", "major_radius_m", "vessel_radius_m", "distance_surface_index", "distance_poloidal_points", "smoothing", "nrho", "ntheta"):
            if settings[key] != reference[key]:
                raise ValueError(f"Cannot combine offset plots with different {key}: {folder}")
        if settings.get("surface_index", settings.get("lcfs_index")) != reference.get("surface_index", reference.get("lcfs_index")):
            raise ValueError(f"Cannot combine different flux surfaces: {folder}")
        data = load_summary(folder)
        ensure_path_metrics(folder, data, settings)
        datasets.append(dict(data, offset_mm=settings["offset_mm"]))
    total = np.concatenate([data[f"lcfs_distance_{statistic}_m"] * 1000 for data in datasets])
    norm, cmap, label = distance_color_scale(total, color_scale, statistic)
    index = reference.get("surface_index", reference.get("lcfs_index"))
    offsets_text = ", ".join(f"{entry['offset_mm']:g}" for entry in entries)
    title = f"Grid {reference['nrho']} x {reference['ntheta']}; exclusion LCFS {index}; distance to LCFS {reference['distance_surface_index']}; {reference.get('solver', 'LSODA')}\n" + fill(f"Offsets [mm]: {offsets_text}", width=60)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    lengths = np.concatenate([data["length_m"].sum(axis=1) for data in datasets])
    ln, lc, ll = connection_color_scale(lengths, color_scale)
    plot_distance_initial_points(plots, reference, datasets, ln, lc, ll, title,
                                 suffix="_all_offsets", statistic="length")
    plot_distance_initial_points(plots, reference, datasets, norm, cmap, label, title,
                                 suffix="_all_offsets", statistic=statistic)
    plot_extra_metrics(plots, reference, datasets, title, color_scale, suffix="_all_offsets")
    print(f"Saved combined initial-point plots: {plots}")
