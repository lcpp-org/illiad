"""Plot saved Horvath He/Li density scans without rerunning the solver.

Run directly for the default IOTA3 scan, or use:
    python misc_scripts/plot_horvath_density_scan.py --input-dir output/MY_RUN

Each matched directory must contain profiles_1d.csv and run_settings.json.
Densities come from metadata, not folder names. Temperatures are T_slow_eV.
The axis is the model's x_outward_m=0 proxy; the LCFS is the outermost
core sample (L_to_wall_m is NaN or infinity inside the LCFS). The duplicate
SOL-side LCFS row is excluded. The fourth panel is T_LCFS / T_axis.
Missing grid points are gray; values are not interpolated between runs.
The figure and a summary CSV are written to --output-dir.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "output/IOTA3_1000sp_atol1e-9"
RUN_PATTERN = "Horvath_He-Li_*-*"
OUTPUT_SUBDIR = "Horvath_density_scan"
PANELS = (
    ("temperature_max_ev", "Maximum temperature (whole profile)", "Temperature [eV]"),
    ("temperature_lcfs_ev", "LCFS temperature", "Temperature [eV]"),
    ("temperature_axis_ev", "Axis temperature", "Temperature [eV]"),
    ("lcfs_axis_ratio", r"$T_{\mathrm{LCFS}} / T_{\mathrm{axis}}$", "Temperature ratio"),
)


def read_run(directory):
    """Extract densities and temperatures using the saved profile convention."""
    settings = json.loads((directory / "run_settings.json").read_text())
    neutrals = settings["neutral_densities_m3"]
    he, li = (float(neutrals.get(species, 0.0)) for species in ("He", "Li"))
    if not np.all(np.isfinite([he, li])) or min(he, li) < 0:
        raise ValueError(f"{directory}: neutral densities must be finite and nonnegative")
    profile = np.atleast_1d(np.genfromtxt(
        directory / "profiles_1d.csv", delimiter=",", names=True))
    required = {"x_outward_m", "L_to_wall_m", "T_slow_eV"}
    if not required.issubset(profile.dtype.names or ()):
        raise ValueError(f"{directory}: missing profile columns {required}")
    x, temperature = profile["x_outward_m"], profile["T_slow_eV"]
    if not len(x) or not np.all(np.isfinite(x)) or not np.all(
            np.isfinite(temperature) & (temperature > 0)):
        raise ValueError(f"{directory}: invalid coordinates or temperatures")
    core = profile[~np.isfinite(profile["L_to_wall_m"])]
    core = core[np.argsort(core["x_outward_m"])]
    if len(core) < 2 or not np.isclose(core["x_outward_m"][0], 0, atol=1e-12):
        raise ValueError(f"{directory}: cannot identify core and axis samples")
    if np.any(np.diff(core["x_outward_m"]) <= 0):
        raise ValueError(f"{directory}: core coordinates must be unique")
    axis = float(core["T_slow_eV"][0])
    lcfs = float(core["T_slow_eV"][-1])
    return dict(run=directory.name, helium_density_m3=he, lithium_density_m3=li,
                temperature_max_ev=float(temperature.max()),
                temperature_lcfs_ev=lcfs, temperature_axis_ev=axis,
                lcfs_axis_ratio=lcfs / axis)


def cell_edges(centers, scale):
    """Place edges halfway between samples in the chosen axis coordinates."""
    values = np.log10(centers) if scale == "log" else np.asarray(centers)
    if len(values) == 1:
        half_width = 0.5 if scale == "log" else max(abs(values[0]) * 0.5, 0.5)
        edges = np.array([values[0] - half_width, values[0] + half_width])
    else:
        mid = (values[:-1] + values[1:]) / 2
        edges = np.r_[2 * values[0] - mid[0], mid, 2 * values[-1] - mid[-1]]
    return 10.0 ** edges if scale == "log" else edges


def plot_scan(records, scale="log", annotate=True):
    helium = np.unique([row["helium_density_m3"] for row in records])
    lithium = np.unique([row["lithium_density_m3"] for row in records])
    if scale == "log" and (helium[0] <= 0 or lithium[0] <= 0):
        raise ValueError("Logarithmic density axes require positive densities; use --axis-scale linear")
    grids = {key: np.full((len(lithium), len(helium)), np.nan) for key, _, _ in PANELS}
    seen = {}
    for row in records:
        pair = (row["helium_density_m3"], row["lithium_density_m3"])
        if pair in seen:
            raise ValueError(f"Duplicate density pair {pair}: {seen[pair]} and {row['run']}")
        seen[pair] = row["run"]
        j, i = np.searchsorted(helium, pair[0]), np.searchsorted(lithium, pair[1])
        for key in grids:
            grids[key][i, j] = row[key]
    missing = len(helium) * len(lithium) - len(records)
    print(f"Loaded {len(records)} runs: {len(helium)} He × {len(lithium)} Li densities; {missing} missing cells")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.85")
    for ax, (key, title, label) in zip(axes.flat, PANELS):
        mesh = ax.pcolormesh(cell_edges(helium, scale), cell_edges(lithium, scale),
                             np.ma.masked_invalid(grids[key]), cmap=cmap, shading="flat")
        ax.set(xscale=scale, yscale=scale, title=title,
               xlabel=r"Helium neutral density [m$^{-3}$]",
               ylabel=r"Lithium neutral density [m$^{-3}$]")
        ax.set_xticks(helium)
        ax.set_yticks(lithium)
        ax.minorticks_off()
        if scale == "log":
            ax.xaxis.set_major_formatter(LogFormatterSciNotation())
            ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        fig.colorbar(mesh, ax=ax, label=label)
        if annotate:
            for i, li in enumerate(lithium):
                for j, he in enumerate(helium):
                    value = grids[key][i, j]
                    if np.isfinite(value):
                        color = "black" if mesh.norm(value) > 0.55 else "white"
                        ax.text(he, li, f"{value:.3g}", ha="center", va="center", color=color)
    fig.suptitle("Horvath slow-electron temperature — prescribed neutral density scan")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--pattern", default=RUN_PATTERN, help="Run-directory glob within input-dir")
    parser.add_argument("--output-dir", type=Path, help=f"Default: INPUT_DIR/{OUTPUT_SUBDIR}")
    parser.add_argument("--axis-scale", choices=("log", "linear"), default="log")
    parser.add_argument("--no-annotations", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    directories = sorted(path for path in args.input_dir.glob(args.pattern) if path.is_dir())
    if not directories:
        parser.error(f"No run directories match {args.input_dir / args.pattern}")
    try:
        records = [read_run(path) for path in directories]
        fig = plot_scan(records, args.axis_scale, not args.no_annotations)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    output = args.output_dir or args.input_dir / OUTPUT_SUBDIR
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        target = output / f"temperature_heatmaps.{suffix}"
        fig.savefig(target, dpi=200)
        print(f"Saved {target}")
    target = output / "temperature_summary.csv"
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {target}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
