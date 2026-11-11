"""Thesis comparisons from saved Boris outputs; no trajectory recomputation.

Run with no arguments for the configured typical and lithium-evaporation scans.
Edit RUNS/styles below; plot or data directories are accepted. See
docs/thesis_collision_scan.md for statistical definitions and output details.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "output/IOTA4_1000sp_atol1e-9/plots"
# Explicit lists preserve the unusual doubled 'neutrals' in the EVAP names.
# Use the matched 60 V typical baseline, as requested.
RUNS = {
    "typ": [BASE / "0mm_LCFS71_1eV_60V_Z1_newTHESIScompare_typ_noColl"]
    + [BASE / f"0mm_LCFS71_1eV_60V_Z1_newTHESIScompare_typ_neutrals_1e{p}"
       for p in (16, 17, 18)]
    + [BASE / f"0mm_LCFS71_1eV_60V_Z1_newTHESIScompare_typ_IONION1e18_neutrals_1e{p}"
       for p in (16, 17, 18)],
    "EVAP": [BASE / "0mm_LCFS71_5eV_120V_Z2_newTHESIScompare_EVAP_neutrals_noColl"]
    + [BASE / f"0mm_LCFS71_5eV_120V_Z2_newTHESIScompare_EVAP_neutrals_neutrals_1e{p}"
       for p in (16, 17, 18)]
    + [BASE / f"0mm_LCFS71_5eV_120V_Z2_newTHESIScompare_EVAP_IONION1e18_neutrals_1e{p}"
       for p in (16, 17, 18)],
}
OUTPUT = PROJECT_ROOT / "output/thesis_collision_scan"
# Add directly to calculated mean residence times, in milliseconds.
# Applies to every run in the regime, including its collisionless baseline.
RESIDENCE_ADDERS_MS = {
    "typ": 0.019,   # Typical cases.
    "EVAP": -0.006,  # Lithium-evaporation cases.
}
TITLES = {"typ": "Typical operation\n$T_i=1eV, V_{plasma}=60V$", "EVAP": "Lithium evaporation\n$T_i=5eV, V_{plasma}=120V$"}
#TITLES = {"typ": None, "EVAP": None}
#DENSITY_COLORS = ["#0072B2", "#E69F00", "#009E73"]
DENSITY_COLORS = ["#13294B", "#FF5F05", "#4A90E2"]
OPERATOR_COLORS = ["#13294B", "#FF5F05"]
FIGSIZE = (7.2, 4.8)
DPI = 300
KG_PER_AMU = 1.66053907e-27  # Match illiad.particle.Ion.
JOULES_PER_EV = 1.602176634e-19


def read_metadata(log_path):
    """Read actual run settings, never infer physical parameters from names."""
    log = log_path.read_text()
    # Logs can be appended; use the final run header and its subsequent output.
    log = log.rsplit("Started Logger:", 1)[-1]

    def value(label, cast=float):
        match = re.search(r"^\|\s*(?:-->\s*)?" + re.escape(label)
                          + r":\s*(\S+)", log, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"{log_path}: missing {label}")
        return cast(match[1])

    meta = {
        "dt_s": value("TIME STEP"), "tmax_s": value("TOTAL TIME"),
        "n_particles": value("TOTAL PARTICLES", int),
        "mass_amu": value("ION MASS"), "charge": value("ION CHARGE", int),
        "ion_temperature_eV": value("ION TEMPERATURE"),
        "potential_V": value("LOADED ELECTRIC FIELD SCALING FACTOR"),
        "neutral_density_m3": value("NEUTRAL GAS DENSITY"),
        "plasma_density_scale_m3": value("PLASMA DENSITY"),
        "ion_neutral_operator": value("ION-NEUTRAL COLLISIONS", str),
        "ion_ion_operator": value("ION-ION COLLISIONS", str),
        "electric_field": value("LOADED ELECTRIC FIELD DATA FROM", str),
        "density_field": value("LOADED DENSITY FIELD DATA FROM", str),
    }
    for key in ("dt_s", "tmax_s", "mass_amu", "n_particles"):
        if not np.isfinite(meta[key]) or meta[key] <= 0:
            raise ValueError(f"{log_path}: invalid {key}")
    return meta


def survival_statistics(last_inside_steps, n_particles, dt, horizon):
    """Restore first-outside times; unobserved hits are censored at horizon.

    Wallpt_OUTPUT row 6 stores the last inside step, so exit is one step later.
    The exact restricted mean is sum(min(exit_time, horizon))/N, including
    horizon for each surviving ion. It is not the mean of wall-hit times alone.
    """
    steps = np.asarray(last_inside_steps)
    if (steps.ndim != 1 or len(steps) > n_particles or not np.isfinite(steps).all()
            or np.any(steps < 0) or np.any(steps != np.floor(steps))):
        raise ValueError("Invalid last-inside steps or wall-hit count.")
    hit_times = (steps + 1) * dt
    if np.any(hit_times > horizon + dt * 1e-4):
        raise ValueError("Wall-hit time exceeds the observation horizon.")
    # Histogram at the physical timestep avoids sorting millions of events.
    nsteps = int(round(horizon / dt))
    counts = np.bincount(steps.astype(np.int64) + 1, minlength=nsteps + 1)
    time = np.arange(nsteps + 1) * dt
    survival = 1 - np.cumsum(counts) / n_particles
    survivors = n_particles - len(steps)
    restricted_ms = (np.minimum(hit_times, horizon).sum()
                     + survivors * horizon) / n_particles * 1000
    # Reproduce the old IonsVtime integration and tail prescription separately.
    legacy_time = np.arange(0, horizon, dt)
    sorted_times = np.sort(steps) * dt
    legacy = 1 - np.searchsorted(sorted_times, legacy_time, side="right") / n_particles
    legacy += 1 - legacy[0]
    trapz = getattr(np, "trapezoid", np.trapz)
    observed = trapz(legacy, dx=dt)
    correction = 0.0
    if legacy[-1] > 0:
        window = min(100, len(legacy) - 1)
        slope = ((np.log(legacy[-1]) - np.log(legacy[-1-window])) / window / dt
                 if window else -0.1)
        correction = -legacy[-1] / min(slope, -0.1)
    return time * 1000, survival, {
        "n_hits": len(steps), "n_survivors": survivors,
        "surviving_fraction": survivors / n_particles,
        "restricted_mean_residence_ms": restricted_ms,
        "legacy_observed_residence_ms": observed * 1000,
        "legacy_tail_correction_ms": correction * 1000,
        "legacy_extrapolated_residence_ms": (observed + correction) * 1000,
    }


def deposition_statistics(wall, mass_amu):
    """Arithmetic means over wall hits; angle from outward toroidal normal."""
    if wall.shape[1] == 0:
        return {"mean_deposition_energy_eV": float("nan"),
                "mean_deposition_angle_deg": float("nan")}
    # Chunk to bound memory for multi-million-ion runs.
    energy_sum = angle_sum = 0.0
    for start in range(0, wall.shape[1], 250_000):
        block = np.asarray(wall[:, start:start + 250_000], dtype=float)
        if not np.isfinite(block).all():
            raise ValueError("Non-finite wall output.")
        theta, phi = block[1:3]
        velocity = block[3:6]
        speed2 = np.sum(velocity**2, axis=0)
        if np.any(speed2 <= 0):
            raise ValueError("Cannot define a deposition angle for zero velocity.")
        # Same radial unit vector as RTP_XYZ_JAC(..., [1, 0, 0], 'rtp2xyz').
        normal_velocity = (velocity[0] * np.cos(theta) * np.cos(phi)
                           - velocity[1] * np.cos(theta) * np.sin(phi)
                           + velocity[2] * np.sin(theta))
        angles = np.degrees(np.arccos(np.clip(normal_velocity / np.sqrt(speed2), -1, 1)))
        energy_sum += np.sum(0.5 * mass_amu * KG_PER_AMU * speed2 / JOULES_PER_EV)
        angle_sum += np.sum(angles)
    return {"mean_deposition_energy_eV": energy_sum / wall.shape[1],
            "mean_deposition_angle_deg": angle_sum / wall.shape[1]}


def load_run(path, regime):
    path = Path(path).expanduser().resolve()
    if path.parent.name not in ("plots", "data"):
        raise ValueError(f"Expected a run under plots/ or data/: {path}")
    root = path.parent.parent
    data = root / "data" / path.name
    log = root / "logs" / path.name / "boris.log"
    meta = read_metadata(log)
    wall = np.load(data / "Wallpt_OUTPUT.npy", mmap_mode="r", allow_pickle=False)
    initial = np.load(data / "initVelPos.npy", mmap_mode="r", allow_pickle=False)
    if wall.ndim != 2 or wall.shape[0] != 7:
        raise ValueError(f"{data}: expected wall output shaped (7, N).")
    if initial.shape != (meta["n_particles"], 6):
        raise ValueError(f"{data}: initial particle count/shape disagrees with log: {initial.shape}")
    time, survival, stats = survival_statistics(
        wall[6], meta["n_particles"], meta["dt_s"], meta["tmax_s"])
    adder_ms = float(RESIDENCE_ADDERS_MS[regime])
    if not np.isfinite(adder_ms):
        raise ValueError(f"{regime}: residence adder must be finite.")
    stats["residence_adder_ms"] = adder_ms
    for key in ("restricted_mean_residence_ms", "legacy_extrapolated_residence_ms"):
        stats[f"unadjusted_{key}"] = stats[key]
        stats[key] += adder_ms
    stats.update(deposition_statistics(wall, meta["mass_amu"]))
    off = {"none", "false", "off"}
    ii = meta["ion_ion_operator"].lower() not in off
    neutral = meta["ion_neutral_operator"].lower() not in off
    if ii and not neutral:
        raise ValueError("This scan expects ion-ion cases to include ion-neutral collisions.")
    mode = "both" if ii else ("neutral" if neutral else "collisionless")
    if ii and not np.isclose(meta["plasma_density_scale_m3"], 1e18):
        warnings.warn(f"{path.name}: log plasma-density scaling is "
                      f"{meta['plasma_density_scale_m3']:.3g} m^-3; directory says IONION1e18.")
    return {"regime": regime, "run": path.name, "data_directory": str(data),
            "log_path": str(log), "mode": mode, **meta, **stats,
            "time_ms": time, "survival": survival}


def density_text(density):
    exponent = int(np.floor(np.log10(density)))
    mantissa = density / 10**exponent
    prefix = "" if np.isclose(mantissa, 1) else rf"{mantissa:g}\times "
    return rf"${prefix}10^{{{exponent}}}$"


def check_group(runs):
    baseline = [r for r in runs if r["mode"] == "collisionless"]
    if len(baseline) != 1:
        raise ValueError("Each regime needs exactly one collisionless baseline.")
    densities = sorted({r["neutral_density_m3"] for r in runs if r["mode"] != "collisionless"})
    for density in densities:
        for mode in ("neutral", "both"):
            if sum(r["mode"] == mode and r["neutral_density_m3"] == density for r in runs) != 1:
                raise ValueError(f"Need one {mode} run at neutral density {density}.")
    if not densities:
        raise ValueError("No neutral-density scan runs selected.")
    for key in ("tmax_s", "dt_s", "mass_amu", "charge", "ion_temperature_eV", "electric_field"):
        if len({r[key] for r in runs}) != 1:
            raise ValueError(f"Runs differ in {key}; use a comparable group.")
    return baseline[0], densities


def figure_note(runs):
    potentials = sorted({r["potential_V"] for r in runs})
    density_scales = sorted({r["plasma_density_scale_m3"] for r in runs if r["mode"] == "both"})
    potential = "/".join(f"{v:g}" for v in potentials)
    density = ", ".join(density_text(n) for n in density_scales)
    return (rf"Potential: {potential} V; ion-ion density scaling: {density} $\mathrm{{m}}^{{-3}}$"
            if density_scales else f"Potential: {potential} V")


def save_figure(fig, output, stem, formats, dpi):
    for extension in formats:
        fig.savefig(output / f"{stem}.{extension}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_figures(runs, output, formats, dpi, residence_estimator):
    baseline, densities = check_group(runs)
    regime = baseline["regime"]
    colors = {n: DENSITY_COLORS[i % len(DENSITY_COLORS)] for i, n in enumerate(densities)}
    # Three curve figures per regime: each operator set and their direct overlay.
    for selection in ("neutral", "both", "all"):
        fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
        ax.plot(baseline["time_ms"], baseline["survival"], color="0.2", lw=1.8,
                label=f"Collisionless ({baseline['potential_V']:g} V)")
        for run in runs:
            if run["mode"] == "collisionless" or selection not in ("all", run["mode"]):
                continue
            ax.plot(run["time_ms"], run["survival"],
                    color=colors[run["neutral_density_m3"]],
                    ls="--" if run["mode"] == "both" else "-", lw=1.7,
                    label=rf"$n_g =$ {density_text(run['neutral_density_m3'])} $\mathrm{{m}}^{{-3}}$"
                    if selection != "all" or run["mode"] == "neutral" else "_nolegend_")
        if selection == "all":
            handles, labels = ax.get_legend_handles_labels()
            handles += [Line2D([], [], color="0.35", ls=style) for style in ("-", "--")]
            ax.legend(handles, labels + ["Ion-neutral only", "Ion-neutral + ion-ion"],
                      loc="upper left", fontsize=9,
                      title=TITLES[regime], title_fontsize=11)
        else:
            ax.legend(loc="upper left", fontsize=9,
                      title=TITLES[regime], title_fontsize=11)
        subtitle = {"neutral": "Ion-neutral collisions", "both": "Ion-neutral + ion-ion collisions",
                    "all": "Collision-operator comparison"}[selection]
        ax.set(title=subtitle, xlabel="Time (ms)",
               ylabel="Fraction of ions remaining", xlim=(0, baseline["tmax_s"] * 1000),
               ylim=(0, 1.025))
        ax.grid(alpha=0.22)
        fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
        shown_runs = [r for r in runs if selection == "all" or r["mode"] in ("collisionless", selection)]
        #fig.text(0.5, 0.025, figure_note(shown_runs), ha="center", fontsize=9)
        save_figure(fig, output, f"{regime}_IonsVtime_{selection}", formats, dpi)

    residence_key = ("restricted_mean_residence_ms" if residence_estimator == "restricted"
                     else "legacy_extrapolated_residence_ms")
    residence_label = (f"Mean residence time (ms)"
                       if residence_estimator == "restricted" else "Extrapolated mean residence time (ms)")
    metrics = [(residence_key, residence_label, "residence_time"),
               ("mean_deposition_energy_eV", "Mean deposition energy (eV)", "deposition_energy"),
               ("mean_deposition_angle_deg", "Mean angle from wall normal (deg)", "deposition_angle")]
    for key, ylabel, stem in metrics:
        fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
        ax.bar([0], [baseline[key]], width=0.64, color="0.45", label="Collisionless")
        ax.bar_label(ax.containers[-1], fmt="%.3g", padding=3, fontsize=9)
        for mode, offset, color, label in zip(
                ("neutral", "both"), (-0.18, 0.18), OPERATOR_COLORS,
                ("Ion-neutral only", "Ion-neutral + ion-ion")):
            values = [next(r[key] for r in runs if r["mode"] == mode
                           and r["neutral_density_m3"] == n) for n in densities]
            bars = ax.bar(np.arange(1, len(densities) + 1) + offset, values,
                          width=0.34, color=color, label=label)#,
                          #hatch="//" if mode == "both" else None)
            ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=9)
        ax.set_xticks(range(len(densities) + 1),
                      [f"Collisionless"] + [density_text(n) for n in densities])
        ax.set(xlabel=r"Neutral density $n_g$ ($\mathrm{m}^{-3}$)", ylabel=ylabel,
               ylim=(0, max(r[key] for r in runs) * 1.5))
        ax.legend(loc="upper left", fontsize=8.5,
                  title=TITLES[regime], title_fontsize=11)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.22)
        fig.get_layout_engine().set(rect=(0, 0.11, 1, 0.89))
        note = ("Restricted mean; survivors censored at simulation end."
                if stem == "residence_time" and residence_estimator == "restricted" else
                "Includes the default IonsVtime exponential-tail estimate."
                if stem == "residence_time" else "Arithmetic mean over deposited ions only.")
        #fig.text(0.5, 0.025, figure_note(runs) + "\n" + note, ha="center", fontsize=9)
        save_figure(fig, output, f"{regime}_{stem}", formats, dpi)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--formats", nargs="+", choices=("pdf", "svg", "png"), default=["png"])
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--regimes", nargs="+", choices=tuple(RUNS), default=list(RUNS))
    parser.add_argument("--residence-estimator", choices=("restricted", "extrapolated"), default="restricted")
    args = parser.parse_args(argv)
    all_runs = []
    for regime in args.regimes:
        runs = [load_run(path, regime) for path in RUNS[regime]]
        check_group(runs)
        all_runs.extend(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    for regime in args.regimes:
        make_figures([r for r in all_runs if r["regime"] == regime], args.output_dir,
                     args.formats, args.dpi, args.residence_estimator)
    rows = [{k: v for k, v in run.items() if k not in ("time_ms", "survival")} for run in all_runs]
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "script": str(Path(__file__).resolve()), "residence_estimator": args.residence_estimator,
        "runs": rows}, indent=2) + "\n")
    for run in all_runs:
        np.savez_compressed(args.output_dir / f"{run['run']}_survival.npz",
                            time_ms=run["time_ms"], fraction_remaining=run["survival"])
    print(f"Saved {6 * len(args.regimes)} figures in {', '.join(args.formats)} formats to {args.output_dir}")
    print("Summary: summary.csv; metadata: provenance.json; curves: *_survival.npz")


if __name__ == "__main__":
    main()
