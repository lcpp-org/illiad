"""Recreate FluxGradientor plots from an already-solved Cartesian electric field.

Reads Efield_<data_subdir>.npy in (XYZ component, rho, theta, phi) order and
the adjacent saved coordinate arrays. No potential, magnetic-field data,
Poincare surfaces, or gradient calculation is needed. The existing RTP/XYZ
transform recovers the components shown by the original gradient plots.

Examples (from the project root)::

    python misc_scripts/plot_electric_field.py
    python misc_scripts/plot_electric_field.py --phi 18 360
    python misc_scripts/plot_electric_field.py --components magnitude radial

By default, replaces magnitude_XXX.png, radial_XXX.png, poloidal_XXX.png,
and toroidal_XXX.png in output/<analysis_dir>/plots/<data_subdir>/Efield.
Use --output-subdir to write a separate set of plots.
"""

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from illiad.io import IOHandler
from illiad.utilities.coordtrans import RTP_XYZ_JAC


# DATA AND OUTPUT SETTINGS
ANALYSIS_DIR = "IOTA3_1000sp_atol1e-9"
DATA_SUBDIR = "SOLtrace_500c_RegularGrid_Potential_full_linear"
FIELD_FILENAME = None  # None uses Efield_<data_subdir>.npy
OUTPUT_SUBDIR = None   # None uses <data_subdir>/Efield under plots/ and logs/
RHO_FILENAME = "rho_grid_m.npy"
THETA_FILENAME = "theta_grid_rad.npy"
PHI_FILENAME = "phi_grid_deg.npy"

# None selects all saved planes. A scalar or iterable selects computational
# phi in degrees, e.g. 18 or [18, 90, 360]. CLI --phi overrides this setting.
PHI_DEG = None
COMPONENTS = ("magnitude", "radial", "poloidal", "toroidal")
OUTPUT_FILENAME = "{component}_{phi_deg:03.0f}.png"

# PLOT SETTINGS: match FluxGradientor.output_phi_plots defaults.
FIGSIZE = (6.4, 4.8)
DPI = 250
COLORMAP = "afmhot_r"
COLOR_LIMITS = {
    "magnitude": (0.0, 200.0),
    "radial": (-200.0, 200.0),
    "poloidal": (-100.0, 100.0),
    "toroidal": (-30.0, 30.0),
}  # A None bound autoscales that component on each plotted plane.
COLORBAR_LABEL = "Flux"  # Original label; edit to match your potential's units.
COLORBAR_PAD = 0.05
SHADING = "gouraud"
# Match the flux CLI style and the serif font used by the saved example plots.
PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}
VESSEL_RADIUS = None     # None uses the outer saved rho coordinate.
PHYSICAL_PHI_OFFSET_DEG = 198.0
TITLE = None             # Custom format: {component}, {phi_deg}, {phi_phys_deg}
TITLE_LOCATION = "left"
SHOW_ANGLE_TICKS = True  # False hides the polar angle tick marks and labels.
SHOW_GRID = False
SHOW_PLOT = False
SHOW_PROGRESS = True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("analysis_dir", nargs="?", default=ANALYSIS_DIR)
    parser.add_argument("data_subdir", nargs="?", default=DATA_SUBDIR)
    parser.add_argument(
        "--field-file", default=FIELD_FILENAME,
        help="Filename relative to the data subdirectory, or an absolute .npy path.",
    )
    parser.add_argument(
        "--output-subdir", default=OUTPUT_SUBDIR,
        help="Plot/log subdirectory (default: <data_subdir>/Efield).",
    )
    parser.add_argument("--phi", nargs="+", type=float, help="Computational phi angles in degrees.")
    parser.add_argument(
        "--components", nargs="+", choices=tuple(COLOR_LIMITS), default=COMPONENTS,
        help="Components to plot (default: all four).",
    )
    parser.add_argument(
        "--angle-ticks", action=argparse.BooleanOptionalAction, default=SHOW_ANGLE_TICKS,
        help="Show polar angle ticks and labels (disable with --no-angle-ticks).",
    )
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=SHOW_PROGRESS,
    )
    return parser.parse_args()


def load_electric_field(data_dir, field_filename=None):
    """Memory-map the XYZ field and validate its saved coordinate contract."""
    data_dir = Path(data_dir)
    field_path = data_dir / (field_filename or f"Efield_{data_dir.name}.npy")
    field = np.load(field_path, mmap_mode="r", allow_pickle=False)
    grids = []
    for name, minimum_size in ((RHO_FILENAME, 2), (THETA_FILENAME, 3), (PHI_FILENAME, 1)):
        grid = np.load(data_dir / name, allow_pickle=False)
        if (grid.ndim != 1 or grid.size < minimum_size
                or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0)):
            raise ValueError(f"{name} must be a finite, strictly increasing 1-D grid "
                             f"with at least {minimum_size} samples.")
        grids.append(grid)
    rho, theta, phi_deg = grids
    expected_shape = (3, rho.size, theta.size, phi_deg.size)
    if field.shape != expected_shape:
        raise ValueError(f"Electric field shape {field.shape} does not match "
                         f"(XYZ, rho, theta, phi) = {expected_shape}.")
    if not np.issubdtype(field.dtype, np.floating):
        raise ValueError("Expected a real floating-point electric-field array.")
    if rho[0] < 0 or theta[-1] - theta[0] >= 2 * np.pi:
        raise ValueError("Expected nonnegative rho and theta spanning one period without a duplicate endpoint.")
    return field, rho, theta, phi_deg, field_path


def resolve_plane_indices(saved_phi_deg, selection):
    """Select exact saved angles, with 0 and 360 treated as equivalent."""
    if selection is None:
        return np.arange(saved_phi_deg.size, dtype=np.int64)
    requested = np.atleast_1d(selection).astype(float)
    if requested.ndim != 1 or requested.size == 0 or not np.all(np.isfinite(requested)):
        raise ValueError("Select at least one finite phi angle.")
    indices = []
    for phi_deg in requested:
        distance = np.abs((saved_phi_deg - phi_deg + 180.0) % 360.0 - 180.0)
        index = int(np.argmin(distance))
        if not np.isclose(distance[index], 0.0, rtol=0.0, atol=1e-8):
            raise ValueError(f"No saved plane exists at phi={phi_deg:g} degrees.")
        if index not in indices:
            indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def plane_components(xyz_plane, theta, phi_deg):
    """Convert one (3, rho, theta) XYZ plane into (theta, rho) plot arrays.

    Use the same transform as FluxGradientor, including its toroidal sign
    convention. Radius does not enter this orthonormal vector-basis rotation.
    Batch all radii at each theta to avoid a Python loop over every grid point.
    """
    xyz = np.asarray(xyz_plane, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[0] != 3 or xyz.shape[2] != len(theta):
        raise ValueError("Expected one XYZ plane with shape (3, rho, theta).")
    if not np.all(np.isfinite(xyz)):
        raise ValueError(f"Saved electric field contains non-finite values at phi={phi_deg:g}.")
    rtp = np.empty((3, len(theta), xyz.shape[1]), dtype=np.float64)
    phi_rad = np.radians(phi_deg)
    for index, angle in enumerate(theta):
        rtp[:, index, :] = RTP_XYZ_JAC(
            (0.0, angle, phi_rad), xyz[:, :, index], form="xyz2rtp",
        )
    return {
        "magnitude": np.sqrt(np.sum(xyz**2, axis=0)).T,
        "radial": rtp[0],
        "poloidal": rtp[1],
        "toroidal": rtp[2],
    }


def plot_component(data, rho, theta, phi_deg, component, vessel_radius, sim_io,
                   output_subdir, show_angle_ticks=None):
    # Close the seam at the last theta sample's equivalent angle.
    plot_theta = np.concatenate(([theta[-1] - 2 * np.pi], theta))
    grid_theta, grid_rho = np.meshgrid(plot_theta, rho, indexing="ij")
    plot_data = np.vstack((data[-1], data))
    value_min, value_max = COLOR_LIMITS[component]
    if value_min is not None and value_max is not None and value_min >= value_max:
        raise ValueError(f"Invalid color limits for {component}: {COLOR_LIMITS[component]}")
    fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={"projection": "polar"})
    artist = ax.pcolormesh(
        grid_theta, grid_rho, plot_data, shading=SHADING, cmap=COLORMAP,
        vmin=value_min, vmax=value_max,
    )
    phi_phys_deg = (phi_deg + PHYSICAL_PHI_OFFSET_DEG) % 360.0
    title = (
        f"{component}\n"
        rf"$\phi_{{phy}}$={phi_phys_deg:02.0f}$\degree$ CW from North Split" "\n"
        rf"$\phi_c$={phi_deg:02.0f}$\degree$"
        if TITLE is None else TITLE.format(
            component=component, phi_deg=phi_deg, phi_phys_deg=phi_phys_deg,
        )
    )
    ax.set_title(title, loc=TITLE_LOCATION)
    ax.set_rmax(vessel_radius)
    ax.set_rticks([])
    ax.set_yticklabels([])
    if not (SHOW_ANGLE_TICKS if show_angle_ticks is None else show_angle_ticks):
        ax.tick_params(axis="x", which="both", bottom=False, top=False,
                       labelbottom=False, labeltop=False)
    ax.grid(SHOW_GRID)
    fig.colorbar(artist, ax=ax, label=COLORBAR_LABEL, pad=COLORBAR_PAD)
    filename = OUTPUT_FILENAME.format(component=component, phi_deg=phi_deg)
    sim_io.saveFig(filename, dpi=DPI, subdir=output_subdir)
    sim_io.log.info("Saved electric-field replot: %s/%s", output_subdir, filename)
    if SHOW_PLOT:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    data_dir = PROJECT_ROOT / "output" / args.analysis_dir / "data" / args.data_subdir
    field, rho, theta, phi_deg, field_path = load_electric_field(data_dir, args.field_file)
    indices = resolve_plane_indices(phi_deg, PHI_DEG if args.phi is None else args.phi)
    components = list(dict.fromkeys(args.components))
    if not components or any(component not in COLOR_LIMITS for component in components):
        raise ValueError("COMPONENTS must select magnitude, radial, poloidal, or toroidal.")
    vessel_radius = float(rho[-1] if VESSEL_RADIUS is None else VESSEL_RADIUS)
    if not np.isfinite(vessel_radius) or vessel_radius <= 0:
        raise ValueError("Vessel radius must be finite and positive.")
    output_subdir = args.output_subdir or str(Path(args.data_subdir) / "Efield")
    sim_io = IOHandler(args.analysis_dir)
    sim_io.startLog(
        log_name="plot_electric_field.log", subdir=output_subdir,
        logger_name=f"{args.analysis_dir}_{output_subdir}_electric_replot",
    )
    settings = {
        "FIELD_PATH": str(field_path), "FIELD_SHAPE": field.shape,
        "OUTPUT_SUBDIR": output_subdir, "PHI_DEG": phi_deg[indices].tolist(),
        "COMPONENTS": components, "COLOR_LIMITS": COLOR_LIMITS,
        "COLORMAP": COLORMAP, "VESSEL_RADIUS": vessel_radius, "DPI": DPI,
        "SHOW_ANGLE_TICKS": args.angle_ticks,
    }
    sim_io.inputsBoilerplate("SAVED ELECTRIC-FIELD REPLOT INPUTS", settings, list(settings))
    print(f"Reading electric field: {field_path}")
    print(f"Planes selected: {len(indices)} of {phi_deg.size}; components: {', '.join(components)}")
    log_context = logging_redirect_tqdm(loggers=[sim_io.log]) if args.progress else nullcontext()
    with plt.rc_context(PLOT_STYLE), log_context:
        for index in tqdm(indices, desc="Replotting electric field", unit="plane",
                          dynamic_ncols=True, disable=not args.progress):
            planes = plane_components(field[:, :, :, index], theta, float(phi_deg[index]))
            for component in components:
                plot_component(planes[component], rho, theta, float(phi_deg[index]),
                               component, vessel_radius, sim_io, output_subdir,
                               show_angle_ticks=args.angle_ticks)
    print(f"Saved {len(indices) * len(components)} plot(s): {Path(sim_io.plot_dir) / output_subdir}")


if __name__ == "__main__":
    main()
