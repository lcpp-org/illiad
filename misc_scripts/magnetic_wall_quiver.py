"""Plot +B directions on the footprint's unwrapped HIDRA wall map.

Run directly for IOTA3 followed by IOTA5, both at a 1 mm wall offset.
Pass --inputs-json for a single case with overrides of DEFAULT_INPUTS (IOTA3).
WALL_OFFSET_M=0 samples the wall; positive offsets sample inside it.
ARROW_MODE selects "normalized", "magnitude", or "both" (default).
Normalized arrows have equal displayed lengths. Magnitude arrows have lengths
proportional to |B_tangent|, with ARROW_LENGTH_INCHES representing the largest
tangent field in this run. Color always represents total |B| in tesla.

Arrows show the tangent projection, not the wall-normal component. Their
displayed directions account for the torus metric and the footprint's aspect
ratio, so they follow field-line slopes on this angular map. The NPZ retains
Cartesian, tangent, and outward-normal components for further analysis.
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from misc_scripts.magnetic_footprint import (
    build_magnetic_field,
    make_initial_conditions,
)
from illiad.io import IOHandler
from illiad.plotting import global_plotPorts
from illiad.utilities.run_config import load_inputs_json, merge_input_params


DEFAULT_INPUTS = {
    "ANALYSIS_NAME": "IOTA3_wall_quiver_1mm",
    "CURRENT_TOR": 0.486,  # [kA]
    "CURRENT_HEL": 0.900,  # [kA]
    "CONFIG_TOR": "default_toroidal",
    "CONFIG_HEL": "default_helical",
    "ENABLE_ERRFIELD": True,
    "RMAJOR": 0.72,  # [m]
    "RMINOR": 0.19,  # [m]
    "WALL_OFFSET_M": 0.001,  # [m]; zero is allowed
    "NPHI": 60,
    "NTHETA": 25,
    # Boris uses phi_wall = (-phi_comp + 180 + offset) % 360 degrees.
    "PHI_WALL_OFFSET_DEG": 18.0,
    "ARROW_MODE": "both",
    "ARROW_LENGTH_INCHES": 0.14,
    "COLORMAP": "viridis",
    "VMIN": None,  # [T]
    "VMAX": None,  # [T]
    "DPI": 250,
}
# Run sequentially with shared plotting settings and separate output folders.
DEFAULT_CASES = (
    {"ANALYSIS_NAME": "IOTA3_wall_quiver_1mm", "CURRENT_HEL": 0.900,
     "WALL_OFFSET_M": 0.001},
    {"ANALYSIS_NAME": "IOTA5_wall_quiver_1mm", "CURRENT_HEL": 0.710,
     "WALL_OFFSET_M": 0.001},
)
MAP_ASPECT = 0.25


def validate_inputs(params):
    name = params["ANALYSIS_NAME"]
    if (not isinstance(name, str) or not name.strip()
            or Path(name).name != name or name in {".", ".."}):
        raise ValueError("ANALYSIS_NAME must be one non-empty directory name.")
    for key in ("NPHI", "NTHETA", "DPI"):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{key} must be an integer >= 2.")
    for key in ("RMAJOR", "RMINOR", "ARROW_LENGTH_INCHES"):
        if not np.isfinite(params[key]) or params[key] <= 0:
            raise ValueError(f"{key} must be positive and finite.")
    if params["RMAJOR"] <= params["RMINOR"]:
        raise ValueError("RMAJOR must exceed RMINOR.")
    offset = params["WALL_OFFSET_M"]
    if not np.isfinite(offset) or not 0 <= offset < params["RMINOR"]:
        raise ValueError("WALL_OFFSET_M must satisfy 0 <= offset < RMINOR.")
    for key in ("CURRENT_TOR", "CURRENT_HEL", "PHI_WALL_OFFSET_DEG"):
        if not np.isfinite(params[key]):
            raise ValueError(f"{key} must be finite.")
    if params["ARROW_MODE"] not in {"both", "normalized", "magnitude"}:
        raise ValueError('ARROW_MODE must be "both", "normalized", or "magnitude".')
    for key in ("VMIN", "VMAX"):
        if params[key] is not None and (
                not np.isfinite(params[key]) or params[key] < 0):
            raise ValueError(f"{key} must be nonnegative and finite, or null.")
    if (params["VMIN"] is not None and params["VMAX"] is not None
            and params["VMIN"] >= params["VMAX"]):
        raise ValueError("VMIN must be smaller than VMAX.")


def project_field(points_rtp, field_xyz, rmajor):
    """Return physical components [B_phi_wall, B_theta, B_normal] in T."""
    theta, phi = points_rtp[:, 1], points_rtp[:, 2]
    bx, by, bz = field_xyz.T
    b_cyl_r = bx * np.cos(phi) - by * np.sin(phi)
    # phi_wall = -phi_comp + constant, so its tangent is -e_phi_comp.
    b_phi_wall = bx * np.sin(phi) + by * np.cos(phi)
    b_theta = -b_cyl_r * np.sin(theta) + bz * np.cos(theta)
    b_normal = b_cyl_r * np.cos(theta) + bz * np.sin(theta)
    components = np.column_stack((b_phi_wall, b_theta, b_normal))
    # dphi_wall/ds and dtheta/ds, apart from the common |B| divisor.
    map_vectors = np.column_stack((
        b_phi_wall / (rmajor + points_rtp[:, 0] * np.cos(theta)),
        b_theta / points_rtp[:, 0],
    ))
    return components, map_vectors


def arrow_vectors(map_vectors, tangent_strength, mode):
    """Preserve angular-map slopes while controlling displayed lengths."""
    screen = map_vectors * np.array([1.0, MAP_ASPECT])
    lengths = np.linalg.norm(screen, axis=1)
    directions = np.divide(screen, lengths[:, None], out=np.zeros_like(screen),
                           where=lengths[:, None] > 0)
    reference = float(np.max(tangent_strength))
    if mode == "magnitude":
        directions *= (tangent_strength / reference if reference > 0
                       else np.zeros_like(tangent_strength))[:, None]
    return directions, reference


def plot_quiver(phi, theta, components, map_vectors, params, sim_io, mode):
    strength = np.linalg.norm(components, axis=1)
    tangent = np.linalg.norm(components[:, :2], axis=1)
    arrows, reference = arrow_vectors(map_vectors, tangent, mode)
    fig, ax = plt.subplots(figsize=(16, 6))
    global_plotPorts(ax, sim_io)
    q = ax.quiver(
        phi.ravel(), theta.ravel(), arrows[:, 0], arrows[:, 1], strength,
        angles="uv", scale_units="inches",
        scale=1.0 / params["ARROW_LENGTH_INCHES"], pivot="mid",
        units="inches", width=0.012, minlength=0,
        cmap=params["COLORMAP"],
        norm=Normalize(vmin=params["VMIN"], vmax=params["VMAX"]),
    )
    ax.set_xlim(0, 360)
    ax.set_ylim(-180, 180)
    ax.set_aspect(MAP_ASPECT)
    ax.set_xticks(np.arange(0, 361, 36))
    ax.set_yticks(np.linspace(-180, 180, 5))
    ax.set_yticklabels(["Inner Midplane", "Bottom", "Outer Midplane", "Top",
                        "Inner Midplane"])
    ax.set_xlabel(r"$\phi$ ($^\circ$ CCW from South-side split)")
    ax.set_ylabel("Poloidal location")
    ax.set_title(
        f"Magnetic field near wall — {mode} arrows "
        f"($I_t={params['CURRENT_TOR']:g}$ kA, "
        f"$I_h={params['CURRENT_HEL']:g}$ kA; "
        f"offset {params['WALL_OFFSET_M'] * 1000:g} mm)"
    )
    ax.grid(linewidth=0.5, color="0.6")
    fig.colorbar(q, ax=ax, pad=0.02).set_label(r"Total $|B|$ [T]")
    if mode == "magnitude" and reference > 0:
        ax.quiverkey(q, 0.82, 1.20, 1, f"{reference:.3g} T tangent", labelpos="E")
    filename = f"magnetic_wall_quiver_{mode}.png"
    sim_io.saveFig(filename, dpi=params["DPI"])
    plt.close(fig)
    return Path(sim_io.plot_dir) / filename


def main(input_overrides=None):
    params = merge_input_params(DEFAULT_INPUTS, input_overrides)
    validate_inputs(params)
    points, phi, theta = make_initial_conditions(params)
    magnetic_field = build_magnetic_field(params)
    if points[0, 0] > magnetic_field.r_max:
        raise ValueError("Requested sampling radius exceeds the magnetic mesh.")
    field_xyz = np.array([
        magnetic_field.interpField(point.copy(), Cart=False)[0].reshape(3)
        for point in points
    ])
    if not np.all(np.isfinite(field_xyz)):
        raise ValueError("Magnetic field contains nonfinite samples.")
    components, map_vectors = project_field(points, field_xyz, params["RMAJOR"])
    (PROJECT_ROOT / "output" / "magnetic_wall_quiver").mkdir(parents=True, exist_ok=True)
    sim_io = IOHandler(f"magnetic_wall_quiver/{params['ANALYSIS_NAME']}")
    data_path = Path(sim_io.data_dir) / "magnetic_wall_quiver.npz"
    shape = (*phi.shape, 3)
    np.savez_compressed(
        data_path, phi_wall_deg=phi, theta_deg=theta,
        points_rtp=points.reshape(shape), B_xyz_T=field_xyz.reshape(shape),
        B_phi_wall_T=components[:, 0].reshape(phi.shape),
        B_theta_T=components[:, 1].reshape(phi.shape),
        B_normal_T=components[:, 2].reshape(phi.shape),
        inputs_json=json.dumps(params),
    )
    modes = ("normalized", "magnitude") if params["ARROW_MODE"] == "both" else (params["ARROW_MODE"],)
    for mode in modes:
        print(plot_quiver(phi, theta, components, map_vectors, params, sim_io, mode))
    print(data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-json", help="JSON overrides of DEFAULT_INPUTS.")
    args = parser.parse_args()
    if args.inputs_json:
        main(load_inputs_json(args.inputs_json, "Wall-quiver inputs"))
    else:
        for case in DEFAULT_CASES:
            print(f"Running {case['ANALYSIS_NAME']}", flush=True)
            main(case)
