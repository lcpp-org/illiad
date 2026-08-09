"""Prototype Boris run with a flat, wall-adjacent lithium source.

The script leaves the ILLIAD package unchanged. It injects a planar ion
initializer into ``illiad.cli.boris`` for this process only, then calls the
normal Boris runner. At zero rotation, the plane width follows local +theta and
its height follows local +phi. Rotation rolls the rectangle within that tangent
plane; the common launch normal always points radially inward.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

import illiad.cli.boris as boris_cli
import illiad.plotting as illiad_plotting
from illiad.particle import Ion
from illiad.utilities.coordtrans import RTP_XYZ_JAC, RTP_to_XYZ, XYZ_to_RTP_many
from illiad.utilities.run_config import load_inputs_json, merge_input_params


DEFAULT_INPUTS_JSON = Path(__file__).with_name("boris_planar_source_inputs.json")


class VesselGeometry:
    R0 = 0.72
    a = 0.19


def _positive_int(value, name):
    value_float = float(value)
    value_int = int(value_float)
    if value_float != value_int or value_int < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value_int


def generate_planar_emitters(params, bfield):
    """Return flat N x N emitter positions and their common inward normal."""
    width = float(params["PLANE_WIDTH_M"])
    height = float(params["PLANE_HEIGHT_M"])
    resolution = _positive_int(params["PLANE_RESOLUTION"], "PLANE_RESOLUTION")
    wall_offset = float(params["PLANE_WALL_OFFSET_M"])
    if width <= 0.0 or height <= 0.0:
        raise ValueError("PLANE_WIDTH_M and PLANE_HEIGHT_M must be positive")
    if not 0.0 < wall_offset < bfield.a:
        raise ValueError(f"PLANE_WALL_OFFSET_M must be between 0 and {bfield.a} m")

    theta = np.deg2rad(float(params["PLANE_THETA_DEG"]))
    phi = np.deg2rad(float(params["PLANE_PHI_DEG"]))
    rotation = np.deg2rad(float(params["PLANE_ROTATION_DEG"]))
    center_rtp = np.array([bfield.a - wall_offset, theta, phi])
    center_xyz = RTP_to_XYZ(center_rtp, bfield.R0)

    outward = RTP_XYZ_JAC(center_rtp, np.array([1.0, 0.0, 0.0]), form="rtp2xyz")
    poloidal = RTP_XYZ_JAC(center_rtp, np.array([0.0, 1.0, 0.0]), form="rtp2xyz")
    toroidal = RTP_XYZ_JAC(center_rtp, np.array([0.0, 0.0, 1.0]), form="rtp2xyz")
    width_axis = np.cos(rotation) * poloidal + np.sin(rotation) * toroidal
    height_axis = -np.sin(rotation) * poloidal + np.cos(rotation) * toroidal
    inward = -outward / np.linalg.norm(outward)

    width_coords = np.array([0.0]) if resolution == 1 else np.linspace(-width / 2, width / 2, resolution)
    height_coords = np.array([0.0]) if resolution == 1 else np.linspace(-height / 2, height / 2, resolution)
    width_grid, height_grid = np.meshgrid(width_coords, height_coords)
    emitters = (
        center_xyz
        + width_grid[..., None] * width_axis
        + height_grid[..., None] * height_axis
    ).reshape(-1, 3)

    radii = XYZ_to_RTP_many(emitters, bfield.R0)[:, 0]
    if np.max(radii) >= bfield.a:
        raise ValueError(
            f"Planar source reaches r={np.max(radii):.6f} m, outside the "
            f"r<{bfield.a:.6f} m vessel. Increase PLANE_WALL_OFFSET_M or "
            "reduce the plane dimensions."
        )
    return emitters, inward, radii


def planar_outline_plot_coordinates(params):
    """Project the four plane corners into the wall-histogram coordinates."""
    corner_params = dict(params)
    corner_params["PLANE_RESOLUTION"] = 2
    corners, _, _ = generate_planar_emitters(corner_params, VesselGeometry())
    grid = corners.reshape(2, 2, 3)
    corners = np.array([grid[0, 0], grid[0, 1], grid[1, 1], grid[1, 0], grid[0, 0]])
    corner_rtp = XYZ_to_RTP_many(corners, VesselGeometry.R0)

    theta = corner_rtp[:, 1]
    theta[theta > np.pi] -= 2.0 * np.pi
    theta_deg = np.rad2deg(theta)

    phi_plot = -corner_rtp[:, 2] + 2.0 * np.pi
    phi_deg = (np.rad2deg(phi_plot) + 180.0 + 18.0) % 360.0
    phi_deg = np.rad2deg(np.unwrap(np.deg2rad(phi_deg)))
    return phi_deg, theta_deg


def boris_plotWallHist_with_emitter(wallPtArray, runString, simIO, cond_string,
                                    emitter_outline):
    """Copy of ``boris_plotWallHist`` with the planar-source outline added."""
    simIO.log.info('Plotting wall hits with emitter outline, total events = {}...'.format(wallPtArray[0].size))

    # cond string decoder
    parts = cond_string.split('_')
    dr_mm = parts[0]
    LCFS_index = parts[1][4:]  # Remove 'LCFS' prefix
    ion_temp_eV = parts[2][:-2]  # Remove 'eV' suffix
    electric_field_V = parts[3][:-1]  # Remove 'V' suffix
    charge_num_Z = parts[4][1:]  # Remove 'Z' prefix

    phi_plot = wallPtArray[2]*(-1) + 2*np.pi # convert to phi= +CCW (viewing from outside VV)
    theta_plot = wallPtArray[1]
    theta_plot[theta_plot>np.pi] -= 2*np.pi #shift so that (theta=0) is centered in the plot

    # shift to physical phi=0 at at the South-side split, convert to deg.
    a_phi = 18. # (deg), phi_comp 18 CW from south-split
    phi_plot_deg = (phi_plot*(180/np.pi) + 180. + a_phi) % 360.
    theta_plot_deg = theta_plot*(180/np.pi)

    # define bin edges for 2d histogram
    phi_edges = np.linspace(0, 360, 361)
    theta_edges = np.linspace(-180, 180, 181)
    H, phi_edges, theta_edges = np.histogram2d(phi_plot_deg, theta_plot_deg, bins=[phi_edges, theta_edges], density=True)
    H = H.T # histogram reverse axes for some reason; transpose

    ## PLOT HISTOGRAM
    plt.rcParams.update({'font.size': 8})
    w, h = plt.figaspect(0.40)
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(polar=False, aspect=0.2)

    plt.grid(which='both', linewidth=0.5)
    illiad_plotting.global_plotPorts(ax, simIO)

    plt.imshow( H, interpolation='nearest', origin='lower',
                extent=[phi_edges[0], phi_edges[-1], theta_edges[0], theta_edges[-1]],
                #cmap=plt.get_cmap('Blues', 6), norm=colors.LogNorm(vmin=1E-6, vmax=1E-3),
                cmap=plt.get_cmap('Blues', 6), norm=colors.LogNorm(vmin=None, vmax=None),
                aspect=0.2 )

    #cbar = plt.colorbar(boundaries=levels, location='top', shrink=0.6)
    cbar = plt.colorbar(location='top', shrink=0.6)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('$\\hat{\\Gamma}_{depo}=\\frac{N_{depo}}{N_{total}}$', fontsize=12)

    ax.set_xlabel(r'$\phi~\mathit{(\degree CCW~from~South\text{-}Split)}$', fontsize=14)
    ax.set_ylabel('Poloidal Location', fontsize=14)
    ax.set_xlim(0, 360)
    ax.set_ylim(-180, 180)

    phi_spacing = 18. # degrees
    xticks = np.arange(phi_spacing, 361-phi_spacing, phi_spacing)
    ax.set_xticks(xticks)
    ax.set_xticklabels([fr'{int(tick)}$\degree$' if i % 2 != 0 else '' for i, tick in enumerate(xticks)])
    ax.xaxis.set_tick_params(labelsize=12)

    ax.set_yticks(np.linspace(-180, 180, 5))
    ax.set_yticklabels(['', 'Bottom', 'Outer', 'Top', ''])
    ax.yaxis.set_tick_params(labelsize=12, labelrotation=0)

    #ax.text(0.995, 0.975, f'$\\mathrm{{{ion_temp_eV}eV, {electric_field_V}V, Z{charge_num_Z}}}$',
    #ax.text(0.9955, 0.9755, f'$\\mathbf{{ T_i = {ion_temp_eV}eV}}$',
    ax.text(0.9945, 0.974, f'$\\mathbf{{ T_i = {ion_temp_eV}eV}}$',
    transform=ax.transAxes,
    ha='right', va='top',
    fontsize=14,
    bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.9))

    emitter_phi, emitter_theta = emitter_outline
    for phi_shift in (-360.0, 0.0, 360.0):
        ax.plot(
            emitter_phi + phi_shift,
            emitter_theta,
            color='#FF5F05',
            linewidth=2.0,
            zorder=5,
            label='Planar emitter' if phi_shift == 0.0 else None,
        )
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plotname = 'Wall_Histogram_withEmitter.png'
    simIO.saveFig(plotname, dpi=600)
    simIO.log.info('OUTPUT PLOT: {}'.format(plotname))
    plt.close()


def make_wall_histogram_pair(params):
    """Return a plotting hook that writes the regular and outlined histograms."""
    original_plotter = illiad_plotting.boris_plotWallHist
    emitter_outline = planar_outline_plot_coordinates(params)

    def plot_pair(wallPtArray, runString, simIO, cond_string):
        original_plotter(wallPtArray, runString, simIO, cond_string)
        boris_plotWallHist_with_emitter(
            wallPtArray.copy(), runString, simIO, cond_string, emitter_outline
        )

    return plot_pair


def generate_planar_velocities(n_particles, normal, width_axis, temperature_ev, mass_amu):
    """Sample Maxwellian speeds over a cosine-weighted inward hemisphere."""
    joules_per_ev = 1.602_176_634e-19
    kg_per_amu = 1.660_539_068e-27
    sigma = np.sqrt(joules_per_ev * temperature_ev / (mass_amu * kg_per_amu))
    speeds = sigma * np.sqrt(np.random.chisquare(3, n_particles))

    mu = np.sqrt(np.random.uniform(0.0, 1.0, n_particles))
    azimuth = np.random.uniform(0.0, 2.0 * np.pi, n_particles)
    tangent_radius = np.sqrt(1.0 - mu * mu)
    width_axis = width_axis / np.linalg.norm(width_axis)
    height_axis = np.cross(normal, width_axis)
    directions = (
        tangent_radius[:, None] * np.cos(azimuth)[:, None] * width_axis
        + tangent_radius[:, None] * np.sin(azimuth)[:, None] * height_axis
        + mu[:, None] * normal
    )
    return speeds[:, None] * directions


def make_planar_initializer(params):
    """Build an initializer matching the existing ``ionInitializer`` call."""
    def planar_initializer(initial_conditions, ion_properties, bfield, efield,
                           outputHandler="simIO"):
        del efield
        mass, charge, temperature = ion_properties
        nparticles_per_emitter = int(initial_conditions[-1])
        emitters, inward, radii = generate_planar_emitters(params, bfield)
        n_emitters = len(emitters)
        n_particles = n_emitters * nparticles_per_emitter

        center_rtp = np.array([
            bfield.a - float(params["PLANE_WALL_OFFSET_M"]),
            np.deg2rad(float(params["PLANE_THETA_DEG"])),
            np.deg2rad(float(params["PLANE_PHI_DEG"])),
        ])
        poloidal = RTP_XYZ_JAC(center_rtp, np.array([0.0, 1.0, 0.0]), form="rtp2xyz")
        toroidal = RTP_XYZ_JAC(center_rtp, np.array([0.0, 0.0, 1.0]), form="rtp2xyz")
        rotation = np.deg2rad(float(params["PLANE_ROTATION_DEG"]))
        width_axis = np.cos(rotation) * poloidal + np.sin(rotation) * toroidal
        velocities = generate_planar_velocities(
            n_particles, inward, width_axis, float(temperature), float(mass)
        )

        # Match the current runner's emitter-major ordering.
        repeated_emitters = np.repeat(emitters, nparticles_per_emitter, axis=0)
        initial_normals = np.repeat(inward[None, :], n_particles, axis=0)
        ions = [Ion(position, mass, charge) for position in repeated_emitters]
        for ion, velocity in zip(ions, velocities):
            ion.initVelocity(velocity)

        init_vel_pos = np.column_stack((velocities, repeated_emitters))
        outputHandler.saveNumpyData(emitters, "PlanarIonSeedPts")
        outputHandler.saveNumpyData(
            np.repeat(inward[None, :], n_emitters, axis=0),
            "PlanarIonSeedNormals",
        )
        outputHandler.log.info(
            "PLANAR SOURCE: %dx%d emitters, %.4f x %.4f m, "
            "theta=%.3f deg, phi=%.3f deg, offset=%.4f m, rotation=%.3f deg",
            int(params["PLANE_RESOLUTION"]), int(params["PLANE_RESOLUTION"]),
            float(params["PLANE_WIDTH_M"]), float(params["PLANE_HEIGHT_M"]),
            float(params["PLANE_THETA_DEG"]), float(params["PLANE_PHI_DEG"]),
            float(params["PLANE_WALL_OFFSET_M"]), float(params["PLANE_ROTATION_DEG"]),
        )
        outputHandler.log.info(
            "PLANAR SOURCE RADIAL RANGE: %.6f to %.6f m", np.min(radii), np.max(radii)
        )
        return ions, init_vel_pos, initial_normals

    return planar_initializer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-json", default=DEFAULT_INPUTS_JSON)
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = load_inputs_json(args.inputs_json, "Planar Boris inputs")
    params = merge_input_params(boris_cli.DEFAULT_INPUTS, overrides)
    resolution = _positive_int(params["PLANE_RESOLUTION"], "PLANE_RESOLUTION")

    # Reuse the runner's existing counting and tracker logic for this N x N grid.
    params["NPHI"] = resolution
    params["NTHETA"] = resolution
    params["DELTRS"] = [float(params["PLANE_WALL_OFFSET_M"])]
    params["LCFS_INDEX"] = 0  # Placeholder; no Poincare/LCFS data are loaded.
    params["TRACK_NPHI"] = min(int(params["TRACK_NPHI"]), resolution)
    params["TRACK_NTHETA"] = min(int(params["TRACK_NTHETA"]), resolution)
    params["TAG"] = "planar_" + str(params["TAG"])

    boris_cli.ionInitializer = make_planar_initializer(params)
    illiad_plotting.boris_plotWallHist = make_wall_histogram_pair(params)
    boris_cli.boris_runner(params)


if __name__ == "__main__":
    main()
