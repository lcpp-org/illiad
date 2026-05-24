"""
#------------------------------------------------------#
# GENERATING POINCARE PLOTS FOR HIDRA'S MAGNETIC FIELD #
#------------------------------------------------------#
#        COIL CURRENTS NORMALLY RUN ON HIDRA           #
#------------------------------------------------------#
#  IOTA  |   I_T   |   I_H   |   I_V   |  PHI FWD/REV  #
#        |  [Amp]  |  [Amp]  |  [Amp]  |     [deg]     #
#  1/3   |   486   |   900   |    00   |    324/???    #
#  1/4   |   486   |   790   |    00   |    180/144    #
#  1/5   |   486   |   710   |    00   |    360/???    #
#  1/7   |   581   |   581   |    00   |    ???/???    #
#  MAX.  |  3500   |  7000   |    ??   |    ???/???    #
#------------------------------------------------------#
##  NTHREADS:
#    > 0: use N threads
#    = 0: use all available threads
#    < 0: use all but the last N threads
## DOUBLE_LINE:
#    True: run each fieldline in both directions from the init pos *!ONLY USE WHEN NTHREADS > NLINES!*
#    False: run each fieldline in +B direction from the init pos
"""
import numpy as np
import matplotlib.pyplot as plt
from classes.iohandler import IOHandler
from classes.mesh import Mesh
from classes.poincare import Poincare
import utility.phi_events as phi_event_defs
from utility.coordtrans import XYZ_to_RTP_many, axisShift

# DEFINE FIELDS #
CURRENT_TOR = 0.486 #[kA]
CURRENT_HEL = 0.790 #[kA]
CONFIG_TOR = "default_toroidal"
CONFIG_HEL = "default_helical"
ENABLE_ERRFIELD = True

# DEFINE INITIAL CONDITIONS #
IC_PHI_DEG = 180. #[deg]
IC_THETA_DEG = 180. #[deg]
START_RADIUS = 0.000 #[m]
END_RADIUS = 0.10 #[m]
NLINES = 11+10+20
NTHETA = 90
SPINS = 1
NPLANES = 180


# DEFINE SOLVER PARAMETERS #
SOLVER = "RK45"#"LSODA"#
RTOL = 2.49e-12
ATOL = 2.49e-8
NTHREADS = -1
DOUBLE_LINE = False
# DEFINE OUTPUT DIRECTORY #
OUTPUT_DIR = f"It-{CURRENT_TOR*1000:04.0f}_Ih-{CURRENT_HEL*1000:04.0f}_PHI{IC_PHI_DEG:03.0f}_{SPINS:0d}spins_{NLINES*NTHETA}Lines_{SOLVER}1e8_SINGLEIOTA"


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _extract_plane_points(poincare_points, plane_index):
    plane_points_xyz = np.full((len(poincare_points), 3), np.nan)
    for line_index, line_planes in enumerate(poincare_points):
        plane_hits = line_planes[plane_index]
        if isinstance(plane_hits, np.ndarray) and plane_hits.size:
            plane_points_xyz[line_index] = plane_hits[-1, :3]
    return plane_points_xyz


def _estimate_magnetic_axis(init_conds_rtp, plane_points_xyz, field, ntheta, nrings=3):
    valid_mask = np.isfinite(plane_points_xyz).all(axis=1)
    if not np.any(valid_mask):
        raise ValueError("Unable to estimate the magnetic axis without valid one-turn return points.")

    final_rtp = np.full_like(init_conds_rtp, np.nan)
    final_rtp[valid_mask] = XYZ_to_RTP_many(plane_points_xyz[valid_mask], field.R0)

    valid_indices = np.flatnonzero(valid_mask)
    start_rtp = init_conds_rtp[valid_indices]
    end_rtp = final_rtp[valid_indices]

    delta_r = end_rtp[:, 0] - start_rtp[:, 0]
    delta_theta = _wrap_to_pi(end_rtp[:, 1] - start_rtp[:, 1])

    start_x = start_rtp[:, 0] * np.cos(start_rtp[:, 1])
    start_z = start_rtp[:, 0] * np.sin(start_rtp[:, 1])
    end_x = end_rtp[:, 0] * np.cos(end_rtp[:, 1])
    end_z = end_rtp[:, 0] * np.sin(end_rtp[:, 1])
    poloidal_mismatch = np.hypot(end_x - start_x, end_z - start_z)

    best_local_index = int(np.argmin(poloidal_mismatch))
    best_index = int(valid_indices[best_local_index])

    axis_x = 0.5 * (start_x[best_local_index] + end_x[best_local_index])
    axis_z = 0.5 * (start_z[best_local_index] + end_z[best_local_index])
    axis_theta_geo = np.arctan2(axis_z, axis_x)
    if axis_theta_geo < 0.0:
        axis_theta_geo += 2.0 * np.pi
    axis_radius = np.hypot(axis_x, axis_z)

    return {
        'axis_thetar': np.array([axis_theta_geo, axis_radius]),
        'axis_index': best_index,
        'start_rtp': init_conds_rtp[best_index].copy(),
        'end_rtp': final_rtp[best_index].copy(),
        'delta_r': float(delta_r[best_local_index]),
        'delta_theta': float(delta_theta[best_local_index]),
        'poloidal_mismatch': float(poloidal_mismatch[best_local_index]),
    }


def calculate_single_spin_iota(init_conds_rtp, poincare_points, field, initial_phi, nplanes, spins, ntheta):
    plot_angles = np.linspace(2.0 * np.pi / nplanes, 2.0 * np.pi, nplanes)
    plane_index = int(np.argmin(np.abs(np.angle(np.exp(1j * (plot_angles - initial_phi))))))
    plane_points_xyz = _extract_plane_points(poincare_points, plane_index)
    valid_mask = np.isfinite(plane_points_xyz).all(axis=1)

    if not np.any(valid_mask):
        raise ValueError("No valid one-turn return points were found at the requested toroidal plane.")

    axis_info = _estimate_magnetic_axis(init_conds_rtp, plane_points_xyz, field, ntheta)
    axis_theta, axis_radius = axis_info['axis_thetar']

    final_rtp = np.full_like(init_conds_rtp, np.nan)
    final_rtp[valid_mask] = XYZ_to_RTP_many(plane_points_xyz[valid_mask], field.R0)

    initial_axis_coords = axisShift(init_conds_rtp[:, 1], init_conds_rtp[:, 0], axis_theta, axis_radius)
    final_axis_coords = np.full((2, len(init_conds_rtp)), np.nan)
    final_axis_coords[:, valid_mask] = axisShift(final_rtp[valid_mask, 1], final_rtp[valid_mask, 0], axis_theta, axis_radius)

    initial_theta_axis = initial_axis_coords[0]
    initial_radius_axis = initial_axis_coords[1]
    final_theta_axis = final_axis_coords[0]
    final_radius_axis = final_axis_coords[1]

    delta_theta = np.full(len(init_conds_rtp), np.nan)
    delta_theta[valid_mask] = _wrap_to_pi(final_theta_axis[valid_mask] - initial_theta_axis[valid_mask])
    iota = delta_theta / (2.0 * np.pi * spins)

    return {
        'plane_index': plane_index,
        'plane_phi': plot_angles[plane_index],
        'axis_thetar': np.array([axis_theta, axis_radius]),
        'axis_index': axis_info['axis_index'],
        'axis_start_rtp': axis_info['start_rtp'],
        'axis_end_rtp': axis_info['end_rtp'],
        'axis_delta_r': axis_info['delta_r'],
        'axis_delta_theta': axis_info['delta_theta'],
        'axis_poloidal_mismatch': axis_info['poloidal_mismatch'],
        'theta_initial_geo': init_conds_rtp[:, 1].copy(),
        'radius_initial_geo': init_conds_rtp[:, 0].copy(),
        'final_rtp': final_rtp,
        'theta_initial_axis': initial_theta_axis,
        'radius_initial_axis': initial_radius_axis,
        'theta_final_axis': final_theta_axis,
        'radius_final_axis': final_radius_axis,
        'delta_theta': delta_theta,
        'iota': iota,
        'valid_mask': valid_mask,
    }


def plot_iota_contours(iota_output, simIO, dpi=300, levels=24):
    valid_mask = (
        iota_output['valid_mask']
        & np.isfinite(iota_output['iota'])
        & np.isfinite(iota_output['theta_initial_geo'])
        & np.isfinite(iota_output['radius_initial_geo'])
    )

    if np.count_nonzero(valid_mask) < 3:
        raise ValueError("Need at least three valid iota samples to build a contour plot.")

    theta_geo = np.asarray(iota_output['theta_initial_geo'][valid_mask], dtype=np.float64)
    radius_geo = np.asarray(iota_output['radius_initial_geo'][valid_mask], dtype=np.float64)
    iota_vals = np.asarray(iota_output['iota'][valid_mask], dtype=np.float64)

    x_geo = radius_geo * np.cos(theta_geo)
    z_geo = radius_geo * np.sin(theta_geo)
    rounded_coords = np.round(np.column_stack((x_geo, z_geo)), decimals=12)
    unique_coords, inverse = np.unique(rounded_coords, axis=0, return_inverse=True)

    counts = np.bincount(inverse)
    iota_unique = np.bincount(inverse, weights=iota_vals) / counts
    x_unique = unique_coords[:, 0]
    z_unique = unique_coords[:, 1]
    radius_unique = np.hypot(x_unique, z_unique)
    theta_unique = np.arctan2(z_unique, x_unique)
    theta_unique = np.where(theta_unique < 0.0, theta_unique + 2.0 * np.pi, theta_unique)

    seam_buffer = np.deg2rad(12.0)
    low_mask = theta_unique < seam_buffer
    high_mask = theta_unique > (2.0 * np.pi - seam_buffer)
    theta_plot = np.concatenate((theta_unique, theta_unique[low_mask] + 2.0 * np.pi, theta_unique[high_mask] - 2.0 * np.pi))
    radius_plot = np.concatenate((radius_unique, radius_unique[low_mask], radius_unique[high_mask]))
    iota_plot = np.concatenate((iota_unique, iota_unique[low_mask], iota_unique[high_mask]))

    if theta_plot.size < 3:
        raise ValueError("Not enough unique iota samples remain after deduplicating the cross-section.")

    iota_min = float(np.nanmin(iota_unique))
    iota_max = float(np.nanmax(iota_unique))
    if np.isclose(iota_min, iota_max):
        contour_levels = np.linspace(iota_min - 1.0e-9, iota_max + 1.0e-9, 3)
    else:
        contour_levels = np.linspace(iota_min, iota_max, levels)

    phi_deg = np.degrees(iota_output['plane_phi'])

    plt.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    contour = ax.tricontourf(theta_plot, radius_plot, iota_plot, levels=contour_levels, cmap='viridis')
    ax.scatter(theta_unique, radius_unique, c='k', s=1.0, alpha=0.20, linewidths=0.0)
    ax.scatter([iota_output['axis_thetar'][0]], [iota_output['axis_thetar'][1]], c='white', marker='x', s=50, linewidths=0.75)

    ax.set_rmax(max(np.nanmax(radius_unique) * 1.05, 1.0e-6))
    ax.grid(linewidth=0.25, linestyle=':', c='k')
    ax.set_rgrids([], angle=0)
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], labels=['', '', '', '', '', '', '', ''], fontsize=12)
    ax.set_title(
        'One-turn iota, geometric cross-section\n'
        + '$\\phi_c$={:.0f}$\\degree$ | $\\iota_{{min}}$={:.5f}, $\\iota_{{max}}$={:.5f}'.format(phi_deg, iota_min, iota_max),
        loc='left',
    )

    cbar = fig.colorbar(contour, ax=ax, pad=0.10, shrink=0.85)
    cbar.set_label(r'$\iota$')

    plt.tight_layout()
    simIO.saveFig('iota_single_spin_contourf.png', dpi=dpi)
    plt.close(fig)

def main():
    """
    Main function to set up the mesh, load magnetic field data, and generate Poincare plots.
    """
    ## SET UP RUN DIRECTORY (*DATA AND PLOTS WILL BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!*)
    simIO = IOHandler(OUTPUT_DIR) 
    simIO.startLog()

    ## DEFINE MESH AND LOAD MAGNETIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField(coilCurrent=CURRENT_TOR, errField=ENABLE_ERRFIELD, att_mult=CONFIG_TOR)
    b_hidra.set_nonPer_errField()
    b_hidra.addFieldPerturbation(coilCurrent=CURRENT_HEL, att_mult=CONFIG_HEL)

    ## SET UP INITIAL CONDITIONS
    ic_radii = np.array(np.linspace(START_RADIUS, END_RADIUS, NLINES))
    # make init_conds_rtp a numpy array. A polar grid of initial radii and thetas at a fixed phi. The radii are linearly spaced from START_RADIUS to END_RADIUS, and the theta is 0-360 degrees. The phi is fixed at IC_PHI_DEG.
    theta_vals = np.linspace(0, 2*np.pi, NTHETA)
    ic_phi = IC_PHI_DEG * np.pi/180.
    init_conds_rtp = np.array([[r, theta, ic_phi] for r in ic_radii for theta in theta_vals])

    ## GENERATE POINCARE PLOTS
    solver_args = [SOLVER, RTOL, ATOL, NTHREADS, DOUBLE_LINE]
    PoinCare = Poincare(simIO, *solver_args)
    PoinCare.set_conditions(init_conds_rtp, SPINS, b_hidra, nplanes=NPLANES)

    poincare_points = PoinCare.run(plot_args={'title_on': True, 'dpi': 250})[1]


    iota_output = calculate_single_spin_iota(init_conds_rtp, poincare_points, b_hidra, 
                                             ic_phi, NPLANES, SPINS, len(theta_vals))

    axis_theta, axis_radius = iota_output['axis_thetar']
    simIO.log.info(
        'Estimated magnetic axis at phi={:.2f} deg: theta={:.2f} deg, r={:.6e} m'.format(
            np.degrees(iota_output['plane_phi']),
            np.degrees(axis_theta),
            axis_radius)
    )
    simIO.log.info(
        'Axis candidate IC {}: start(r={:.6e}, theta={:.2f} deg), end(r={:.6e}, theta={:.2f} deg), dr={:.3e}, dtheta={:.3e} rad, poloidal mismatch={:.3e} m'.format(
            iota_output['axis_index'],
            iota_output['axis_start_rtp'][0],
            np.degrees(iota_output['axis_start_rtp'][1]),
            iota_output['axis_end_rtp'][0],
            np.degrees(iota_output['axis_end_rtp'][1]),
            iota_output['axis_delta_r'],
            iota_output['axis_delta_theta'],
            iota_output['axis_poloidal_mismatch'],
        )
    )
    simIO.log.info(
        'Computed one-turn iota for {} of {} initial conditions.'.format(
            int(np.count_nonzero(iota_output['valid_mask'])),
            len(init_conds_rtp))
    )

    plot_iota_contours(iota_output, simIO, dpi=300)

    axis_data = np.array([[iota_output['plane_index'], iota_output['plane_phi'], axis_theta, axis_radius]])
    # simIO.saveCSV(axis_data, 'magnetic_axis_single_spin.csv',
    #                 header='plane_index,plane_phi_rad,axis_theta_rad,axis_r_m')

    iota_data = np.column_stack((
        np.arange(len(init_conds_rtp), dtype=float),
        init_conds_rtp,
        iota_output['final_rtp'],
        iota_output['theta_initial_axis'],
        iota_output['theta_final_axis'],
        iota_output['delta_theta'],
        iota_output['iota'],
    ))
    # simIO.saveCSV(
    #     iota_data,
    #     'iota_single_spin.csv',
    #     header='line_index,init_r_m,init_theta_rad,init_phi_rad,final_r_m,final_theta_rad,final_phi_rad,theta0_about_axis_rad,theta1_about_axis_rad,delta_theta_rad,iota',
    # )


    ## END RUN ##
    simIO.log.info('## SIM FINISHED ##\n\n\n\n')

if __name__ == '__main__': main()