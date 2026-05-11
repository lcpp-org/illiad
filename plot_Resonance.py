from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classes.iohandler import IOHandler
from classes.mesh import Mesh


REPO_ROOT = Path(__file__).resolve().parent

FIELD_FILE_TOR = "input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy"
FIELD_FILE_HEL = "input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy"

CURRENT_TOR = 0.581 # [kA]
CURRENT_HEL = 0.581 #0.790 # [kA]
CONFIG_TOR = "ideal_toroidal" #"default_toroidal"
CONFIG_HEL = "ideal_poloidal" #"default_poloidal"
ENABLE_ERRFIELD = False

RESONANCE_B_T = 0.0875 # [Tela] ECRH resonance for 2.45 GHz

#ANLYS_DIR = "AcceptedIota4_1500spins_atole-8_eng"
#ANLYS_DIR = "AcceptedIota3_1500spins_atole-9"
ANLYS_DIR = "It-0486_Ih-0900_noErr_1500sp_LSODA1e8"
ANLYS_DIR = "Resonance_test"
ANLYS_SUBDIR = "Resonance_test2"

PHI_COUNT = 40
PHI_START_DEG = 9.0
CONTOUR_LEVELS = 20

PLOT_POINCARE = True
POINCARE_ANLYS_NAME = "Poincare"
POINCARE_MARKER_SIZE = 0.5
POINCARE_MARKER_COLOR = "k"


def contour_levels(data, nlevels):
    data_min = float(np.nanmin(data))
    data_max = float(np.nanmax(data))
    if np.isclose(data_min, data_max):
        pad = max(abs(data_min) * 0.01, 1.0e-6)
        data_min -= pad
        data_max += pad
    return np.linspace(data_min, data_max, nlevels)


def load_poincare_data(simIO, phi_deg):
    fname = POINCARE_ANLYS_NAME + '_{:03.0f}.npy'.format(phi_deg)
    try:
        radtheta_pts = simIO.loadNumpyData(fname)
    except FileNotFoundError:
        simIO.log.warning(f"Poincare data not found for phi={phi_deg:.0f} deg: {fname}")
        return None, None

    num_sets = radtheta_pts.shape[0]
    point_total = np.zeros(num_sets, dtype=int)
    for i in range(num_sets):
        these_radtheta_pts = radtheta_pts[i]
        point_total[i] = np.sum(~np.isnan(these_radtheta_pts).all(axis=0))

    return radtheta_pts, point_total


def overlay_poincare_points(ax, radtheta_pts, point_total):
    if radtheta_pts is None or point_total is None:
        return

    num_sets = len(radtheta_pts)
    for i in range(num_sets):
        ax.scatter(radtheta_pts[i][0][:point_total[i]], radtheta_pts[i][1][:point_total[i]],
                    marker='.', s=POINCARE_MARKER_SIZE, c=POINCARE_MARKER_COLOR,
                    linewidths=0.0, zorder=5)

 
def plot_resonance_xsections(simIO, b_hidra, b_magnitude,
    r_grid, theta_grid, phi_grid, toroidal_current, helical_current, levels):
    
    rr, tt = np.meshgrid(r_grid, theta_grid)
    mesh_dtheta = b_hidra.dtheta * 2.0

    wrapped_theta = np.concatenate((tt, tt[-1:] + mesh_dtheta))
    wrapped_radius = np.concatenate((rr, rr[-1:]))
    filled_levels = contour_levels(b_magnitude, levels)

    for k, phi_comp in enumerate(phi_grid):
        plot_data = b_magnitude[:, :, k].T
        wrapped_data = np.concatenate((plot_data, plot_data[0:1, :]), axis=0)
        loc_min = float(np.nanmin(plot_data))
        loc_max = float(np.nanmax(plot_data))

        fig = plt.figure(figsize=(6.0, 5.2))
        ax = fig.add_subplot(111, polar=True)

        contourf = ax.contourf(wrapped_theta.T, wrapped_radius.T, wrapped_data.T,
            levels=filled_levels, cmap="viridis", extend="both")

        if loc_min <= RESONANCE_B_T <= loc_max:
            resonance_contour = ax.contour(wrapped_theta.T, wrapped_radius.T, wrapped_data.T,
                levels=[RESONANCE_B_T], colors="purple", linewidths=1.1)
            ax.clabel(resonance_contour, fmt={RESONANCE_B_T: f"{RESONANCE_B_T:.4f} T"}, fontsize=8)
        else:
            print(f"No {RESONANCE_B_T:.4f} T contour at phi_c={np.degrees(phi_comp):.0f} deg (slice range: {loc_min:.4f} T to {loc_max:.4f} T).")

        if PLOT_POINCARE:
            phi_deg = np.degrees(phi_comp)
            radtheta_pts, point_total = load_poincare_data(simIO, phi_deg)
            overlay_poincare_points(ax, radtheta_pts, point_total)

        ax.set_rmax(b_hidra.r_max)
        ax.set_rticks(np.arange(0.0, b_hidra.r_max, 0.02))
        ax.yaxis.set_tick_params(labelsize=5)
        ax.grid(linewidth=0.25, linestyle=":", color="k")

        cbar = fig.colorbar(contourf, ax=ax, pad=0.12)
        cbar.set_label(r"$|B|$ [T]")

        phi_phys = (phi_comp + np.radians(198.0)) % (2.0 * np.pi)
        title_text = (f"$\\phi_{{phy}}$={np.degrees(phi_phys):02.0f}$\\degree$ CW from North Split ($\\phi_c$={np.degrees(phi_comp):02.0f}$\\degree)$\n"
            f"$I_t$={toroidal_current * 1000.0:4.0f} A, $I_h$={helical_current * 1000.0:4.0f} A")
        ax.set_title(title_text, loc="left", fontsize=8)

        fig.tight_layout()
        plot_name = f"Bmag_resonance_phi={np.degrees(phi_comp):02.0f}.png"
        simIO.saveFig(f"{ANLYS_SUBDIR}/{plot_name}")
        plt.close(fig)


def plotResonance(input_params=None):
    ## LOAD INPUT PARAMETERS
    if input_params is not None:
        print(f'{input_params.keys()=}')
        for key, value in input_params.items():
            print(f'{key}: {value}')
            globals()[str(key)] = value

    if PHI_COUNT < 1:
        raise ValueError("PHI_COUNT must be at least 1")
    if CONTOUR_LEVELS < 2:
        raise ValueError("CONTOUR_LEVELS must be at least 2")

    ## DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
    simIO = IOHandler(ANLYS_DIR)
    simIO.startLog()
    simIO.createSubDir(ANLYS_SUBDIR)
    simIO.log.info(f"Toroidal current [kA]: {CURRENT_TOR}, Helical current [kA]: {CURRENT_HEL}, Resonance contour [T]: {RESONANCE_B_T}")
    simIO.log.info(f"Analysis directory: {ANLYS_DIR}, Analysis subdirectory: {ANLYS_SUBDIR}")

    ## DEFINE MESH AND LOAD MAGNETIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField(file_path=REPO_ROOT / FIELD_FILE_TOR, coilCurrent=CURRENT_TOR, errField=ENABLE_ERRFIELD, att_mult=CONFIG_TOR)
    b_hidra.set_nonPer_errField()
    b_hidra.addFieldPerturbation(file_path=REPO_ROOT / FIELD_FILE_HEL, coilCurrent=CURRENT_HEL, att_mult=CONFIG_HEL)

    mesh_nr = int(b_hidra.nr // 2 + 1)
    mesh_ntheta = int(b_hidra.ntheta / 2)
    r_grid = np.linspace(b_hidra.r_min * 2, b_hidra.r_max, mesh_nr)
    theta_grid = np.linspace(b_hidra.theta_min, b_hidra.theta_max, mesh_ntheta)
    phi_grid = np.linspace(np.radians(PHI_START_DEG), 2.0 * np.pi, PHI_COUNT)


    print("Calculating overall B-field magnitude...")
    mesh_size = (r_grid.size, theta_grid.size, phi_grid.size)
    b_magnitude = np.zeros(mesh_size)
    for j, theta in enumerate(theta_grid):
        for k, phi in enumerate(phi_grid):
            for i, r in enumerate(r_grid):
                bxyz, _ = b_hidra.interpField(np.asarray([r, theta, phi]), Cart=False)
                b_magnitude[i, j, k] = np.linalg.norm(bxyz)
    print("Fields calculated.")


    plot_resonance_xsections(simIO, b_hidra, b_magnitude,
        r_grid, theta_grid, phi_grid,
        CURRENT_TOR, CURRENT_HEL, CONTOUR_LEVELS)
    
    print(f"Saved plots in {Path(simIO.plot_dir) / ANLYS_SUBDIR}")



if __name__ == "__main__":

    plotResonance()
