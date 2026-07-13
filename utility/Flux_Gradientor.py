## IN THIS FILE, WE WILL TAKE THE CALCULATED FLUX FOR EACH SURFACE,
## APPLY IT TO THE POINTS ON THE THEIR RESPECTIVE SURFACE, 
## AND THEN INTERPOLATE IT ONTO A MESH THE SAME SHAPE/RESOLUTION AS THE BFIELD MESH
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from classes.mesh import *
from classes.iohandler import IOHandler
from utility.coordtrans import RTP_XYZ_JAC
from utility.gradient_utils import scalar_gradient_periodic_angles

def fluxGradientor(input_params=None):
    ## LOAD INPUT PARAMETERS
    if input_params is not None:
        print(f'{input_params.keys()=}')
        for key, value in input_params.items():
            print(f'{key}: {value}')
            globals()[str(key)] = value

    ## DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
    simIO = IOHandler(ANLYS_DIR)
    simIO.setActiveSubDir(ANLYS_SUBDIR)
    simIO.startLog(log_name="fluxGradientor.log", subdir=ANLYS_SUBDIR, logger_name="FluxGradientor")
    simIO.inputsBoilerplate(
        "FLUX GRADIENTOR INPUTS",
        globals(),
        [
            "ANLYS_DIR",
            "ANLYS_SUBDIR",
            "TAG",
            "CURRENT_TOR",
            "CURRENT_HEL",
            "CONFIG_TOR",
            "CONFIG_HEL",
            "ENABLE_ERRFIELD",
            "LCFS_INDEX",
            "PHI_GENs",
            "OUTPUT_FILE_NAME",
        ],
    )
    ## DEFINE MESH AND LOAD MAGNETIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField(coilCurrent=CURRENT_TOR, errField=ENABLE_ERRFIELD, att_mult=CONFIG_TOR)
    b_hidra.addFieldPerturbation(coilCurrent=CURRENT_HEL, att_mult=CONFIG_HEL)
    b_hidra.set_nonPer_errField()

    # load density data from flux inteprolator step
    #density_grid = simIO.loadNumpyData(ANLYS_SUBDIR + '/density_field.npy') #'/big_grid_linear.npy')
    density_grid = simIO.loadNumpyData(ANLYS_SUBDIR + '/' + 'nField_' + OUTPUT_FILE_NAME + '.npy') #'/big_grid_linear.npy')

    # Create a meshgrid for the interpolation
    RADS = np.linspace(b_hidra.r_min, b_hidra.r_max, b_hidra.nr)
    THETAS = np.linspace(b_hidra.theta_min, b_hidra.theta_max, b_hidra.ntheta)
    grid_theta, grid_rad = np.meshgrid(THETAS, RADS, indexing='ij')

    # GRADIENT CALCULATION: remember to divide by Jacobian determinant:
    #  gradF = [dF/dr] * R_HAT + [(1/r) * df/dtheta] * THETA_HAT + [( 1/(R0+rcos(theta)) ) * df/dphi] * PHI_HAT
    simIO.log.info("## Starting flux gradient calculation. ##")
    dp_dphi, dp_dtheta, dp_drho = scalar_gradient_periodic_angles(
        density_grid, PHI_GENs, THETAS, RADS
    )
    simIO.log.info("## Flux gradient calculation complete. ##")

    # Calculate RTP basis vectors and apply the Jacobian factors to get the physical gradient in each direction
    flux_gradient_radial = -dp_drho  # E = -grad[V]

    flux_gradient_poloidal = np.zeros_like(dp_dtheta)
    flux_gradient_poloidal[:,:,1:] = -dp_dtheta[:,:,1:] / grid_rad[:,1:]

    flux_gradient_toroidal = -dp_dphi / (b_hidra.R0 + grid_rad * np.cos(grid_theta))
    simIO.log.info(f'{flux_gradient_radial.shape=}')

    # # Load LCFS file:
    # lcfs_filename = ANLYS_SUBDIR + '/fSurf_{:03d}_POINTmesh.npy'.format(int(LCFS_INDEX))
    # lcfs_points_full = simIO.loadNumpyData(lcfs_filename)

    # The exterior decay in Flux_Interpolator makes the old LCFS masking unnecessary.
    # for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):
    #     filename = 'Poincare_{:03d}.npy'.format(int(PHI_GEN_DEG))
    #     lcfs_points = simIO.loadNumpyData(filename)[LCFS_INDEX]
    #     th_in, r_in = lcfs_points
    #     r_in = r_in[~np.isnan(r_in)]
    #     th_in = th_in[~np.isnan(th_in)]
    #
    #     for theta_index, this_theta in enumerate(THETAS):
    #         mintheta1 = np.abs(th_in - this_theta)
    #         mintheta2 = np.abs(th_in - this_theta + 2*np.pi)
    #         mintheta3 = np.fmin(mintheta1, mintheta2)
    #         lcfs_theta_index = np.argmin(mintheta3)
    #         lcfs_rad = r_in[lcfs_theta_index]
    #         mask = RADS > (lcfs_rad + 0.01)
    #         flux_gradient_radial[phi_index][theta_index][mask] = 0.0
    #         flux_gradient_poloidal[phi_index][theta_index][mask] = 0.0
    #         flux_gradient_toroidal[phi_index][theta_index][mask] = 0.0

    flux_gradient_magnitude = np.sqrt(flux_gradient_radial**2 + flux_gradient_poloidal**2 + flux_gradient_toroidal**2)

    # RESHAPE THE ARRAYS TO MATCH THE DIMNENSIONS OF INPUT BFIELDS
    reshaped_flux_gradient_radial = np.transpose(flux_gradient_radial, (2, 1, 0))
    reshaped_flux_gradient_poloidal = np.transpose(flux_gradient_poloidal, (2, 1, 0))
    reshaped_flux_gradient_toroidal = np.transpose(flux_gradient_toroidal, (2, 1, 0))
    Efield_rtpArray_linear = np.array([reshaped_flux_gradient_radial, reshaped_flux_gradient_poloidal, reshaped_flux_gradient_toroidal])

    # save the array using simIO method
    # simIO.saveNumpyData(Efield_rtpArray_linear, ANLYS_SUBDIR + '/Efield_rtpArray_linear.npy')

    Efield_xyzArray_linear = np.zeros_like(Efield_rtpArray_linear)
    xform_rad, xform_theta, xform_phi= np.meshgrid(RADS, THETAS, PHI_GENs, indexing='ij')

    flattened_shape = Efield_xyzArray_linear[0].flatten().shape
    Ex_linear = np.zeros(flattened_shape)
    Ey_linear = np.zeros(flattened_shape)
    Ez_linear = np.zeros(flattened_shape)
    for i, (rad, theta, phi, EradLin, EthetaLin, EphiLin) in enumerate(zip(xform_rad.flatten(),
                                                                           xform_theta.flatten(),
                                                                           xform_phi.flatten(),
                                                                           reshaped_flux_gradient_radial.flatten(),
                                                                           reshaped_flux_gradient_poloidal.flatten(), 
                                                                           reshaped_flux_gradient_toroidal.flatten())):
        ErtpLin = np.array([EradLin, EthetaLin, EphiLin])
        p_RTP = np.array([rad, theta, np.radians(phi)])
        Ex_linear[i], Ey_linear[i], Ez_linear[i] = RTP_XYZ_JAC(p_RTP, ErtpLin, form='rtp2xyz')

    # reshape the arrays to match the dimensions of input Bfields
    Efield_xyzArray_linear[0] = Ex_linear.reshape(Efield_xyzArray_linear[0].shape)
    Efield_xyzArray_linear[1] = Ey_linear.reshape(Efield_xyzArray_linear[1].shape)
    Efield_xyzArray_linear[2] = Ez_linear.reshape(Efield_xyzArray_linear[2].shape)
    simIO.saveNumpyData(Efield_xyzArray_linear, ANLYS_SUBDIR + '/' + 'Efield_' + OUTPUT_FILE_NAME + '.npy')

    ## LOOP THROUGH PHI ANGLES forplotting
    #colortest = 'seismic'
    colortest = 'afmhot_r'
    for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):
        #output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_grid_linear[phi_index], 'FluxNorm', ANLYS_SUBDIR, simIO, 'inferno', 0.0, 1.0)
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, flux_gradient_magnitude[phi_index], 'FluxGradMagnitude', ANLYS_SUBDIR, simIO, colortest, 0.0, 200.0)
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, flux_gradient_radial[phi_index], 'FluxGradRadial', ANLYS_SUBDIR, simIO, colortest, -200., 200)
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, flux_gradient_poloidal[phi_index], 'FluxGradPoloidal', ANLYS_SUBDIR, simIO, colortest, -100.0, 100.0)
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, flux_gradient_toroidal[phi_index], 'FluxGradToroidal', ANLYS_SUBDIR, simIO, colortest, -3.0, 3.0)

    simIO.log.info("## Flux gradienting complete. ##")


def output_phi_plots(phi_deg, grid_theta, grid_rad, data, name, subdir, output_handler, colormap='inferno', plotmin=None, plotmax=None):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    data = np.vstack((data[-1], data))
    grid_rad = np.vstack((grid_rad[-1], grid_rad))
    grid_theta = np.vstack((grid_theta[-1], grid_theta))
    grid_theta[0] = 0

    c = ax.pcolormesh(grid_theta, grid_rad, data, shading='gouraud', cmap=colormap, vmin=plotmin, vmax=plotmax)

    ax.set_title(name + '\n$\phi_{{phy}}$={:02.0f}$\degree$ CW from North Split\n$\phi_c$={:02.0f}$\degree$'.format((phi_deg+198.)%360., phi_deg), loc='left')
    ax.set_rmax(0.19)
    ax.set_rticks([])
    # set the r-labels to an empty list
    ax.set_yticklabels([])
    fig.colorbar(c, ax=ax, label='Flux')
    #plt.grid(True, which='both', linewidth=0.5, color='grey')
    plt.grid(False)

    output_handler.saveFig(subdir + '/' + name +'_{:03d}deg.png'.format(int(phi_deg)), dpi=250)
    output_handler.log.info('Saved figure: ' + subdir + '/' + name +'_{:03d}deg.png'.format(int(phi_deg)))
    plt.close("All")

if __name__ == '__main__':
    #### DEFINE ANALYSIS PARAMETERS ####
    ## RUN DIRECTORY AND SUBDIRECTORY

    #ANLYS_DIR = "AcceptedIota3_1500spins_atole-9"
    # ANLYS_SUBDIR = ""
    # ANLYS_DIR = "AcceptedIota4_1500spins_atole-8_eng"
    # ANLYS_SUBDIR = "LCFS40_3x360x360mesh_UPDATED"
    # ANLYS_DIR = "ChangeToIota3_1500spins_atole-9"
    # ANLYS_SUBDIR = 'LCFS29_3x360x360mesh_CORRECTCURR_lotol'

    # ANLYS_DIR = "AcceptedIota4_1500spins_atole-8_eng"
    # ANLYS_SUBDIR = "LCFS35_360x90_tol_5e1_5e2_LOMEM"

    # ANLYS_DIR = "AcceptedIota3_1500spins_atole-9"
    # ANLYS_SUBDIR = "LCFS19_360x180_tol_5e1_5e2_APS2025"

    # ANLYS_DIR = "It-0486_Ih-0900_noErr_1500sp_LSODA1e8"
    # ANLYS_SUBDIR = "LCFS30_360x180_smooth1e-4"

    ANLYS_DIR = "It-0486_Ih-0790_PHI324_1500sp_LSODA2p49e8"
    ANLYS_SUBDIR = "LCFS15_360x180_smooth1e-4"

    ## DEFINE FIELDS
    FIELD_FILE_TOR = 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy'
    FIELD_FILE_HEL = 'input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy'
    CURRENT_TOR = 0.486 #[kA]
    CURRENT_HEL = 0.790 #[kA]
    CONFIG_TOR = 'default_toroidal'
    CONFIG_HEL = 'default_helical'
    ENABLE_ERRFIELD = True

    ## DEFINE LCFS AND ANGLES TO EVALUATE
    LCFS_INDEX = 15 #29 #100  #1f00 #40 #22 #29?
    NPHI = 360
    NTHETA = 180
    PHI_GENs = np.linspace(360//NPHI, 360, NPHI)

    ## FLUX INTEGRATION PARAMETERS
    MAX_SUBSETS = 4
    SMALLEST_ISLAND_INDEX = None #104 #53 #39

    OUTPUT_FILE_NAME = 'Efield_IdealIota3_lcfs30'
    fluxGradientor()
