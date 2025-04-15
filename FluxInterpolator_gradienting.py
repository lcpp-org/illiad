## IN THIS FILE, WE WILL TAKE THE CALCULATED FLUX FOR EACH SURFACE,
## APPLY IT TO THE POINTS ON THE THEIR RESPECTIVE SURFACE, 
## AND THEN INTERPOLATE IT ONTO A MESH THE SAME SHAPE/RESOLUTION AS THE BFIELD MESH
import classes.class_outputHandler as out
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import gc

from classes.mesh import *
from utility.anlys_funcs import identifyLCFS
from utility.coordtrans import RTP_XYZ_JAC

def main():
    ## DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
    simIO = out.IOHandler(ANLYS_DIR)
    simIO.startLog()
    simIO.createSubDir(ANLYS_SUBDIR)

    ## DEFINE MESH AND LOAD FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField(FIELD_FILE_TOR, errField=True, att_mult=FIELD_SCALE_TOR)
    b_hidra.addFieldPerturbation(FIELD_FILE_HEL, att_mult=FIELD_SCALE_HEL)
    b_hidra.set_nonPer_errField(FIELD_ERR_MAG, FIELD_ERR_DIR)

    # load numpy data using simIO method: big_grid_linear, big_grid_parabolic
    big_grid_linear = simIO.loadNumpyData(ANLYS_SUBDIR + '/big_grid_linear.npy')
    big_grid_parabolic = simIO.loadNumpyData(ANLYS_SUBDIR + '/big_grid_parabolic.npy')
    print(f'{big_grid_linear.shape=}, {big_grid_parabolic.shape=}')

    # Create a meshgrid for the interpolation
    RADS = np.linspace(b_hidra.r_min, b_hidra.r_max, b_hidra.nr)
    THETAS = np.linspace(b_hidra.theta_min, b_hidra.theta_max, b_hidra.ntheta)
    #grid_phi, grid_theta, grid_rad = np.meshgrid(PHI_GENs, THETAS, RADS, indexing='ij')
    grid_theta, grid_rad = np.meshgrid(THETAS, RADS, indexing='ij')





    # GRADIENT CALCULATION: remember to divide by Jacobian determinant gradF = [dF/dr] * R_HAT + [(1/r) * df/dtheta] * THETA_HAT + [( 1/(R0+rcos(theta)) ) * df/dphi] * PHI_HAT
    big_flux_Lingrad = np.gradient(big_grid_linear, PHI_GENs, THETAS, RADS)#, [grid_rad, grid_theta])
    # print(f'{big_flux_Lingrad[0] .shape=}')
    # print(f'{grid_rad.shape=}')

    big_flux_Lingrad_radial = -big_flux_Lingrad[2]  # E = -grad[V]
    big_flux_Lingrad_poloidal =np.zeros_like(big_flux_Lingrad[1])
    big_flux_Lingrad_poloidal[:,:,1:] = -big_flux_Lingrad[1][:,:,1:] / grid_rad[:,1:]
    big_flux_Lingrad_toroidal = -big_flux_Lingrad[0] / (b_hidra.R0 + grid_rad * np.cos(grid_theta))
    big_flux_Lingrad_magnitude = np.sqrt(big_flux_Lingrad_radial**2 + big_flux_Lingrad_poloidal**2 + big_flux_Lingrad_toroidal**2)


    # GRADIENT CALCULATION: remember to divide by Jacobian determinant
    big_flux_Pargrad = np.gradient(big_grid_parabolic, PHI_GENs, THETAS, RADS)#, [grid_rad, grid_theta])

    big_flux_Pargrad_radial = -big_flux_Pargrad[2]  # E = -grad[V]
    big_flux_Pargrad_poloidal = np.zeros_like(big_flux_Pargrad[1])
    big_flux_Pargrad_poloidal[:,:,1:] =  -big_flux_Pargrad[1][:,:,1:] / grid_rad[:,1:]
    big_flux_Pargrad_toroidal = -big_flux_Pargrad[0] / (b_hidra.R0 + grid_rad * np.cos(grid_theta))
    big_flux_Pargrad_magnitude = np.sqrt(big_flux_Pargrad_radial**2 + big_flux_Pargrad_poloidal**2 + big_flux_Pargrad_toroidal**2)



    # RESHAPE THE ARRAYS TO MATCH THE DIMNENSIONS OF INPUT BFIELDS
    reshaped_big_grid_linear_r = np.transpose(big_flux_Pargrad_radial, (2, 1, 0))
    reshaped_big_grid_linear_pol = np.transpose(big_flux_Pargrad_poloidal, (2, 1, 0))
    reshaped_big_grid_linear_tor = np.transpose(big_flux_Pargrad_toroidal, (2, 1, 0))
    Efield_rtpArray_linear = np.array([reshaped_big_grid_linear_r, reshaped_big_grid_linear_pol, reshaped_big_grid_linear_tor])
    print(f'{Efield_rtpArray_linear.shape=}')
    # save the array using simIO method
    simIO.saveNumpyData(Efield_rtpArray_linear, ANLYS_SUBDIR + '/Efield_rtpArray_linear.npy')

    reshaped_big_grid_parabolic_r = np.transpose(big_flux_Lingrad_radial, (2, 1, 0))
    reshaped_big_grid_parabolic_pol = np.transpose(big_flux_Lingrad_poloidal, (2, 1, 0))
    reshaped_big_grid_parabolic_tor = np.transpose(big_flux_Lingrad_toroidal, (2, 1, 0))
    Efield_rtpArray_parabolic = np.array([reshaped_big_grid_parabolic_r, reshaped_big_grid_parabolic_pol, reshaped_big_grid_parabolic_tor])
    
    # save the array using simIO method
    simIO.saveNumpyData(Efield_rtpArray_parabolic, ANLYS_SUBDIR + '/Efield_rtpArray_parabolic.npy')


    
    Efield_xyzArray_linear = np.zeros_like(Efield_rtpArray_linear)
    Efield_xyzArray_parabolic = np.zeros_like(Efield_rtpArray_parabolic)
    print(f'{Efield_rtpArray_parabolic.shape=}')

    #xform_phi, xform_theta, xform_rad = np.meshgrid(PHI_GENs, THETAS, RADS, indexing='ij')
    xform_rad, xform_theta, xform_phi= np.meshgrid(RADS, THETAS, PHI_GENs, indexing='ij')
    print(f'{xform_rad.shape=}')

    flattened_shape = Efield_xyzArray_linear[0].flatten().shape
    Ex_linear = np.zeros(flattened_shape)
    Ey_linear = np.zeros(flattened_shape)
    Ez_linear = np.zeros(flattened_shape)
    Ex_parabolic = np.zeros(flattened_shape)
    Ey_parabolic = np.zeros(flattened_shape)
    Ez_parabolic = np.zeros(flattened_shape)

    for i, (rad, theta, phi, EradLin, EthetaLin, EphiLin, EradPar, EthetaPar, EphiPar) in enumerate(zip(xform_rad.flatten(), xform_theta.flatten(), xform_phi.flatten(),
                                                                      reshaped_big_grid_linear_r.flatten(), reshaped_big_grid_linear_pol.flatten(), reshaped_big_grid_linear_tor.flatten(),
                                                                      reshaped_big_grid_parabolic_r.flatten(), reshaped_big_grid_parabolic_pol.flatten(), reshaped_big_grid_parabolic_tor.flatten())):
        ErtpLin = np.array([EradLin, EthetaLin, EphiLin])
        ErtpPar = np.array([EradPar, EthetaPar, EphiPar])
        #print(f'{ErtpLin=}, {ErtpPar=}')
        p_RTP = np.array([rad, theta, np.radians(phi)])
        #print(f'{p_RTP=}')

        Ex_linear[i], Ey_linear[i], Ez_linear[i] = RTP_XYZ_JAC(p_RTP, ErtpLin, form='rtp2xyz')
        Ex_parabolic[i], Ey_parabolic[i], Ez_parabolic[i] = RTP_XYZ_JAC(p_RTP, ErtpPar, form='rtp2xyz')


    print(f'{Ex_linear.max()=}, {Ex_linear.min()=}, {Ey_linear.max()=}, {Ey_linear.min()=}, {Ez_linear.max()=}, {Ez_linear.min()=}')
    print(f'{Ex_parabolic.max()=}, {Ex_parabolic.min()=}, {Ey_parabolic.max()=}, {Ey_parabolic.min()=}, {Ez_parabolic.max()=}, {Ez_parabolic.min()=}')
    # reshape the arrays to match the dimensions of input Bfields

    Efield_xyzArray_linear[0] = Ex_linear.reshape(Efield_xyzArray_linear[0].shape)
    Efield_xyzArray_linear[1] = Ey_linear.reshape(Efield_xyzArray_linear[1].shape)
    Efield_xyzArray_linear[2] = Ez_linear.reshape(Efield_xyzArray_linear[2].shape)
    Efield_xyzArray_parabolic[0] = Ex_parabolic.reshape(Efield_xyzArray_parabolic[0].shape)
    Efield_xyzArray_parabolic[1] = Ey_parabolic.reshape(Efield_xyzArray_parabolic[1].shape)
    Efield_xyzArray_parabolic[2] = Ez_parabolic.reshape(Efield_xyzArray_parabolic[2].shape)

    print(f'{Efield_xyzArray_linear.shape=}')
    #print(f'{Efield_xyzArray_parabolic}')

    ## SAVE THE ARRAYS
    simIO.saveNumpyData(Efield_xyzArray_linear, ANLYS_SUBDIR + '/Efield_xyzArray_linear.npy')
    simIO.saveNumpyData(Efield_xyzArray_parabolic, ANLYS_SUBDIR + '/Efield_xyzArray_parabolic.npy')

    plot_Efield_XArray_parabolic = np.transpose(Efield_xyzArray_parabolic[0], (2, 1, 0))
    plot_Efield_YArray_parabolic = np.transpose(Efield_xyzArray_parabolic[1], (2, 1, 0))
    plot_Efield_ZArray_parabolic = np.transpose(Efield_xyzArray_parabolic[2], (2, 1, 0))


    ## LOOP THROUGH PHI ANGLES forplotting
    colortest = 'seismic'
    for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):
        #output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_grid_linear[phi_index], 'LinearFluxNorm', ANLYS_SUBDIR, simIO, 'inferno', 0.0, 1.0)
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_flux_Lingrad_magnitude[phi_index], 'LinearFluxGradMagnitude', ANLYS_SUBDIR, simIO, 'inferno', 0.0, 200.0)
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_flux_Lingrad_radial[phi_index], 'LinearFluxGradRadial', ANLYS_SUBDIR, simIO, colortest, -200., 200)
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_flux_Lingrad_poloidal[phi_index], 'LinearFluxGradPoloidal', ANLYS_SUBDIR, simIO, colortest, -100.0, 100.0)
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_flux_Lingrad_toroidal[phi_index], 'LinearFluxGradToroidal', ANLYS_SUBDIR, simIO, colortest, -0.3, 0.3)

        #output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_grid_parabolic[phi_index], 'ParabolicFluxNorm', ANLYS_SUBDIR, simIO, 'inferno', 0.0, 1.0)
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_flux_Pargrad_magnitude[phi_index], 'ParabolicFluxGradMagnitude', ANLYS_SUBDIR, simIO, 'inferno', 0.0, 200.0)
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_flux_Pargrad_radial[phi_index], 'ParabolicFluxGradRadial', ANLYS_SUBDIR, simIO, colortest, -200., 200)
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_flux_Pargrad_poloidal[phi_index], 'ParabolicFluxGradPoloidal', ANLYS_SUBDIR, simIO, colortest, -100.0, 100.0)
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_flux_Pargrad_toroidal[phi_index], 'ParabolicFluxGradToroidal', ANLYS_SUBDIR, simIO, colortest, -0.3, 0.3)

        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, plot_Efield_XArray_parabolic[phi_index], 'ParabolicFluxGradX', ANLYS_SUBDIR, simIO, colortest, -200., 200)
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, plot_Efield_YArray_parabolic[phi_index], 'ParabolicFluxGradY', ANLYS_SUBDIR, simIO, colortest, -200., 200)
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, plot_Efield_ZArray_parabolic[phi_index], 'ParabolicFluxGradZ', ANLYS_SUBDIR, simIO, colortest, -200., 200)




def output_phi_plots(phi_deg, grid_theta, grid_rad, data, name, subdir, output_handler, colormap='inferno', plotmin=None, plotmax=None):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    c = ax.pcolormesh(grid_theta, grid_rad, data, shading='auto', cmap=colormap, vmin=plotmin, vmax=plotmax)

    ax.set_title(name + '\n$\phi_{{phy}}$={:02.0f}$\degree$ CW from North Split\n$\phi_c$={:02.0f}$\degree$'.format((phi_deg+198.)%360., phi_deg), loc='left')
    #ax.set_rmax(b_hidra.a)
    ax.set_rmax(0.19)
    #ax.set_rticks(np.arange(0.0, b_hidra.a, 0.02))
    ax.set_rticks(np.arange(0.0, 0.19, 0.02))
    # set the r-labels to an empty list
    #ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.colorbar(c, ax=ax, label='Flux')

    output_handler.saveFig(subdir + '/' + name +'_{:03d}deg.png'.format(int(phi_deg)), dpi=250)
    output_handler.log.info('Saved figure: ' + subdir + '/' + name +'_{:03d}deg.png'.format(int(phi_deg)))
    #plt.show()
    plt.close("All")
    #gc.collect()

if __name__ == '__main__':
    #### DEFINE ANALYSIS PARAMETERS ####
    ## RUN DIRECTORY AND SUBDIRECTORY
    # ANLYS_DIR = "AcceptedIota3_1500spins_atole-9"
    # ANLYS_SUBDIR = 'LCFS22_3x360x60mesh_PRODUCTION2'

    ANLYS_DIR = "ChangeToIota3_1500spins_atole-9"
    ANLYS_SUBDIR = 'LCFS18_3x360x60mesh_Production1'

    ## DEFINE FIELDS
    FIELD_FILE_TOR = 'input_files/It486_Ih000_Iv000_1p000_1p000_64bit.npy'
    FIELD_SCALE_TOR = 0.9452
    FIELD_FILE_HEL = 'input_files/It000_Ih900_Iv000_1p000_1p000_64bit.npy'
    FIELD_SCALE_HEL = 0.955 * FIELD_SCALE_TOR
    FIELD_ERR_MAG = 1.5939e-4 #3.168e-4
    FIELD_ERR_DIR = np.radians(272.)
    ## IDENTIFY LAST-CLOSED FLUX SURFACE
    LCFS_INPUT = 18
    ## DEFINE ANGLES TO EVALUATE AND PLOT
    NPHI = 360
    NTHETA = 60
    PHI_GENs = np.linspace(360//NPHI, 360, NPHI) # = np.array([18])
    MAX_SUBSETS = 3

    main()