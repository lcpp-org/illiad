## IN THIS FILE, WE WILL TAKE THE CALCULATED FLUX FOR EACH SURFACE,
## APPLY IT TO THE POINTS ON THE THEIR RESPECTIVE SURFACE, 
## AND THEN INTERPOLATE IT ONTO A MESH THE SAME SHAPE/RESOLUTION AS THE BFIELD MESH
import classes.class_outputHandler as out
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
#import gc
import time

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
    b_hidra.loadCartesianField(FIELD_FILE_TOR, att_mult=FIELD_SCALE_TOR, errField=True )
    b_hidra.addFieldPerturbation(FIELD_FILE_HEL, att_mult=FIELD_SCALE_HEL)
    b_hidra.set_nonPer_errField(ERRFIELD_MAG, ERRFIELD_DIR_DEG*np.pi/180.)
    lcfs_index = identifyLCFS(LCFStype='input', num=LCFS_INPUT, outputHandler=simIO)

    # load numpy data using simIO method: big_grid_linear, big_grid_parabolic
    big_grid_linear = simIO.loadNumpyData(ANLYS_SUBDIR + '/big_grid_linear2.npy')
    print(f'{big_grid_linear.shape=}')#, {big_grid_parabolic.shape=}')

    # Create a meshgrid for the interpolation
    RADS = np.linspace(b_hidra.r_min, b_hidra.r_max, b_hidra.nr)
    THETAS = np.linspace(b_hidra.theta_min, b_hidra.theta_max, b_hidra.ntheta)
    #THETAS = np.linspace(0, b_hidra.theta_max, b_hidra.ntheta+1)
    grid_theta, grid_rad = np.meshgrid(THETAS, RADS, indexing='ij')

    # GRADIENT CALCULATION: remember to divide by Jacobian determinant gradF = [dF/dr] * R_HAT + [(1/r) * df/dtheta] * THETA_HAT + [( 1/(R0+rcos(theta)) ) * df/dphi] * PHI_HAT
    big_flux_Lingrad = np.gradient(big_grid_linear, PHI_GENs, THETAS, RADS, edge_order=2)#, [grid_rad, grid_theta])

    big_flux_Lingrad_radial = -big_flux_Lingrad[2]  # E = -grad[V]
    big_flux_Lingrad_poloidal =np.zeros_like(big_flux_Lingrad[1])
    big_flux_Lingrad_poloidal[:,:,1:] = -big_flux_Lingrad[1][:,:,1:] / grid_rad[:,1:]
    big_flux_Lingrad_toroidal = -big_flux_Lingrad[0] / (b_hidra.R0 + grid_rad * np.cos(grid_theta))

    print(f'{big_flux_Lingrad_radial.shape=}')
    # Load LCFS file:
    lcfs_filename = ANLYS_SUBDIR + '/fSurf_{:03d}_POINTmesh.npy'.format(int(lcfs_index+1))
    lcfs_points_full = simIO.loadNumpyData(lcfs_filename)
    
    # set all points outside the LCFS to zero
    for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):
        lcfs_points = lcfs_points_full[phi_index]
        # keep only 1st subset of points (360 points)
        lcfs_points= lcfs_points[:360].T
        print(f'{phi_index=}')

        for theta_index, this_theta in enumerate(THETAS):
            # find the index of the value in lcfs_points[0] closest to this_theta
            #lcfs_theta_index = np.argmin(np.abs(lcfs_points[0] - this_theta))
            # find the index of the value in lcfs_points[0] closest to this_theta
            mintheta1 = np.abs(lcfs_points[0] - this_theta)
            mintheta2 = np.abs(lcfs_points[0] - this_theta + 2*np.pi)
            # calculate the minimum of the two
            mintheta3 = np.fmin(mintheta1, mintheta2)
            #print(f'{mintheta1.shape=}\n{mintheta2.shape=}\n{mintheta3.shape=}')
            lcfs_theta_index = np.argmin(mintheta3)
            #print(f'LCFS theta index for {this_theta} is {lcfs_theta_index} (theta={lcfs_points[0][lcfs_theta_index]})')
            lcfs_rad = lcfs_points[1][lcfs_theta_index]

            # Use boolean indexing to set all radii greater than (lcfs_rad - 0.01) to zero for this theta
            mask = RADS > (lcfs_rad + 0.001) # add buffer to avoid numerical issues
            #grid_linear[theta_index][mask] = 0.0
            big_flux_Lingrad_radial[phi_index][theta_index][mask] = 0.0
            big_flux_Lingrad_poloidal[phi_index][theta_index][mask] = 0.0
            big_flux_Lingrad_toroidal[phi_index][theta_index][mask] = 0.0
            # for r_index, radius in enumerate(RADS):
            #     lcfs_rad = lcfs_points[1][lcfs_theta_index]
            #     #lcfs_theta = lcfs_points[0][lcfs_theta_index]
            #     #print(f'Checking point at theta={this_theta}, r={radius}\nagainst LCFS theta={lcfs_theta}, r={lcfs_rad}')

            #     if radius > lcfs_rad+0.004:
            #         big_flux_Lingrad_radial[phi_index][theta_index][r_index] = 0.0
            #         big_flux_Lingrad_poloidal[phi_index][theta_index][r_index] = 0.0
            #         big_flux_Lingrad_toroidal[phi_index][theta_index][r_index] = 0.0
                    #print(f'Setting point at theta={THETAS[theta_index]}, r={RADS[r_index]} to zero (outside LCFS)')



    big_flux_Lingrad_magnitude = np.sqrt(big_flux_Lingrad_radial**2 + big_flux_Lingrad_poloidal**2 + big_flux_Lingrad_toroidal**2)

    # RESHAPE THE ARRAYS TO MATCH THE DIMNENSIONS OF INPUT BFIELDS
    reshaped_big_grid_linear_r = np.transpose(big_flux_Lingrad_radial, (2, 1, 0))
    reshaped_big_grid_linear_pol = np.transpose(big_flux_Lingrad_poloidal, (2, 1, 0))
    reshaped_big_grid_linear_tor = np.transpose(big_flux_Lingrad_toroidal, (2, 1, 0))
    Efield_rtpArray_linear = np.array([reshaped_big_grid_linear_r, reshaped_big_grid_linear_pol, reshaped_big_grid_linear_tor])
    print(f'{Efield_rtpArray_linear.shape=}')
    # save the array using simIO method
    simIO.saveNumpyData(Efield_rtpArray_linear, ANLYS_SUBDIR + '/Efield_rtpArray_linear.npy')

    Efield_xyzArray_linear = np.zeros_like(Efield_rtpArray_linear)
    xform_rad, xform_theta, xform_phi= np.meshgrid(RADS, THETAS, PHI_GENs, indexing='ij')
    print(f'{xform_rad.shape=}')

    flattened_shape = Efield_xyzArray_linear[0].flatten().shape
    Ex_linear = np.zeros(flattened_shape)
    Ey_linear = np.zeros(flattened_shape)
    Ez_linear = np.zeros(flattened_shape)
    for i, (rad, theta, phi, EradLin, EthetaLin, EphiLin) in enumerate(zip(xform_rad.flatten(), xform_theta.flatten(), xform_phi.flatten(),
                                                                      reshaped_big_grid_linear_r.flatten(), reshaped_big_grid_linear_pol.flatten(), reshaped_big_grid_linear_tor.flatten())):
        ErtpLin = np.array([EradLin, EthetaLin, EphiLin])
        p_RTP = np.array([rad, theta, np.radians(phi)])
        Ex_linear[i], Ey_linear[i], Ez_linear[i] = RTP_XYZ_JAC(p_RTP, ErtpLin, form='rtp2xyz')
    print(f'{Ex_linear.max()=}, {Ex_linear.min()=}, {Ey_linear.max()=}, {Ey_linear.min()=}, {Ez_linear.max()=}, {Ez_linear.min()=}')
    
    # reshape the arrays to match the dimensions of input Bfields
    Efield_xyzArray_linear[0] = Ex_linear.reshape(Efield_xyzArray_linear[0].shape)
    Efield_xyzArray_linear[1] = Ey_linear.reshape(Efield_xyzArray_linear[1].shape)
    Efield_xyzArray_linear[2] = Ez_linear.reshape(Efield_xyzArray_linear[2].shape)
    #print(f'{Efield_xyzArray_linear.shape=}')

    ## SAVE THE ARRAYS
    simIO.saveNumpyData(Efield_xyzArray_linear, ANLYS_SUBDIR + '/Efield_SOFE2.npy')

    ## LOOP THROUGH PHI ANGLES forplotting
    colortest = 'afmhot_r'
    for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):
        #output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_grid_linear[phi_index], 'LinearFluxNorm', ANLYS_SUBDIR, simIO, 'inferno', 0.0, 1.0)
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_flux_Lingrad_magnitude[phi_index], 'LinearFluxGradMagnitude', ANLYS_SUBDIR, simIO, colortest, 0.0, 200.0)
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_flux_Lingrad_radial[phi_index], 'LinearFluxGradRadial', ANLYS_SUBDIR, simIO, colortest, -200., 200)
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_flux_Lingrad_poloidal[phi_index], 'LinearFluxGradPoloidal', ANLYS_SUBDIR, simIO, colortest, -100.0, 100.0)
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_flux_Lingrad_toroidal[phi_index], 'LinearFluxGradToroidal', ANLYS_SUBDIR, simIO, colortest, -0.3, 0.3)


def output_phi_plots(phi_deg, grid_theta, grid_rad, data, name, subdir, output_handler, colormap='infernp', plotmin=None, plotmax=None):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    #c = ax.pcolormesh(grid_theta, grid_rad, data, shading='auto', cmap=colormap, vmin=plotmin, vmax=plotmax)
    data = np.vstack((data[-1], data))
    grid_rad = np.vstack((grid_rad[-1], grid_rad))
    grid_theta = np.vstack((grid_theta[-1], grid_theta))
    grid_theta[0] = 0

    c = ax.pcolormesh(grid_theta, grid_rad, data, shading='gouraud', cmap=colormap, vmin=plotmin, vmax=plotmax)

    ax.set_title(name + '\n$\phi_{{phy}}$={:02.0f}$\degree$ CW from North Split\n$\phi_c$={:02.0f}$\degree$'.format((phi_deg+198.)%360., phi_deg), loc='left')
    ax.set_rmax(0.19)
    #ax.set_rticks(np.arange(0.0, 0.19, 0.02))
    ax.set_rticks([])
    # set the r-labels to an empty list
    #ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.colorbar(c, ax=ax, label='Flux')
    #plt.grid(True, which='both', linewidth=0.5, color='grey')
    plt.grid(False)

    output_handler.saveFig(subdir + '/' + name +'_{:03d}deg.png'.format(int(phi_deg)), dpi=250)
    output_handler.log.info('Saved figure: ' + subdir + '/' + name +'_{:03d}deg.png'.format(int(phi_deg)))
    #plt.show()
    plt.close("All")

if __name__ == '__main__':
     #### DEFINE ANALYSIS PARAMETERS ####
    ## RUN DIRECTORY AND SUBDIRECTORY
    ANLYS_DIR = "AcceptedIota3_1500spins_atole-9_older"
    #ANLYS_SUBDIR = 'LCFS22_3x360x360mesh_SMOOTHER_7p5e6'
    ANLYS_SUBDIR = 'LCFS29_3x360x360mesh_SOFE1'

    # ANLYS_DIR = "ChangeToIota3_1500spins_atole-9"
    # ANLYS_SUBDIR = 'LCFS18_3x360x60mesh_Production1'

    ## DEFINE FIELDS
    FIELD_FILE_TOR = 'input_files/It486_Ih000_Iv000_1p000_1p000_64bit.npy'
    FIELD_SCALE_TOR = 0.9448 #0.9452
    FIELD_FILE_HEL = 'input_files/It000_Ih900_Iv000_1p000_1p000_64bit.npy'
    FIELD_SCALE_HEL = -0.955 * FIELD_SCALE_TOR
    ERRFIELD_MAG = 1.5654e-4 # [Tesla]
    ERRFIELD_DIR_DEG = 271.5 # [degrees]

    ## IDENTIFY LAST-CLOSED FLUX SURFACE
    LCFS_INPUT = 33 #25 #22
    ## DEFINE ANGLES TO EVALUATE AND PLOT
    NPHI = 360
    NTHETA = 360

    PHI_GENs = np.linspace(360//NPHI, 360, NPHI)
    MAX_SUBSETS = 3

    main()
