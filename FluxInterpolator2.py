## IN THIS FILE, WE WILL TAKE THE CALCULATED FLUX FOR EACH SURFACE,
## APPLY IT TO THE POINTS ON THE THEIR RESPECTIVE SURFACE, 
## AND THEN INTERPOLATE IT ONTO A MESH THE SAME SHAPE/RESOLUTION AS THE BFIELD MESH
import classes.class_outputHandler as out
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from classes.mesh import *
from utility.anlys_funcs import identifyLCFS

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

    RADS = np.linspace(b_hidra.r_min, b_hidra.r_max, b_hidra.nr)
    THETAS = np.linspace(b_hidra.theta_min, b_hidra.theta_max, b_hidra.ntheta)

    ## LOAD FLUX DATA
    filepath = ANLYS_SUBDIR  + '/'
    flux_name = filepath + 'CalculatedFLuxes.npy'
    flux_array = simIO.loadNumpyData(flux_name)
    flux_norm_name = filepath + 'CalculatedFLuxes-normalized.npy'
    flux_norm_array = simIO.loadNumpyData(flux_norm_name)
    N_surfaces, N_phis = flux_array.shape
    print(f'{N_surfaces=}, {N_phis=}')

    # LOAD VALID SURFACE DATA
    validSurf_name = filepath + 'ValidSurfaces.npy'
    valid_surface = simIO.loadNumpyData(validSurf_name)
    #valid_surface[[38,46,58]] = True # manually set some surfaces to valid
    valid_surface[[38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58]] = True # manually set some surfaces to valid

    # Load Magnetic Axis point:
    filename_center = filepath + 'fSurf_{:03d}_center.npy'.format(N_surfaces-1)
    axis_array = simIO.loadNumpyData(filename_center)
    filename_center_island = filepath + 'fSurf_{:03d}_center.npy'.format(39)
    island_axis_array = simIO.loadNumpyData(filename_center_island)

    # Load LCFS file:
    lcfs_filename = ANLYS_SUBDIR + '/fSurf_{:03d}_POINTmesh.npy'.format(int(lcfs_index+1))
    lcfs_points_full = simIO.loadNumpyData(lcfs_filename)

    # Choosing one 'well-behaved' angle for the calculation (no failed calculations)
    linear_flux_array = flux_norm_array[:, 14] #6, 13, 19, 20, 22, 24,26

    ## DEBUG plot filtered_flux_array with matplotlib
    # # # fig, ax = plt.subplots()
    # # # ax.plot(np.arange(N_surfaces), filtered_flux_array)
    # # # ax.set_xlabel('Surface Index')
    # # # ax.set_ylabel('Flux')
    # # # ax.set_title('Filtered Flux Array')
    # # # plt.show()

    # Create a meshgrid for the interpolation
    grid_theta, grid_rad = np.meshgrid(THETAS, RADS, indexing='ij')
    big_grid_linear = np.zeros([len(PHI_GENs), len(THETAS), len(RADS)])

    ## LOOP THROUGH PHI ANGLES
    for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):
        # axes points
        points = np.zeros([4,2])
        points[0] = axis_array[phi_index][0]
        print(f'Central axis point: {axis_array[phi_index][0]}')
        points[1:] = island_axis_array[phi_index]
        print(f'Island axis points: {points[1]}, {points[2]}, {points[3]}')
        # linear values for the axes points
        linear_values = np.ones([4])

        ## LOAD SCATTER POINTS (POINCARE DATA)
        filename = 'Poincare_{:03d}.npy'.format(int(PHI_GEN_DEG))
        flux_surfaces = simIO.loadNumpyData(filename)
        ## GET LCFS POINTS
        #print(f'{lcfs_points_full.shape=}')
        lcfs_points = lcfs_points_full[phi_index].T
        # # make a polar plot of the LCFS points
        # plt.figure()
        # plt.polar(lcfs_points[0], lcfs_points[1], 'o', markersize=1, label='LCFS Points')
        # plt.show()

        #print(f'LCFS points: {lcfs_points.shape=}, {lcfs_points[0]=}, {lcfs_points[1]=}')
        #print(f'{b_hidra.ntheta=}')
        ## LOOP THROUGH SURFACES
        for surface_index in range(lcfs_index, N_surfaces):
            if valid_surface[surface_index] == False:
                print(f'Skipping surface {surface_index} (not valid)')
            else:
                # ## LOAD SCATTER POINTS (SPLINED VALUES)
                # filename = ANLYS_SUBDIR + '/fSurf_{:03d}_POINTmesh.npy'.format(int(surface_index))
                # flux_surfaces = simIO.loadNumpyData(filename)[phi_index].T
                # #print(f'{flux_surfaces.shape=}')
                # thetas = flux_surfaces[0]
                # rads = flux_surfaces[1]
                # thetas = thetas[~np.isnan(thetas)]
                # rads = rads[~np.isnan(rads)]
                # N_pts = len(thetas)
                # #print(f'{N_pts=}')

                ### GET VALUES
                thetas = flux_surfaces[surface_index][0]
                rads = flux_surfaces[surface_index][1]
                # filter NaNs
                thetas = thetas[~np.isnan(thetas)]
                rads = rads[~np.isnan(rads)]
                N_pts = len(thetas)

                # concatenate to big array of points
                these_points = np.array([thetas, rads]).T
                points = np.concatenate((points, these_points))

                these_lin_values = np.full(N_pts, linear_flux_array[surface_index])
                linear_values = np.concatenate((linear_values, these_lin_values))

        grid_linear = griddata(points, linear_values, (grid_theta, grid_rad), method='linear', fill_value=0.0, rescale=True)

        ## HACKY SOLUTIONS HERE!!!
        # averaging out for theta=2pi
        grid_linear[-1] = (grid_linear[-2] + grid_linear[0]) / 2
        # copying values out for r=0.0
        fred3 = grid_linear.T[1]
        fred4 = grid_linear.T[2]
        fred3[fred3==0] = fred4[fred3==0]
        grid_linear.T[1] = fred3
        grid_linear.T[0] = grid_linear.T[1]

        # set all points outside the LCFS to zero
        #grid_linear[theta_index][r_index]
        for theta_index, this_theta in enumerate(THETAS):
            # find the index of the value in lcfs_points[0] closest to this_theta
            lcfs_theta_index = np.argmin(np.abs(lcfs_points[0] - this_theta))
            # get the corresponding radius from lcfs_points[1]

            for r_index, radius in enumerate(RADS):
                lcfs_rad = lcfs_points[1][lcfs_theta_index]
                lcfs_theta = lcfs_points[0][lcfs_theta_index]
                #print(f'Checking point at theta={THETAS[theta_index]}, r={radius}\nagainst LCFS theta={lcfs_theta}, r={lcfs_rad}')
                if radius > lcfs_rad+0.0005:
                    grid_linear[theta_index][r_index] = 0.0
                    #print(f'Setting point at theta={THETAS[theta_index]}, r={RADS[r_index]} to zero (outside LCFS)')



        # Add to big mesh array (3D)
        big_grid_linear[phi_index] = grid_linear

    #### END OF LOOP THROUGH PHI ANGLES ####

    # save numpy data using simIO method
    simIO.saveNumpyData(big_grid_linear, ANLYS_SUBDIR + '/big_grid_linear2.npy')

    ## LOOP THROUGH PHI ANGLES for plotting
    for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_grid_linear[phi_index], 'LinearFluxNorm', ANLYS_SUBDIR, simIO, 'inferno', 0.0, 1.0)


def output_phi_plots(phi_deg, grid_theta, grid_rad, data, name, subdir, output_handler, colormap='inferno', plotmin=None, plotmax=None):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title(name + '\n$\phi_{{phy}}$={:02.0f}$\degree$ CW from North Split\n$\phi_c$={:02.0f}$\degree$'.format((phi_deg+198.)%360., phi_deg), loc='left')

    c = ax.pcolormesh(grid_theta, grid_rad, data, shading='auto', cmap=colormap, vmin=plotmin, vmax=plotmax)

    ax.set_rmax(0.19)
    ax.set_rticks(np.arange(0.0, 0.19, 0.02))
    fig.colorbar(c, ax=ax, label='Flux')

    output_handler.saveFig(subdir + '/' + name +'_{:03d}deg.png'.format(int(phi_deg)), dpi=300)
    plt.close()


if __name__ == '__main__':
    #### DEFINE ANALYSIS PARAMETERS ####
    ## RUN DIRECTORY AND SUBDIRECTORY
    ANLYS_DIR = "AcceptedIota3_1500spins_atole-9"
    ANLYS_SUBDIR = 'LCFS22_3x360x360mesh_SMOOTHER_7p5e6'

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
    LCFS_INPUT = 22
    ## DEFINE ANGLES TO EVALUATE AND PLOT
    NPHI = 360
    NTHETA = 360

    PHI_GENs = np.linspace(360//NPHI, 360, NPHI)
    MAX_SUBSETS = 3

    main()