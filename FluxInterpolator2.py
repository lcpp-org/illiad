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
    ## DEFINE MESH AND LOAD MAGNETIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField(coilCurrent=CURRENT_TOR, errField=True, att_mult=CONFIG_TOR)
    b_hidra.addFieldPerturbation(coilCurrent=CURRENT_HEL, att_mult=CONFIG_HEL)
    b_hidra.set_nonPer_errField()
    lcfs_index = identifyLCFS(LCFStype='input', num=LCFS_INPUT, outputHandler=simIO)


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
    valid_surface[lcfs_index:] = True # manually set some surfaces to valid
    valid_surface[39] = False # manually set some surfaces to valid

    # Load Magnetic Axis point:
    filename_center = filepath + 'fSurf_{:03d}_center.npy'.format(N_surfaces-1)
    axis_array = simIO.loadNumpyData(filename_center)
    filename_center_island = filepath + 'fSurf_{:03d}_center.npy'.format(SMALLEST_ISLAND_INDEX)
    island_axis_array = simIO.loadNumpyData(filename_center_island)

    # Load LCFS file:
    lcfs_filename = ANLYS_SUBDIR + '/fSurf_{:03d}_POINTmesh.npy'.format(int(lcfs_index+1))
    lcfs_points_full = simIO.loadNumpyData(lcfs_filename)
    print(f'{lcfs_points_full.shape=}')
    # Choosing one 'well-behaved' angle for the calculation (no failed calculations)
    linear_flux_array = flux_norm_array[:, 4]

    # find the indices where valid_surface is True
    valid_indices = np.where(valid_surface)[0]

    ## DEBUG plot filtered_flux_array with matplotlib
    fig, ax = plt.subplots()
    ax.plot(valid_indices, linear_flux_array[valid_indices])
    ax.set_xlabel('Surface Index')
    ax.set_ylabel('Flux')
    ax.set_title('Filtered Flux Array')
    plt.show()

    # Create a meshgrid for the interpolation
    RADS = np.linspace(b_hidra.r_min, b_hidra.r_max, b_hidra.nr)
    THETAS = np.linspace(0, b_hidra.theta_max, b_hidra.ntheta+1) #add theta=0 for proper interpolation

    grid_theta, grid_rad = np.meshgrid(THETAS, RADS, indexing='ij')
    big_grid_linear = np.zeros([len(PHI_GENs), len(THETAS)-1, len(RADS)])

    ## LOOP THROUGH PHI ANGLES
    for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):
        # axes points
        points = np.zeros([MAX_SUBSETS+1,2])
        points[0] = axis_array[phi_index][0]
        points[1:] = island_axis_array[phi_index]

        print(f'Central axis point: {axis_array[phi_index][0]}')
        print(f'Island axis points: {points}')
        
        # linear values for the axes points
        flux_norm = np.ones([MAX_SUBSETS+1])
    
        ## LOAD SCATTER POINTS (POINCARE DATA)
        filename = 'Poincare_{:03d}.npy'.format(int(PHI_GEN_DEG))
        flux_surfaces = simIO.loadNumpyData(filename)

        ## GET LCFS POINTS
        lcfs_points = lcfs_points_full[phi_index][:NPHI].T# LCFS IS ONLY 1 SUBSET OF POINTS
        #print(f'LCFS points: {lcfs_points.shape=}, {lcfs_points[0]=}, {lcfs_points[1]=}')

        # # make a polar plot of the LCFS points
        # plt.figure()
        # plt.polar(lcfs_points[0], lcfs_points[1], 'o', markersize=1, label='LCFS Points')
        # plt.show()

        ## LOOP THROUGH SURFACES
        for surface_index in range(lcfs_index, N_surfaces):
            if valid_surface[surface_index] == False:
                print(f'Skipping surface {surface_index} (not valid)')
            else:
                ### GET VALUES
                thetas = flux_surfaces[surface_index][0]
                rads = flux_surfaces[surface_index][1]
                # filter NaNs
                thetas = thetas[~np.isnan(thetas)]
                rads = rads[~np.isnan(rads)]

                thetas = np.concatenate((thetas,thetas+np.pi*2,thetas-np.pi*2))
                rads = np.concatenate((rads,rads,rads))
                N_pts = len(thetas)

                # concatenate to big array of points
                these_points = np.array([thetas, rads]).T
                points = np.concatenate((points, these_points))

                these_flux_norms = np.full(N_pts, linear_flux_array[surface_index])
                flux_norm = np.concatenate((flux_norm, these_flux_norms))

        #grid_linear = griddata(points, flux_norm, (grid_theta, grid_rad), method='linear', fill_value=0.0, rescale=True)
        grid_linear = griddata(points, flux_norm, (grid_theta, grid_rad), method='linear', fill_value=0.0, rescale=True)
        #print(f'{grid_linear.shape=}')

        ## HACKY SOLUTIONS HERE!!!
        # copying values out for r=0.0
        fred3 = grid_linear.T[1]
        fred4 = grid_linear.T[2]
        fred3[fred3==0] = fred4[fred3==0]
        grid_linear.T[1] = fred3
        grid_linear.T[0] = grid_linear.T[1]

        # set all points outside the LCFS to zero
        for theta_index, this_theta in enumerate(THETAS):

            # find the index of the value in lcfs_points[0] closest to this_theta
            mintheta1 = np.abs(lcfs_points[0] - this_theta)
            mintheta2 = np.abs(lcfs_points[0] - this_theta + 2*np.pi)
            # calculate the minimum of the two
            mintheta3 = np.fmin(mintheta1, mintheta2)
            #print(f'{mintheta1.shape=}\n{mintheta2.shape=}\n{mintheta3.shape=}')
            lcfs_theta_index = np.argmin(mintheta3)
            #print(f'LCFS theta index for {this_theta} is {lcfs_theta_index} (theta={lcfs_points[0][lcfs_theta_index]})')
            lcfs_rad = lcfs_points[1][lcfs_theta_index]
            #lcfs_theta = lcfs_points[0][lcfs_theta_index]
            #print(f'Checking point at theta={THETAS[theta_index]}, r={radius}\nagainst LCFS theta={lcfs_theta}, r={lcfs_rad}')

            # Use boolean indexing to set all radii greater than (lcfs_rad - 0.01) to zero for this theta
            mask = RADS > (lcfs_rad + 0.001) # add buffer to avoid numerical issues
            grid_linear[theta_index][mask] = 0.0
            #print(f'Setting points at theta={THETAS[theta_index]}, r={RADS[mask]} to zero (outside LCFS)')

        # Add to big mesh array (3D)
        big_grid_linear[phi_index] = grid_linear[1:]  # skip the first row (theta=0) to match the shape of the b_hidra mesh

    #### END OF LOOP THROUGH PHI ANGLES ####

    # save numpy data using simIO method
    simIO.saveNumpyData(big_grid_linear, ANLYS_SUBDIR + '/big_grid_linear.npy')

    ## LOOP THROUGH PHI ANGLES for plotting
    for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):
        output_phi_plots(PHI_GEN_DEG, grid_theta, grid_rad, big_grid_linear[phi_index], 'LinearFluxNorm', ANLYS_SUBDIR, simIO, 'Blues', 0.0, 1.0)

def output_phi_plots(phi_deg, mesh_theta, mesh_rad, data, name, subdir, output_handler, colormap='inferno', plotmin=None, plotmax=None):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title(name + '\n$\phi_{{phy}}$={:02.0f}$\degree$ CW from North Split\n$\phi_c$={:02.0f}$\degree$'.format((phi_deg+198.)%360., phi_deg), loc='left')
    plot_data = np.vstack((data[-1], data))
    c = ax.pcolormesh(mesh_theta, mesh_rad, plot_data, shading='gouraud', cmap=colormap, vmin=plotmin, vmax=plotmax)

    ax.set_rmax(0.19)
    ax.set_rticks([])
    plt.grid(False)
    fig.colorbar(c, ax=ax, label='Flux')

    output_handler.saveFig(subdir + '/' + name +'_{:03d}deg.png'.format(int(phi_deg)), dpi=300)
    plt.close()


if __name__ == '__main__':
    #### DEFINE ANALYSIS PARAMETERS ####
    ## RUN DIRECTORY AND SUBDIRECTORY

    #ANLYS_DIR = "AcceptedIota3_1500spins_atole-9"

    # ANLYS_DIR = "AcceptedIota4_1500spins_atole-8_eng"
    # ANLYS_SUBDIR = "LCFS40_3x360x360mesh_UPDATED"

    ANLYS_DIR = "ChangeToIota3_1500spins_atole-9"
    ANLYS_SUBDIR = 'LCFS29_3x360x360mesh_CORRECTCURR_lotol'

    ## DEFINE FIELDS
    FIELD_FILE_TOR = 'input_files/It486_Ih000_Iv000_1p000_1p000_64bit.npy'
    FIELD_FILE_HEL = 'input_files/It000_Ih900_Iv000_1p000_1p000_64bit.npy'
    CURRENT_TOR = 0.486 #[kA]
    CURRENT_HEL = 0.900 #[kA]
    CONFIG_TOR = 'default_toroidal'
    CONFIG_HEL = 'default_helical_rev'

    ## DEFINE LCFS AND ANGLES TO EVALUATE
    LCFS_INPUT = 29 #22 #29?
    NPHI = 360
    NTHETA = 360
    PHI_GENs = np.linspace(360//NPHI, 360, NPHI)
    MAX_SUBSETS = 3
    SMALLEST_ISLAND_INDEX = 47 #53 #39

    main()