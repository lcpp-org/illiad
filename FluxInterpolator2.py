## IN THIS FILE, WE WILL TAKE THE CALCULATED FLUX FOR EACH SURFACE,
## AND APPLY IT TO THE POINTS ON THE THEIR ESPECTIVE SURFACE, 
## AND THEN INTERPOLATE IT ONTO A MESH THE SAME SHAPE/RESOLUTION AS THE BFIELD MESH
import logging
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
    b_hidra.loadCartesianField(FIELD_FILE_TOR, errField=True, att_mult=FIELD_SCALE_TOR)
    b_hidra.addFieldPerturbation(FIELD_FILE_HEL, att_mult=FIELD_SCALE_HEL)
    b_hidra.set_nonPer_errField(FIELD_ERR_MAG, FIELD_ERR_DIR)
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

    # Load Magnetic Axis point:
    filename_center = filepath + 'fSurf_{:03d}_center.npy'.format(N_surfaces-1)
    axis_array = simIO.loadNumpyData(filename_center)

    # Choosing one 'well-behaved' angle for the calculation (no failed calculations)
    filtered_flux_array = flux_norm_array[:, 6] #6, 13, 19, 20, 22, 24, 

    grid_theta, grid_rad = np.meshgrid(THETAS, RADS, indexing='ij')

    ## LOOP THROUGH PHI ANGLES
    for phi_index, PHI_GEN_DEG in enumerate(PHI_GENs):
        #phi_index = 9 ## For TESTING one Phi angle!!!

        # axis point
        points = np.zeros([1,2])
        points[0] = axis_array[phi_index][0]
        values = np.array([1.0])

        ## LOAD POINCARE DATA
        filename = 'Poincare_{:03d}.npy'.format(int(PHI_GEN_DEG))
        flux_surfaces = simIO.loadNumpyData(filename)
        #print(f'{flux_surfaces.shape=}')

        ## LOOP THROUGH SURFACES
        for surface_index in range(lcfs_index, N_surfaces):

            # get values
            thetas = flux_surfaces[surface_index][0]
            rads = flux_surfaces[surface_index][1]
            # filter NaNs
            thetas = thetas[~np.isnan(thetas)]
            rads = rads[~np.isnan(rads)]
            N_pts = len(thetas)
            fluxNorms = np.full(N_pts, filtered_flux_array[surface_index])

            these_points = np.array([thetas, rads]).T
            these_values = fluxNorms

            points = np.concatenate((points, these_points))
            values = np.concatenate((values, these_values))

            these_points_2 = these_points
            these_points_2.T[0] += 2*np.pi
            these_values_2 = these_values
            # these_points_3 = these_points
            # these_points_3.T[0] -= 2*np.pi
            # these_values_3 = these_values
            points = np.concatenate((points, these_points_2))
            values = np.concatenate((values, these_values_2))

        grid_test = griddata(points, values, (grid_theta, grid_rad), method='linear', fill_value=0.0)

        ## HACKY SOLUTIONS HERE
        # averaging out for theta=2pi
        grid_test[-1] = (grid_test[-2] + grid_test[0]) / 2
        # copying values out for r=0.0
        fred1 = grid_test.T[1]
        fred2 = grid_test.T[2]
        fred1[fred1==0] = fred2[fred1==0]
        grid_test.T[1] = fred1
        grid_test.T[0] = grid_test.T[1]
        #print(f'{grid_test.shape=}')

        # QUICK 'n DIRTY GRADIENT CALCULATION
        print(f'{grid_test.shape=}')
        print(f'{RADS.shape=}')
        print(f'{THETAS.shape=}')
        flux_grad = np.gradient(grid_test, THETAS, RADS)#, [grid_rad, grid_theta])
        flux_grad_radial = -flux_grad[1]  # E = -grad[V]
        flux_grad_poloidal = -flux_grad[0]
        flux_grad_magnitude = np.sqrt(flux_grad_radial**2 + flux_grad_poloidal**2)
        
        # PLOT THE SURFACE PARAM AS A POLAR PLOT
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_title('Normalized Flux\n$\phi_{{phy}}$={:02.0f}$\degree$ CW from North Split\n$\phi_c$={:02.0f}$\degree$'.format((PHI_GEN_DEG+198.)%360., PHI_GEN_DEG), loc='left')
        c = ax.pcolormesh(grid_theta, grid_rad, grid_test, shading='auto', cmap='inferno')

        ax.set_rmax(b_hidra.a)
        ax.set_rticks(np.arange(0.0, b_hidra.a, 0.02))
        fig.colorbar(c, ax=ax, label='Flux')
        simIO.saveFig(ANLYS_SUBDIR+'/FluxContour_{:03d}deg.png'.format(int(PHI_GEN_DEG)), dpi=300)
        #plt.show()

        # PLOT THE GRADIENT AS A POLAR PLOT
        fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
        ax2.set_title('Flux Gradient Magnitude\n$\phi_{{phy}}$={:02.0f}$\degree$ CW from North Split\n$\phi_c$={:02.0f}$\degree$'.format((PHI_GEN_DEG+198.)%360., PHI_GEN_DEG), loc='left')
        c = ax2.pcolormesh(grid_theta, grid_rad, flux_grad_magnitude, shading='auto', cmap='inferno') #, vmin=0, vmax=9)

        ax2.set_rmax(b_hidra.a)
        ax2.set_rticks(np.arange(0.0, b_hidra.a, 0.02))
        fig2.colorbar(c, ax=ax2, label='Flux Gradient')
        simIO.saveFig(ANLYS_SUBDIR+'/FluxGrad2_{:03d}deg.png'.format(int(PHI_GEN_DEG)), dpi=300)
        #plt.show()


if __name__ == '__main__':
    #### DEFINE ANALYSIS PARAMETERS ####
    ## RUN DIRECTORY AND SUBDIRECTORY
    ANLYS_DIR = "AcceptedIota3_1500spins_atole-9"
    ANLYS_SUBDIR = 'LCFS22_3x180x60mesh_FluxTest4-NEW4_epsabs1e-5_epsrel=1e-3'

    ## DEFINE FIELDS
    FIELD_FILE_TOR = 'input_files/It486_Ih000_Iv000_1p000_1p000_64bit.npy'
    FIELD_SCALE_TOR = 0.9452
    FIELD_FILE_HEL = 'input_files/It000_Ih900_Iv000_1p000_1p000_64bit.npy'
    FIELD_SCALE_HEL = -0.955 * FIELD_SCALE_TOR
    FIELD_ERR_MAG = 1.5939e-4 #3.168e-4
    FIELD_ERR_DIR = np.radians(272.)

    ## IDENTIFY LAST-CLOSED FLUX SURFACE
    LCFS_INPUT = 22
    ## DEFINE ANGLES TO EVALUATE AND PLOT
    NPHI = 180
    NTHETA = 60

    PHI_GENs = np.linspace(360//NPHI, 360, NPHI)
    #PHI_GENs = np.array([18])
    MAX_SUBSETS = 3

    ## PLOTTING FLAG
    ##PLOT_ALL = True

    main()