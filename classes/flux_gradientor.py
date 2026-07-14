import numpy as np
import matplotlib.pyplot as plt
from utility.coordtrans import RTP_XYZ_JAC


class FluxGradientor:
    """Class to handle flux gradient calculations.

    This class loads the interpolated normalized flux field, computes
    E = -grad[flux] on the RTP grid, filts values outside the LCFS,
    converts the RTP vector components to XYZ coordinates, and saves the
    resulting field array.

    """


    def __init__(self, IO_handler, b_hidra, input_params):
        """Initialize the FluxGradientor.

        Args:
            IO_handler: An object responsible for handling output operations, such as logging and directory creation.
            b_hidra: Magnetic field mesh/object.  
            input_params (dict): Dictionary of flux calculation input parameters. 
        """
        self.simIO = IO_handler
        self.field = b_hidra
        self.input_params = input_params

        # Input parameters
        self.anlys_subdir = input_params['ANLYS_SUBDIR']
        self.output_file_name = input_params['OUTPUT_FILE_NAME']
        self.phi_gens = input_params['PHI_GENs']
        self.lcfs_index = input_params['LCFS_INDEX']

        # Coordinate arrays
        self.rads = None
        self.thetas = None
        self.grid_rad = None
        self.grid_theta = None

        # Gradient arrays in RTP coordinates
        self.flux_grad_radial = None
        self.flux_grad_poloidal = None
        self.flux_grad_toroidal = None
        self.flux_grad_magnitude = None

        # Final output array in XYZ coordinates
        self.efield_xyz_array_linear = None


    def build_coordinate_grid(self):
        """Build the radial and poloidal coordinate grids used by the gradient."""
        rads = np.linspace(self.field.r_min, self.field.r_max, self.field.nr)
        thetas = np.linspace(self.field.theta_min, self.field.theta_max, self.field.ntheta)
        self.grid_theta, self.grid_rad = np.meshgrid(thetas, rads, indexing='ij')
        self.rads = rads
        self.thetas = thetas



    def calculate_gradients(self):
        """Calculate the RTP components of -grad(flux).

        Stores:
            self.flux_grad_radial: Radial component of -grad(flux).
            self.flux_grad_poloidal: Poloidal component of -grad(flux).
            self.flux_grad_toroidal: Toroidal component of -grad(flux).
        """

        # GRADIENT CALCULATION: remember to divide by Jacobian determinant:
        # gradF = [dF/dr] * R_HAT + [(1/r) * df/dtheta] * THETA_HAT + [( 1/(R0+rcos(theta)) ) * df/dphi] * PHI_HAT
        density_grid = self.load_density_data()
        self.simIO.log.info("## Starting flux gradient calculation. ##")
        flux_gradient = np.gradient(density_grid, self.phi_gens, self.thetas, self.rads, edge_order=2)#, [grid_rad, grid_theta])
        self.simIO.log.info("## Flux gradient calculation complete. ##")

        # Calculate RTP basis vectors and apply the Jacobian factors to get the physical gradient in each direction
        flux_grad_radial = -flux_gradient[2]  # E = -grad[V]

        flux_grad_poloidal = np.zeros_like(flux_gradient[1])
        flux_grad_poloidal[:,:,1:] = -flux_gradient[1][:,:,1:] / self.grid_rad[:,1:]

        flux_grad_toroidal = -flux_gradient[0] / (self.field.R0 + self.grid_rad * np.cos(self.grid_theta))
        self.simIO.log.info(f'{flux_grad_radial.shape=}')
        self.flux_grad_radial = flux_grad_radial
        self.flux_grad_poloidal = flux_grad_poloidal
        self.flux_grad_toroidal = flux_grad_toroidal


    def load_density_data(self):
        """Load the interpolated normalized flux field."""
        return self.simIO.loadNumpyData(self.anlys_subdir + '/' + 'nField_' + self.output_file_name + '.npy') #'/big_grid_linear.npy')

    
    def filter_gradients(self):
        """
        For each phi plane, this method loads the LCFS Poincare points,
        finds the LCFS radius closest to each theta grid location, and zeros the
        radial, poloidal, and toroidal gradient components for radii outside the
        LCFS plus a small buffer.

        Modifies:
            self.flux_grad_radial: Values outside the LCFS are set to zero.
            self.flux_grad_poloidal: Values outside the LCFS are set to zero.
            self.flux_grad_toroidal: Values outside the LCFS are set to zero.
        """
        # set all points outside the LCFS to zero
        for phi_index, phi_gen_deg in enumerate(self.phi_gens):
            th_in, r_in = self.load_poincare_data(phi_gen_deg)

            for theta_index, this_theta in enumerate(self.thetas):
                # find the index of the value in lcfs_points[0] closest to this_theta
                mintheta1 = np.abs(th_in - this_theta) #lcfs_points[0]
                mintheta2 = np.abs(th_in - this_theta + 2*np.pi) #lcfs_points[0]

                # calculate the minimum of the two
                mintheta3 = np.fmin(mintheta1, mintheta2)
                lcfs_theta_index = np.argmin(mintheta3)
                lcfs_rad = r_in[lcfs_theta_index] #lcfs_points[1]

                # Use boolean indexing to set all radii greater than (lcfs_rad - 0.01) to zero for this theta
                mask = self.rads > (lcfs_rad + 0.01) # add buffer to avoid numerical issues
                self.flux_grad_radial[phi_index][theta_index][mask] = 0.0
                self.flux_grad_poloidal[phi_index][theta_index][mask] = 0.0
                self.flux_grad_toroidal[phi_index][theta_index][mask] = 0.0


    def load_poincare_data(self, phi_gen_deg):
        """Load LCFS Poincare data for one toroidal angle."""
        filename = 'Poincare_{:03d}.npy'.format(int(phi_gen_deg))
        lcfs_points = self.simIO.loadNumpyData(filename, subdir="Poincare")[self.lcfs_index]
        th_in, r_in = lcfs_points
        r_in = r_in[~np.isnan(r_in)]
        th_in = th_in[~np.isnan(th_in)]
        
        return th_in, r_in
        

    def calculate_gradient_magnitude(self):
        """Calculate the magnitude of the flux gradient."""
        self.flux_grad_magnitude = np.sqrt(self.flux_grad_radial**2 
                                           + self.flux_grad_poloidal**2 
                                           + self.flux_grad_toroidal**2)


    def convert_gradient_to_xyz(self):
        """Convert the RTP gradient components to XYZ coordinates."""
        # reshape the arrays to match the dimensions of input B-fields
        reshaped_flux_grad_radial = np.transpose(self.flux_grad_radial, (2, 1, 0))
        reshaped_flux_grad_poloidal = np.transpose(self.flux_grad_poloidal, (2, 1, 0))
        reshaped_flux_grad_toroidal = np.transpose(self.flux_grad_toroidal, (2, 1, 0))
        Efield_rtpArray_linear = np.array([reshaped_flux_grad_radial, reshaped_flux_grad_poloidal, reshaped_flux_grad_toroidal])


        self.efield_xyz_array_linear = np.zeros_like(Efield_rtpArray_linear)
        xform_rad, xform_theta, xform_phi= np.meshgrid(self.rads, self.thetas, self.phi_gens, indexing='ij')

        flattened_shape = self.efield_xyz_array_linear[0].flatten().shape
        Ex_linear = np.zeros(flattened_shape)
        Ey_linear = np.zeros(flattened_shape)
        Ez_linear = np.zeros(flattened_shape)
        for i, (rad, theta, phi, E_rad_lin, E_theta_lin, E_phi_lin) in enumerate(zip(xform_rad.flatten(),
                                                                            xform_theta.flatten(),
                                                                            xform_phi.flatten(),
                                                                            reshaped_flux_grad_radial.flatten(),
                                                                            reshaped_flux_grad_poloidal.flatten(), 
                                                                            reshaped_flux_grad_toroidal.flatten())):
            ErtpLin = np.array([E_rad_lin, E_theta_lin, E_phi_lin])
            p_RTP = np.array([rad, theta, np.radians(phi)])
            Ex_linear[i], Ey_linear[i], Ez_linear[i] = RTP_XYZ_JAC(p_RTP, ErtpLin, form='rtp2xyz')

        # reshape the arrays to match the dimensions of input Bfields
        self.efield_xyz_array_linear[0] = Ex_linear.reshape(self.efield_xyz_array_linear[0].shape)
        self.efield_xyz_array_linear[1] = Ey_linear.reshape(self.efield_xyz_array_linear[1].shape)
        self.efield_xyz_array_linear[2] = Ez_linear.reshape(self.efield_xyz_array_linear[2].shape)
    

    def save_and_plot_data(self):
        """Save the XYZ field array gradient plots."""
        self.simIO.saveNumpyData(self.efield_xyz_array_linear, self.anlys_subdir + '/' + 'Efield_' + self.output_file_name + '.npy')

        # loop through PHI ANGLES for plotting
        colortest = 'afmhot_r'
        for phi_index, phi_gen_deg in enumerate(self.phi_gens):
            self.output_phi_plots(phi_gen_deg, self.flux_grad_magnitude[phi_index], 'FluxGradMagnitude', self.anlys_subdir, self.simIO, colortest, 0.0, 200.0)
            self.output_phi_plots(phi_gen_deg, self.flux_grad_radial[phi_index], 'FluxGradRadial', self.anlys_subdir, self.simIO, colortest, -200., 200)
            self.output_phi_plots(phi_gen_deg, self.flux_grad_poloidal[phi_index], 'FluxGradPoloidal', self.anlys_subdir, self.simIO, colortest, -100.0, 100.0)
            self.output_phi_plots(phi_gen_deg, self.flux_grad_toroidal[phi_index], 'FluxGradToroidal', self.anlys_subdir, self.simIO, colortest, -0.3, 0.3)

        self.simIO.log.info("## Flux gradienting complete. ##")


    def output_phi_plots(self, phi_deg, data, name, subdir, output_handler, colormap='inferno', plotmin=None, plotmax=None):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        data = np.vstack((data[-1], data))
        plot_grid_rad = np.vstack((self.grid_rad[-1], self.grid_rad))
        plot_grid_theta = np.vstack((self.grid_theta[-1], self.grid_theta))
        plot_grid_theta[0] = 0

        c = ax.pcolormesh(plot_grid_theta, plot_grid_rad, data, shading='gouraud', cmap=colormap, vmin=plotmin, vmax=plotmax)

        ax.set_title(name + '\n$\phi_{{phy}}$={:02.0f}$\degree$ CW from North Split\n$\phi_c$={:02.0f}$\degree$'.format((phi_deg+198.)%360., phi_deg), loc='left')
        ax.set_rmax(0.19)
        ax.set_rticks([])
        # set the r-labels to an empty list
        ax.set_yticklabels([])
        fig.colorbar(c, ax=ax, label='Flux')
        plt.grid(False)

        output_handler.saveFig(subdir + '/' + name +'_{:03d}deg.png'.format(int(phi_deg)), dpi=250)
        output_handler.log.info('Saved figure: ' + subdir + '/' + name +'_{:03d}deg.png'.format(int(phi_deg)))
        plt.close("All")


    def run(self):
        """Run the full flux gradient calculation workflow."""
        self.build_coordinate_grid()
        self.calculate_gradients()
        self.filter_gradients()
        self.calculate_gradient_magnitude()
        self.convert_gradient_to_xyz()
        self.save_and_plot_data()