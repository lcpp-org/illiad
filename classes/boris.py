import logging
from time import perf_counter
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
from math import degrees

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout':True})

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utility.coordtrans import XYZ_to_RTP, RTP_to_XYZ
#from classes.particle import FieldLine

class Boris():
    """Class to handle Boris analysis of magnetic field lines."""
    def __init__(self, io_handler, anlys_name='Boris', tag=None):
        """Initializes the Boris class with the specified solver parameters and writes to log.

        Args:
            io_handler: An object responsible for handling output operations, such as logging and directory creation.
            anlys_name (str, optional): Name of the analysis or subdirectory. Defaults to 'Poincare'.
        """
        self.IO = io_handler
        self.anlys_name = anlys_name
        self.solver = 'boris_buneman'
        # self.dt = dt
        # self.tmax = tmax
        # self.nsteps = int(tmax // dt) + 1
        self.tag = tag

        self.IO.createSubDir(anlys_name)
        self.IO.log.info("+----------------+-------------------------+")
        self.IO.log.info("| Parameter      | Value                   |")
        self.IO.log.info("+----------------+-------------------------+")
        self.IO.log.info(f"| SOLVER         | {self.solver:<23} |")
        self.IO.log.info(f"| ANLYS_NAME     | {str(self.anlys_name):<23} |")
        self.IO.log.info(f"| TAG            | {str(self.tag):<23} |")
        self.IO.log.info("+----------------+-------------------------+")

    def set_conditions(self, ion_list, dt=1e-8, tmax=1e-3):
        """Sets the initial conditions and events for Poincare analysis.

        Args:
            ion_list (list): List of Ion objects containing initial conditions and properties.
            dt (float, optional): Time step for the simulation. Defaults to 1e-8.
            tmax (float, optional): Maximum time for the simulation. Defaults to 1e-3.

        Returns:
            None
        """
        self.dt = dt
        self.tmax = tmax
        self.nsteps = int(tmax // dt) + 1

        self.IO.log.info("+----------------+-------------------------+")
        self.IO.log.info(f"| DT             | {self.dt:<23} |")
        self.IO.log.info(f"| TMAX           | {self.tmax:<23} |")
        self.IO.log.info(f"| NSTEPS         | {self.nsteps:<23} |")
        self.IO.log.info("+----------------+-------------------------+")

    def parallel_solver(self, ions, Bfield, Efield=None, trace_IDs=[]):
        """
        Function to take in a particle and field object and solves the particle path until termination event or tmax
        using a fixed-step Boris-Buneman Solver, based on (Birdsall, 4-3&4).

        Parameters:
            -ions (list): List of ion objects containing initial conditions and properties.
            -Bfield (object): Magnetic field object providing field interpolation methods.
            -Efield (object, optional): Electric field object providing field interpolation methods. Defaults to None.
            -track_ID (list, optional): List of particle IDs to track. Defaults to [10, 20].
        Returns:
            -wallPts (torch.Tensor): XYZ Positions where particles terminate (e.g., hit the wall), shape (Nparticles, 3).
            -wallVelocities (torch.Tensor): Velocities of particles at termination, shape (Nparticles, 3).
            -maxStep (torch.Tensor): Step index at which each particle terminated, shape (Nparticles,).
        """
        log = logging.getLogger()
        log.info('Start ICs: {}-{}'.format(ions[0].particleID, ions[-1].particleID))

        t_startInd = perf_counter()
        Nparticles = len(ions)
        trace_output = torch.zeros([self.nsteps+1, len(trace_IDs), 3], dtype=torch.float64, device=device)
        with torch.no_grad():
            wallPts = torch.zeros([Nparticles, 3], dtype=torch.float64, device=device)
            wallVelocities = torch.zeros([Nparticles, 3], dtype=torch.float64, device=device)
            maxStep = torch.zeros(Nparticles, dtype=torch.int, device=device)

            tvec = torch.empty([Nparticles, 3], dtype=torch.float64, device=device)
            qdt2m = torch.tensor([ion.charge_mass_ratio * self.dt / 2 for ion in ions], dtype=torch.float64, device=device)
            v_k = torch.tensor(np.array([ion.vel0_XYZ for ion in ions]), dtype=torch.float64, device=device)

            [ion.setPosition(0, ion.pos0_XYZ) for ion in ions]
            pos_k = torch.tensor(np.array([ion.pos0_XYZ for ion in ions]), dtype=torch.float64, device=device)

            # NEED v_n-1/2 TO START
            if Efield:
                Evec = (Efield.interpField(pos_k) * qdt2m).T
            else:
                Evec = torch.zeros([Nparticles, 3], dtype=torch.float64, device=device)

            tvec = (Bfield.interpField(pos_k) * qdt2m).T
            tmag = torch.linalg.norm(tvec, axis=-1)

            vminus = v_k + Evec
            vprime = vminus + torch.linalg.cross(vminus, tvec)
            svec = 2 * tvec / (1 + (tmag * tmag)[:, None])
            vplus = vminus - torch.linalg.cross(vprime, svec) / 2
            v_k = vplus + Evec

            x2 = pos_k.T[0] * pos_k.T[0]
            y2 = pos_k.T[1] * pos_k.T[1]
            z2 = pos_k.T[2] * pos_k.T[2]
            r_k = torch.sqrt(x2 + y2 + z2 + Bfield.R0 * Bfield.R0
                              - 2 * Bfield.R0 * torch.sqrt(x2 + y2))

            running = torch.arange(0, Nparticles, 1, dtype=torch.int, device=device)
            Nrunning = Nparticles

            # ADD SELECTED PARTICLE TRACING
            trace_output[0] = pos_k[trace_IDs]

            log.info('START STEPPING...')
            logging.basicConfig(level=logging.INFO)
            with logging_redirect_tqdm(loggers=[log]):
                pbar = tqdm(range(1, self.nsteps), ncols=100, mininterval=2.0)
                for k in pbar:
                    if Efield:
                        Evec[running] = (Efield.interpField(pos_k[running]) * qdt2m[running]).T
                    tvec[running] = (Bfield.interpField(pos_k[running]) * qdt2m[running]).T
                    tmag[running] = torch.linalg.norm(tvec[running], axis=-1)

                    vminus[running] = v_k[running] + Evec[running]
                    vprime[running] = vminus[running] + torch.linalg.cross(vminus[running], tvec[running])
                    svec[running] = 2 * tvec[running] / (1 + (tmag[running] * tmag[running])[:, None])
                    vplus[running] = vminus[running] + torch.linalg.cross(vprime[running], svec[running])
                    v_k[running] = vplus[running] + Evec[running]

                    pos_k[running] = pos_k[running] + v_k[running] * self.dt

                    # ADD SELECTED PARTICLE TRACING
                    trace_output[k] = pos_k[trace_IDs]

                    x2[running] = pos_k[running].T[0]**2
                    y2[running] = pos_k[running].T[1]**2
                    z2[running] = pos_k[running].T[2]**2
                    r_k[running] = torch.sqrt(x2[running] + y2[running] + z2[running] + Bfield.R0 * Bfield.R0
                                               - 2 * Bfield.R0 * torch.sqrt(x2[running] + y2[running]))

                    running = torch.where(r_k < Bfield.a)[0]

                    maxStep[running] = k # +1?
                    Nrunning = running.size(0)
                    if Nrunning == 0:
                        log.info('All particles terminated at step {}'.format(k))
                        break

                    pbar.set_postfix({'#Particles running': Nrunning}, refresh=False)

            terminated = torch.where(r_k >= Bfield.a)[0]
            wallPts[terminated] = pos_k[terminated]
            wallVelocities[terminated] = v_k[terminated]

        t_stopInd = perf_counter()
        elapsed_timeInd = t_stopInd - t_startInd
        min_, sec_ = divmod(elapsed_timeInd, 60)
        hr_, min_ = divmod(min_, 60)

        log.info(
            'ELAPSED TIME({} Particles): {:02.0f}H:{:02.0f}M:{:02.3f}S'.format(
                Nparticles, hr_, min_, sec_
            )
        )

        return wallPts, wallVelocities, maxStep, trace_output

    def single_solver(self, particle):
        pass

    def post_solver(self, solver_output):
        """Processes the solver output to extract path lengths and Poincare data,
        and prepares the data for plotting and output.

        Args:
            solver_output (iterator): The output from the solver, containing tuples of path lengths and event data.

        Returns:
            tuple: (path_lengths, poincare_points, wall_points)
                path_lengths (list): List of path lengths for each particle.
                poincare_points (list): List of Poincare data for each particle.
                wall_points (list): List of wall intersection data for each particle.
        """
        # ## PARSE OUTPUT INTO LISTS
        # path_lengths = []
        # poincare_points = []
        # wall_points = []
        # for pLngth, out in solver_output:
        #     path_lengths += [pLngth]
        #     poincare_points += [out[1:]]
        #     if isinstance(out[0], np.ndarray) and out[0].any():
        #         wall_points += [XYZ_to_RTP(out[0][0], self.field.R0)]

        # if self.double_line:
        #     # Combine the positive and negative fieldlines into one
        #     path_lengths = [path_lengths[i]+path_lengths[i+self.nlines] for i in range(0,self.nlines)]
        #     for line_index in range(0,self.nlines):
        #         for event_index in range(len(poincare_points[line_index])):
        #             arr_a = poincare_points[line_index][event_index]
        #             arr_b = poincare_points[line_index+self.nlines][event_index]
        #             if arr_a.any() and arr_b.any():
        #                 poincare_points[line_index][event_index] = np.vstack((arr_a, arr_b))
        #     poincare_points = poincare_points[:self.nlines]

        # self.IO.log.info('PLOTTING AND OUTPUTTING PHI-ANGLE DATA:')
        # save_output_partial = partial(self.save_output, xyz_list=poincare_points, saveData=True)
        # plot_workers = min(self.workers, 16)
        # iter_in = enumerate(self.plot_angles)
        # with cf.ProcessPoolExecutor(max_workers=plot_workers) as executor:
        #     list(executor.map(save_output_partial, iter_in))

        # return path_lengths, poincare_points, wall_points

    def save_output(self, iter, xyz_list, saveData=True):
        """
        Generates and saves Poincare plots and associated data for a given phi angle, and logs the operation.
            
        Args:
            iter (tuple): A tuple containing the index and phi angle (in radians).
            xyz_list (list): A list of Poincare data arrays for each particle.
            saveData (bool, optional): If True, saves the computed data to disk. Defaults to True.
        
        Returns:
            None

        Side Effects:
            - Saves plot images and data files to disk.
            - Logs the phi angle information.
        """
        num_sets = len(xyz_list)
        rminor = self.field.a
        rmajor = self.field.R0
        n, phi = iter
        phi_deg = phi*180/np.pi
        plt.rcParams.update({'font.size': 10})
        plt.rcParams.update({'figure.autolayout':True})

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)

        maxLength = max(len(xyz_list[i][n]) for i in range(num_sets))
        radtheta_pts = np.full([num_sets, 2, maxLength], fill_value=np.nan)
        for i in range(num_sets):
            xyz_points = xyz_list[i][n]
            point_total = max(0, len(xyz_points)-1)
            for j in range(point_total):
                radtheta_pts[i][1][j], radtheta_pts[i][0][j] = XYZ_to_RTP(xyz_points[j][:3], rmajor)[:2]
            plt.scatter(radtheta_pts[i][0][:point_total], radtheta_pts[i][1][:point_total], marker='.', s=1.00, c='k', linewidths=0.0)

        if saveData:
            fname = self.anlys_name + '_{:03.0f}'.format(degrees(phi))
            self.IO.saveNumpyData(radtheta_pts, fname)

        ax.set_rmax(rminor)
        ax.set_rticks(np.arange(0.0, rminor, 0.02))
        ax.yaxis.set_tick_params(labelsize=5)
        ax.grid(linewidth = 0.25, linestyle=':', c='k')
        phi_phys = (phi + (198 * np.pi/180.)) % (2*np.pi)
        phi_phys_deg = phi_phys*180/np.pi
        plt.title('$\phi_{{phy}}$={:02.0f}$\degree$ CW from North Split\n$\phi_c$={:02.0f}$\degree$'.format(phi_phys_deg, phi_deg), loc='left')
        plot_name = self.anlys_name +'/'+ self.anlys_name + '_phi={:03.0f}.png'.format(phi_deg)
        self.IO.saveFig(plot_name, dpi=250)
        plt.close()
        self.IO.log.info('\tPHI: {:.2f} degrees'.format(phi_deg))

    def run(self):
        """Generates Poincare plots based on the initial conditions and magnetic field.

        Returns:
            tuple: (pathLength_test, Poincare_output_test, wall_output_test)
            pathLength_test (list): List of path lengths for each particle.
            Poincare_output_test (list): List of Poincare data for each particle.
            wall_output_test (list): List of wall intersection data for each particle.
        """
        solv_out = self.parallel_solver()
        pathLength, Poincare_output, wall_output = self.post_solver(solv_out)

        return pathLength, Poincare_output, wall_output
