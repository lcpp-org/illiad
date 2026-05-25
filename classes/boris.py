import logging
from time import perf_counter
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
from math import degrees

import matplotlib.pyplot as plt

from plot_funcs import plotFuncs
plt.rcParams.update({'font.size': 10})
#plt.rcParams.update({'figure.autolayout':True})

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utility.coordtrans import XYZ_to_RTP, RTP_XYZ_JAC#, RTP_to_XYZ
#from plot_funcs import plotFuncs


class Boris():
    """Class to handle Boris analysis of magnetic field lines."""
    def __init__(self, io_handler, anlys_name='Boris', tag=None):
        """Initializes the Boris class with the specified solver parameters and writes to log.

        Args:
            io_handler: An object responsible for handling output operations, such as logging and directory creation.
            anlys_name (str, optional): Name of the analysis or subdirectory. Defaults to 'Poincare'.
        """

        # attach plotting function to class instance
        for name in dir(plotFuncs):
                    func = getattr(plotFuncs, name)
                    if callable(func) and not name.startswith("__"):
                        if name.startswith("global_"):
                            new_name = name.replace("global_", "")  # Remove prefix
                        elif name.startswith("boris_"):
                            new_name = name.replace("boris_", "")  # Remove prefix
                            setattr(self, new_name, func)  # Attach to the instance with the new name

        self.IO = io_handler
        self.anlys_name = anlys_name
        self.solver = 'boris_buneman'
        self.tag = tag

        # self.IO.createSubDir(anlys_name)
        # self.IO.log.info("+----------------+-------------------------+")
        # self.IO.log.info("| Parameter      | Value                   |")
        # self.IO.log.info("+----------------+-------------------------+")
        # self.IO.log.info(f"| SOLVER         | {self.solver:<23} |")
        # self.IO.log.info(f"| ANLYS_NAME     | {str(self.anlys_name):<23} |")
        # self.IO.log.info(f"| TAG            | {str(self.tag):<23} |")
        # self.IO.log.info("+----------------+-------------------------+")


    def setConditions(self, ion_list, cond_string, dt=1e-8, tmax=1e-3):
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
        self.ion_list = ion_list
        self.cond_string = cond_string
        ## SET OUTPUT
        # for ion in ion_list:
        #     ion.initOutput(dt, tmax)

        # self.IO.log.info("+----------------+-------------------------+")
        # self.IO.log.info(f"| DT             | {self.dt:<23} |")
        # self.IO.log.info(f"| TMAX           | {self.tmax:<23} |")
        # self.IO.log.info(f"| NSTEPS         | {self.nsteps:<23} |")
        # self.IO.log.info("+----------------+-------------------------+")

    def parallel_solver(self, ions, Bfield, Efield=None, trace_IDs=[], freq_corr=False):
        """
        Function to take in a particle and field object and solves the particle path until termination event or tmax
        using a fixed-step Boris-Buneman Solver, based on (Birdsall, 4-3&4).

        Parameters:
            -ions (list): List of ion objects containing initial conditions and properties.
            -Bfield (object): Magnetic field object providing field interpolation methods.
            -Efield (object, optional): Electric field object providing field interpolation methods. Defaults to None.
            -track_ID (list, optional): List of particle IDs to track. Defaults to [10, 20].
            -freq_corr (bool, optional): Flag to enable frequency correction. Defaults to False.
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

            if freq_corr:
                Bvec = torch.empty([Nparticles, 3], dtype=torch.float64, device=device)
                Bvec = Bfield.interpField(pos_k).T
                Bmag = torch.linalg.norm(Bvec, axis=-1)
                Bhat = Bvec / Bmag[:, None]
                tvec = torch.tan(qdt2m * Bmag)[:, None] * Bhat
            else:
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
                    pos_active = pos_k[running]
                    qdt2m_active = qdt2m[running]
                    v_k_active = v_k[running]

                    actv_weights, actv_corner_indices, actv_ph_localN = Bfield.get_weights(pos_active)
                    b_vecs_active = Bfield.return_vecs(actv_weights, actv_corner_indices, actv_ph_localN)

                    if freq_corr:
                        Bmag_active = torch.linalg.norm(b_vecs_active, axis=-1)
                        Bhat_active = b_vecs_active / Bmag_active[:, None]
                        tvec_active = torch.tan(qdt2m_active * Bmag_active)[:, None] * Bhat_active
                    else:
                        tvec_active = (b_vecs_active * qdt2m_active).T

                    tmag_active = torch.linalg.norm(tvec_active, axis=-1)

                    if Efield:
                        sector = torch.remainder(actv_ph_localN.to(torch.long), Bfield.periodicity[2])
                        phi_offset = sector.unsqueeze(0) * Bfield.nphi
                        e_phi_idx = actv_corner_indices[2] + phi_offset
                        # e_corner_indices = actv_corner_indices.clone()
                        # e_corner_indices[2] = e_phi_idx

                        #e_vecs_active = Efield.return_vecs(actv_weights, e_corner_indices, torch.zeros_like(actv_ph_localN))
                        e_vecs_active = Efield.return_vecs(actv_weights, torch.stack([actv_corner_indices[0], actv_corner_indices[1], e_phi_idx]), ph_localN=None)
                        Evec_active = (e_vecs_active * qdt2m_active).T

                    vminus_active = v_k_active + Evec_active
                    vprime_active = vminus_active + torch.linalg.cross(vminus_active, tvec_active)
                    svec_active = 2 * tvec_active / (1 + (tmag_active * tmag_active)[:, None])
                    vplus_active = vminus_active + torch.linalg.cross(vprime_active, svec_active)
                    
                    v_k_active = vplus_active + Evec_active
                    pos_k[running] = pos_active + v_k_active * self.dt
                    v_k[running] = v_k_active

                    # ADD SELECTED PARTICLE TRACING
                    trace_output[k] = pos_k[trace_IDs]

                    x2 = pos_k.T[0]**2
                    y2 = pos_k.T[1]**2
                    z2 = pos_k.T[2]**2
                    r_k = torch.sqrt(x2 + y2 + z2 + Bfield.R0 * Bfield.R0
                                               - 2 * Bfield.R0 * torch.sqrt(x2 + y2))

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

    def post_solver(self, solver_output, Bfield):
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
        ## SOME PHYSICAL CONSTANTS
        #kg_per_amu = 1.660_539_068E-27
        kboltz = 1.602_176_634E-19 # Joules/eV

        wallPts_, wallVelocities_, maxStep_, trace_output_ = solver_output

        tic = perf_counter()
        wallPt_output = wallPts_.cpu().numpy()
        velocity_output = wallVelocities_.cpu().numpy()
        max_timeStep = maxStep_.cpu().numpy()
        ion_traces = trace_output_.cpu().numpy()

        # filter out rows containing all zeros
        wallPt_output = wallPt_output[~np.all(wallPt_output == 0, axis=1)]
        # Filter velocity_output and get the indices of nonzero rows
        nonzero_indices = ~np.all(velocity_output == 0, axis=1)
        velocity_output = velocity_output[nonzero_indices]
        max_timeStep = max_timeStep[nonzero_indices]

        speed_output = np.linalg.norm(velocity_output, axis=1)
        ion_mass_kg = self.ion_list[0].mass #* kg_per_amu
        energy_output = 0.5 * ion_mass_kg * speed_output**2 / kboltz #convert speed to energy in eV
        self.IO.log.info('Energy output stats: min={:.2f} eV, max={:.2f} eV, avg={:.2f} eV'.format(
            np.min(energy_output), np.max(energy_output), np.mean(energy_output)))

        wallPtArray = np.asarray( [XYZ_to_RTP(wall_point, Bfield.R0) for wall_point in wallPt_output] ).T
        outputArray = np.vstack((wallPtArray, velocity_output.T, max_timeStep[None, :]))

        ## CALCULATE UNIT VECTORS
        unit_vec_xyz = velocity_output/speed_output[:, None]  # Normalize the velocity vectors to get unit vectors
        radial_vec_xyz = np.asarray( [RTP_XYZ_JAC(wall_point, np.array([1,0,0]), form='rtp2xyz') for wall_point in wallPtArray.T] )# Convert unit vectors to RTP coordinates
        toroidal_vec_xyz = np.asarray( [RTP_XYZ_JAC(wall_point, np.array([0,0,1]), form='rtp2xyz') for wall_point in wallPtArray.T] )# Convert unit vectors to RTP coordinates

        ## CALCULATE ANGLE FROM NORMAL    
        deposition_angles = np.arccos(np.einsum('ij,ij->i', unit_vec_xyz, radial_vec_xyz))  # Calculate angles between unit vectors and radial vectors
        deposition_angles_deg = np.degrees(deposition_angles)  # Convert angles to degrees
        ## CALCULATE TOROIDAL ANGLE    
        cos_toroidal_angles = np.einsum('ij,ij->i', unit_vec_xyz, toroidal_vec_xyz)
        toroidal_angles = np.arccos(cos_toroidal_angles)
        toroidal_angles_deg = np.degrees(toroidal_angles)  # Convert angles to degrees


        self.IO.log.info('deposition_angles_deg min: {:.2f} deg, max: {:.2f} deg, avg: {:.2f} deg'.format(
            np.min(deposition_angles_deg), np.max(deposition_angles_deg), np.mean(deposition_angles_deg)))
        self.IO.log.info('toroidal_angles_deg min: {:.2f} deg, max: {:.2f} deg, avg: {:.2f} deg'.format(
            np.min(toroidal_angles_deg), np.max(toroidal_angles_deg), np.mean(toroidal_angles_deg)))
        toc = perf_counter()
        self.IO.log.info('OUTPUT SENT TO CPU AND CONVERTED TO RTP IN {}SEC'.format(toc-tic))

        return outputArray, energy_output, deposition_angles_deg, toroidal_angles_deg, ion_traces

    def save_output(self, outputArray, ion_traces):
        """Saves the output data to files in the specified output directory."""
        trace_filename = 'Ion_traces_' + self.cond_string+self.tag
        self.IO.saveNumpyData(ion_traces, trace_filename)
        self.IO.log.info('OUTPUT ION TRACES: {}'.format(trace_filename))

        wallpts_filename = 'Wallpt_OUTPUT_' + self.cond_string+self.tag
        self.IO.saveNumpyData(outputArray, wallpts_filename)
        self.IO.log.info('OUTPUT RESULT DATA: {}'.format(wallpts_filename))

    def run(self, Bfield, Efield=None, trace_IDs=[]):
        """Runs the Boris solver and processes the results.

        Args:
            Bfield: Magnetic field object providing field interpolation methods.
            Efield: Electric field object providing field interpolation methods. Defaults to None.
            trace_IDs: List of particle IDs to trace. Defaults to [].

        Returns:
            Tuple containing:
                outputArray (np.ndarray): Array of wall point and velocity data.
                energy_output (np.ndarray): Array of particle energies at termination (in eV).
                deposition_angles_deg (np.ndarray): Array of deposition angles (in degrees).
                toroidal_angles_deg (np.ndarray): Array of toroidal angles (in degrees).
                ion_traces (np.ndarray): Array of traced particle positions.
        """
        solv_out = self.parallel_solver(self.ion_list, Bfield, Efield, trace_IDs=trace_IDs)

        outputArray, energy_output, deposition_angles_deg, toroidal_angles_deg, ion_traces = self.post_solver(solv_out, Bfield)

        self.save_output(outputArray, ion_traces)

        return outputArray, energy_output, deposition_angles_deg, toroidal_angles_deg, ion_traces
