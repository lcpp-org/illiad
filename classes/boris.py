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

from illiad.utilities.coordtrans import XYZ_to_RTP, RTP_XYZ_JAC#, RTP_to_XYZ
from classes.collisions import Collisions, kg_per_amu, kboltz, eps0, sqrt_pi, Li_mass, He_mass
#from plot_funcs import plotFuncs


class Boris(Collisions):
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

    def setConditions(self, ion_list, cond_string, dt=1e-8, tmax=1e-3, T_gas_eV=0.025, Ti_eV=2.0, n_gas=3e18, n_e=1e18, m_gas_amu=4.002603):
        """Sets the initial conditions and events for Poincare analysis.

        Args:
            ion_list (list): List of Ion objects containing initial conditions and properties.
            dt (float, optional): Time step for the simulation. Defaults to 1e-8.
            tmax (float, optional): Maximum time for the simulation. Defaults to 1e-3.
            n_gas (float, optional): Neutral gas density for neutral collision models.
            n_e (float, optional): Plasma density for ion-ion collision models.
            m_gas_amu (float, optional): Mass of the background gas species in atomic mass units.

        Returns:
            None
        """
        self.dt = dt
        self.tmax = tmax
        self.nsteps = int(tmax // dt) + 1
        self.ion_list = ion_list
        
        self.T_gas_eV = T_gas_eV # eV, room temperature
        self.m_gas_amu = m_gas_amu #amu, background gas mass
        self.Ti_eV = Ti_eV # eV, ion temperature for ion-ion collision model
        self.m_ion_amu = m_gas_amu #amu, ion mass for ion-ion collision model
        self.n_gas = n_gas
        self.n_e = n_e

        self.cond_string = cond_string
        ## SET OUTPUT
        # for ion in ion_list:
        #     ion.initOutput(dt, tmax)

        # self.IO.log.info("+----------------+-------------------------+")
        # self.IO.log.info(f"| DT             | {self.dt:<23} |")
        # self.IO.log.info(f"| TMAX           | {self.tmax:<23} |")
        # self.IO.log.info(f"| NSTEPS         | {self.nsteps:<23} |")
        # self.IO.log.info("+----------------+-------------------------+")

    def parallel_solver(self, ions, Bfield, Efield=None, nfield=None, trace_IDs=[],
                        trace_stride=1,
                        freq_corr=False, ion_neutral_collisions=None, ion_ion_collisions=None):
        """
        Function to take in a particle and field object and solves the particle path until termination event or tmax
        using a fixed-step Boris-Buneman Solver, based on (Birdsall, 4-3&4).

        Parameters:
            -ions (list): List of ion objects containing initial conditions and properties.
            -Bfield (object): Magnetic field object providing field interpolation methods.
            -Efield (object, optional): Electric field object providing field interpolation methods. Defaults to None.
            -nfield (object, optional): Neutral field object providing field interpolation methods. Defaults to None.
            -track_ID (list, optional): List of particle IDs to track. Defaults to [10, 20].
            -freq_corr (bool, optional): Flag to enable frequency correction. Defaults to False.
            -ion_neutral_collisions (str or None): None, 'viscous_drag_hstep', or 'langevin_in_hstep'.
            -ion_ion_collisions (str or None): None, 'linearFP_ii_hstep', or 'fokker_planck_ii_hstep'.
            -trace_stride (int): Save one trace sample every trace_stride timesteps.
        Returns:
            -wallPts (torch.Tensor): XYZ Positions where particles terminate (e.g., hit the wall), shape (Nparticles, 3).
            -wallVelocities (torch.Tensor): Velocities of particles at termination, shape (Nparticles, 3).
            -maxStep (torch.Tensor): Step index at which each particle terminated, shape (Nparticles,).
        """
        try:
            trace_stride = int(trace_stride)
        except (TypeError, ValueError) as exc:
            raise ValueError('trace_stride must be a positive integer') from exc
        if trace_stride < 1:
            raise ValueError('trace_stride must be a positive integer')

        log = logging.getLogger()
        log.info('Start ICs: {}-{}'.format(ions[0].particleID, ions[-1].particleID))

        ion_neutral_collision_model = self._resolve_ion_neutral_collision_model(ion_neutral_collisions)
        ion_ion_collision_model = self._resolve_ion_ion_collision_model(ion_ion_collisions)
        if ion_neutral_collision_model:
            log.info('Ion-neutral collision model: {}'.format(ion_neutral_collision_model))
        if ion_ion_collision_model:
            log.info('Ion-ion collision model: {}'.format(ion_ion_collision_model))

        t_startInd = perf_counter()
        Nparticles = len(ions)
        max_trace_saves = 1 + ((self.nsteps - 1) // trace_stride) + 1
        trace_output = torch.zeros([max_trace_saves, len(trace_IDs), 3], dtype=torch.float64, device=device)
        trace_write_idx = 0
        last_trace_step = None
        final_step = 0
        with torch.no_grad():
            wallPts = torch.zeros([Nparticles, 3], dtype=torch.float64, device=device)
            wallVelocities = torch.zeros([Nparticles, 3], dtype=torch.float64, device=device)
            maxStep = torch.zeros(Nparticles, dtype=torch.int, device=device)
            tvec = torch.empty([Nparticles, 3], dtype=torch.float64, device=device)

            qdt2m = torch.tensor([ion.charge_mass_ratio * self.dt / 2 for ion in ions], dtype=torch.float64, device=device)
            kbTgasqMi = torch.tensor(
                (kboltz * self.T_gas_eV) / (self.m_gas_amu * kg_per_amu),
                dtype=torch.float64,
                device=device,
            ) # convert to Joules and divide by mass to get velocity squared units
            # kbTqMi = torch.tensor(
            #     (kboltz * 2.0) / (self.m_gas_amu * kg_per_amu),
            #     dtype=torch.float64,
            #     device=device,
            # ) # convert to Joules and divide by mass to get velocity squared units


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
            trace_output[trace_write_idx] = pos_k[trace_IDs]
            trace_write_idx += 1
            last_trace_step = 0

            log.info('START STEPPING...')
            log.info('Trace output stride: saving every {} timestep(s) plus final positions'.format(trace_stride))
            logging.basicConfig(level=logging.INFO)
            with logging_redirect_tqdm(loggers=[log]):
                pbar = tqdm(range(1, self.nsteps), ncols=100, mininterval=2.0)



                for k in pbar:
                    final_step = k
                    pos_active = pos_k[running]
                    qdt2m_active = qdt2m[running]
                    v_k_active = v_k[running]


                    actv_weights, actv_corner_indices, actv_ph_localN = Bfield.get_weights(pos_active)
                    b_vecs_active = Bfield.return_vecs(actv_weights, actv_corner_indices, actv_ph_localN)
                    full_phi_corner_indices = None
                    if Efield or (ion_ion_collision_model and nfield):
                        sector = torch.remainder(actv_ph_localN.to(torch.long), Bfield.periodicity[2])
                        phi_offset = sector.unsqueeze(0) * Bfield.nphi
                        full_phi_corner_indices = torch.stack([
                            actv_corner_indices[0],
                            actv_corner_indices[1],
                            actv_corner_indices[2] + phi_offset,
                        ])

                    if Efield:
                        e_vecs_active = Efield.return_vecs(actv_weights, full_phi_corner_indices, ph_localN=None)
                        Evec_active = (e_vecs_active * qdt2m_active).T
                    else:
                        Evec_active = torch.zeros_like(v_k_active)

                    # INSERT (FIRST) COLLISION HALF-STEP HERE IF DESIRED
                    ne_active = None
                    if ion_ion_collision_model:
                        if nfield:
                            ne_active = nfield.return_scalars(actv_weights, full_phi_corner_indices)
                        else:
                            ne_active = torch.full((pos_active.shape[0],), self.n_e, dtype=v_k_active.dtype, device=v_k_active.device)

                    if ion_neutral_collision_model:
                        v_k_active = self._apply_collision_hstep(
                            ion_neutral_collision_model,
                            pos_active,
                            v_k_active,
                            n_gas=self.n_gas,
                            kbTgasqMi=kbTgasqMi,
                        )

                    if ion_ion_collision_model:
                        v_k_active = self._apply_collision_hstep(
                            ion_ion_collision_model,
                            pos_active,
                            v_k_active,
                            n_e=ne_active,
                            Ti_ev=self.Ti_eV,
                        )

                    if freq_corr:
                        Bmag_active = torch.linalg.norm(b_vecs_active, axis=-1)
                        Bhat_active = b_vecs_active / Bmag_active[:, None]
                        tvec_active = torch.tan(qdt2m_active * Bmag_active)[:, None] * Bhat_active
                    else:
                        tvec_active = (b_vecs_active * qdt2m_active).T

                    tmag_active = torch.linalg.norm(tvec_active, axis=-1)



                    vminus_active = v_k_active + Evec_active
                    vprime_active = vminus_active + torch.linalg.cross(vminus_active, tvec_active)
                    svec_active = 2 * tvec_active / (1 + (tmag_active * tmag_active)[:, None])
                    vplus_active = vminus_active + torch.linalg.cross(vprime_active, svec_active)
                    
                    v_k_active = vplus_active + Evec_active

                    pos_active += v_k_active * self.dt
    
                    # INSERT (SECOND)COLLISION HALF-STEP HERE IF DESIRED
                    ##-----------------------------------------##
                    if ion_neutral_collision_model:
                        v_k_active = self._apply_collision_hstep(
                            ion_neutral_collision_model,
                            pos_active,
                            v_k_active,
                            n_gas=self.n_gas,
                            kbTgasqMi=kbTgasqMi,
                        )
                    if ion_ion_collision_model:
                        v_k_active = self._apply_collision_hstep(
                            ion_ion_collision_model,
                            pos_active,
                            v_k_active,
                            n_e=ne_active,
                            Ti_ev=self.Ti_eV,
                        )
                    ##-----------------------------------------##

                    # Update particle positions and velocities
                    pos_k[running] = pos_active
                    v_k[running] = v_k_active



                    x2 = pos_k.T[0]**2
                    y2 = pos_k.T[1]**2
                    z2 = pos_k.T[2]**2
                    r_k = torch.sqrt(x2 + y2 + z2 + Bfield.R0 * Bfield.R0
                                               - 2 * Bfield.R0 * torch.sqrt(x2 + y2))
                    running = torch.where(r_k < Bfield.a)[0]


                    maxStep[running] = k # +1?


                    # ADD SELECTED PARTICLE TRACING
                    if k % trace_stride == 0:
                        trace_output[trace_write_idx] = pos_k[trace_IDs]
                        trace_write_idx += 1
                        last_trace_step = k

                    Nrunning = running.size(0)
                    if Nrunning == 0:
                        log.info('All particles terminated at step {}'.format(k))
                        break

                    pbar.set_postfix({'#Particles running': Nrunning}, refresh=False)




            terminated = torch.where(r_k >= Bfield.a)[0]
            wallPts[terminated] = pos_k[terminated]
            wallVelocities[terminated] = v_k[terminated]

            if last_trace_step != final_step:
                trace_output[trace_write_idx] = pos_k[trace_IDs]
                trace_write_idx += 1
                last_trace_step = final_step
            trace_output = trace_output[:trace_write_idx]

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
        trace_filename = 'Ion_traces'
        self.IO.saveNumpyData(ion_traces, trace_filename)
        self.IO.log.info('OUTPUT ION TRACES: {}'.format(trace_filename))

        wallpts_filename = 'Wallpt_OUTPUT'
        self.IO.saveNumpyData(outputArray, wallpts_filename)
        self.IO.log.info('OUTPUT RESULT DATA: {}'.format(wallpts_filename))

    def run(self, Bfield, Efield=None, nfield=None,
            ion_neutral_collisions=None, ion_ion_collisions=None, trace_IDs=[],
            trace_stride=1):
        """Runs the Boris solver and processes the results.

        Args:
            Bfield: Magnetic field object providing field interpolation methods.
            Efield: Electric field object providing field interpolation methods. Defaults to None.
            nfield: Neutral field object providing field interpolation methods. Defaults to None.
            ion_neutral_collisions: Ion-neutral collision model name, or None.
            ion_ion_collisions: Ion-ion collision model name, or None.
            trace_IDs: List of particle IDs to trace. Defaults to [].
            trace_stride: Save one trace sample every trace_stride timesteps.

        Returns:
            Tuple containing:
                outputArray (np.ndarray): Array of wall point and velocity data.
                energy_output (np.ndarray): Array of particle energies at termination (in eV).
                deposition_angles_deg (np.ndarray): Array of deposition angles (in degrees).
                toroidal_angles_deg (np.ndarray): Array of toroidal angles (in degrees).
                ion_traces (np.ndarray): Array of traced particle positions.
        """
        solv_out = self.parallel_solver(
            ions = self.ion_list,
            Bfield = Bfield,
            Efield = Efield,
            nfield = nfield,
            ion_neutral_collisions = ion_neutral_collisions,
            ion_ion_collisions = ion_ion_collisions,
            trace_IDs = trace_IDs,
            trace_stride = trace_stride
        )

        outputArray, energy_output, deposition_angles_deg, toroidal_angles_deg, ion_traces = self.post_solver(solv_out, Bfield)

        self.save_output(outputArray, ion_traces)

        return outputArray, energy_output, deposition_angles_deg, toroidal_angles_deg, ion_traces
