import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from math import degrees

from functools import partial
import concurrent.futures as cf
from time import perf_counter
import matplotlib.pyplot as plt

#from utility.phi_events import *
from utility.phi_events import inVV
from utility.coordtrans import XYZ_to_RTP, XYZ_to_RTP_many, RTP_to_XYZ_many #RTP_to_XYZ
from classes.mesh import Mesh
from classes.particle import FieldLine
from plot_funcs import plotFuncs
import gc

def _phi_from_xyz(points_xyz):
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    phi = -np.arctan2(points_xyz[..., 1], points_xyz[..., 0])
    return np.where(phi < 0.0, phi + 2.0 * np.pi, phi)


def _unwrap_phi_local(phi_wrapped, phi_ref_wrapped, phi_ref_unwrapped):
    delta = np.arctan2(
        np.sin(phi_wrapped - phi_ref_wrapped),
        np.cos(phi_wrapped - phi_ref_wrapped),
    )
    return phi_ref_unwrapped + delta


def _crossing_ids(phi0, phi1, plane_spacing):
    if phi1 > phi0:
        first_cross = int(np.floor(phi0 / plane_spacing)) + 1
        last_cross = int(np.floor(phi1 / plane_spacing))
        if last_cross < first_cross:
            return ()
        return range(first_cross, last_cross + 1)

    first_cross = int(np.ceil(phi0 / plane_spacing)) - 1
    last_cross = int(np.ceil(phi1 / plane_spacing))
    if first_cross < last_cross:
        return ()
    return range(first_cross, last_cross - 1, -1)


def _extract_plane_crossings(solution, t_values, xyz_values, plot_angles):
    nplanes = len(plot_angles)
    if solution is None or nplanes == 0 or t_values.size < 2:
        return [np.empty((0, 3), dtype=np.float64) for _ in range(nplanes)]

    plane_spacing = (2.0 * np.pi) / nplanes
    plane_hits = [[] for _ in range(nplanes)]
    wrapped_phi = _phi_from_xyz(xyz_values)
    unwrapped_phi = np.unwrap(wrapped_phi)

    def solve_segment_crossings(t0, t1, xyz0, xyz1, phi0_wrapped, phi1_wrapped, phi0_unwrapped, phi1_unwrapped):
        delta_phi = phi1_unwrapped - phi0_unwrapped
        if delta_phi == 0.0:
            return

        if abs(delta_phi) > (np.pi / 2.0):
            tm = 0.5 * (t0 + t1)
            xyz_mid = np.asarray(solution(tm), dtype=np.float64)
            phi_mid_wrapped = float(_phi_from_xyz(xyz_mid))
            phi_mid_unwrapped = _unwrap_phi_local(phi_mid_wrapped, phi0_wrapped, phi0_unwrapped)
            solve_segment_crossings(t0, tm, xyz0, xyz_mid, phi0_wrapped, phi_mid_wrapped, phi0_unwrapped, phi_mid_unwrapped)
            solve_segment_crossings(tm, t1, xyz_mid, xyz1, phi_mid_wrapped, phi1_wrapped, phi_mid_unwrapped, phi1_unwrapped)
            return

        for cross_id in _crossing_ids(phi0_unwrapped, phi1_unwrapped, plane_spacing):
            target_phi = cross_id * plane_spacing
            plane_index = (cross_id - 1) % nplanes

            def phi_residual(t, target=target_phi, phi_ref_wrapped=phi0_wrapped, phi_ref_unwrapped=phi0_unwrapped):
                xyz = np.asarray(solution(t), dtype=np.float64)
                phi_wrapped = float(_phi_from_xyz(xyz))
                phi_unwrapped = _unwrap_phi_local(phi_wrapped, phi_ref_wrapped, phi_ref_unwrapped)
                return phi_unwrapped - target

            try:
                t_cross = brentq(phi_residual, t0, t1, xtol=1e-10, rtol=1e-10, maxiter=50)
                xyz_cross = np.asarray(solution(t_cross), dtype=np.float64)
            except ValueError:
                alpha = (target_phi - phi0_unwrapped) / delta_phi
                xyz_cross = xyz0 + alpha * (xyz1 - xyz0)

            plane_hits[plane_index].append(xyz_cross)

    for seg_idx in range(t_values.size - 1):
        solve_segment_crossings(
            t_values[seg_idx],
            t_values[seg_idx + 1],
            xyz_values[seg_idx],
            xyz_values[seg_idx + 1],
            wrapped_phi[seg_idx],
            wrapped_phi[seg_idx + 1],
            unwrapped_phi[seg_idx],
            unwrapped_phi[seg_idx + 1],
        )

    return [
        np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if points else np.empty((0, 3), dtype=np.float64)
        for points in plane_hits
    ]

class Poincare():
    """Class to handle Poincare analysis of magnetic field lines."""
    def __init__(self, io_handler, solvr='LSODA', r_tol=1e-6, a_tol=1e-16, workers=-1, double_line=False, anlys_name='Poincare'):
        """Initializes the Poincare class with the specified solver parameters and writes to log.

        workers > 0: use N threads
        workers = 0: use all available threads
        workers < 0: use all but the last N threads

        double_line True: run each fieldline in both directions from the init pos (!ONLY USE WHEN (NTHREADS > NLINES)!)
        double_line False: run each fieldline in +B direction from the init pos

        Args:
            io_handler: An object responsible for handling output operations, such as logging and directory creation.
            solvr (str, optional): The ODE solver to use. Defaults to 'LSODA'.
            r_tol (float, optional): Relative tolerance for the solver. Defaults to 1e-6.
            a_tol (float, optional): Absolute tolerance for the solver. Defaults to 1e-16.
            workers (int, optional): Number of worker threads to use. Defaults to -1.
            double_line (bool, optional): Whether to use double line integration. Defaults to False.
            anlys_name (str, optional): Name of the analysis or subdirectory. Defaults to 'Poincare'.
        """
        self.IO = io_handler
        self.anlys_name = anlys_name
        self.solver = solvr
        self.r_tol = r_tol
        self.a_tol = a_tol
        self.double_line = double_line
        if workers <= 0: 
            self.workers = os.cpu_count() + workers
        else:
            self.workers = workers

        self.IO.createSubDir(anlys_name)
        self.IO.log.info("+----------------+-------------------------+")
        self.IO.log.info("| Parameter      | Value                   |")
        self.IO.log.info("+----------------+-------------------------+")
        self.IO.log.info(f"| SOLVER         | {self.solver:<23} |")
        self.IO.log.info(f"| RTOL           | {self.r_tol:<23} |")
        self.IO.log.info(f"| ATOL           | {self.a_tol:<23} |")
        self.IO.log.info(f"| THREADS        | {self.workers:<23} |")
        self.IO.log.info("+----------------+-------------------------+")

        
        # attach plotting function to class instance
    
        for name in dir(plotFuncs):
            func = getattr(plotFuncs, name)
            if callable(func) and not name.startswith("__"):
                if name.startswith("global_"):
                    new_name = name.replace("global_", "")  # Remove prefix
                    setattr(self, new_name, func)  # Attach to the instance with the new name
                elif name.startswith("poincare_"):
                    new_name = name.replace("poincare_", "")  # Remove prefix
                    setattr(self, new_name, func)  # Attach to the instance with the new name
        

    def set_conditions(self, init_pos_arr=np.zeros([1, 3]), spins=100, field: Mesh = None, events=None, nplanes=360):
        """Sets the initial conditions and events for Poincare analysis.

        Args:
            init_pos_arr (np.ndarray): Array of initial conditions in RTP (radius, theta, phi) format.
            spins (int): Number of spins for the field lines.
            field (Mesh): The magnetic field mesh object.
            events (list, optional): List of event functions to be used in the solver. If None, a default set of Poincare events is used.

        Returns:
            None
        """
        self.IC_rtp_arr = np.atleast_2d(np.asarray(init_pos_arr, dtype=np.float64))
        self.nlines = len(self.IC_rtp_arr)
        self.spins = spins

        if field: self.field = field
        else: raise ValueError("Field mesh is required.")

        ## CONVERT TO XYZ COORDS
        ICs_XYZ = RTP_to_XYZ_many(self.IC_rtp_arr, self.field.R0)
        length = (2 * np.pi * self.field.R0) * spins

        self.fieldlines = [FieldLine(init_cond, length, direction = 1.0) for init_cond in ICs_XYZ]
        if self.double_line: 
            self.fieldlines += [FieldLine(init_cond, length, direction = -1.0) for init_cond in ICs_XYZ]

        """
        if events is None:
            self.solver_events = poincare_events
            self.plot_angles = np.linspace(np.pi/180., 2*np.pi, 360)
        else:
            self.solver_events = events
            n_angles = len(events) - 1
            self.plot_angles = np.linspace(np.pi/180., 2*np.pi, n_angles)
        """

        if events is None:
            if nplanes <= 0:
                raise ValueError("nplanes must be positive.")
            self.use_plane_reconstruction = True
            self.solver_events = [inVV]
            self.plot_angles = np.linspace(2 * np.pi / nplanes, 2 * np.pi, nplanes)
        else:
            self.use_plane_reconstruction = False
            self.solver_events = events
            n_angles = len(events) - 1
            self.plot_angles = np.linspace(np.pi / 180.0, 2 * np.pi, n_angles)


        self.IO.log.info("+----------------+-------------------------+")
        self.IO.log.info(f"| NLINES         | {self.nlines:<23} |")
        self.IO.log.info(f"| SPINS          | {self.spins:<23} |")
        self.IO.log.info(f"| # OF EVENTS    | {len(self.solver_events):<23} |")
        self.IO.log.info("| Initial Conditions (RTP):                |")
        for ic in init_pos_arr:
            self.IO.log.info(f"|     {str(ic):<23}   |")
        self.IO.log.info("+----------------+-------------------------+")

    def parallel_solver(self) -> iter:
        """Runs the solver in parallel for each particle.

        This method uses a process pool of 'self.workers' to execute the solver for each field line in parallel,
        allowing for efficient computation of multiple field lines simultaneously.

        Returns:
            iterator: An iterator of tuples containing the maximum time and event data for each particle.
        """
        length = self.fieldlines[0].maxLife

        self.IO.log.info('Begin running {} ICs for max. {} spins...'.format(self.nlines, int(length/(2*np.pi * self.field.R0))))
        tic = perf_counter()
        with cf.ProcessPoolExecutor(max_workers=self.workers) as executor:
            collected_output = executor.map(self.single_solver, self.fieldlines)
        toc = perf_counter()
        self.IO.log.info('ALL SOLVERS FINISHED IN {} seconds\n###############\n\n'.format(toc - tic))

        return collected_output

    def single_solver(self, particle):
        """Runs the solver for a single particle.

        Args:
            particle (FieldLine): The particle object containing initial conditions and properties.

        Returns:
            tuple: (tmax, data) where tmax is the maximum time reached by the solver,
                and data is the event data collected by the solver.
        """
        self.IO.log.info('Start IC: {}'.format(particle.particleID))
        init_cond = particle.pos0_XYZ
        maxLength = particle.maxLife

        tic = perf_counter()
        if self.use_plane_reconstruction:
            fieldlines = solve_ivp(particle.pushXYZ, (0.0, maxLength), init_cond,
                                    args=([self.field]), dense_output=True,
                                    events=self.solver_events,
                                    method=self.solver, rtol=self.r_tol, atol=self.a_tol)
            plane_output = _extract_plane_crossings( fieldlines.sol, fieldlines.t, fieldlines.y.T, self.plot_angles)
            wall_output = fieldlines.y_events[0] if fieldlines.y_events else np.empty((0, 3), dtype=np.float64)
        else:
            for event in self.solver_events:
                if event.__name__ == 'inVV':
                    event.direction = -1.0
                elif event.__name__ == 'isphi360':
                    event.direction = -particle.direction
                else:
                    event.direction = particle.direction

        
            fieldlines = solve_ivp(particle.pushXYZ, (0.0, maxLength), init_cond,
                                    args = ([self.field]),
                                    dense_output=False,
                                    events = self.solver_events, 
                                    method=self.solver, rtol=self.r_tol, atol=self.a_tol)
            plane_output = fieldlines.y_events[1:]
            wall_output = fieldlines.y_events[0] if fieldlines.y_events else np.empty((0, 3), dtype=np.float64)

        toc = perf_counter()
        elapsed_timeInd = toc - tic
        tmax = float(fieldlines.t[-1])

        if fieldlines.status == 0: #solver ran to max. time
            self.IO.log.info(
                'Success!: Particle {} of {} took {:.4f} sec.\tEnd at tmax={:.3f}'.format(
                    particle.particleID,
                    particle.particleCount,
                    elapsed_timeInd,
                    tmax))
        elif fieldlines.status == 1: #termination event
            self.IO.log.info(
                'Success!: Particle {} of {} took {:.4f} sec.\tWall Event at t={}'.format(
                    particle.particleID,
                    particle.particleCount,
                    elapsed_timeInd,
                    fieldlines.t_events[0]))
        else: #solver failure
            self.IO.log.critical('FAILURE!: Particle {}'
                .format(particle.particleID))

        return tmax, plane_output, wall_output

    def post_solver(self, solver_output, plot_args=None):
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
        ## PARSE OUTPUT INTO LISTS
        path_lengths = []
        poincare_points = []
        wall_points = []
        for pLngth, plane_out, wall_out in solver_output:
            path_lengths.append(pLngth)
            poincare_points.append(plane_out)
            if isinstance(wall_out, np.ndarray) and wall_out.size:
                wall_points.append(XYZ_to_RTP(wall_out[0], self.field.R0))



        if self.double_line:
            # Combine the positive and negative fieldlines into one
            path_lengths = [path_lengths[i]+path_lengths[i+self.nlines] for i in range(0,self.nlines)]
            for line_index in range(0,self.nlines):
                for event_index in range(len(poincare_points[line_index])):
                    arr_a = poincare_points[line_index][event_index]
                    arr_b = poincare_points[line_index+self.nlines][event_index]
                    if arr_a.any() and arr_b.any():
                        poincare_points[line_index][event_index] = np.vstack((arr_a, arr_b))
            poincare_points = poincare_points[:self.nlines]

        self.IO.log.info('PLOTTING AND OUTPUTTING PHI-ANGLE DATA:')
        # self.IO.log.info('path_lengths: {}'.format(path_lengths))
        save_output_partial = partial(self.save_output, xyz_list=poincare_points, saveData=True, plot_args=plot_args)
        plot_workers = min(self.workers, 9)
        iter_in = enumerate(self.plot_angles)
        with cf.ProcessPoolExecutor(max_workers=plot_workers) as executor:
            #list(executor.map(save_output_partial, iter_in))
            executor.map(save_output_partial, iter_in)

        return path_lengths, poincare_points, wall_points

    
    def identifyLCFS(self, LCFStype='inner', t_maxs=[100], index=0):
        """Returns the index of the Last-Closed Flux Surface (LCFS).

        Args:
            LCFStype (str): Method to identify LCFS. One of 'inner', 'outer', or 'input'.
            t_maxs (list): List of connection lengths for each initial condition.
            index_in (int): Index to use if LCFStype is 'input'.

        Returns:
            int: Index of the identified LCFS.

        Raises:
            ValueError: If LCFStype is not one of ['inner', 'outer', 'input'].
        """
        LCFStypes = ['inner', 'outer', 'input']
        if LCFStype not in LCFStypes:
            raise ValueError("Invalid LCFS type. Expected one of: %s" % LCFStypes)

        elif LCFStype == 'input':
                LCFS_index = index

        elif LCFStype == 'inner':
            # Assuming surfaces are ordered from 'out' to 'in':
            ## This returns the LCFS 'inside' ALL open flux surfaces
            maxTime = np.max(t_maxs)
            # Get indices of open flux surfaces
            openSurface_ind = [i for i, t in enumerate(t_maxs) if t != maxTime]
            if openSurface_ind:
                LCFS_index = max(openSurface_ind) + 1
            else:
                LCFS_index = 1

            plt.figure()
            plt.plot(self.IC_rtp_arr.T[0], t_maxs, '-o', c='k')
            ## CHECK
            plt.plot(self.IC_rtp_arr[LCFS_index][0], t_maxs[LCFS_index], '^', c='b')

            plt.title(r'Connection length vs. $r_{initial} (@{}\phi=324\degree)$')
            plt.yscale('log')
            plt.grid(True, which='both')
            plt.xlabel('Connection length [m]')
            self.IO.saveFig('connectLengths')
            plt.close()

        elif LCFStype == 'outer':
            maxTime = np.max(t_maxs)
            LCFS_index = t_maxs.index(maxTime)

            plt.figure()
            plt.plot(self.IC_rtp_arr, t_maxs, '-o', c='k')
            plt.plot(self.IC_rtp_arr[LCFS_index], maxTime, '^', c='b')

            plt.title(r'Connection length vs. $r_{initial} (@{}\phi=324\degree)$')
            plt.yscale('log')
            plt.grid(True, which='both')
            plt.xlabel('Connection length [m]')
            self.IO.saveFig('connectLengths')
            plt.close()

        self.IO.log.info('LCFS_index = {}'.format(LCFS_index))
        return LCFS_index

    def save_output(self, iter, xyz_list, saveData=True, plot_args=None):
        """
        Generates and saves Poincare plots and associated data for a given phi angle, and logs the operation.

        Note:
            This method uses self.field and self.anlys_name, which are required for correct operation in a multiprocessing context.

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

        n, phi = iter
        phi_deg = phi*180/np.pi

        num_sets = len(xyz_list)
        maxLength = max((len(xyz_list[i][n]) for i in range(num_sets)), default=0)
        radtheta_pts = np.full([num_sets, 2, maxLength], fill_value=np.nan)
        point_total = np.zeros(num_sets, dtype=int)

        for i in range(num_sets):
            xyz_points = xyz_list[i][n]
            point_total[i] = len(xyz_points)
            if point_total[i] == 0:
                continue

            rtp_points = XYZ_to_RTP_many(xyz_points[:, :3], self.field.R0)
            radtheta_pts[i][0, :point_total[i]] = rtp_points[:, 1]
            radtheta_pts[i][1, :point_total[i]] = rtp_points[:, 0]


        if saveData:
            fname = self.anlys_name + '_{:03.0f}'.format(phi_deg)
            self.IO.saveNumpyData(radtheta_pts, fname)

        # plotting
        self.plotPoincareBW(radtheta_pts, point_total, phi_deg, self.field, self.anlys_name, simIO=self.IO, plot_args=plot_args)


    def run(self, plot_args=None):
        """Generates Poincare plots based on the initial conditions and magnetic field.

        Returns:
            tuple: (pathLength_test, Poincare_output_test, wall_output_test)
            pathLength_test (list): List of path lengths for each particle.
            Poincare_output_test (list): List of Poincare data for each particle.
            wall_output_test (list): List of wall intersection data for each particle.
        """
        solv_out = self.parallel_solver()
        pathLength, Poincare_output, wall_output = self.post_solver(solv_out, plot_args)
        # self.IO.log.info('pathLength: {}'.format(pathLength))
        return pathLength, Poincare_output, wall_output