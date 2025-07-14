import numpy as np
from scipy.integrate import solve_ivp
from math import degrees

from functools import partial
import concurrent.futures as cf
from time import perf_counter
import matplotlib.pyplot as plt

from phi_events import *
from utility.coordtrans import XYZ_to_RTP, RTP_to_XYZ
from classes.particle import fieldLine


class Poincare():
    """Class to handle Poincare analysis of magnetic field lines."""

    def __init__(self, outputHandler, solvr='LSODA', r_tol=1e-6, a_tol=1e-16, workers=6, double_line=False, anlys_name='Poincare'):
        """Initializes the Poincare class with the specified solver parameters and writes to log.

        Args:
            outputHandler: An object responsible for handling output operations, such as logging and directory creation.
            solvr (str, optional): The ODE solver to use. Defaults to 'LSODA'.
            r_tol (float, optional): Relative tolerance for the solver. Defaults to 1e-6.
            a_tol (float, optional): Absolute tolerance for the solver. Defaults to 1e-16.
            workers (int, optional): Number of worker threads to use. Defaults to 6.
            double_line (bool, optional): Whether to use double line integration. Defaults to False.
            anlys_name (str, optional): Name of the analysis or subdirectory. Defaults to 'Poincare'.
        """
        self.IO = outputHandler
        self.anlys_name = anlys_name
        self.solver = solvr
        self.r_tol = r_tol
        self.a_tol = a_tol
        self.workers = workers
        self.double_line = double_line

        self.IO.createSubDir(anlys_name)
        self.IO.log.info("+----------------+-------------------------+")
        self.IO.log.info("| Parameter      | Value                   |")
        self.IO.log.info("+----------------+-------------------------+")
        self.IO.log.info(f"| SOLVER         | {self.solver:<23} |")
        self.IO.log.info(f"| RTOL           | {self.r_tol:<23} |")
        self.IO.log.info(f"| ATOL           | {self.a_tol:<23} |")
        self.IO.log.info(f"| THREADS        | {self.workers:<23} |")
        self.IO.log.info("+----------------+-------------------------+")

    def set_conditions(self, ic_rtp_arr, spins, field, events=None):
        """Sets the initial conditions and events for Poincare analysis.

        Args:
            ic_rtp_arr (np.ndarray): Array of initial conditions in RTP (radius, theta, phi) format.
            spins (int): Number of spins for the field lines.
            field (Mesh): The magnetic field mesh object.
            events (list, optional): List of event functions to be used in the solver. If None, a default set of Poincare events is used.

        Returns:
            None
        """
        self.nlines = len(ic_rtp_arr)
        self.spins = spins
        self.field = field

        ## CONVERT TO XYZ COORDS
        ICs_XYZ = np.zeros(shape=(self.nlines, 3))
        for i in range(self.nlines):
            ICs_XYZ[i] = RTP_to_XYZ(ic_rtp_arr[i], self.field.R0)
        length = (2*np.pi * self.field.R0) * spins

        self.fieldlines = [fieldLine(init_cond, length, direction = 1.0) for init_cond in ICs_XYZ]
        if self.double_line: 
            self.fieldlines += [fieldLine(init_cond, length, direction = -1.0) for init_cond in ICs_XYZ]

        poincare_events = [ inVV, 
                        isphi1, 
                        isphi2, 
                        isphi3, 
                        isphi4, 
                        isphi5, 
                        isphi6, 
                        isphi7, 
                        isphi8, 
                        isphi9, 
                        isphi10, 
                        isphi11, 
                        isphi12, 
                        isphi13, 
                        isphi14, 
                        isphi15, 
                        isphi16, 
                        isphi17, 
                        isphi18, 
                        isphi19, 
                        isphi20, 
                        isphi21, 
                        isphi22, 
                        isphi23, 
                        isphi24, 
                        isphi25, 
                        isphi26, 
                        isphi27, 
                        isphi28, 
                        isphi29, 
                        isphi30, 
                        isphi31, 
                        isphi32, 
                        isphi33, 
                        isphi34, 
                        isphi35, 
                        isphi36, 
                        isphi37, 
                        isphi38, 
                        isphi39, 
                        isphi40, 
                        isphi41, 
                        isphi42, 
                        isphi43, 
                        isphi44, 
                        isphi45, 
                        isphi46, 
                        isphi47, 
                        isphi48, 
                        isphi49, 
                        isphi50, 
                        isphi51, 
                        isphi52, 
                        isphi53, 
                        isphi54, 
                        isphi55, 
                        isphi56, 
                        isphi57, 
                        isphi58, 
                        isphi59, 
                        isphi60, 
                        isphi61, 
                        isphi62, 
                        isphi63, 
                        isphi64, 
                        isphi65, 
                        isphi66, 
                        isphi67, 
                        isphi68, 
                        isphi69, 
                        isphi70, 
                        isphi71, 
                        isphi72, 
                        isphi73, 
                        isphi74, 
                        isphi75, 
                        isphi76, 
                        isphi77, 
                        isphi78, 
                        isphi79, 
                        isphi80, 
                        isphi81, 
                        isphi82, 
                        isphi83, 
                        isphi84, 
                        isphi85, 
                        isphi86, 
                        isphi87, 
                        isphi88, 
                        isphi89, 
                        isphi90, 
                        isphi91, 
                        isphi92, 
                        isphi93, 
                        isphi94, 
                        isphi95, 
                        isphi96, 
                        isphi97, 
                        isphi98, 
                        isphi99, 
                        isphi100, 
                        isphi101, 
                        isphi102, 
                        isphi103, 
                        isphi104, 
                        isphi105, 
                        isphi106, 
                        isphi107, 
                        isphi108, 
                        isphi109, 
                        isphi110, 
                        isphi111, 
                        isphi112, 
                        isphi113, 
                        isphi114, 
                        isphi115, 
                        isphi116, 
                        isphi117, 
                        isphi118, 
                        isphi119, 
                        isphi120, 
                        isphi121, 
                        isphi122, 
                        isphi123, 
                        isphi124, 
                        isphi125, 
                        isphi126, 
                        isphi127, 
                        isphi128, 
                        isphi129, 
                        isphi130, 
                        isphi131, 
                        isphi132, 
                        isphi133, 
                        isphi134, 
                        isphi135, 
                        isphi136, 
                        isphi137, 
                        isphi138, 
                        isphi139, 
                        isphi140, 
                        isphi141, 
                        isphi142, 
                        isphi143, 
                        isphi144, 
                        isphi145, 
                        isphi146, 
                        isphi147, 
                        isphi148, 
                        isphi149, 
                        isphi150, 
                        isphi151, 
                        isphi152, 
                        isphi153, 
                        isphi154, 
                        isphi155, 
                        isphi156, 
                        isphi157, 
                        isphi158, 
                        isphi159, 
                        isphi160, 
                        isphi161, 
                        isphi162, 
                        isphi163, 
                        isphi164, 
                        isphi165, 
                        isphi166, 
                        isphi167, 
                        isphi168, 
                        isphi169, 
                        isphi170, 
                        isphi171, 
                        isphi172, 
                        isphi173, 
                        isphi174, 
                        isphi175, 
                        isphi176, 
                        isphi177, 
                        isphi178, 
                        isphi179, 
                        isphi180, 
                        isphi181, 
                        isphi182, 
                        isphi183, 
                        isphi184, 
                        isphi185, 
                        isphi186, 
                        isphi187, 
                        isphi188, 
                        isphi189, 
                        isphi190, 
                        isphi191, 
                        isphi192, 
                        isphi193, 
                        isphi194, 
                        isphi195, 
                        isphi196, 
                        isphi197, 
                        isphi198, 
                        isphi199, 
                        isphi200, 
                        isphi201, 
                        isphi202, 
                        isphi203, 
                        isphi204, 
                        isphi205, 
                        isphi206, 
                        isphi207, 
                        isphi208, 
                        isphi209, 
                        isphi210, 
                        isphi211, 
                        isphi212, 
                        isphi213, 
                        isphi214, 
                        isphi215, 
                        isphi216, 
                        isphi217, 
                        isphi218, 
                        isphi219, 
                        isphi220, 
                        isphi221, 
                        isphi222, 
                        isphi223, 
                        isphi224, 
                        isphi225, 
                        isphi226, 
                        isphi227, 
                        isphi228, 
                        isphi229, 
                        isphi230, 
                        isphi231, 
                        isphi232, 
                        isphi233, 
                        isphi234, 
                        isphi235, 
                        isphi236, 
                        isphi237, 
                        isphi238, 
                        isphi239, 
                        isphi240, 
                        isphi241, 
                        isphi242, 
                        isphi243, 
                        isphi244, 
                        isphi245, 
                        isphi246, 
                        isphi247, 
                        isphi248, 
                        isphi249, 
                        isphi250, 
                        isphi251, 
                        isphi252, 
                        isphi253, 
                        isphi254, 
                        isphi255, 
                        isphi256, 
                        isphi257, 
                        isphi258, 
                        isphi259, 
                        isphi260, 
                        isphi261, 
                        isphi262, 
                        isphi263, 
                        isphi264, 
                        isphi265, 
                        isphi266, 
                        isphi267, 
                        isphi268, 
                        isphi269, 
                        isphi270, 
                        isphi271, 
                        isphi272, 
                        isphi273, 
                        isphi274, 
                        isphi275, 
                        isphi276, 
                        isphi277, 
                        isphi278, 
                        isphi279, 
                        isphi280, 
                        isphi281, 
                        isphi282, 
                        isphi283, 
                        isphi284, 
                        isphi285, 
                        isphi286, 
                        isphi287, 
                        isphi288, 
                        isphi289, 
                        isphi290, 
                        isphi291, 
                        isphi292, 
                        isphi293, 
                        isphi294, 
                        isphi295, 
                        isphi296, 
                        isphi297, 
                        isphi298, 
                        isphi299, 
                        isphi300, 
                        isphi301, 
                        isphi302, 
                        isphi303, 
                        isphi304, 
                        isphi305, 
                        isphi306, 
                        isphi307, 
                        isphi308, 
                        isphi309, 
                        isphi310, 
                        isphi311, 
                        isphi312, 
                        isphi313, 
                        isphi314, 
                        isphi315, 
                        isphi316, 
                        isphi317, 
                        isphi318, 
                        isphi319, 
                        isphi320, 
                        isphi321, 
                        isphi322, 
                        isphi323, 
                        isphi324, 
                        isphi325, 
                        isphi326, 
                        isphi327, 
                        isphi328, 
                        isphi329, 
                        isphi330, 
                        isphi331, 
                        isphi332, 
                        isphi333, 
                        isphi334, 
                        isphi335, 
                        isphi336, 
                        isphi337, 
                        isphi338, 
                        isphi339, 
                        isphi340, 
                        isphi341, 
                        isphi342, 
                        isphi343, 
                        isphi344, 
                        isphi345, 
                        isphi346, 
                        isphi347, 
                        isphi348, 
                        isphi349, 
                        isphi350, 
                        isphi351, 
                        isphi352, 
                        isphi353, 
                        isphi354, 
                        isphi355, 
                        isphi356, 
                        isphi357, 
                        isphi358, 
                        isphi359, 
                        isphi360] 

        if events is None:
            self.events = poincare_events
            self.plot_angles = np.linspace(np.pi/180., 2*np.pi, 360)
        else:
            self.events = events

        self.IO.log.info("+----------------+-------------------------+")
        self.IO.log.info(f"| NLINES         | {self.nlines:<23} |")
        self.IO.log.info(f"| SPINS          | {self.spins:<23} |")
        self.IO.log.info("| Initial Conditions (RTP):                |")
        for ic in ic_rtp_arr:
            self.IO.log.info(f"|     {str(ic):<23}   |")
        self.IO.log.info("+----------------+-------------------------+")

    def parallel_solver(self):
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
            particle (fieldLine): The particle object containing initial conditions and properties.

        Returns:
            tuple: (tmax, data) where tmax is the maximum time reached by the solver,
                and data is the event data collected by the solver.
        """
        self.IO.log.info('Start IC: {}'.format(particle.particleID))
        init_cond = particle.pos0_XYZ
        maxLength = particle.maxLife

        for event in self.solver_events:
            if event.__name__ == 'inVV':
                event.direction = -1.0
            elif event.__name__ == 'isphi360':
                event.direction = -particle.direction
            else:
                event.direction = particle.direction

        tic = perf_counter()
        fieldlines = solve_ivp(particle.pushXYZ, (0.0, maxLength), init_cond,
                                args = ([self.field]),
                                dense_output=False,
                                events = self.solver_events, 
                                method=self.solver, rtol=self.r_tol, atol=self.a_tol)
        toc = perf_counter()
        elapsed_timeInd = toc - tic

        tmax = np.max(fieldlines.t)

        if fieldlines.status == 0: #solver ran to max. time
            self.IO.log.info('Success!: Particle {} of {} took {:.4f} sec.\tEnd at tmax={:.3f}'
                .format(particle.particleID,
                        particle.particleCount,
                        elapsed_timeInd,
                        tmax))
        elif fieldlines.status == 1: #termination event
            self.IO.log.info('Success!: Particle {} of {} took {:.4f} sec.\tWall Event at t={}'
                .format(particle.particleID,
                        particle.particleCount,
                        elapsed_timeInd,
                        fieldlines.t_events[0]))
        else: #solver failure
            self.IO.log.critical('FAILURE!: Particle {}'
                .format(particle.particleID))

        data = fieldlines.y_events[:]

        return tmax, data

    def post_solver(self, solver_output):
        """Processes the solver output to extract path lengths and Poincare data,
        and prepares the data for plotting and output.

        Args:
            solver_output (list): The output from the solver, containing tuples of path lengths and event data.

        Returns:
            tuple: (pathLength_, Poincare_output_, wall_output_)
                pathLength_ (list): List of path lengths for each particle.
                Poincare_output_ (list): List of Poincare data for each particle.
                wall_output_ (list): List of wall intersection data for each particle.
        """
        ## PARSE OUTPUT INTO LISTS
        pathLength_=[]
        Poincare_output_ = []
        wall_output_ = []
        for pLngth, out in solver_output:
            pathLength_ += [pLngth]
            Poincare_output_ += [out[1:]]
            if isinstance(out[0], np.ndarray) and out[0].any():
                wall_output_ += [XYZ_to_RTP(out[0][0], self.field.R0)]

        if self.double_line:
            # Combine the positive and negative fieldlines into one
            pathLength_ = [pathLength_[i]+pathLength_[i+self.nlines] for i in range(0,self.nlines)]
            for line_index in range(0,self.nlines):
                for event_index in range(len(Poincare_output_[line_index])):
                    arr_a = Poincare_output_[line_index][event_index]
                    arr_b = Poincare_output_[line_index+self.nlines][event_index]
                    if arr_a.any() and arr_b.any():
                        Poincare_output_[line_index][event_index] = np.vstack((arr_a, arr_b))
            Poincare_output_ = Poincare_output_[:self.nlines]

        ## POST-SOLVER PLOTTING AND OUTPUT
        plt.rcParams.update({'font.size': 10})
        plt.rcParams.update({'figure.autolayout':True})

        self.IO.log.info('PLOTTING AND OUTPUTTING PHI-ANGLE DATA:')
        plot_workers = min(self.workers, 16)
        iter_in = enumerate(self.plot_angles)

        save_output_x = partial(self.save_output, Pdata=Poincare_output_, saveData=True)
        with cf.ProcessPoolExecutor(max_workers=plot_workers) as executor:
            outs = executor.map(save_output_x, iter_in)
        for out in outs:
            self.IO.log.info(out)

        return pathLength_, Poincare_output_, wall_output_

    def save_output(self, iter, Pdata, saveData=True):
        """Outputs Poincare plots and data set at a given phi angle.

        Args:
            iter (tuple): Tuple of (index, phi angle).
            Pdata (list): List of Poincare data for each particle.
            saveData (bool, optional): Whether to save the data. Defaults to True.

        Returns:
            str: Log message indicating the phi angle processed.
        """
        num_sets = len(Pdata)
        rminor = self.field.a
        rmajor = self.field.R0
        n, phi_ = iter

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)

        maxLength = 0
        for i in range(num_sets):
            maxLength = max(maxLength, len(Pdata[i][n]))

        scatter_points = np.full([num_sets, 2, maxLength], fill_value=np.nan)
        for i in range(num_sets):
            t_pts = Pdata[i][n]
            point_total = max(0, len(t_pts)-1)

            for j in range(point_total):
                scatter_points[i][1][j], scatter_points[i][0][j], dum = XYZ_to_RTP(t_pts[j][:3], rmajor)

            plt.scatter(scatter_points[i][0][:point_total], scatter_points[i][1][:point_total], marker='.', s=1.00, c='k', linewidths=0.0)

        if saveData:
            fname = self.anlys_name + '_{:03.0f}'.format(degrees(phi_))
            self.IO.saveNumpyData(scatter_points, fname)

        ax.set_rmax(rminor)
        ax.set_rticks(np.arange(0.0, 0.19, 0.02))
        ax.yaxis.set_tick_params(labelsize=5)
        ax.grid(linewidth = 0.25, linestyle=':', c='k')
        phi_phys = (phi_ + (198 * np.pi/180.)) % (2*np.pi)  
        plt.title('$\phi_{{phy}}$={:02.0f}$\degree$ CW from North Split\n$\phi_c$={:02.0f}$\degree$'.format(phi_phys*180/np.pi, phi_*180/np.pi), loc='left')
        plot_name = self.anlys_name +'/'+ self.anlys_name + '_phi={:03.0f}.png'.format(phi_*180/np.pi)
        self.IO.saveFig(plot_name, dpi=250)
        plt.close()

        return '\tPHI: {}'.format(phi_*(180/np.pi))

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
