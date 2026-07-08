import os as os
import numpy as np

import matplotlib.pyplot as plt

import logging
import logging.config

class IOHandler:
    """
    Class to handle output and logging.
    Creates an output folder underneath the module directory if none exists.
    Creates a 'run_name' sub-directory in '\output' with further sub-dirs for plot and data output.
    Includes a method to instantiate and configure logging.
    Includes methods for storing data output and plots output in proper sub-directories.
    """
    def __init__(self, run_name):
        # get script directory
        self.module_path = os.path.realpath(os.path.dirname(__file__))
        self.module_path = os.path.join(self.module_path, '..')
        self.module_path = os.path.abspath(self.module_path)
        print('Executing script in {}'.format(self.module_path))
              
        self.log = logging.getLogger("Poincare")

        # Create output directories if none exist
        #self.output_dir = os.path.join(self.module_path, '..', 'output')
        self.output_dir = os.path.join(self.module_path, 'output')
        self.output_dir = os.path.abspath(self.output_dir)
        
        try:
            print('Creating output directory if none exists...')
            os.mkdir(self.output_dir)
        except OSError as error:
            pass#print()

        self.run_dir = os.path.join(self.output_dir, run_name)
        print('Creating Run Directory: "{}"'.format(run_name))
        try:
            os.mkdir(self.run_dir)
        except OSError as error:
            print('Run Directory already exists!')

        self.data_dir = os.path.join(self.run_dir, 'data')
        try:
            os.mkdir(self.data_dir)
        except OSError as error:
            pass #print(error)

        self.plot_dir = os.path.join(self.run_dir, 'plots')
        try:
            os.mkdir(self.plot_dir)
        except OSError as error:
            pass #print(error)

        self.log_dir = os.path.join(self.run_dir, 'logs')
        try:
            os.mkdir(self.log_dir)
        except OSError as error:
            pass #print(error)

        self.active_subdir = None

    def startLog(self, log_name="simLog.log", subdir=None, mode="w", logger_name="Poincare"):
        # Creates a logger instance and configures the logging opions, handlers, formatting, etc.        
        self.log = logging.getLogger(logger_name)

        if subdir:
            log_dir = os.path.join(self.log_dir, subdir)
            try:
                os.makedirs(log_dir)
            except OSError as error:
                pass
            log_path = os.path.join(log_dir, log_name)
        else:
            log_path = os.path.join(self.run_dir, log_name)

        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,

            "formatters": {
                "simple": {
                    "format": "[%(levelname)s]: %(message)s",
                },
                "detailed": {
                    "format": "[%(levelname)s] %(asctime)s: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S%z"
                }
            },

            "handlers": {
                "stdout": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout",
                },
                "stderr": {
                    "class": "logging.StreamHandler",
                    "level": "WARNING",
                    "formatter": "simple",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "detailed",
                    "filename": log_path,
                    "mode": mode,
                    "maxBytes": 100000000, #100MB
                    "backupCount": 5
                }

            },

            "root": {
                "level": "DEBUG",
                "handlers": ["stdout", "file"]
            },
        }

        logging.config.dictConfig(config=logging_config)
        self.log.info(f"Started Logger: {log_path}")

        # create runStats file
        # create output file?

    def createSubDir(self, name, plots=True, data=True, logs=True):
        if self.active_subdir:
            active_prefix = self.active_subdir + os.sep
            if name != self.active_subdir and not name.startswith(active_prefix):
                name = os.path.join(self.active_subdir, name)

        sub_dir1 = os.path.join(self.plot_dir, name)
        sub_dir2 = os.path.join(self.data_dir, name)
        sub_dir3 = os.path.join(self.log_dir, name)
        if plots:
            try:
                os.makedirs(sub_dir1)
            except OSError as error:
                pass #print('Plot subDirectory already exists!')
        if data:
            try:
                os.makedirs(sub_dir2)
            except OSError as error:
                pass #print('Data subDirectory already exists!')
        if logs:
            try:
                os.makedirs(sub_dir3)
            except OSError as error:
                pass #print('Log subDirectory already exists!')

    def setActiveSubDir(self, name, plots=True, data=True, logs=True):
        self.createSubDir(name, plots=plots, data=data, logs=logs)
        self.active_subdir = name

    def _outputPath(self, base_dir, name, subdir=None):
        name = os.fspath(name)
        if os.path.isabs(name):
            return name

        active_subdir = self.active_subdir if subdir is None else subdir
        if active_subdir:
            active_prefix = active_subdir + os.sep
            if name == active_subdir or name.startswith(active_prefix):
                return os.path.join(base_dir, name)
            return os.path.join(base_dir, active_subdir, name)

        return os.path.join(base_dir, name)

    def saveNumpyData(self, data, name, subdir=None):
        # method to store a numpy array in the \data sub-directory
        name_loc = self._outputPath(self.data_dir, name, subdir=subdir)
        os.makedirs(os.path.dirname(name_loc), exist_ok=True)
        #self.log.info(f'saving numpy file: "{name_loc}"')
        np.save(name_loc, data)

    def loadNumpyData(self, name, subdir=None, mmap_mode=None):
        # method to load a numpy array from the \data sub-directory
        if subdir:
            name_loc = os.path.join(self.data_dir, subdir, name)
        else:
            name_loc = os.path.join(self.data_dir, name)

        self.log.info('loading numpy file: "{}"'.format(name_loc))

        return np.load(name_loc, mmap_mode=mmap_mode)

    def saveFig(self, name, dpi=300, subdir=None):
        # method to store  a plot in the \plots sub-directory
        name_loc = self._outputPath(self.plot_dir, name, subdir=subdir)
        os.makedirs(os.path.dirname(name_loc), exist_ok=True)
        plt.savefig(name_loc, dpi=dpi, bbox_inches=None)

    def loadPorts_fromCSV(self, name):
        """ 
        Method to load the HIDRA port locations and sizes for plotting.
        Returns array of locations in phi, theta coordinates, and height and width in degrees.
        """
        name_loc = name #os.path.join('self.module_path', name)
        portdata = np.loadtxt(name_loc, delimiter=',', skiprows=1, usecols=[2,3,4,5,6])

        ## PARSE DATA
        p_phi = portdata[:,0]
        p_theta = portdata[:,1]
        p_rmaj = portdata[:,2]
        p_rmin = portdata[:,3]
        p_dia = portdata[:,4]/1000 # convert mm to meters
        p_height = np.degrees(np.arcsin(p_dia/p_rmin)) # calculate height in degrees
        p_width =  np.degrees(np.arcsin(p_dia/p_rmaj)) # calculate width in degrees

        return np.array([p_phi, p_theta, p_width, p_height])
    
    def loadCSV(self, name):
        """ 
        Method to load arbitrary data from a CSV file.
        Logs the header and returns a numpy array with the data.
        """
        name_loc = name
        with open(name_loc, 'r') as f:
            header = f.readline().strip()
        self.log.info(f'Loading [{header}] from CSV file: "{name_loc}"')
        data = np.loadtxt(name_loc, delimiter=',', skiprows=1)
        return data
    
    def saveCSV(self, data, name, header=None):
        """ 
        Method to save data to a CSV file.
        If header is provided, it will be written as the first line.
        """
        name_loc = self._outputPath(self.data_dir, name)
        os.makedirs(os.path.dirname(name_loc), exist_ok=True)
        if header:
            np.savetxt(name_loc, data, delimiter=',', header=header, comments='')
        else:
            np.savetxt(name_loc, data, delimiter=',')
        self.log.info(f'Saved data to CSV file: "{name_loc}"')

    def inputsBoilerplate(self, title, param_dict=None, input_parameters=None):
        if input_parameters is None:
            input_parameters = sorted(param_dict.keys()) if param_dict is not None else []

        param_values = {}
        for param in input_parameters:
            if param_dict is not None and param in param_dict:
                param_values[param] = param_dict[param]
            elif hasattr(self, param):
                param_values[param] = getattr(self, param)
            elif param in globals():
                param_values[param] = globals()[param]
            else:
                param_values[param] = '*DEFAULT*'

        lines = [
            "",
            "|=======================================================================================|",
            f"| {title}",
            "|---------------------------------------------------------------------------------------|",
        ]
        for param in input_parameters:
            lines.append(f"| {param}: {param_values[param]}")
        lines.append("|=======================================================================================|\n")
        self.log.info("\n".join(lines))

    def borisBoilerplate(self, param_dict=None):
        """
        Logs the values of a predefined set of simulation input parameters, indicating whether each was set or is using a default value.
        Useful for recording the configuration of a simulation run for reproducibility and debugging.

        This method checks for parameter values first in the provided param_dict (e.g., globals() or locals() from the caller),
        then as instance attributes of the IOHandler object, and if not found, attempts to retrieve them from global variables.
        If none is found, '*DEFAULT*' is used.
        """
        # List of all possible input parameters to check if user has set them
        possible_input_parameters = [
            "FIELD_FILE_TOR",
            "CONFIG_TOR",
            "TOROIDAL_CURRENT",
            "FIELD_SCALE_TOR",
            "FIELD_FILE_HEL",
            "CONFIG_HEL",
            "HELICAL_CURRENT",
            "FIELD_SCALE_HEL",
            "ENABLE_ERRFIELD",
            "ERRFIELD_MAG",
            "ERRFIELD_DIR_DEG",
            "FIELD_FILE_ELECTRIC",
            "ELECTRIC_POTENTIAL",
            "ELECTRON_TEMP_EV",
            "BACKGROUND_GAS_SPECIES",
            "M_GAS_AMU",
            "FIELD_FILE_DENSITY",
            "ION_NEUTRAL_COLLISIONS",
            "ION_ION_COLLISIONS",
            "NEUTRAL_GAS_DENSITY",
            "PLASMA_DENSITY",
            "LCFS_INDEX",
            "ION_TEMP",
            "ION_MASS",
            "CHARGE_NUM",
            "DELTRS",
            "NPHI",
            "NTHETA",
            "NPARTICLES_PER_EMITTER",
            "DT",
            "TMAX",
            "NSTEPS",
            "TRACK_NPHI",
            "TRACK_NTHETA",
            "TRACK_NPARTICLES_PER_EMITTER",
            "STRIDE",
            "TRACE_STRIDE",
            "OUTPUT_DIRECTORY_NAME",
            "TAG"
        ]
    
        # Build a dictionary of parameter values or '*DEFAULT*'
        param_values = {}
        for param in possible_input_parameters:
            if param_dict is not None and param in param_dict:
                param_values[param] = param_dict[param]
            elif hasattr(self, param):
                param_values[param] = getattr(self, param)
            elif param in globals():
                param_values[param] = globals()[param]
            else:
                param_values[param] = '*DEFAULT*'
    
        self.log.info('\n|=======================================================================================|'
                      +'\n| LOADED TOROIDAL FIELD DATA FROM: {}'.format(param_values["FIELD_FILE_TOR"])
                      +'\n| LOADED TOROIDAL FIELD CONFIG: {}'.format(param_values["CONFIG_TOR"])
                      +'\n| LOADED TOROIDAL COIL CURRENT: {}'.format(param_values["TOROIDAL_CURRENT"])
                      +'\n| LOADED TOROIDAL FIELD SCALING FACTOR: {}'.format(param_values["FIELD_SCALE_TOR"])
                      +'\n| LOADED HELICAL FIELD DATA FROM: {}'.format(param_values["FIELD_FILE_HEL"])
                      +'\n| LOADED HELICAL FIELD CONFIG: {}'.format(param_values["CONFIG_HEL"])
                      +'\n| LOADED HELICAL COIL CURRENT: {}'.format(param_values["HELICAL_CURRENT"])
                      +'\n| LOADED HELICAL FIELD SCALING FACTOR: {}'.format(param_values["FIELD_SCALE_HEL"])
                      +'\n| ENABLE ERROR FIELD: {}'.format(param_values["ENABLE_ERRFIELD"])
                      +'\n| LOADED ERRFIELD MAG: {}'.format(param_values["ERRFIELD_MAG"])
                      +'\n| LOADED ERRFIELD DIR: {}'.format(param_values["ERRFIELD_DIR_DEG"])
                      +'\n| LOADED ELECTRIC FIELD DATA FROM: {}'.format(param_values["FIELD_FILE_ELECTRIC"])
                      +'\n| LOADED ELECTRIC FIELD SCALING FACTOR: {}'.format(param_values["ELECTRIC_POTENTIAL"])
                      +'\n| LOADED ELECTRON TEMPERATURE: {} eV'.format(param_values["ELECTRON_TEMP_EV"])
                      +'\n| LOADED BACKGROUND GAS SPECIES: {}'.format(param_values["BACKGROUND_GAS_SPECIES"])
                      +'\n| LOADED BACKGROUND GAS MASS: {} amu'.format(param_values["M_GAS_AMU"])
                      +'\n| LOADED DENSITY FIELD DATA FROM: {}'.format(param_values["FIELD_FILE_DENSITY"])
                      +'\n| ION-NEUTRAL COLLISIONS: {}'.format(param_values["ION_NEUTRAL_COLLISIONS"])
                      +'\n| ION-ION COLLISIONS: {}'.format(param_values["ION_ION_COLLISIONS"])
                      +'\n| NEUTRAL GAS DENSITY: {} m^-3'.format(param_values["NEUTRAL_GAS_DENSITY"])
                      +'\n| PLASMA DENSITY: {} m^-3'.format(param_values["PLASMA_DENSITY"])
                      +'\n|---------------------------------------------------------------------------------------|'
                      +'\n| LAST-CLOSED FLUX SURFACE INDEX: {}'.format(param_values["LCFS_INDEX"])
                      +'\n| ION TEMPERATURE: {} eV'.format(param_values["ION_TEMP"])
                      +'\n| ION MASS: {} amu'.format(param_values["ION_MASS"])
                      +'\n| ION CHARGE: {}'.format(param_values["CHARGE_NUM"])
                      +'\n|---------------------------------------------------------------------------------------|'
                      +'\n| RUNNING {} EMITTERS WITH {} PARTICLES PER EMITTER'.format(
                            len(param_values["DELTRS"]) if param_values["DELTRS"] != '*DEFAULT*' else '*DEFAULT*',
                            param_values["NPARTICLES_PER_EMITTER"])
                      +'\n|  --> TOTAL PARTICLES: {}'.format(
                            (len(param_values["DELTRS"]) * param_values["NPHI"] * param_values["NTHETA"] * param_values["NPARTICLES_PER_EMITTER"])
                            if all(param_values[x] != '*DEFAULT*' for x in ["DELTRS", "NPHI", "NTHETA", "NPARTICLES_PER_EMITTER"])
                            else '*DEFAULT*')
                      +'\n|---------------------------------------------------------------------------------------|'
                      +'\n| TIME STEP: {} sec'.format(param_values["DT"])
                      +'\n| TOTAL TIME: {:.6f} sec'.format(param_values["TMAX"] if param_values["TMAX"] != '*DEFAULT*' else 0)
                      +'\n|  --> # OF TIME STEPS: {}'.format(param_values["NSTEPS"])
                      +'\n| TRACE TRACK_NPHI: {}'.format(param_values["TRACK_NPHI"])
                      +'\n| TRACE TRACK_NTHETA: {}'.format(param_values["TRACK_NTHETA"])
                      +'\n| TRACE TRACK_NPARTICLES_PER_EMITTER: {}'.format(param_values["TRACK_NPARTICLES_PER_EMITTER"])
                      +'\n| TRACE OUTPUT STRIDE: {}'.format(
                            param_values["TRACE_STRIDE"] if param_values["TRACE_STRIDE"] != '*DEFAULT*'
                            else param_values["STRIDE"])
                      +'\n| OUTPUT DIRECTORY NAME: {}'.format(param_values["OUTPUT_DIRECTORY_NAME"])
                      +'\n| TAG: {}'.format(param_values["TAG"])
                      +'\n|=======================================================================================|\n\n\n')
