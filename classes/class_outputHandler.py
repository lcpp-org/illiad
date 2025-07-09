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

        self.log = logging.getLogger("Poincare")

        #self.module_path = os.path.realpath(os.path.dirname(__file__))
        print('Executing script in {}'.format(self.module_path))

        # Create output directories if none exist
        self.output_dir = os.path.join(self.module_path, '..', 'output')
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
            print(error)

        self.plot_dir = os.path.join(self.run_dir, 'plots')
        try:
            os.mkdir(self.plot_dir)
        except OSError as error:
            print(error)

    def startLog(self):
        # Creates a logger instance and configures the logging opions, handlers, formatting, etc.        
        self.log = logging.getLogger("Poincare")

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
                    "filename": self.run_dir + "/simLog.log",
                    "maxBytes": 100000000, #100MB
                    "backupCount": 5
                }

            },

            "loggers": {
                "root": {"level": "DEBUG", "handlers": ["stdout", "file"]}
            },
        }

        logging.config.dictConfig(config=logging_config)
        self.log.info(f"Started Logger in {self.run_dir}")

        # create runStats file
        # create output file?

    def createSubDir(self, name, plots=True, data=True):
        sub_dir1 = os.path.join(self.plot_dir, name)
        sub_dir2 = os.path.join(self.data_dir, name)
        if plots:
            try:
                os.mkdir(sub_dir1)
            except OSError as error:
                pass #print('Plot subDirectory already exists!')
        if data:
            try:
                os.mkdir(sub_dir2)
            except OSError as error:
                pass #print('Data subDirectory already exists!')

    def saveNumpyData(self, data, name):
        # method to store a numpy array in the \data sub-directory
        name_loc = os.path.join( self.data_dir, name)
        #self.log.info(f'saving numpy file: "{name_loc}"')
        np.save(name_loc, data)

    def loadNumpyData(self, name, subdir=None):
        # method to load a numpy array from the \data sub-directory
        if subdir:
            name_loc = os.path.join(self.data_dir, subdir, name)
        else:
            name_loc = os.path.join(self.data_dir, name)

        self.log.info('loading numpy file: "{}"'.format(name_loc))

        return np.load(name_loc)

    def saveFig(self, name, dpi=300):
        # method to store  a plot in the \plots sub-directory
        name_loc = os.path.join( self.plot_dir, name)
        plt.savefig(name_loc, dpi=dpi)

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
        name_loc = os.path.join(self.data_dir, name)
        if header:
            np.savetxt(name_loc, data, delimiter=',', header=header, comments='')
        else:
            np.savetxt(name_loc, data, delimiter=',')
        self.log.info(f'Saved data to CSV file: "{name_loc}"')

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
            "TOROIDAL_CURRENT",
            "FIELD_SCALE_TOR",
            "FIELD_FILE_HEL",
            "HELICAL_CURRENT",
            "FIELD_SCALE_HEL",
            "ERRFIELD_MAG",
            "ERRFIELD_DIR_DEG",
            "FIELD_FILE_ELECTRIC",
            "FIELD_SCALE_ELECTRIC",
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
            "NSTEPS"
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
                      +'\n| LOADED TOROIDAL COIL CURRENT: {}'.format(param_values["TOROIDAL_CURRENT"])
                      +'\n| LOADED TOROIDAL FIELD SCALING FACTOR: {}'.format(param_values["FIELD_SCALE_TOR"])
                      +'\n| LOADED HELICAL FIELD DATA FROM: {}'.format(param_values["FIELD_FILE_HEL"])
                      +'\n| LOADED HELICAL COIL CURRENT: {}'.format(param_values["HELICAL_CURRENT"])
                      +'\n| LOADED HELICAL FIELD SCALING FACTOR: {}'.format(param_values["FIELD_SCALE_HEL"])
                      +'\n| LOADED ERRFIELD MAG: {}'.format(param_values["ERRFIELD_MAG"])
                      +'\n| LOADED ERRFIELD DIR: {}'.format(param_values["ERRFIELD_DIR_DEG"])
                      +'\n| LOADED ELECTRIC FIELD DATA FROM: {}'.format(param_values["FIELD_FILE_ELECTRIC"])
                      +'\n| LOADED ELECTRIC FIELD SCALING FACTOR: {}'.format(param_values["FIELD_SCALE_ELECTRIC"])
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
                      +'\n|=======================================================================================|\n\n\n')
