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
        Returns a numpy array with the data.
        """
        name_loc = name #os.path.join('self.module_path', name)
        data = np.loadtxt(name_loc, delimiter=',', skiprows=0)#, usecols=[2,3,4,5,6])
        
        self.log.info('Loading {} from file: "{}"'.format(data[0,:], name_loc))

        return data[1:,:]  # return data without header