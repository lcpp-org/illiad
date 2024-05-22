import os as os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import logging
import logging.config

class IOHandler:
    # Class to handle output and logging
    # Creates an output folder underneath the module directory if none exists
    # Creates a 'run_name' sub-directory in '\output' with further sub-dirs for plot and data output
    # includes a method to instantiate and configure logging
    # includes methods for storing data output and plots output in proper sub-directories
    def __init__(self, run_name):
        # get script directory
        self.module_path = os.path.realpath(os.path.dirname(__file__))

        self.log = logging.getLogger("Poincare")

        #self.module_path = os.path.realpath(os.path.dirname(__file__))
        print(f'Executing script in {self.module_path}')

        # Create output directories if none exist
        self.output_dir = os.path.join(self.module_path,'output')
        try:
            print('Creating output directory if none exists...')
            os.mkdir(self.output_dir)
        except OSError as error:
            pass#print()

        self.run_dir = os.path.join(self.output_dir, run_name)
        print(f'Creating Run Directory: "{run_name}"')
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


    def createSubDir(self, name):
        self.sub_dir = os.path.join(self.plot_dir, name)
        try:
            os.mkdir(self.sub_dir)
        except OSError as error:
            pass#print('Run Directory already exists!')

    def saveNumpyData(self, data, name):
        # method to store a numpy array in the \data sub-directory
        name_loc = os.path.join( self.data_dir, name)
        #self.log.info(f'saving numpy file: "{name_loc}"')
        np.save(name_loc, data)

    def loadNumpyData(self, name):
        # method to load a numpy array from the \data sub-directory
        name_loc = os.path.join( self.data_dir, name)
        self.log.info(f'loading numpy file: "{name_loc}"')

        return np.load(name_loc)
        #try:
        #    return np.load(name_loc)
        #except OSError as error:
        #    print('FILE DOES NOT EXIST!')
    
    def saveFig(self, name):
        # method to store  a plot in the \plots sub-directory
        name_loc = os.path.join( self.plot_dir, name)
        plt.savefig(name_loc, dpi=400)

    def loadPorts_fromCSV(self, name):
        # method to load the HIDRA port locations and sizes
        name_loc = name #os.path.join('self.module_path', name)
 
        portdata = pd.read_csv(
            name_loc,
            header=None,
            index_col=None,
            skiprows=1,
            delim_whitespace=False,
            engine='python')
        
        ## PARSE DATA
        #p_type = portdata.loc[:,0].values
        p_phi = np.array(portdata.loc[:,2].values)
        p_theta = np.array(portdata.loc[:,3].values)
        p_rmaj = np.array(portdata.loc[:,4].values)
        p_rmin = np.array(portdata.loc[:,5].values)
        p_dia = np.array(portdata.loc[:,6].values/1000) # convert mm to meters
        p_height = np.degrees(np.arcsin(p_dia/p_rmin)) # calculate height in degrees
        p_width =  np.degrees(np.arcsin(p_dia/p_rmaj)) # calculate width in degrees

        return np.array([p_phi, p_theta, p_width, p_height])



    #def wallOutput(self, )