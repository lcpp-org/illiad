import os as os
import numpy as np
import matplotlib.pyplot as plt

import logging
import logging.config

class outputHandler:
    def __init__(self):
        # get script directory
        self.module_path = os.path.realpath(os.path.dirname(__file__))

        self.log = logging.getLogger("Poincare")


    def setupOutputDirectory(self, run_name):
       
        self.module_path = os.path.realpath(os.path.dirname(__file__))
        print(f'Executing script in {self.module_path}')

        # Create output directories if none exist
        self.output_dir = os.path.join(self.module_path,'output')
        try:
            os.mkdir(self.output_dir)
        except OSError as error:
            print(error)

        self.run_dir = os.path.join(self.output_dir, run_name)
        try:
            os.mkdir(self.run_dir)
        except OSError as error:
            print(error)

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
            #def setupLogger:
        log = logging.getLogger("Poincare")

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
                    "formatter": "detailed",
                    "stream": "ext://sys.stdout",
                },
                "stderr": {
                    "class": "logging.StreamHandler",
                    "level": "WARNING",
                    "formatter": "detailed",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "simple",
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
        self.log.info("Started Logger")




    def saveNumpyData(self, data, name):
        name_loc = os.path.join( self.data_dir, name)
        np.save(name_loc, data)


    def saveFig(self, name):
        name_loc = os.path.join( self.plot_dir, name)
        plt.savefig(name_loc, dpi=900)

