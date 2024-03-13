import os as os
import numpy as np
import matplotlib.pyplot as plt

class outputHandler:
    def __init__(self):
        # get script directory
        self.module_path = os.path.realpath(os.path.dirname(__file__))


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


    def saveNumpyData(self, data, name):
        name_loc = os.path.join( self.data_dir, name)
        np.save(name_loc, data)


    def saveFig(self, name):
        name_loc = os.path.join( self.plot_dir, name)
        plt.savefig(name_loc, dpi=900)

