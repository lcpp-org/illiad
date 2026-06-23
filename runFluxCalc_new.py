"""
#------------------------------------------------------#
# RUNNING FLUX CALCULATOR FOR HIDRA MAGNETIC SURFACES #
#------------------------------------------------------#
#        COIL CURRENTS NORMALLY RUN ON HIDRA           #
#------------------------------------------------------#
#  IOTA  |   I_T   |   I_H   |   I_V   |  PHI FWD/REV  #
#        |  [Amp]  |  [Amp]  |  [Amp]  |     [deg]     #
#  1/3   |   486   |   900   |    00   |    324/???    #
#  1/4   |   486   |   790   |    00   |    180/144    #
#  1/5   |   486   |   710   |    00   |    360/???    #
#  1/7   |   581   |   581   |    00   |    ???/???    #
#  MAX.  |  3500   |  7000   |    ??   |    ???/???    #
#------------------------------------------------------#
"""
import numpy as np
from classes.iohandler import IOHandler
from classes.mesh import Mesh
from classes.calculate_flux import FluxCalculator

#################
## USER INPUTS ##
#################

NPHI = 10 # Number of phi planes to evaluate
input_params = {
    'ANLYS_DIR': "iota3_1200spins_53Lines_LSODA_165deg", # Existing Poincare input directory
    'ANLYS_SUBDIR': "iota3_test12_170deg", # Name of new output subdirectory inside ANLYS_DIR
    'FIELD_FILE_TOR': 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy',
    'FIELD_FILE_HEL': 'input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy',
    'CURRENT_TOR': 0.486, #[kA]
    'CURRENT_HEL': 0.900, #[kA]
    'CONFIG_TOR': 'default_toroidal',
    'CONFIG_HEL': 'default_helical',
    'ENABLE_ERRFIELD': True,
    'LCFS_INDEX': 10, # Surface index selected from Poincare log
    'NPHI': NPHI,
    'NTHETA': 180,
    'PHI_GENs': np.linspace(360//NPHI, 360, NPHI),
    'MAX_SUBSETS': 3, # Number of magnetic islands
    'SMOOTH_FCTR': 7.5e-5, #7.5e-6, 3e-5 #baseline 1e-6
    'INTEGRATE_EPSABS': 5e-2,
    'INTEGRATE_EPSREL': 5e-3,
    'ISLAND_ALGORITHM': 'histogram', # 'kmeans', 'spectral'
    'HIST_BINS': 90,
    'PLOT_ALL': True,
    'BIG_MESH': True,
}


def main():
    ## SET UP RUN DIRECTORY (*DATA AND PLOTS WILL BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!*)
    ANLYS_DIR = input_params['ANLYS_DIR']
    ANLYS_SUBDIR = input_params['ANLYS_SUBDIR']
    simIO = IOHandler(ANLYS_DIR)
    simIO.setActiveSubDir(ANLYS_SUBDIR)
    simIO.startLog(log_name="fluxCalc.log", subdir=ANLYS_SUBDIR, logger_name="FluxCalculator")
    simIO.inputsBoilerplate(
        "FLUX CALCULATOR INPUTS",
        input_params,
        [
            "ANLYS_DIR",
            "ANLYS_SUBDIR",
            "FIELD_FILE_TOR",
            "FIELD_FILE_HEL",
            "CURRENT_TOR",
            "CURRENT_HEL",
            "CONFIG_TOR",
            "CONFIG_HEL",
            "ENABLE_ERRFIELD",
            "LCFS_INDEX",
            "NPHI",
            "NTHETA",
            "PHI_GENs",
            "MAX_SUBSETS",
            "SMOOTH_FCTR",
            "INTEGRATE_EPSABS",
            "INTEGRATE_EPSREL",
            "ISLAND_ALGORITHM",
            "HIST_BINS",
            "PLOT_ALL",
            "BIG_MESH",
        ],
    )

    ## DEFINE MESH AND LOAD MAGNETIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.setErrorField()
    b_hidra.loadCartesianField(coilCurrent=input_params['CURRENT_TOR'],
                               errField=input_params['ENABLE_ERRFIELD'],
                               att_mult=input_params['CONFIG_TOR'])
    b_hidra.addFieldPerturbation(coilCurrent=input_params['CURRENT_HEL'],
                                 att_mult=input_params['CONFIG_HEL'])
    ## RUN ANALYSIS
    flux_calc = FluxCalculator(simIO, b_hidra, input_params)
    flux_calc.run()


if __name__ == '__main__':
    main()