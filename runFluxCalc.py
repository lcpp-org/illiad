import numpy as np
import utility.Flux_Calculator as fc
import utility.Flux_Gradientor as fg

NUMBER_PHI = 20
input_params = {
    #'ANLYS_DIR': "AAAnewIO_iota3FWD_phi306_LSODA",
    'ANLYS_DIR': "It-0486_Ih-0790_PHI180_1500spins_105Lines_LSODA1e9_newEvents",
    'ANLYS_SUBDIR': "LCFS15_20x180_FIXED",
    'FIELD_FILE_TOR': 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy',
    'FIELD_FILE_HEL': 'input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy',
    'CURRENT_TOR': 0.486, #[kA]
    'CURRENT_HEL': 0.790, #[kA]
    'CONFIG_TOR': 'default_toroidal',
    'CONFIG_HEL': 'default_helical',
    'ENABLE_ERRFIELD': True,
    'LCFS_INDEX': 15,
    'NPHI': NUMBER_PHI,
    'NTHETA': 180,
    'PHI_GENs': np.linspace(360//NUMBER_PHI, 360, NUMBER_PHI),
    'MAX_SUBSETS': 4,
    'SMOOTH_FCTR': 1e-6, #3e-5, #7.5e-6 #baseline 1e-6
    'INTEGRATE_EPSABS': 1e-2,
    'INTEGRATE_EPSREL': 1e-1,
    'ISLAND_ALGORITHM': 'histogram', # 'kmeans', 'spectral'
    'HIST_BINS': 90,
    'PLOT_ALL': True,
    'BIG_MESH': True
}


## RUN ANALYSIS
island_index = fc.fluxCalculator(input_params)