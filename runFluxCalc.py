import numpy as np
import utility.Flux_Calculator as fc
import utility.Flux_Gradientor as fg

NUMBER_PHI = 360
input_params = {
    'ANLYS_DIR': "AAAnewIO_iota3FWD_phi306",
    'ANLYS_SUBDIR': "LCFS8_360x180",
    'FIELD_FILE_TOR': 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy',
    'FIELD_FILE_HEL': 'input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy',
    'CURRENT_TOR': 0.486, #[kA]
    'CURRENT_HEL': 0.900, #[kA]
    'CONFIG_TOR': 'default_toroidal',
    'CONFIG_HEL': 'default_helical',
    'ENABLE_ERRFIELD': True,
    'LCFS_INDEX': 9,
    'NPHI': NUMBER_PHI,
    'NTHETA': 180,
    'PHI_GENs': np.linspace(360//NUMBER_PHI, 360, NUMBER_PHI),
    'MAX_SUBSETS': 3,
    'SMOOTH_FCTR': 3e-5, #7.5e-6 #baseline 1e-6
    'INTEGRATE_EPSABS': 5e-1,
    'INTEGRATE_EPSREL': 5e-2,
    'ISLAND_ALGORITHM': 'histogram', # 'kmeans', 'spectral'
    'HIST_BINS': 90,
    'PLOT_ALL': True,
    'BIG_MESH': True
}


## RUN ANALYSIS
island_index = fc.fluxCalculator(input_params)