import numpy as np
import utility.Flux_Calculator as fc
import utility.Flux_Interpolator as fi
import utility.Flux_Gradientor as fg

N_phi = 360
input_params = {
    # 'ANLYS_DIR': "AcceptedIota3_1500spins_atole-9",
    # 'ANLYS_SUBDIR': "LCFS19_360x180_ARTICLE_smooth3e-5",
    #'ANLYS_SUBDIR': "LCFS14_360x180_ARTICLE",
    #'ANLYS_DIR': "AAAnewIO_iota3FWD_phi306_LSODA",
    'ANLYS_DIR': "It-0486_Ih-0790_PHI180_1500spins_105Lines_LSODA1e9_newEvents",
    #'ANLYS_SUBDIR': "LCFS21_20x180",
    #'ANLYS_SUBDIR': "LCFS40_20x360_fixd2_abserr1e-2_relerr1e-1_meshErrTest2",
    'ANLYS_SUBDIR': "LCFS15_20x180_FIXED",


    'FIELD_FILE_TOR': 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy',
    'FIELD_FILE_HEL': 'input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy',
    'CURRENT_TOR': 0.486, #[kA]
    'CURRENT_HEL': 0.790, #[kA]
    'CONFIG_TOR': 'default_toroidal',
    'CONFIG_HEL': 'default_helical',
    'ENABLE_ERRFIELD': True,
    'LCFS_INDEX': 15,
    'NPHI': N_phi,
    'NTHETA': 180,
    'PHI_GENs': np.linspace(360//N_phi, 360, N_phi),
    'MAX_SUBSETS': 4,
    'SMALLEST_ISLAND_INDEX': 61, #61, #53 #39
    # 'SMOOTH_FCTR': 3e-5, #7.5e-6 #baseline 1e-6
    # 'INTEGRATE_EPSABS': 5e-1,
    # 'INTEGRATE_EPSREL': 5e-2,
    # 'ISLAND_ALGORITHM': 'histogram', # 'kmeans', 'spectral'
    # 'HIST_BINS': 90,
    # 'PLOT_ALL': True,
    # 'BIG_MESH': False

    }

## RUN ANALYSIS
#smallest_island_index = fc.fluxCalculator(input_params)
#input_params['SMALLEST_ISLAND_INDEX'] = 27 #61

## RUN ANALYSIS
input_params['ALPHA'] = 1.0
input_params['DEBUG'] = True
input_params['LCFS_INDEX'] = 15
input_params['INV_SURF_INDICES'] = [20]
input_params['GUESS_PHI_INDEX'] = -3

fi.fluxInterpolator(input_params)

# ## RUN ANALYSIS
input_params['OUTPUT_FILE_NAME'] = "Efield_LCFS15"
fg.fluxGradientor(input_params)