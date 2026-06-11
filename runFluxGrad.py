import numpy as np
import utility.Flux_Calculator as fc
import utility.Flux_Interpolator as fi
import utility.Flux_Gradientor as fg

#------------------------------------------------------#
# GENERATING POINCARE PLOTS FOR HIDRA'S MAGNETIC FIELD #
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

N_phi = 90 #360
input_params = {
    # 'ANLYS_DIR': "AcceptedIota3_1500spins_atole-9",
    # 'ANLYS_SUBDIR': "LCFS19_360x180_ARTICLE_smooth3e-5",
    #'ANLYS_SUBDIR': "LCFS14_360x180_ARTICLE",
    'ANLYS_DIR': "iota4FWD_1000spins_53Lines_LSODA_flux", #iota3FWD_1000spins_53Lines_LSODA_flux, 
    'ANLYS_SUBDIR': "first_flux_run", # Name of existing output subdirectory inside ANLYS_DIR
    'FIELD_FILE_TOR': 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy',
    'FIELD_FILE_HEL': 'input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy',
    'CURRENT_TOR': 0.486, #[kA]
    'CURRENT_HEL': 0.790, #[kA]
    'CONFIG_TOR': 'default_toroidal',
    'CONFIG_HEL': 'default_helical',
    'ENABLE_ERRFIELD': True,
    'LCFS_INDEX': 4, # start with the value from the logfile
    'NPHI': N_phi,
    'NTHETA': 180,
    'PHI_GENs': np.linspace(360//N_phi, 360, N_phi),
    'MAX_SUBSETS': 4, # have to make number of islands
    'SMALLEST_ISLAND_INDEX': 38, #61, #53 #39
    'SMOOTH_FCTR': 3e-5, #7.5e-6, 3e-5 #baseline 1e-6
    'INTEGRATE_EPSABS': 5e-2,
    'INTEGRATE_EPSREL': 5e-3,
    'ISLAND_ALGORITHM': 'histogram', # 'kmeans', 'spectral'
    'HIST_BINS': 90,
    'PLOT_ALL': True,
    'BIG_MESH': False,
    }

## RUN ANALYSIS
#smallest_island_index = fc.fluxCalculator(input_params)
input_params['SMALLEST_ISLAND_INDEX'] = 38 #61

## RUN ANALYSIS
input_params['ALPHA'] = 1.0
input_params['DEBUG'] = True #Debug helper 
# New input parameters
input_params['SAVE_BEST_PROFILE'] = True
input_params['DPI'] = 100
input_params['LCFS_INDEX'] = 4
input_params['INV_SURF_INDICES'] = [28] #[55, 56, 57, 58] #[18,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]
input_params['GUESS_PHI_INDEX'] = -5

fi.fluxInterpolator(input_params)

# ## RUN ANALYSIS
input_params['OUTPUT_FILE_NAME'] = "Efield_iota4_test1"
fg.fluxGradientor(input_params)