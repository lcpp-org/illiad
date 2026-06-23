import argparse
import numpy as np
import utility.Flux_Calculator as fc
import utility.Flux_Interpolator as fi
import utility.Flux_Gradientor as fg
from utility.run_config import load_inputs_json, merge_input_params, normalize_phi_gens

<<<<<<< HEAD
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
=======
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

>>>>>>> origin/main

N_phi = 10 #360
input_params = {
    'ANLYS_DIR': "iota3_1200spins_53Lines_LSODA_165deg", #iota3FWD_1000spins_53Lines_LSODA_flux, 
    'ANLYS_SUBDIR': "iota3_test4_165deg", # Name of existing output subdirectory inside ANLYS_DIR
    'FIELD_FILE_TOR': 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy',
    'FIELD_FILE_HEL': 'input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy',
    'CURRENT_TOR': 0.486, #[kA]
    'CURRENT_HEL': 0.790, #[kA]
    'CONFIG_TOR': 'default_toroidal',
    'CONFIG_HEL': 'default_helical',
    'ENABLE_ERRFIELD': True,
<<<<<<< HEAD
    'LCFS_INDEX': 10, # start with the value from the logfile
    'NPHI': N_phi,
    'NTHETA': 180,
    'PHI_GENs': np.linspace(360//N_phi, 360, N_phi),
    'MAX_SUBSETS': 3, # have to make number of islands
    'SMALLEST_ISLAND_INDEX': 29, #61, #53 #39
    'SMOOTH_FCTR': 7.5e-5, #7.5e-6, 3e-5 #baseline 1e-6
    'INTEGRATE_EPSABS': 5e-2,
    'INTEGRATE_EPSREL': 5e-3,
    'ISLAND_ALGORITHM': 'histogram', # 'kmeans', 'spectral'
    'HIST_BINS': 90,
    'PLOT_ALL': True,
    'BIG_MESH': False,
    }

## RUN ANALYSIS
#smallest_island_index = fc.fluxCalculator(input_params)
input_params['SMALLEST_ISLAND_INDEX'] = 25 

## RUN ANALYSIS
input_params['ALPHA'] = 1.0
input_params['DEBUG'] = False #Debug helper 
input_params['LCFS_INDEX'] = 10
input_params['INV_SURF_INDICES'] = [2] #[55, 56, 57, 58] #[18,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]
input_params['GUESS_PHI_INDEX'] = -1 #16, 18
=======
    'LCFS_INDEX': 15,
    'NPHI': N_phi,
    'NTHETA': 180,
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

_CLI_INPUTS = object()

def parse_args():
    parser = argparse.ArgumentParser(description="Run ILLIAD flux interpolation and gradient generation.")
    parser.add_argument(
        "--inputs-json",
        default=None,
        help="Optional path to a JSON object overriding runFluxGrad.py defaults.",
    )
    return parser.parse_args()
>>>>>>> origin/main


<<<<<<< HEAD
# ## RUN ANALYSIS
input_params['OUTPUT_FILE_NAME'] = "Efield_iota3_test1_165deg"
fg.fluxGradientor(input_params)
=======
def main(input_params_override=_CLI_INPUTS):
    if input_params_override is _CLI_INPUTS:
        args = parse_args()
        input_params_override = load_inputs_json(args.inputs_json, "Flux gradient inputs") if args.inputs_json else None
    params = merge_input_params(input_params, input_params_override)
    ## RUN ANALYSIS
    #smallest_island_index = fc.fluxCalculator(input_params)
    #input_params['SMALLEST_ISLAND_INDEX'] = 27 #61

    ## RUN ANALYSIS
    params.setdefault('ALPHA', 1.0)
    params.setdefault('DEBUG', True)
    params.setdefault('LCFS_INDEX', 15)
    params.setdefault('INV_SURF_INDICES', [20])
    params.setdefault('GUESS_PHI_INDEX', -3)
    params.setdefault('OUTPUT_FILE_NAME', "Efield_LCFS15")
    normalize_phi_gens(params)

    ## RUN ANALYSIS
    fi.fluxInterpolator(params)
    fg.fluxGradientor(params)


if __name__ == "__main__":
    main()
>>>>>>> origin/main
