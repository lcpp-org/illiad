import numpy as np
import utility.Flux_Calculator as fc

fc.ANLYS_DIR = "It-0486_Ih-0790_1500sp_LSODA2p49e8"
fc.ANLYS_SUBDIR = "LCFS6_10x180_atol-8_rtol-2_testTest1again"

## DEFINE FIELDS
fc.FIELD_FILE_TOR = 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy'
fc.FIELD_FILE_HEL = 'input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy'
fc.CURRENT_TOR = 0.486 #[kA]
fc.CURRENT_HEL = 0.790 #[kA]
fc.CONFIG_TOR = 'default_toroidal'
fc.CONFIG_HEL = 'default_helical'

## DEFINE LCFS AND ANGLES TO EVALUATE
fc.LCFS_INDEX = 6 #26
fc.NPHI = 10
fc.NTHETA = 180
fc.PHI_GENs = np.linspace(360//fc.NPHI, 360, fc.NPHI)

## FLUX INTEGRATION PARAMETERS
fc.MAX_SUBSETS = 4
fc.SMOOTH_FCTR = 1e-5 #7.5e-6 #baseline 1e-6
fc.INTEGRATE_EPSABS = 5e-8
fc.INTEGRATE_EPSREL = 5e-2

## PLOTTING FLAG
fc.ISLAND_ALGORITHM = 'histogram' # 'kmeans', 'spectral'
fc.HIST_BINS = 120
fc.PLOT_ALL = True
fc.BIG_MESH = False

## RUN ANALYSIS
fc.fluxCalculator()