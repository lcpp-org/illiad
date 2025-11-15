import numpy as np
import utility.Flux_Calculator as fc

# fc.ANLYS_DIR = "LSODA-2p49e8_Iota4FWD_1200spins_320Lines"
# fc.ANLYS_SUBDIR = "LCFS100_4x360x180mesh_tole1e2_LOMEM"

# fc.ANLYS_DIR = "AcceptedIota4_1500spins_atole-8_eng"
# fc.ANLYS_SUBDIR = "LCFS35_180x360_tol_5e1_5e2_APS2025"
fc.ANLYS_DIR = "AcceptedIota3_1500spins_atole-9"
fc.ANLYS_SUBDIR = "LCFS19_360x180_tol_5e1_5e2_APS2025"

## DEFINE FIELDS
fc.FIELD_FILE_TOR = 'input_files/It1000_Ih000_Iv000_1p000_1p000_64bit.npy'
fc.FIELD_FILE_HEL = 'input_files/It000_Ih1000_Iv000_1p000_1p000_64bit.npy'
fc.CURRENT_TOR = 0.486 #[kA]
fc.CURRENT_HEL = 0.900 #[kA]
fc.CONFIG_TOR = 'default_toroidal'
fc.CONFIG_HEL = 'default_helical'

## DEFINE LCFS AND ANGLES TO EVALUATE
fc.LCFS_INDEX = 19 #100  #1f00 #40 #22 #29?
fc.NPHI = 360
fc.NTHETA = 180
fc.PHI_GENs = np.linspace(360//fc.NPHI, 360, fc.NPHI)

## FLUX INTEGRATION PARAMETERS
fc.MAX_SUBSETS = 3
fc.SMOOTH_FCTR = 7.5e-6 #baseline 1e-6
fc.INTEGRATE_EPSABS=5e-1
fc.INTEGRATE_EPSREL=5e-2

## PLOTTING FLAG
fc.ISLAND_ALGORITHM = 'histogram' # 'kmeans', 'spectral'
fc.HIST_BINS = 120
fc.PLOT_ALL = True
fc.BIG_MESH = False

## RUN ANALYSIS
fc.fluxCalculator()