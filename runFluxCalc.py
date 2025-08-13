
import numpy as np
import utility.Flux_Calculator as fc

fc.ANLYS_DIR = "ChangetoIota4_1500spins_atole-8_eng"
fc.ANLYS_SUBDIR = "LCFS39_4x36x360mesh_OLDclstr"

## DEFINE FIELDS
fc.FIELD_FILE_TOR = 'input_files/It486_Ih000_Iv000_1p000_1p000_64bit.npy'
fc.FIELD_FILE_HEL = 'input_files/It000_Ih900_Iv000_1p000_1p000_64bit.npy'
fc.CURRENT_TOR = 0.486 #[kA]
fc.CURRENT_HEL = 0.790 #[kA]
fc.CONFIG_TOR = 'default_toroidal'
fc.CONFIG_HEL = 'default_helical_rev'

## DEFINE LCFS AND ANGLES TO EVALUATE
fc.LCFS_INDEX = 39 #40 #22 #29?
fc.NPHI = 36
fc.NTHETA = 360
fc.PHI_GENs = np.linspace(360//fc.NPHI, 360, fc.NPHI)

## FLUX INTEGRATION PARAMETERS
fc.MAX_SUBSETS = 4
#SMOOTH_FCTR = 8.0e-6 #7.5e-6 #baseline 1e-6
fc.SMOOTH_FCTR = 1.0e-5 #7.5e-6 #baseline 1e-6
# INTEGRATE_EPSABS=1e-5 #1.49e-5
# INTEGRATE_EPSREL=1e-3 #4.49e-3
fc.INTEGRATE_EPSABS=1e-3
fc.INTEGRATE_EPSREL=1e-2

## PLOTTING FLAG
fc.PLOT_ALL = True

fc.fluxCalculator()