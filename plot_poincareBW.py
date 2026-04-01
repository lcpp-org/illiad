"""This file generates poincare plots from the output of a previously-run field line tracing simulation."""

import numpy as np
import matplotlib.pyplot as plt
from classes.iohandler import IOHandler
from classes.mesh import Mesh
from classes.poincare import Poincare


# DEFINE OUTPUT DIRECTORY #
#OUTPUT_DIR = f"It-{CURRENT_TOR*1000:04.0f}_Ih-{CURRENT_HEL*1000:04.0f}_SEMI-IDEAL_{SPINS:04d}sp_LSODA5e8"
#OUTPUT_DIR = "AcceptedIota5_500spins_103Lines"
OUTPUT_DIR = "It-0486_Ih-0790_1500sp_LSODA2p49e8"

def main():
    """
    Main function to set up the mesh, load magnetic field data, and generate Poincare plots.
    """
    ## SET UP RUN DIRECTORY (*DATA AND PLOTS WILL BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!*)
    simIO = IOHandler(OUTPUT_DIR) 
    simIO.startLog()

    ## DEFINE MESH AND LOAD MAGNETIC FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)

    # DEFINE POINCARE ANALYSIS OBJECT
    solved_PoinCare = Poincare(simIO)#, *solver_args)
    solved_PoinCare.set_conditions(field=b_hidra)

    ## LOOP OVER PHI ANGLES
    for phi in solved_PoinCare.plot_angles:
        phi_deg = phi*180/np.pi

        # LOAD NUMPY DATA PROPER ANALYSIS DIRECTORY
        fname = solved_PoinCare.anlys_name + '_{:03.0f}.npy'.format(phi_deg)
        radtheta_pts = simIO.loadNumpyData(fname)
        num_sets = radtheta_pts.shape[0]
        point_total = np.zeros(num_sets, dtype=int)
        
        for i in range(num_sets):
            these_radtheta_pts = radtheta_pts[i] # (2, max_pts)
            point_total[i] = np.sum(~np.isnan(these_radtheta_pts).all(axis=0))
        # PLOT POINCARE
        solved_PoinCare.plotPoincareBW(radtheta_pts, point_total, phi_deg,
                                        solved_PoinCare.field, solved_PoinCare.anlys_name, simIO)

if __name__ == "__main__":
    main()