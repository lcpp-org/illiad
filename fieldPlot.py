import numpy as np
import matplotlib.pyplot as plt

import classes.class_outputHandler as out
from classes.mesh import *
from utility.coordtrans import RTP_XYZ_JAC

def main():

    ## SET UP RUN DIRECTORY
    ## DATA AND PLOTS *WILL* BE OVERWRITTEN IF THE DIRECTORY ALREADY EXISTS!!
    simIO = out.IOHandler("FIELD_STRENGTH_TESTING")
    simIO.startLog()
    anlys_dir = 'It486_Ih900_Iv000'
    simIO.createSubDir(anlys_dir)

    ## DEFINE MESH AND LOAD FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)
    b_hidra.loadCartesianField('input_files/It486_Ih900_Iv000_0p955_hires.npy', errField=True)

    # MEASURED VALUES (Gio 1-25-2025)
    # measure +CW from North Split, in Gauss
    BTOR_exp_1 = np.array([27., 674])
    BTOR_exp_2 = np.array([72., 673])
    BTOR_exp_3 = np.array([216., 678])

    BTOR_predict_1 = np.array([36., 674])
    BTOR_predict_2 = np.array([108., 673.5])
    BTOR_predict_3 = np.array([324., 677])


    #BX_exp_1 = np.array([27., 22./10_000])
    #BX_exp_2 = np.array([72., 21./10_000])
    #BX_exp_3 = np.array([216., 25./10_000])
    #BY_exp_1 = np.array([27., -20./10_000])
    #BY_exp_2 = np.array([72., -30./10_000])
    #BY_exp_3 = np.array([216., -77./10_000])

    # phi adders to align with Gio's meaurement coords (+CW from North Split)
    a_phi =  162. # assuming phi_c = 0 @ 18deg CW from South Split
    #a_phi =   90.
    #a_phi =   18.
    #a_phi =  -54.
    #a_phi = -126.


    PHI_PLOT = np.linspace(1, 360., 360)
    B_rtp = np.zeros((len(PHI_PLOT), 3))

    for i, phi in enumerate(PHI_PLOT):
        # points along the major radius
        point = np.array([0.0, 0., (phi+a_phi)*np.pi/180.] )

        B_xyz = b_hidra.interpField(point, Cart=False)[0]
        #print(temp)
        B_rtp[i] = RTP_XYZ_JAC(point, B_xyz) # Br, Bpol, Btor
        

    fig = plt.figure()
    # ax1 = fig.add_subplot(311)
    # ax2 = fig.add_subplot(312)
    # ax1.plot(PHI_PLOT, B_rtp[:, 0])#*0.9615)#, label='B_RADIAL')
    # ax2.plot(PHI_PLOT, B_rtp[:, 1])#*0.9615)#, label='B_POLOIDAL')
    # ax2.scatter(*BX_exp_1)
    # ax2.scatter(*BX_exp_2)
    # ax2.scatter(*BX_exp_3)
    # ax2.scatter(*BY_exp_1)
    # ax2.scatter(*BY_exp_2)
    # ax2.scatter(*BY_exp_3)
    # ax1.set_title('Radial Field')
    # ax1.grid(which='both')
    # ax2.set_title('Poloidal Field')
    # ax2.grid(which='both')

    ax3 = fig.add_subplot()

    ax3.plot(PHI_PLOT, B_rtp[:, 2]*10_000*0.9616)#, label='B_TOROIDAL')

    ax3.scatter(*BTOR_exp_1, label='Measured: {} (PFC B-port)'.format(BTOR_exp_1[1]))
    ax3.scatter(*BTOR_exp_2, label='Measured: {} (Blank A-port)'.format(BTOR_exp_2[1]))
    ax3.scatter(*BTOR_exp_3, label='Measured: {} (Magnetron A-port)'.format(BTOR_exp_3[1]))

    ax3.axvline(x=36, color='grey', linestyle='--')
    ax3.axvline(x=108, color='grey', linestyle='--')
    ax3.axvline(x=324, color='grey', linestyle='--')

    ax3.scatter(*BTOR_predict_1, facecolors='none', edgecolors='r', label='Prediction: {} (PFC A-port)'.format(BTOR_predict_1[1]))
    ax3.scatter(*BTOR_predict_2, facecolors='none', edgecolors='b', label='Prediction: {} (Hall Probe)'.format(BTOR_predict_2[1]))
    ax3.scatter(*BTOR_predict_3, facecolors='none', edgecolors='g', label='Prediction: {} (Specs)'.format(BTOR_predict_3[1]))

    ax3.set_title('Toroidal Field, $I_T=486 A, I_H=900 A$')
    ax3.set_xlabel('Toroidal Angle [deg], +CW from North Split')
    ax3.set_ylabel('Field Strength [Gauss]')
    ax3.grid(which='both')
    ax3.legend(loc='upper left', prop={'size': 8})
    plt.tight_layout()
    plt.savefig("test1.png")
    #plt.show()

if __name__ == '__main__':
    main()