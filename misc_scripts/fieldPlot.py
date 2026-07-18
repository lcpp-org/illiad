import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from scipy.optimize import curve_fit

import classes.class_outputHandler as out
from classes.mesh import *
from utility.coordtrans import RTP_XYZ_JAC



def main():
    ## SET UP RUN DIRECTORY (DATA AND PLOTS *WILL* BE OVERWRITTEN IF DIR ALREADY EXISTS!!)
    simIO = out.IOHandler("FIELD_STRENGTH_TESTING")
    simIO.startLog()
    anlys_dir = 'It486_Ih900_Iv000'
    simIO.createSubDir(anlys_dir)

    ## DEFINE MESH AND LOAD FIELD
    b_hidra = Mesh(R0=0.72, a=0.19)

    #b_hidra.loadCartesianField('input_files/It486_Ih900_Iv000_0p955_hires.npy', errField=True)
    b_hidra.loadCartesianField('input_files/It486_Ih900_Iv000_1p0_hires.npy', errField=True)
    #b_hidra.loadCartesianField('input_files/FITTED_02092025_hel-0p950.npy', errField=True)
    assumed_rad =  0. #0.01
    assumed_theta = 0. #1. * np.pi/180

    def compBfield(phi_deg, attMult = 1.0, errMag = 0.0, errDir=0.0):
        a_phi =  162. # assuming phi_c = 0 @ 18deg CW from South Split, #a_phi =   18., 90.,  -54.,  -126.,
        b_hidra.err_mag = errMag
        b_hidra.err_dir = errDir
        b_hidra.att_mult = attMult #0.943, 0.9616        
        
        phi_radians = (phi_deg+a_phi)*np.pi/180.
        point = np.array([assumed_rad, assumed_theta, phi_radians])

        B_xyz = b_hidra.interpField(point, Cart=False)[0]
        B_rtp = RTP_XYZ_JAC(point, B_xyz) # Br, Bpol, Btor
        B_rtp[1] *= -1

        return B_rtp * 10_000 # gauss

    def return_Btor(phi_deg, attMult = 1.0, errMag = 0.0, errDir=0.0):
        a_phi =  162. # assuming phi_c = 0 @ 18deg CW from South Split, #a_phi =   18., 90.,  -54.,  -126., 
        b_hidra.err_mag = errMag
        b_hidra.err_dir = errDir
        b_hidra.att_mult = attMult #0.943, 0.9616
        b_tor = np.zeros(len(phi_deg))

        for i, phi in enumerate(phi_deg):
            phi_radians = (phi+a_phi)*np.pi/180.
            point = np.array([assumed_rad, assumed_theta, phi_radians] )

            B_xyz = b_hidra.interpField(point, Cart=False)[0]
            B_rtp = RTP_XYZ_JAC(point, B_xyz) # Br, Bpol, Btor

            b_tor[i] = B_rtp[2] * 10_000 # gauss

        return b_tor
    
    def return_Bnorm(phi_deg, attMult = 1.0, errMag = 0.0, errDir=0.0):
        a_phi =  162. # assuming phi_c = 0 @ 18deg CW from South Split, #a_phi =   18., 90.,  -54.,  -126., 
        b_hidra.err_mag = errMag
        b_hidra.err_dir = errDir
        b_hidra.att_mult = attMult #0.943, 0.9616
        bnorm = np.zeros(len(phi_deg))

        for i, phi in enumerate(phi_deg):
            phi_radians = (phi+a_phi)*np.pi/180.
            point = np.array([assumed_rad, assumed_theta, phi_radians] )

            B_xyz = b_hidra.interpField(point, Cart=False)[0]
            bnorm[i] = np.sqrt( B_xyz[0]**2 + B_xyz[1]**2 + B_xyz[2]**2 ) * 10_000 #gauss

        return bnorm

    ## MEASUREMENT DATA
    ###################
    measured_Btor = np.array([ [36., 666.], [108., 662.6], [216., 660.], [324., 664.6] ])
    measured_Bx = np.array([ [36.,-27.9], [108., -13.2], [216.,35.9], [324., -39] ])
    measured_By = np.array([ [36., -0.6], [108., -24.5], [216., -19.3], [324., -6] ])

    #measured_Bnorm = np.sqrt(  measured_Bx[:, 1]**2 + measured_By[:,1]**2 )
    measured_Bnorm = np.sqrt( measured_Btor[:, 1]**2 + measured_Bx[:, 1]**2 + measured_By[:,1]**2 )
    
    measured_Btor9cm = np.array([ [36., 588.8], [108., 583.7], [216., 575.], [324., 586.5] ])
    measured_Bx9cm = np.array([ [36., -161.3], [108., -162.4], [216., 206.2], [324.,-165.5] ])
    measured_By9cm = np.array([ [36., -13.1], [108., -23.5], [216., 6.], [324., -19.9] ])

    #measured_Bnorm9cm = np.sqrt( measured_Bx9cm[:, 1]**2 + measured_By9cm[:,1]**2 )
    measured_Bnorm9cm = np.sqrt( measured_Btor9cm[:, 1]**2 + measured_Bx9cm[:, 1]**2 + measured_By9cm[:,1]**2 )
    
    # independent and dependent values, and std. dev. values for fitting
    ind_measured = measured_Btor[:, 0]
    dep_measured = measured_Btor[:, 1]    
   
    simIO.log.info('ind_measured: {}'.format(ind_measured))
    simIO.log.info('dep_measured: {}'.format(dep_measured))

    ## FITTING
    ##########
    test_errDir = 270.* np.pi/180
    sigma_measured = np.array([0.5, 0.9, 0.6, 0.9])     
    simIO.log.info('sigma_measured: {}'.format(sigma_measured))    
    Jon = curve_fit(return_Btor, ind_measured, dep_measured,
                     p0=[1.0, 0.0, test_errDir], sigma=sigma_measured, absolute_sigma=True,
                     bounds=( [0.8, 0.0, np.pi], [1.2, 1e-3 , 2*np.pi] ),  method='dogbox')

    # FITTED VALUES
    calc_attMult = Jon[0][0]
    calc_errMag = Jon[0][1]
    calc_errDir = Jon[0][2] * 180./np.pi
    perr = np.sqrt(np.diag(Jon[1]))       
    simIO.log.info('calc_attMult: {}'.format(calc_attMult))
    simIO.log.info('calc_errMag: {}'.format(calc_errMag))
    simIO.log.info('calc_errDir: {}'.format(calc_errDir))
    simIO.log.info('perr: {}'.format(perr))

    # COMPUTE THE FITTED FIELD VALUES (FOR PLOTTING)
    PHI_PLOT = np.linspace(1, 360., 360)
    B_rtp = np.zeros((len(PHI_PLOT), 3))
    B_rtp_ideal = np.zeros((len(PHI_PLOT), 3))
    B_rtp_plus = np.zeros((len(PHI_PLOT), 3))
    B_rtp_minus = np.zeros((len(PHI_PLOT), 3))
    for i, phi in enumerate(PHI_PLOT):
        assumed_rad =  0.0
        assumed_theta = 0.0
        B_rtp[i] = compBfield(phi_deg=phi, attMult = calc_attMult, errMag = calc_errMag, errDir=calc_errDir)
        B_rtp_ideal[i] = compBfield(phi_deg=phi, attMult = 1.0, errMag = 0.0, errDir=0.0)

        assumed_rad =  0.10 
        assumed_theta = 0.
        B_rtp_plus[i] = compBfield(phi_deg=phi, attMult = calc_attMult, errMag = calc_errMag, errDir=calc_errDir)

        # assumed_rad =  0.10
        # assumed_theta = np.pi
        # B_rtp_minus[i] = compBfield(phi_deg=phi, attMult = calc_attMult, errMag = calc_errMag, errDir=calc_errDir)


    ## PLOTTING
    ###########
    plt.rcParams.update({'font.size': 6, 'axes.labelsize': 6, 'legend.fontsize': 4, 'axes.titlesize': 6, 'lines.linewidth': 0.9})
    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    ax1.plot(PHI_PLOT, B_rtp_ideal[:, 2], label='Ideal (no error field)', zorder=4)    
    ax1.plot(PHI_PLOT, B_rtp[:, 2], label='Fitted Error Field', zorder=5)
    ax1.plot(PHI_PLOT, B_rtp_plus[:, 2], label='Fitted Error Field (9cm)', zorder=5)
    #ax1.plot(PHI_PLOT, B_rtp_minus[:, 2], label='minus 1cm', zorder=5)

    ax1.scatter(measured_Btor[:, 0], measured_Btor[:, 1], marker='x', color='k', label='Measurement Data', zorder=5)   
    ax1.scatter(measured_Btor9cm[:, 0], measured_Btor9cm[:, 1], marker='o', color='k', s=1.0, label='Measurement Data (9cm)', zorder=5)   
    ax1.set_ylabel('Toroidal (Z) Field Strength [Gauss]')
    ax1.grid(which='both', zorder=3)
    ax1.legend(loc='best')


    ax2 = fig.add_subplot(222)
    ax2.plot(PHI_PLOT, B_rtp_ideal[:, 0], label='Ideal (no error field)', zorder=4)    
    ax2.plot(PHI_PLOT, B_rtp[:, 0], label='Fitted Error Field', zorder=5)
    ax2.plot(PHI_PLOT, B_rtp_plus[:, 0], label='plus 1cm', zorder=5)
    #ax2.plot(PHI_PLOT, B_rtp_minus[:, 0], label='minus 1cm', zorder=5)

    ax2.scatter(measured_Bx[:, 0], measured_Bx[:, 1], marker='x', color='k', label='Measurement Data', zorder=5)
    ax2.scatter(measured_Bx9cm[:, 0], measured_Bx9cm[:, 1], marker='o', s=1.0, color='k', label='Measurement Data(9cm)', zorder=5)
    ax2.set_ylabel('Radial (X) Field Strength [Gauss]')
    ax2.grid(which='both')


    ax4 = fig.add_subplot(224)
    ax4.plot(PHI_PLOT, B_rtp_ideal[:, 1], label='Ideal (no error field)', zorder=4)    
    ax4.plot(PHI_PLOT, B_rtp[:, 1], label='Fitted Error Field', zorder=5)
    ax4.plot(PHI_PLOT, B_rtp_plus[:, 1], label='plus 1cm', zorder=5)
    #ax4.plot(PHI_PLOT, B_rtp_minus[:, 1], label='minus 1cm', zorder=5)

    ax4.scatter(measured_By[:, 0], measured_By[:, 1], marker='x', color='k', label='Measurement Data', zorder=5)
    ax4.scatter(measured_By9cm[:, 0], measured_By9cm[:, 1], marker='o', s=1.0, color='k', label='Measurement Data', zorder=5)
    ax4.set_xlabel('Toroidal Angle [deg], +CW from North Split')
    ax4.set_ylabel('Poloidal (Y) Field Strength [Gauss]')
    ax4.grid(which='both')


    ax3 = fig.add_subplot(223)
    ax3.plot(PHI_PLOT, np.sqrt( B_rtp_ideal[:, 0]**2 + B_rtp_ideal[:,1]**2 + B_rtp_ideal[:,2]**2), label='Ideal (no error field)', zorder=4)
    ax3.plot(PHI_PLOT, np.sqrt( B_rtp[:, 0]**2 + B_rtp[:,1]**2 + B_rtp[:,2]**2), label='Fitted Error Field', zorder=5)
    ax3.plot(PHI_PLOT, np.sqrt( B_rtp_plus[:, 0]**2 + B_rtp_plus[:,1]**2 + B_rtp_plus[:,2]**2), label='plus 1cm', zorder=5)
    # ax3.plot(PHI_PLOT, np.sqrt( B_rtp_ideal[:, 0]**2 + B_rtp_ideal[:,1]**2 ), label='Ideal (no error field)', zorder=4)
    # ax3.plot(PHI_PLOT, np.sqrt( B_rtp[:, 0]**2 + B_rtp[:,1]**2), label='Fitted Error Field', zorder=5)
    # ax3.plot(PHI_PLOT, np.sqrt( B_rtp_plus[:, 0]**2 + B_rtp_plus[:,1]**2), label='plus 1cm', zorder=5)

    ax3.scatter( measured_By[:, 0], measured_Bnorm[:], marker='x', color='k', label='Measurement Data', zorder=5)
    ax3.scatter( measured_By[:, 0], measured_Bnorm9cm[:], marker='o', s=1.0, color='k', label='Measurement Data (9cm)', zorder=5)
    ax3.set_xlabel('Toroidal Angle [deg], +CW from North Split')
    ax3.set_ylabel('Scalar Field Strength [Gauss]')
    ax3.grid(which='both')


    plt.tight_layout()
    plt.savefig("test2.png", dpi=300)
    #plt.show()

if __name__ == '__main__':
    main()
