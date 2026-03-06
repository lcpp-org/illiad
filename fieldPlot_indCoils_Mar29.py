from matplotlib.ticker import StrMethodFormatter
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from scipy.optimize import curve_fit, least_squares

import classes.class_outputHandler as out
from classes.mesh import *
from utility.coordtrans import RTP_XYZ_JAC


def main():
    ## SET UP RUN DIRECTORY (DATA AND PLOTS *WILL* BE OVERWRITTEN IF DIR ALREADY EXISTS!!)
    simIO = out.IOHandler("BFIELD_VALIDATION_PLOTS")
    simIO.startLog()
    anlys_dir = 'It486_Ih900_Iv000'
    simIO.createSubDir(anlys_dir)

    ## DEFINE MESH AND LOAD FIELDS
    # Load Toroidal Field Coils
    b_TCoils = Mesh(R0=0.72, a=0.19)
    b_TCoils.loadCartesianField('input_files/It486_Ih000_Iv000_1p000_1p000.npy', errField=True, att_mult=1.0)
    b_TCoils.set_nonPer_errField(err_mag=0.0, err_dir=0.0)
    b_TCoils.att_mult = 1.0
    # Load Helical Field Coils
    b_HCoils = Mesh(R0=0.72, a=0.19)
    b_HCoils.loadCartesianField('input_files/It000_Ih900_Iv000_1p000_1p000.npy', errField=True, att_mult=1.0)
    b_HCoils.set_nonPer_errField(err_mag=0.0, err_dir=0.0)
    b_HCoils.att_mult = 1.0


    def compBfield(phi_deg, errMag=0.0, errDir=0.0, torMult=1.0, helMult=1.0, assumed_rad=0.0, assumed_theta=0.0):
        #computational-to-physical angle transform, 
        a_phi =  162. # assuming phi_c = 0 @ 18deg CW from South Split, #a_phi =   18., 90.,  -54.,  -126.,

        phi_radians = (phi_deg+a_phi)*np.pi/180.
        point = np.array([assumed_rad, assumed_theta, phi_radians])

        b_TCoils.set_nonPer_errField(err_mag=errMag, err_dir=errDir)
        b_HCoils.set_nonPer_errField(err_mag=errMag, err_dir=errDir)

        B_xyz = b_TCoils.interpField(point, Cart=False)[0] * torMult
        B_xyz += b_HCoils.interpField(point, Cart=False)[0] * helMult

        B_rtp = RTP_XYZ_JAC(point, B_xyz) # Br, Bpol, Btor

        return B_rtp * 10_000 # Tesla -> gauss

    def return_Bnorm(independents, errMag=0.0, errDir=0.0, torMult=1.0, helMult=1.0):
        a_phi = 162. * np.pi / 180. # assuming phi_c = 0 @ 18deg CW from South Split, #a_phi =   18., 90.,  -54.,  -126., 
        N = len(independents)
        b_norm = np.zeros(N)

        # manually add the error field as a Cartesian perturbation to the Bfield at each point, to avoid scaling by torMult & helMult
        xerr_adder = errMag * np.cos(errDir)
        yerr_adder = -errMag * np.sin(errDir)
        err_adder = np.array([xerr_adder, yerr_adder, 0.0]).T

        for i in range(N):
            phi = independents[i, 1]
            phi_radians = phi + a_phi #*np.pi/180.       
            r_minor = independents[i, 0]
            if r_minor < 0.:
                r_minor *= -1
                assumed_theta = np.pi
            else:
                assumed_theta = 0.0
            point = np.array([r_minor, assumed_theta, phi_radians] )

            B_xyz = b_TCoils.interpField(point, Cart=False)[0] * torMult
            B_xyz += b_HCoils.interpField(point, Cart=False)[0] * helMult

            B_xyz += err_adder

            b_norm[i] = np.sqrt(B_xyz[0]**2 + B_xyz[1]**2 + B_xyz[2]**2)

        return b_norm * 10_000 # gauss


    ## MEASUREMENT DATA & ERRORS
    ############################
    measurements = np.array([
        #r,   phi,  Btor,     Bx,     By    chipErr, dBtor/dr
        [ 0.0,    36., 666.,   -27.9,  -00.6, 1.5, 9.1],
        [ 0.0,   108., 662.6,  -13.2,  -24.5, 1.5, 9.1],
        [ 0.0,   216., 660.,    35.9,  -19.3, 1.5, 9.1],
        [ 0.0,   324., 664.6,  -39.0,  -06.0, 1.5, 9.1],
        [ 0.10,  36., 588.8,  -161.3, -13.1, 1.5, 7.1],
        [ 0.10, 108., 583.7,  -162.4, -23.5, 1.9, 7.1],
        [ 0.10, 216., 570.,    200.2,  00.0, 1.6, 7.1],
        [ 0.10, 324., 586.5,  -165.5, -19.9, 1.9, 7.1],
        [-0.10, 108., 757.8,  112.9,  -9.4, 1.9, 12.2],
        [-0.10, 216., 778.5,  -80.5,  -9.2, 1.6, 12.2],
        [-0.10, 324., 754.5,   80.9,   8.7, 1.9, 12.2],
    ])
    measurements[:, 1] *= np.pi/180.

    # dBtor/dr listed as +/- 1 cm. True translational measurement error is +/- 0.5cm
    # (if we are confident that the its actually +/-1 mm, divide "measurements[:, 6]"" by 10)
    sigma_measured = np.sqrt(measurements[:, 5]**2 + (measurements[:, 6]/10)**2)
    print('sigma_measured shape: {}'.format(sigma_measured.shape))

    dBtor_dArm = 2.5 # gauss/deg for +5deg.
    dBpol_dArm = 60. # gauss/deg for +5deg.
    dBrad_dArm = 0.0 # gauss/deg for +5deg.

    dBtor_dChip = 0.0 # gauss/deg for +5deg.
    dBpol_dChip = 2.0 # gauss/deg for +5deg.
    dBrad_dChip = 0.8 # gauss/deg for +5deg.

    dBtor_dMount = 2.5 #0.0 # gauss/deg for +5deg.
    dBpol_dMount = 0.0 # gauss/deg for +5deg.
    dBrad_dMount = 60.0 #0.0 # gauss/deg for +5deg.

    # CHOOSE WHICH MEASUREMENT POINTS TO FEED INTO FITTING ROUTINE
    i_r0_points = [0,1,2,3]                 #only r=0.0
    i_lf_points = [4,5,6,7]                 #r=10cm LF    
    i_hf_points = [8,9,10]                  #r=10cm HF
    i_fitpoints = i_r0_points + i_lf_points #+ i_hf_points

    plot_name = "bfield_fitting_021826.png"
    #plot_title = "Fitting only r=0 points:"
    #plot_title = "Fitting All Measurement points:"
    plot_title = "Fitting r=0 and LF points:"

    # INDEPENDENT PARAMETERS, DEPENDENT PARAMETERS, AND ERROR ESTIMATES FOR FITTING
    ind_measured = measurements[:, :2]
    dep_measured = np.sqrt(measurements[:, 2]**2 + measurements[:, 3]**2 + measurements[:, 4]**2)

    simIO.log.info('sigma_measured: {}'.format(sigma_measured[i_fitpoints]))    ## GET REAL VALUES FOR THE 9cm MEASUREMENTS!:
    simIO.log.info('ind_measured: {}'.format(ind_measured[i_fitpoints]))
    simIO.log.info('dep_measured: {}'.format(dep_measured[i_fitpoints]))

    ##########
    ## FITTING
    ##########
    # initial guesses
    mag_guess = 1.5e-4
    dir_guess = (210.* np.pi/180)
    s_tor_guess = 0.9 #0.96347 #0.955
    s_hel_guess = 0.90

    # no helical scalar
    # popt, pcov = curve_fit(return_Bnorm, ind_measured, dep_measured,
    #                 sigma=sigma_fitPts, absolute_sigma=True,
    #                 p0=[mag_guess, dir_guess, s_tor_guess],
    #                 bounds=([5e-5,    0., 0.80],
    #                         [5e-3, 2*np.pi, 1.0]),
    #                 max_nfev=1e4, method='trf', x_scale=[1e-4, 1, 1])
    
    #with helical scalar
    popt, pcov = curve_fit(return_Bnorm, ind_measured[i_fitpoints], dep_measured[i_fitpoints],
                    sigma=sigma_measured[i_fitpoints], absolute_sigma=True,
                    p0=[mag_guess, dir_guess, s_tor_guess, s_hel_guess], 
                    bounds=( [5e-5,   0.0, 0.80, 0.90],# 
                             [1.5646e-4, 2.*np.pi, 0.9448, 1.0] ),
                    max_nfev=1e4, method='dogbox', x_scale=[1e-4, 1, 1e-1, 1e-1]) #'dogbox'

    # FITTED VALUES
    # calc_errMag = 1.5654e-4 #popt[0] 
    # #calc_errDir = (271.5 + 162.) % 360.  #popt[1]
    # calc_errDir = 271.5 * np.pi/180.  #popt[1]
    # calc_errDir_deg = calc_errDir*180./np.pi # convert to degrees
    # calc_torMult = 0.9448 #popt[2]
    # calc_helMult = 1.0 * 0.9448 #popt[2] # popt[3]

    # FITTED VALUES
    calc_errMag = popt[0] 
    calc_errDir = popt[1]
    calc_errDir_deg = calc_errDir*180./np.pi # convert to degrees
    calc_torMult = popt[2]
    calc_helMult = popt[3]

    perr = np.sqrt(np.diag(pcov))   
    condition_num = np.linalg.cond(pcov)

    simIO.log.info('\n** RESULTS ** :')
    simIO.log.info('calc_errMag: {}'.format(calc_errMag))
    simIO.log.info('calc_errDir: {}'.format(calc_errDir*180./np.pi ))    
    simIO.log.info('calc_torMult: {}'.format(calc_torMult))
    simIO.log.info('calc_helMult: {}'.format(calc_helMult))
    simIO.log.info('perr: {}'.format(perr))
    simIO.log.info('condition_num: {}'.format(condition_num))

    resultString = f'{calc_errMag=:.4e}[Tesla], {calc_errDir_deg=:.3f}[deg], {calc_torMult=:.4f}, {calc_helMult=:.4f}\n{perr=}'

    ###########
    ## PLOTTING
    ###########
    # COMPUTE THE FITTED FIELD 
    PHI_PLOT = np.linspace(1, 360., 360)
    B_rtp = np.zeros((len(PHI_PLOT), 3))
    B_rtp_ideal = np.zeros((len(PHI_PLOT), 3))
    B_rtp_LF = np.zeros((len(PHI_PLOT), 3))
    B_rtp_LF_ideal = np.zeros((len(PHI_PLOT), 3))
    B_rtp_HF = np.zeros((len(PHI_PLOT), 3))
    for i, phi in enumerate(PHI_PLOT):

        B_rtp[i] = compBfield(phi_deg=phi, errMag=calc_errMag, errDir=calc_errDir, torMult=calc_torMult, helMult=calc_helMult, assumed_rad=0.0, assumed_theta=0.0)
        B_rtp_ideal[i] = compBfield(phi_deg=phi, errMag = 0.0, errDir=0.0, torMult = 1.0, helMult = 1.0, assumed_rad=0.0, assumed_theta=0.0)

        B_rtp_LF[i] = compBfield(phi_deg=phi, errMag = calc_errMag, errDir=calc_errDir, torMult = calc_torMult, helMult = calc_helMult, assumed_rad=0.10, assumed_theta=0.0)
        B_rtp_LF_ideal[i] = compBfield(phi_deg=phi, errMag = 0.0, errDir=0.0, torMult = 1.0, helMult = 1.0, assumed_rad=0.10, assumed_theta=0.0)
        #B_rtp_LF[i][:2] *= -1

        B_rtp_HF[i] = compBfield(phi_deg=phi, errMag = calc_errMag, errDir=calc_errDir, torMult = calc_torMult, helMult = calc_helMult, assumed_rad=0.10, assumed_theta=np.pi)
        B_rtp_HF[i][:2] *= -1


    plt.rcParams.update({
        # --- fonts & text (IOP-friendly, ~8–12 pt at final size) ---
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavusans",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.labelweight": "bold",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.0,
    })

    fig = plt.figure()
    fig.suptitle(plot_title+'\n'+resultString, fontsize=4)

    ax1 = fig.add_subplot(221)
    ax1.plot(PHI_PLOT, np.sqrt( B_rtp_ideal[:,0]**2 + B_rtp_ideal[:,1]**2 + B_rtp_ideal[:,2]**2), '--b', label='Ideal B-field (on-axis)', zorder=4)
    ax1.plot(PHI_PLOT, np.sqrt( B_rtp[:,0]**2 + B_rtp[:,1]**2 + B_rtp[:,2]**2), 'b', label='w/ fitted error field (on-axis)', zorder=5)
    ax1.plot(PHI_PLOT, np.sqrt( B_rtp_LF[:,0]**2 + B_rtp_LF[:,1]**2 + B_rtp_LF[:,2]**2), 'g', label='w/ fitted error field (10cm LF)', zorder=5)
    #ax1.plot(PHI_PLOT, np.sqrt( B_rtp_HF[:,0]**2 + B_rtp_HF[:,1]**2 + B_rtp_HF[:,2]**2), 'r', label='With fitted error field (10cm HF)', zorder=5)
    ax1.errorbar( np.degrees(ind_measured[i_fitpoints,1]), dep_measured[i_fitpoints], yerr=sigma_measured[i_fitpoints],
                  fmt='s', markersize=0.0, color='k', capsize=2, ecolor='k', zorder=5, label='Measurements')
    ax1.set_ylabel('$|\mathbf{B}|\:[G]$')
    ax1.grid(which='both')
    ax1.legend(loc='upper left', bbox_to_anchor=(-0.24, 1.2), fontsize=6, ncol=2)

    ax2 = fig.add_subplot(222)
    ax2.plot(PHI_PLOT, B_rtp_ideal[:, 2], '--k', label='Ideal B-field (on-axis)', zorder=4)    
    ax2.plot(PHI_PLOT, B_rtp[:, 2], 'b', label='With fitted error field (on-axis)', zorder=5)
    ax2.plot(PHI_PLOT, B_rtp_LF[:, 2], 'g', label='With fitted error field (10cm LF)', zorder=5)
    #ax2.plot(PHI_PLOT, B_rtp_HF[:, 2], 'r', label='Fitted 10cm HF', zorder=5)
    ax2.errorbar(np.degrees(measurements[i_fitpoints, 1]), measurements[i_fitpoints, 2], yerr=np.sqrt(sigma_measured[i_fitpoints]**2 + dBtor_dArm**2 + dBtor_dChip**2 + dBtor_dMount**2),
                  fmt='s', markersize=0.0, color='k', capsize=2, ecolor='k', zorder=5, label='Measurements')
    ax2.set_ylabel('$B_\phi\:[G]$')
    ax2.grid(which='both', zorder=3)


    ax3 = fig.add_subplot(223)
    ax3.plot(PHI_PLOT, B_rtp_ideal[:, 0], '--k', label='Ideal (no error field)', zorder=4)    
    ax3.plot(PHI_PLOT, B_rtp[:, 0], 'b', label='Fitted Error Field', zorder=5)
    ax3.plot(PHI_PLOT, B_rtp_LF[:, 0], 'g', label='Fitted 10cm LF', zorder=5)
    #ax3.plot(PHI_PLOT, B_rtp_HF[:, 0], 'r', label='Fitted 10cm HF', zorder=5)
    ax3.errorbar(np.degrees(measurements[i_fitpoints, 1]), measurements[i_fitpoints, 3], yerr=np.sqrt(sigma_measured[i_fitpoints]**2 + dBrad_dArm**2 + dBrad_dChip**2 +  dBrad_dMount**2),
                  fmt='s', markersize=0.0, color='k', capsize=2, ecolor='k', zorder=5)
    ax3.set_xlabel('$\phi$ (CW from North Split)')
    ax3.set_ylabel('$B_r\:[G]$', labelpad=0)
    ax3.grid(which='both')

    ax4 = fig.add_subplot(224)
    ax4.plot(PHI_PLOT, B_rtp_ideal[:, 1], '--k', label='Ideal (no error field)', zorder=4)    
    ax4.plot(PHI_PLOT, B_rtp[:, 1], 'b', label='Fitted Error Field', zorder=5)
    ax4.plot(PHI_PLOT, B_rtp_LF[:, 1], 'g', label='Fitted 10cm LF', zorder=5)
    #ax4.plot(PHI_PLOT, B_rtp_HF[:, 1], 'r', label='Fitted 10cm HF', zorder=5)
    ax4.errorbar(np.degrees(measurements[i_fitpoints, 1]), measurements[i_fitpoints, 4], yerr=np.sqrt(sigma_measured[i_fitpoints]**2 + dBpol_dArm**2 + dBpol_dChip**2 + dBpol_dMount**2),
                  fmt='s', markersize=0.0, color='k', capsize=2, ecolor='r', zorder=5)   
    ax4.set_xlabel('$\phi$ (CW from North Split)')
    ax4.set_ylabel('$B_\\theta\:[G]$', labelpad=0)
    ax4.grid(which='both')

    ax1.set_xlim(0, 360)
    ax1.set_xticks(np.arange(0, 361, 90))
    #ax1.xaxis.set_major_formatter(StrMethodFormatter(u"$\\mathbf{{{x:.0f}°}}$"))
    ax1.set_xticklabels([])

    ax2.set_xlim(0, 360)
    ax2.set_xticks(np.arange(0, 361, 90))
    #ax2.xaxis.set_major_formatter(StrMethodFormatter(u"$\\mathbf{{{x:.0f}°}}$"))
    ax2.set_xticklabels([])

    ax3.set_xlim(0, 360)
    ax3.set_xticks(np.arange(0, 361, 90))
    ax3.xaxis.set_major_formatter(StrMethodFormatter(u"$\\mathbf{{{x:.0f}°}}$"))

    ax4.set_xlim(0, 360)
    ax4.set_xticks(np.arange(0, 361, 90))
    ax4.xaxis.set_major_formatter(StrMethodFormatter(u"$\\mathbf{{{x:.0f}°}}$"))

    plt.subplots_adjust(wspace=0.3, hspace=0.15)
    #plt.tight_layout()
    simIO.saveFig(plot_name)

    ### ARTICLE PLOT ###
    fig = plt.figure()
    #fig.suptitle(plot_title +'\n'+resultString, fontsize=4)
    axSolo = fig.add_subplot(111)
    r0_ideal, = axSolo.plot(PHI_PLOT, np.sqrt( B_rtp_ideal[:,0]**2 + B_rtp_ideal[:,1]**2 + B_rtp_ideal[:,2]**2), '--b', label='Ideal Field ($\\rho=0$)', zorder=4)
    r0_fit, = axSolo.plot(PHI_PLOT, np.sqrt( B_rtp[:,0]**2 + B_rtp[:,1]**2 + B_rtp[:,2]**2), 'b', label='LS-Fit ($\\rho=0$)', zorder=5)
    r0_data = axSolo.errorbar( np.degrees(ind_measured[i_r0_points,1]), dep_measured[i_r0_points], yerr=sigma_measured[i_r0_points],
                  fmt='^', markersize=2, color='b', fillstyle='none', capsize=2, ecolor='k', zorder=5, label='Measurements ($\\rho=0$)')

    lf_ideal, = axSolo.plot(PHI_PLOT, np.sqrt( B_rtp_LF_ideal[:,0]**2 + B_rtp_LF_ideal[:,1]**2 + B_rtp_LF_ideal[:,2]**2), '-.g', label='Ideal Field ($\\rho=10cm\,LF$)', zorder=5)
    lf_fit, = axSolo.plot(PHI_PLOT, np.sqrt( B_rtp_LF[:,0]**2 + B_rtp_LF[:,1]**2 + B_rtp_LF[:,2]**2), 'g', label='LS-Fit ($\\rho=10cm\,LF$)', zorder=5)
    lf_data = axSolo.errorbar( np.degrees(ind_measured[i_lf_points,1]), dep_measured[i_lf_points], yerr=sigma_measured[i_lf_points],
                  fmt='o', markersize=2, color='g', fillstyle='none', capsize=2, ecolor='k', zorder=5, label='Measurements($\\rho=10 cm\,LF$)')

    axSolo.set_xlabel('$\phi \: [ \degree CW\,from\:North\:Split] $')
    axSolo.set_ylabel('$\| \mathbf{B} \| \:[gauss]$')
    axSolo.grid(which='both')
    axSolo.legend(handles=[r0_ideal, r0_fit, r0_data, lf_ideal, lf_fit, lf_data], loc='best', fontsize=7, ncol=2)

    axSolo.set_xlim(0, 360)
    axSolo.set_xticks(np.arange(0, 361, 90))
    plt.tight_layout()
    simIO.saveFig('solo_' + plot_name)
    
    '''UNUSED FUNCTIONS
    def return_Btor(independents, errMag=0.0, errDir=0.0, torMult=1.0, helMult=1.0):
        a_phi =  162. *np.pi/180. # assuming phi_c = 0 @ 18deg CW from South Split, #a_phi =   18., 90.,  -54.,  -126.,
        #helMult = 1.0

        N = len(independents)
        b_tor = np.zeros(N)

        b_TCoils.err_mag = errMag
        b_TCoils.err_dir = errDir
        b_TCoils.att_mult = torMult #0.943, 0.9616
        
        b_HCoils.err_mag = errMag
        b_HCoils.err_dir = errDir
        b_HCoils.att_mult = helMult #0.943, 0.9616

        for i in range(N):
            phi = independents[i]#, 0]
            r_minor = 0. #independents[i, 1]
            phi_radians = phi+a_phi #*np.pi/180.
            point = np.array([r_minor, assumed_theta, phi_radians] )

            Btc_xyz = b_TCoils.interpField(point, Cart=False)[0]
            Bhc_xyz = b_HCoils.interpField(point, Cart=False)[0]
            B_xyz = Btc_xyz + Bhc_xyz

            B_rtp = RTP_XYZ_JAC(point, B_xyz) # Br, Bpol, Btor

            b_tor[i] = B_rtp[2] * 10_000 # gauss

        return b_tor

    def return_Brad(independents, errMag=0.0, errDir=0.0, torMult=1.0, helMult=1.0):
        a_phi =  162. *np.pi/180. # assuming phi_c = 0 @ 18deg CW from South Split, #a_phi =   18., 90.,  -54.,  -126.,
        #helMult = 1.0

        N = len(independents)
        b_rad = np.zeros(N)

        b_TCoils.err_mag = errMag
        b_TCoils.err_dir = errDir
        b_TCoils.att_mult = torMult #0.943, 0.9616
        
        b_HCoils.err_mag = errMag
        b_HCoils.err_dir = errDir
        b_HCoils.att_mult = helMult #0.943, 0.9616

        for i in range(N):
            phi = independents[i]#, 0]
            r_minor = 0. #independents[i, 1]
            phi_radians = phi+a_phi #*np.pi/180.
            point = np.array([r_minor, assumed_theta, phi_radians] )

            Btc_xyz = b_TCoils.interpField(point, Cart=False)[0]
            Bhc_xyz = b_HCoils.interpField(point, Cart=False)[0]
            B_xyz = Btc_xyz + Bhc_xyz

            B_rtp = RTP_XYZ_JAC(point, B_xyz) # Br, Bpol, Btor

            b_rad[i] = B_rtp[0] * 10_000 # gauss

        return b_rad
    
    def return_Bperp(independents, errMag=0.0, errDir=0.0, torMult=1.0, helMult=1.0):

        a_phi =  162. *np.pi/180.# assuming phi_c = 0 @ 18deg CW from South Split, #a_phi =   18., 90.,  -54.,  -126., 

        N = len(independents)
        b_perp = np.zeros(N)

        b_TCoils.err_mag = errMag
        b_TCoils.err_dir = errDir
        b_TCoils.att_mult = torMult #0.943, 0.9616
        
        b_HCoils.err_mag = errMag
        b_HCoils.err_dir = errDir
        b_HCoils.att_mult = helMult #0.943, 0.9616

        for i in range(N):
            phi = independents[i] #, 0]
            r_minor = 0. #independents[i, 1]
            phi_radians = phi+a_phi #*np.pi/180.

            point = np.array([r_minor, assumed_theta, phi_radians] )

            B_xyz = b_TCoils.interpField(point, Cart=False)[0]
            B_xyz += b_HCoils.interpField(point, Cart=False)[0]

            B_rtp = RTP_XYZ_JAC(point, B_xyz) # Br, Bpol, Btor

            b_perp[i] = np.sqrt(B_rtp[1]*B_rtp[1] + B_rtp[2]*B_rtp[2]) * 10_000 # gauss

        return b_perp'''
    
if __name__ == '__main__':
    main()
