## IMPORTS
import numpy as np
from numpy.polynomial import Polynomial
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import illiad.io as out
from illiad.mesh import Mesh
#import "UIUC" color definitions from \plot_funcs\plotFuncs.py
from illiad.plotting import UIUC

plt.rcParams.update({
    # --- fonts & text (IOP-friendly, ~8–12 pt at final size) ---
    #"font.family": "serif",
    #"font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavusans",
    "axes.titlesize": 10,
    "axes.labelsize": 12,
    #"axes.labelweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 1.0,
})

def ion_ion_collision_freq(ne, Ti, m_bg, z_bg, m_imp, z_imp, ln_lambda):
    """
    Calculate the collision frequency between background and impurity species.
    
    Parameters:
    ne (float): Electron density (m^-3)
    Ti (float): Ion temperature (eV)
    m_bg (float): Mass of background species (amu)
    z_bg (int): Charge state of background species
    m_imp (float): Mass of impurity species (amu)
    z_imp (int): Charge state of impurity species
    ln_lambda (float): Coulomb logarithm
    
    Returns:
    float: Collision frequency (s^-1)
    """

    coefficient = 4.80e-8
    reduced_mass = (m_bg * m_imp) / (m_bg + m_imp)
    ne_cm3 = ne * 1e-6  # Convert from m^-3 to cm^-3

    # Calculate the collision frequency using the formula
    f_coll = coefficient * (ne_cm3 * z_bg**2 * z_imp**2 * ln_lambda) / ( reduced_mass**0.5 * Ti**1.5)

    return f_coll

def ion_neutral_collision_freq(ng, Ti):
    """
    Calculate the ion-neutral collision frequency.
    
    Parameters:
    ng (float): Neutral density (m^-3)
    Ti (float): Ion temperature (eV)
    
    Returns:
    float: Ion-neutral collision frequency (s^-1)
    """
    # Convert neutral density from m^-3 to cm^-3
    ng_cm3 = ng * 1e-6
    
    # Calculate the ion-neutral collision frequency using a typical formula
    f_in = 5.2e-15 * ng * np.sqrt(Ti)  # Example formula, adjust as needed
    
    return f_in

def plot_collision_times(Ti_ev, tau_coll_ne18_ms, tau_coll_ne17_ms,
                          tau_ionNeutral_coll_ms, tau_ionNeutral_coll_ms_18, 
                          tau_residence_lo_ms, tau_residence_hi_ms, 
                          typical_ionTemp_lo, typical_ionTemp_hi, 
                          transition_temp_lo, transition_temp_hi):
    plt.figure()
    plt.plot(Ti_ev, tau_coll_ne18_ms, marker='none', linewidth=2,
             linestyle='--', color='blue', label='$\\tau_{i\\text{-}i},~\\it{Core-Edge~Plasma}$')
             #linestyle='-', color='blue', label='$\\tau_{i\\text{-}i},~n_e=10^{18}$')
    plt.plot(Ti_ev, tau_coll_ne17_ms, marker='none', linewidth=2,
              linestyle='-', color='blue', label='$\\tau_{i\\text{-}i},~\\it{Far~SOL}$')

    # # Text labels directly on the blue ion-ion collision time lines
    # _lbl_Ti = 3.5
    # _idx = np.argmin(np.abs(Ti_ev - _lbl_Ti))
    # plt.text(Ti_ev[_idx], tau_coll_ne18_ms[_idx], 'Core/Edge',
    #          fontsize=9, color='blue', ha='left', va='bottom')
    # plt.text(Ti_ev[_idx], tau_coll_ne17_ms[_idx], 'Far SOL',
    #          fontsize=9, color='blue', ha='left', va='bottom')

    plt.plot(Ti_ev, tau_ionNeutral_coll_ms, marker='none', linewidth=2,
              linestyle='-', color='darkorange', label='$\\tau_{i\\text{-}n},~\\it{Lithium~Evaporation}$')
              #linestyle='--', color='red', label='$\\tau_{i\\text{-}n}$')
    plt.plot(Ti_ev, tau_ionNeutral_coll_ms_18, marker='none', linewidth=2,
              linestyle='--', color='darkorange', label='$\\tau_{i\\text{-}n},~\\it{Typical~Operation}$')
                  #linestyle='--', color='red', label='$\\tau_{i\\text{-}n}$')

    collisional_bottom_line = np.maximum(tau_coll_ne17_ms, tau_residence_lo_ms)
    #collisional_bottom_line = tau_coll_ne17_ms
    collisional_top_line = np.full_like(tau_coll_ne17_ms, tau_residence_hi_ms)
    #collisional_span = (Ti_ev >= typical_ionTemp_lo) & (Ti_ev <= transition_temp_hi)
    collisional_span = (Ti_ev >= typical_ionTemp_lo) & (Ti_ev <= transition_temp_hi)
    plt.fill_between(Ti_ev[collisional_span],
                        collisional_bottom_line[collisional_span],
                        collisional_top_line[collisional_span],
                        color='red', alpha=0.25, label='$\\it{Collisional~Regime}$')

    #collisionless_top_line = np.minimum(tau_coll_ne17_ms, tau_residence_hi_ms)
    collisionless_top_line = np.minimum(tau_coll_ne17_ms, tau_ionNeutral_coll_ms)
    #collisionless_top_line = tau_coll_ne17_ms
    collisionless_botom_line = np.full_like(tau_coll_ne17_ms, tau_residence_lo_ms)
    collisionless_span = (Ti_ev <= typical_ionTemp_hi) & (Ti_ev >= transition_temp_lo)
    plt.fill_between(Ti_ev[collisionless_span], 
                        collisionless_botom_line[collisionless_span],
                        collisionless_top_line[collisionless_span],
                        color='green', alpha=0.25, label='$\\it{Collisionless~Regime}$')


    plt.axvspan(transition_temp_lo, transition_temp_hi,
                 color='darkgray', alpha=0.4, label='$\\it{Collisional/Collisionless~Transition}$', zorder=0)
    plt.axvline(transition_temp_lo, color='gray', linestyle=':')#, label='Typical Operating Range')
    plt.axvline(transition_temp_hi, color='gray', linestyle=':')#, label='Typical Operating Range')
    # plt.axhspan(tau_residence_lo_ms, tau_residence_hi_ms, color='gray', alpha=0.3, label='Residence Time Range')
    # plt.axvspan(typical_ionTemp_lo, typical_ionTemp_hi, color='gray', alpha=0.3, label='Operating Range')
    
    plt.xlabel('$T_i~\\text{[eV]}$')
    plt.ylabel('$\\tau~\\text{[ms]}$')#_{i\\text{-}i}~\\text{[ms]}$')

    plt.grid(which='both')
    plt.xlim(0, 7)#ti_plot_high)
    plt.ylim(1e-3, 1e0)#ti_plot_high)
    plt.yscale('log')
    plt.legend(loc='lower right', fontsize=8, ncol=1)
    #plt.tight_layout()
    #plt.show()
    plt.savefig("collision_time_vs_Ti.png", dpi=300)

def plot_tau_starSOL(Ti_ev, f_typical_edge, f_typical_far, f_evap_edge, f_evap_far, tau_residence_lo_ms, tau_residence_hi_ms, typical_ionTemp_lo, typical_ionTemp_hi):
    tau_typical_edge_ms = 1e3 / f_typical_edge
    tau_typical_far_ms = 1e3 / f_typical_far
    tau_evap_edge_ms = 1e3 / f_evap_edge
    tau_evap_far_ms = 1e3 / f_evap_far

    tau_star_typical_edge = tau_residence_hi_ms / tau_typical_edge_ms
    tau_star_typical_far = tau_residence_hi_ms / tau_typical_far_ms
    tau_star_evap_edge = tau_residence_lo_ms / tau_evap_edge_ms
    tau_star_evap_far = tau_residence_lo_ms / tau_evap_far_ms

    plt.figure()

    z_grid = 1
    z_lines = 1
    z_labels = 1
    z_shading = 5
    # fill between tau_star_typical_edge and tau_star_typical_far
    plt.fill_between(Ti_ev, tau_star_typical_edge, tau_star_typical_far, color=UIUC['il_blue'], alpha=0.5,  zorder=z_shading,
                     label='$\\bf{Typical~Operation}$\n$\\left(T_e~5\\text{eV},~n_g\\sim10^{18}~\\text{m}^{-3}\\right)$')
    plt.plot(Ti_ev, tau_star_typical_edge, marker='none', linewidth=1.5, linestyle='-', color=UIUC['il_blue'], zorder=z_lines)
    plt.plot(Ti_ev, tau_star_typical_far, marker='none', linewidth=1.5, linestyle=':', color=UIUC['il_blue'], zorder=z_lines)

    label_Ti_typ = 3.3
    _idx = np.argmin(np.abs(Ti_ev - label_Ti_typ))

    label_Ti_typ2 = 3.2
    _idx2 = np.argmin(np.abs(Ti_ev - label_Ti_typ2))

    plt.text(Ti_ev[_idx], tau_star_typical_edge[_idx]-1.5, 'Core/Edge',
             fontsize=7, color=UIUC['il_blue'], weight='bold', style='italic',
             ha='left', va='center', rotation=-5, backgroundcolor='white', zorder=z_labels)
    plt.text(Ti_ev[_idx2], tau_star_typical_far[_idx2]+0.22, 'Far SOL',
             fontsize=7, color=UIUC['il_blue'], weight='bold', style='italic',
             ha='left', va='center', rotation=1, backgroundcolor='white', zorder=z_labels)
    
    label_Ti_typ_ne = 4.5
    _idx = np.argmin(np.abs(Ti_ev - label_Ti_typ_ne))
    plt.text(Ti_ev[_idx], tau_star_typical_edge[_idx], '$\\mathbf{n_e=10^{18}~m^{-3}}$',
             fontsize=6, color=UIUC['il_blue'], weight='bold', style='italic',
             ha='center', va='center', rotation=-4,
             backgroundcolor='white', zorder=z_labels)
    plt.text(Ti_ev[_idx], tau_star_typical_far[_idx], '$\\mathbf{n_e=10^{17}~m^{-3}}$',
             fontsize=6, color=UIUC['il_blue'], weight='bold', style='italic',
             ha='center', va='center', rotation=1,
             backgroundcolor='white', zorder=z_labels)

    # fill between tau_star_evap_edge and tau_star_evap_far
    plt.fill_between(Ti_ev, tau_star_evap_edge, tau_star_evap_far, color=UIUC['il_orange'], alpha=0.5, zorder=z_shading,
                     label='$\\bf{During~Evaporation}$\n$\\left(T_e~40\\text{eV},~n_g\\sim10^{17}~\\text{m}^{-3}\\right)$') 
    plt.plot(Ti_ev, tau_star_evap_edge, marker='none', linewidth=1.5, linestyle='-', color=UIUC['il_orange'], zorder=z_lines)
    plt.plot(Ti_ev, tau_star_evap_far, marker='none', linewidth=1.5, linestyle=':', color=UIUC['il_orange'], zorder=z_lines)

    label_Ti_evap = 3.4
    _idx = np.argmin(np.abs(Ti_ev - label_Ti_evap))
    label_Ti_evap_2 = 2.6
    _idx2 = np.argmin(np.abs(Ti_ev - label_Ti_evap_2))

    plt.text(Ti_ev[_idx], tau_star_evap_edge[_idx], 'Core/Edge',
             fontsize=7, color=UIUC['il_orange'], weight='bold', style='italic',
             ha='center', va='center', rotation=-11, backgroundcolor='white', zorder=z_labels)
    plt.text(Ti_ev[_idx2], tau_star_evap_far[_idx2], 'Far SOL',
             fontsize=7, color=UIUC['il_orange'], weight='bold', style='italic',
             ha='center', va='center', rotation=-11, backgroundcolor='white', zorder=z_labels)

    label_Ti_evap_ne = 4.5
    _idx = np.argmin(np.abs(Ti_ev - label_Ti_evap_ne))
    label_Ti_evap_ne2 = 4.5
    _idx2 = np.argmin(np.abs(Ti_ev - label_Ti_evap_ne2))   

    plt.text(Ti_ev[_idx], tau_star_evap_edge[_idx], '$\\mathbf{n_e=10^{18}~m^{-3}}$',
             fontsize=6, color=UIUC['il_orange'], weight='bold', style='italic',
             ha='center', va='center', rotation=-8,
             backgroundcolor='white', zorder=z_labels)
    plt.text(Ti_ev[_idx2], tau_star_evap_far[_idx2], '$\\mathbf{n_e=10^{17}~m^{-3}}$',
             fontsize=6, color=UIUC['il_orange'], weight='bold', style='italic',
             ha='center', va='center', rotation=-6,
             backgroundcolor='white', zorder=z_labels)

    # vertical lines at x=1 and x=5
    plt.axvline(typical_ionTemp_lo, color='k', linestyle='-', zorder=2)#, label='Typical Operating Range')
    plt.axvline(typical_ionTemp_hi, color='k', linestyle='-', zorder=2)#, label='Typical Operating Range')

    plt.xlabel('$\\rm{T_i}~\\rm{[eV]}$', fontsize=12)
    plt.ylabel('$\\rm{\\tau^*}$', fontsize=12)
    plt.grid(which='both', zorder=z_grid, linewidth=0.5, color='k', alpha=0.35)
    plt.xlim(0, 6)#ti_plot_high)
    plt.ylim(3e-2, 3e2)
    plt.yscale('log')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    plt.legend(loc='lower center', fontsize=10, ncol=2, framealpha=1)
    #plt.tight_layout()
    #plt.show()
    plt.savefig("tau_star_vs_Ti.png", dpi=400)  

    plt.figure()



def main():
    # Example parameters
    m_bg_amu = 4.002602  # Mass of background species (Helium) in amu
    m_imp_amu = 6.941    # Mass of impurity species (Lithium) in amu
    z_imp = 1       # Charge state of impurity species (Lithium)
    ln_lambda = 10   # Coulomb logarithm (typical value)

    #z_bg = 1.0        # Effective charge state of background species (Helium)
    #ne_m3 = 1e18  # Electron density in m^-3
    ti_plot_low = 0.05   # Ion temperature in eV
    ti_plot_high = 7.0 # Ion temperature in eV
    Ti_ev = np.linspace(ti_plot_low, ti_plot_high, 2000)   # Ion temperature in eV

    f_coll_ne18 = ion_ion_collision_freq(1e18, Ti_ev, m_bg_amu, 1.0, m_imp_amu, z_imp, ln_lambda)
    f_coll_ne17 = ion_ion_collision_freq(1e17, Ti_ev, m_bg_amu, 1.0, m_imp_amu, z_imp, ln_lambda)
    #f_coll_ne16 = ion_ion_collision_freq(1e16, Ti_ev, m_bg_amu, 1.0, m_imp_amu, z_imp, ln_lambda)

    f_coll_ne18_z2He = ion_ion_collision_freq(1e18, Ti_ev, m_bg_amu, 2.0, m_imp_amu, z_imp, ln_lambda)
    f_coll_ne17_z2He = ion_ion_collision_freq(1e17, Ti_ev, m_bg_amu, 2.0, m_imp_amu, z_imp, ln_lambda)

    tau_coll_ne18_ms = 1e3 / f_coll_ne18
    tau_coll_ne17_ms = 1e3 / f_coll_ne17
    #tau_coll_ne16_ms = 1e3 / f_coll_ne16
    tau_coll_ne18_z2He_ms = 1e3 / f_coll_ne18_z2He
    tau_coll_ne17_z2He_ms = 1e3 / f_coll_ne17_z2He

    ## 'operating' domain
    tau_residence_hi_ms = 0.3 # E ms
    tau_residence_lo_ms = 0.1 # E ms
    typical_ionTemp_lo = 1.0 # eV
    typical_ionTemp_hi = 5.0 # eV
    transition_temp_lo = Ti_ev[tau_coll_ne17_ms >= tau_residence_lo_ms][0]
    transition_temp_hi = Ti_ev[tau_coll_ne17_ms >= tau_residence_hi_ms][0]

    # tau_coll_lo = tau_coll_ne18_ms[np.argmin(np.abs(Ti_ev - typical_ionTemp_lo))]
    # print(f'Collision time at Ti={typical_ionTemp_lo} eV and ne=1e18 m^-3: {tau_coll_lo:.2e} ms')
    # print(f'Collision frequency at Ti={typical_ionTemp_lo} eV and ne=1e18 m^-3: {f_coll_ne18[np.argmin(np.abs(Ti_ev - typical_ionTemp_lo))]:.2e} s^-1')

    tau_coll_lo = tau_coll_ne18_ms[np.argmin(np.abs(Ti_ev - typical_ionTemp_lo))]
    tau_coll_hi = tau_coll_ne17_ms[np.argmin(np.abs(Ti_ev - typical_ionTemp_hi))]
    print(f'Ion-ion collision time at Ti={typical_ionTemp_lo} eV and ne=1e18 m^-3: {tau_coll_lo:.2e} ms')
    print(f'Ion-ion collision time at Ti={typical_ionTemp_hi} eV and ne=1e17 m^-3: {tau_coll_hi:.2e} ms')
    #print(f'Collision frequency at Ti={typical_ionTemp_hi} eV and ne=1e17 m^-3: {f_coll_ne17[np.argmin(np.abs(Ti_ev - typical_ionTemp_hi))]:.2e} s^-1')

    f_ionNeutral_coll = ion_neutral_collision_freq(3e17, Ti_ev)  # Example neutral density of 3e17 m^-3
    tau_ionNeutral_coll_ms = 1e3 / f_ionNeutral_coll
    print('\nFAR SOL:')
    print(f'Ion-neutral collision frequency at Ti={typical_ionTemp_lo} eV and ng=3e17 m^-3: {f_ionNeutral_coll[np.argmin(np.abs(Ti_ev - typical_ionTemp_lo))]/1000.:.2e} kHz')
    print(f'Ion-neutral collision frequency at Ti={typical_ionTemp_hi} eV and ng=3e17 m^-3: {f_ionNeutral_coll[np.argmin(np.abs(Ti_ev - typical_ionTemp_hi))]/1000.:.2e} kHz')
    print(f'Ion-neutral collision time at Ti={typical_ionTemp_lo} eV and ng=3e17 m^-3: {tau_ionNeutral_coll_ms[np.argmin(np.abs(Ti_ev - typical_ionTemp_lo))]:.2e} ms') 
    print(f'Ion-neutral collision time at Ti={typical_ionTemp_hi} eV and ng=3e17 m^-3: {tau_ionNeutral_coll_ms[np.argmin(np.abs(Ti_ev - typical_ionTemp_hi))]:.2e} ms') 

    f_ionNeutral_coll_18 = ion_neutral_collision_freq(3e18, Ti_ev)  # Example neutral density of 3e18 m^-3
    tau_ionNeutral_coll_ms_18 = 1e3 / f_ionNeutral_coll_18
    print('\nEDGE PLASMA:')
    print(f'Ion-neutral collision frequency at Ti={typical_ionTemp_lo} eV and ng=3e18 m^-3: {f_ionNeutral_coll_18[np.argmin(np.abs(Ti_ev - typical_ionTemp_lo))]/1000.:.2e} kHz')
    print(f'Ion-neutral collision frequency at Ti={typical_ionTemp_hi} eV and ng=3e18 m^-3: {f_ionNeutral_coll_18[np.argmin(np.abs(Ti_ev - typical_ionTemp_hi))]/1000.:.2e} kHz')
    print(f'Ion-neutral collision time at Ti={typical_ionTemp_lo} eV and ng=3e18 m^-3: {tau_ionNeutral_coll_ms_18[np.argmin(np.abs(Ti_ev - typical_ionTemp_lo))]:.2e} ms') 
    print(f'Ion-neutral collision time at Ti={typical_ionTemp_hi} eV and ng=3e18 m^-3: {tau_ionNeutral_coll_ms_18[np.argmin(np.abs(Ti_ev - typical_ionTemp_hi))]:.2e} ms')


    tau_star_lo_lowTemp = tau_residence_lo_ms / tau_coll_lo
    tau_star_hi_lowTemp = tau_residence_hi_ms / tau_coll_lo
    tau_star_lo_highTemp = tau_residence_lo_ms / tau_coll_hi
    tau_star_hi_highTemp = tau_residence_hi_ms / tau_coll_hi
    print(f'At Ti={typical_ionTemp_lo} eV and ne=1e17 m^-3, tau* ranges from {tau_star_lo_lowTemp:.2e} to {tau_star_hi_lowTemp:.2e}')
    print(f'At Ti={typical_ionTemp_hi} eV and ne=1e17 m^-3, tau* ranges from {tau_star_lo_highTemp:.2e} to {tau_star_hi_highTemp:.2e}')

    plot_collision_times(Ti_ev, tau_coll_ne18_ms, tau_coll_ne17_ms, tau_ionNeutral_coll_ms, tau_ionNeutral_coll_ms_18,
                          tau_residence_lo_ms, tau_residence_hi_ms,
                          typical_ionTemp_lo, typical_ionTemp_hi,
                          transition_temp_lo, transition_temp_hi)

    f_typical_edge = f_coll_ne18 + f_ionNeutral_coll_18
    f_typical_far = f_coll_ne17 + f_ionNeutral_coll_18

    f_evap_edge = f_coll_ne18 + f_ionNeutral_coll
    f_evap_far = f_coll_ne17 + f_ionNeutral_coll

    plot_tau_starSOL(Ti_ev, f_typical_edge, f_typical_far, 
                    f_evap_edge, f_evap_far,
                    tau_residence_lo_ms, tau_residence_hi_ms, 
                    typical_ionTemp_lo, typical_ionTemp_hi)




if __name__ == "__main__":
    main()