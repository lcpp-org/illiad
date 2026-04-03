## IMPORTS
import numpy as np
from numpy.polynomial import Polynomial
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import classes.class_outputHandler as out
from classes.mesh import *
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

def collision_freq(ne, Ti, m_bg, z_bg, m_imp, z_imp, ln_lambda):
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


def main():
    # Example parameters
    m_bg_amu = 4.002602  # Mass of background species (Helium) in amu
    m_imp_amu = 6.941    # Mass of impurity species (Lithium) in amu
    z_imp = 1       # Charge state of impurity species (Lithium)
    ln_lambda = 10   # Coulomb logarithm (typical value)

    #z_bg = 1.0        # Effective charge state of background species (Helium)
    #ne_m3 = 1e18  # Electron density in m^-3
    ti_plot_low = 0.5   # Ion temperature in eV
    ti_plot_high = 7.0 # Ion temperature in eV
    Ti_ev = np.linspace(ti_plot_low, ti_plot_high, 1000)   # Ion temperature in eV

    f_coll_ne18 = collision_freq(1e18, Ti_ev, m_bg_amu, 1.0, m_imp_amu, z_imp, ln_lambda)
    f_coll_ne17 = collision_freq(1e17, Ti_ev, m_bg_amu, 1.0, m_imp_amu, z_imp, ln_lambda)
    f_coll_ne16 = collision_freq(1e16, Ti_ev, m_bg_amu, 1.0, m_imp_amu, z_imp, ln_lambda)

    f_coll_ne18_z2He = collision_freq(1e18, Ti_ev, m_bg_amu, 2.0, m_imp_amu, z_imp, ln_lambda)
    f_coll_ne17_z2He = collision_freq(1e17, Ti_ev, m_bg_amu, 2.0, m_imp_amu, z_imp, ln_lambda)

    tau_coll_ne18_ms = 1e3 / f_coll_ne18
    tau_coll_ne17_ms = 1e3 / f_coll_ne17
    tau_coll_ne16_ms = 1e3 / f_coll_ne16
    tau_coll_ne18_z2He_ms = 1e3 / f_coll_ne18_z2He
    tau_coll_ne17_z2He_ms = 1e3 / f_coll_ne17_z2He

    ## 'operating' domain
    tau_residence_hi_ms = 0.3 # E ms
    tau_residence_lo_ms = 0.1 # E ms
    typical_ionTemp_lo = 0.5 # eV
    typical_ionTemp_hi = 7.0 # eV
    transition_temp_lo = Ti_ev[tau_coll_ne17_ms >= tau_residence_lo_ms][0]
    transition_temp_hi = Ti_ev[tau_coll_ne17_ms >= tau_residence_hi_ms][0]

    # tau_coll_lo = tau_coll_ne18_ms[np.argmin(np.abs(Ti_ev - typical_ionTemp_lo))]
    # print(f'Collision time at Ti={typical_ionTemp_lo} eV and ne=1e18 m^-3: {tau_coll_lo:.2e} ms')
    # print(f'Collision frequency at Ti={typical_ionTemp_lo} eV and ne=1e18 m^-3: {f_coll_ne18[np.argmin(np.abs(Ti_ev - typical_ionTemp_lo))]:.2e} s^-1')

    tau_coll_lo = tau_coll_ne17_ms[np.argmin(np.abs(Ti_ev - typical_ionTemp_lo))]
    tau_coll_hi = tau_coll_ne17_ms[np.argmin(np.abs(Ti_ev - typical_ionTemp_hi))]
    print(f'Collision time at Ti={typical_ionTemp_lo} eV and ne=1e17 m^-3: {tau_coll_lo:.2e} ms')
    print(f'Collision time at Ti={typical_ionTemp_hi} eV and ne=1e17 m^-3: {tau_coll_hi:.2e} ms')
    #print(f'Collision frequency at Ti={typical_ionTemp_hi} eV and ne=1e17 m^-3: {f_coll_ne17[np.argmin(np.abs(Ti_ev - typical_ionTemp_hi))]:.2e} s^-1')

    f_ionNeutral_coll = ion_neutral_collision_freq(3e17, Ti_ev)  # Example neutral density of 1e19 m^-3
    tau_ionNeutral_coll_ms = 1e3 / f_ionNeutral_coll
    #print(f'Ion-neutral collision frequency at Ti={typical_ionTemp_lo} eV and ng=3e17 m^-3: {f_ionNeutral_coll[np.argmin(np.abs(Ti_ev - typical_ionTemp_lo))]:.2e} s^-1')
    print(f'Ion-neutral collision time at Ti={typical_ionTemp_lo} eV and ng=3e17 m^-3: {tau_ionNeutral_coll_ms[np.argmin(np.abs(Ti_ev - typical_ionTemp_lo))]:.2e} ms') 
    print(f'Ion-neutral collision time at Ti={typical_ionTemp_hi} eV and ng=3e17 m^-3: {tau_ionNeutral_coll_ms[np.argmin(np.abs(Ti_ev - typical_ionTemp_hi))]:.2e} ms') 

    tau_star_lo_lowTemp = tau_residence_lo_ms / tau_coll_lo
    tau_star_hi_lowTemp = tau_residence_hi_ms / tau_coll_lo
    tau_star_lo_highTemp = tau_residence_lo_ms / tau_coll_hi
    tau_star_hi_highTemp = tau_residence_hi_ms / tau_coll_hi
    print(f'At Ti={typical_ionTemp_lo} eV and ne=1e17 m^-3, tau* ranges from {tau_star_lo_lowTemp:.2e} to {tau_star_hi_lowTemp:.2e}')
    print(f'At Ti={typical_ionTemp_hi} eV and ne=1e17 m^-3, tau* ranges from {tau_star_lo_highTemp:.2e} to {tau_star_hi_highTemp:.2e}')





    plt.figure()
    plt.plot(Ti_ev, tau_coll_ne18_ms, marker='none', linewidth=2,
             linestyle='-', color='blue', label='$\\tau_{i\\text{-}i,Core}$')
             #linestyle='-', color='blue', label='$\\tau_{i\\text{-}i},~n_e=10^{18}$')
    plt.plot(Ti_ev, tau_coll_ne17_ms, marker='none', linewidth=2,
              linestyle='-', color='black', label='$\\tau_{i\\text{-}i,far~SOL}$')
    plt.plot(Ti_ev, tau_ionNeutral_coll_ms, marker='none', linewidth=2,
              linestyle='-', color='darkorange', label='$\\tau_{i\\text{-}n}$')
              #linestyle='--', color='red', label='$\\tau_{i\\text{-}n}$')


    #collisional_bottom_line = np.maximum(tau_coll_ne17_ms, tau_residence_lo_ms)
    collisional_bottom_line = tau_coll_ne17_ms
    collisional_top_line = np.full_like(tau_coll_ne17_ms, tau_residence_hi_ms)
    collisional_span = (Ti_ev >= typical_ionTemp_lo) & (Ti_ev <= transition_temp_hi)
    plt.fill_between(Ti_ev[collisional_span],
                        collisional_bottom_line[collisional_span],
                        collisional_top_line[collisional_span],
                        color='red', alpha=0.25, label='Collisional Regime')
    
    #collisionless_top_line = np.minimum(tau_coll_ne17_ms, tau_residence_hi_ms)
    #collisionless_top_line = np.minimum(tau_coll_ne17_ms, tau_ionNeutral_coll_ms)
    collisionless_top_line = tau_coll_ne17_ms
    collisionless_botom_line = np.full_like(tau_coll_ne17_ms, tau_residence_lo_ms)
    collisionless_span = (Ti_ev <= typical_ionTemp_hi) & (Ti_ev >= transition_temp_lo)
    plt.fill_between(Ti_ev[collisionless_span], 
                        collisionless_botom_line[collisionless_span],
                        collisionless_top_line[collisionless_span],
                        color='green', alpha=0.25, label='Collisionless Regime')


    plt.axvspan(transition_temp_lo, transition_temp_hi,
                 color='darkgray', alpha=0.4, label='Collisionless-Collisional Transition Region', zorder=0)
    plt.axvline(transition_temp_lo, color='gray', linestyle=':')#, label='Typical Operating Range')
    plt.axvline(transition_temp_hi, color='gray', linestyle=':')#, label='Typical Operating Range')
    # plt.axhspan(tau_residence_lo_ms, tau_residence_hi_ms, color='gray', alpha=0.3, label='Residence Time Range')
    # plt.axvspan(typical_ionTemp_lo, typical_ionTemp_hi, color='gray', alpha=0.3, label='Operating Range')

    plt.xlabel('$T_i~\\text{[eV]}$')
    plt.ylabel('$\\tau~\\text{[ms]}$')#_{i\\text{-}i}~\\text{[ms]}$')
  
    plt.grid(which='both')
    plt.xlim(0, 8)#ti_plot_high)
    plt.ylim(1e-3, 1e0)#ti_plot_high)
    plt.yscale('log')
    plt.legend(loc='lower right', fontsize=8, ncol=2)
    #plt.tight_layout()
    #plt.show()
    plt.savefig("collision_time_vs_Ti.png", dpi=300)




if __name__ == "__main__":
    main()