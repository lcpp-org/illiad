## IMPORTS
import numpy as np
from numpy.polynomial import Polynomial
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import classes.class_outputHandler as out
from classes.mesh import *

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

def main():
    # Example parameters
    m_bg_amu = 4.002602  # Mass of background species (Helium) in amu
    m_imp_amu = 6.941    # Mass of impurity species (Lithium) in amu
    z_imp = 1       # Charge state of impurity species (Lithium)
    ln_lambda = 10   # Coulomb logarithm (typical value)

    #z_bg = 1.0        # Effective charge state of background species (Helium)
    #ne_m3 = 1e18  # Electron density in m^-3
    ti_plot_low = 0.2   # Ion temperature in eV
    ti_plot_high = 14.0 # Ion temperature in eV
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

    # extract the value of tau_coll_ne18_ms at Ti = 2 eV

    typical_ionTemp_lo = 1.0 # eV
    typical_ionTemp_hi = 7.0 # eV


    tau_coll_lo = tau_coll_ne18_ms[np.argmin(np.abs(Ti_ev - typical_ionTemp_lo))]
    print(f'Collision time at Ti={typical_ionTemp_lo} eV and ne=1e18 m^-3: {tau_coll_lo:.2e} ms')
    print(f'Collision frequency at Ti={typical_ionTemp_lo} eV and ne=1e18 m^-3: {f_coll_ne18[np.argmin(np.abs(Ti_ev - typical_ionTemp_lo))]:.2e} s^-1')

    tau_coll_hi = tau_coll_ne17_ms[np.argmin(np.abs(Ti_ev - typical_ionTemp_hi))]
    print(f'Collision time at Ti={typical_ionTemp_hi} eV and ne=1e17 m^-3: {tau_coll_hi:.2e} ms')
    print(f'Collision frequency at Ti={typical_ionTemp_hi} eV and ne=1e17 m^-3: {f_coll_ne17[np.argmin(np.abs(Ti_ev - typical_ionTemp_hi))]:.2e} s^-1')

    plt.figure()
    plt.plot(Ti_ev, tau_coll_ne18_ms, marker='none', linewidth=2,
              linestyle='-', color='blue', label='$n_e=1^{18} \, m^{-3}, \\, Z_{He}=1$')
    plt.plot(Ti_ev, tau_coll_ne18_z2He_ms, marker='none', linewidth=2,
              linestyle=':', color='blue', label='$n_e=1^{18} \, m^{-3}, \\, Z_{He}=2$')  


    plt.plot(Ti_ev, tau_coll_ne17_ms, marker='none', linewidth=2,
              linestyle='-', color='green', label='$n_e=1^{17} \, m^{-3}, \\, Z_{He}=1$')
    plt.plot(Ti_ev, tau_coll_ne17_z2He_ms, marker='none', linewidth=2,
              linestyle=':', color='green', label='$n_e=1^{17} \, m^{-3}, \\, Z_{He}=2$')

    plt.scatter(typical_ionTemp_lo, tau_coll_lo, marker='^', color='black', zorder=6)#, label='$n_e=1^{16} \, m^{-3}, \\, Z_{He}=1$')
    plt.scatter(typical_ionTemp_hi, tau_coll_hi, marker='v', color='black', zorder=6)#, label='$n_e=1^{16} \, m^{-3}, \\, Z_{He}=1$')

    plt.xlabel('Ion Temperature (eV)')
    plt.ylabel('Collision Time, $\\tau_{ii}$ (ms)')
    #plt.title('Collision Time vs Ion Temperature')
    plt.grid(which='both')
    plt.axvspan(typical_ionTemp_lo, typical_ionTemp_hi, color='gray', alpha=0.3)#, label='Typical Core-Edge Ion Temperature Range')
    plt.xlim(0, ti_plot_high)
    plt.yscale('log')
    plt.legend(loc='lower right', fontsize=8, ncol=2)
    #plt.show()
    plt.savefig("collision_time_vs_Ti.png", dpi=300)




if __name__ == "__main__":
    main()