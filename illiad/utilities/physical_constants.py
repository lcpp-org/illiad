"""Physical constants and background-species data used by ILLIAD."""

KG_PER_AMU       = 1.660_539_068e-27  # kg/amu
JOULES_PER_EV    = 1.602_176_634e-19  # J/eV
ELECTRON_MASS_KG = 9.109_383_701e-31  # kg

SPECIES_MASS_AMU = {
        # Hydrogen isotopes
        "H":    1.007825,
        "D":    2.014101,
        "T":    3.016049,

        # Helium isotopes
        "He":   4.002603,
        "He-3": 3.016029,
        "He-4": 4.002603,

        # Lithium isotopes
        "Li":   6.941000,
        "Li-6": 6.015122,
        "Li-7": 7.016004,

        # Other species
        "Ar":   39.94800,
    }


def get_species_mass_amu(species):
    """Return the atomic mass of a supported species in atomic mass units."""
    try:
        return SPECIES_MASS_AMU[species]
    except KeyError as exc:
        valid_species = ", ".join(SPECIES_MASS_AMU)
        raise ValueError(
            f"Unknown species '{species}'. "
            f"Valid options are: {valid_species}"
        ) from exc
