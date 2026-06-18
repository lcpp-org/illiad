"""Public collision-model namespace for ILLIAD."""

__all__ = [
    "Collisions",
    "kg_per_amu",
    "kboltz",
    "eps0",
    "sqrt_pi",
    "Li_mass",
    "He_mass",
]


def __getattr__(name):
    if name in __all__:
        from classes import collisions

        return getattr(collisions, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
