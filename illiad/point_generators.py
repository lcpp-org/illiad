"""Public initial-condition generation namespace for ILLIAD."""

__all__ = ["generateSeedShells", "generate_MB_velocities", "ionInitializer"]


def __getattr__(name):
    if name in __all__:
        from utility import point_generators

        return getattr(point_generators, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
