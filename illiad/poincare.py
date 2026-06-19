"""Public Poincare namespace for ILLIAD."""

__all__ = ["Poincare"]


def __getattr__(name):
    if name == "Poincare":
        from classes.poincare import Poincare

        return Poincare
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
