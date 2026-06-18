"""Public Boris solver namespace for ILLIAD."""

__all__ = ["Boris"]


def __getattr__(name):
    if name == "Boris":
        from classes.boris import Boris

        return Boris
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
