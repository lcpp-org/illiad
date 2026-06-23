"""Public plotting namespace for ILLIAD."""

__all__ = ["plotFuncs"]


def __getattr__(name):
    if name == "plotFuncs":
        from plot_funcs import plotFuncs

        return plotFuncs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
