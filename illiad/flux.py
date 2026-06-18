"""Public flux-analysis namespace for ILLIAD."""

__all__ = ["fluxCalculator", "fluxInterpolator", "fluxGradientor"]


def __getattr__(name):
    if name == "fluxCalculator":
        from utility.Flux_Calculator import fluxCalculator

        return fluxCalculator
    if name == "fluxInterpolator":
        from utility.Flux_Interpolator import fluxInterpolator

        return fluxInterpolator
    if name == "fluxGradientor":
        from utility.Flux_Gradientor import fluxGradientor

        return fluxGradientor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
