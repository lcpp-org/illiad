"""Public flux-analysis namespace for ILLIAD."""

__all__ = [
    "calculate_flux",
    "interpolate_flux",
    "build_electric_field",
    "fluxCalculator",
    "fluxInterpolator",
    "fluxGradientor",
]


def calculate_flux(input_params=None):
    from utility.Flux_Calculator import fluxCalculator

    return fluxCalculator(input_params)


def interpolate_flux(input_params=None):
    from utility.Flux_Interpolator import fluxInterpolator

    return fluxInterpolator(input_params)


def build_electric_field(input_params=None):
    from utility.Flux_Gradientor import fluxGradientor

    return fluxGradientor(input_params)


def fluxCalculator(input_params=None):
    return calculate_flux(input_params)


def fluxInterpolator(input_params=None):
    return interpolate_flux(input_params)


def fluxGradientor(input_params=None):
    return build_electric_field(input_params)


def __getattr__(name):
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
