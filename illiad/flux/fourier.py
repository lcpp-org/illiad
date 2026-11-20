"""Periodic geometric radius fits for regular, star-shaped poloidal contours.

This is a contour representation, not a straight-field-line angle transform.
Angles are geometric radians about the same origin used by the spline caller.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class FourierContour:
    """r(theta) = a0 + sum(a_m cos(m theta) + b_m sin(m theta))."""

    coefficients: np.ndarray
    diagnostics: dict

    @property
    def modes(self):
        return (len(self.coefficients) - 1) // 2

    def __call__(self, theta, derivative=0):
        if derivative not in (0, 1, 2):
            raise ValueError("derivative must be 0, 1, or 2")
        theta = np.asarray(theta, dtype=float)
        m = np.arange(1, self.modes + 1)
        phase = theta[..., None] * m + derivative * np.pi / 2
        values = ((np.cos(phase) * m**derivative) @ self.coefficients[1::2]
                  + (np.sin(phase) * m**derivative) @ self.coefficients[2::2])
        return values + (self.coefficients[0] if derivative == 0 else 0.0)


def validate_fourier_options(modes, regularization, max_gap):
    if isinstance(modes, bool) or not isinstance(modes, (int, np.integer)) or modes < 1:
        raise ValueError("FOURIER_MODES must be a positive integer")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("FOURIER_REGULARIZATION must be finite and nonnegative")
    if not np.isfinite(max_gap) or not 0 < max_gap <= 2 * np.pi:
        raise ValueError("Fourier maximum angular gap must be in (0, 2*pi]")


def fit_fourier_contour(theta, radius, *, modes=8, regularization=1e-6,
                        max_gap=np.pi / 2):
    """Fit with angular-cell weights and an m^4 curvature penalty.

    Minimize sum(w_i (r_fit-r_i)^2) + lambda sum(m^4 (a_m^2+b_m^2)),
    with sum(w_i)=1. Lambda is dimensionless and does not scale with the
    sample count. The mean radius is unpenalized. Solve the augmented least
    squares system directly, without forming normal equations.

    Reject underdetermined/rank-deficient fits, large unsampled angular gaps,
    and nonpositive fitted radii. These checks cannot certify surface topology.
    """
    validate_fourier_options(modes, regularization, max_gap)
    theta, radius = np.asarray(theta, dtype=float), np.asarray(radius, dtype=float)
    if theta.ndim != 1 or radius.shape != theta.shape:
        raise ValueError("theta and radius must be matching one-dimensional arrays")
    finite = np.isfinite(theta) & np.isfinite(radius)
    theta, radius = theta[finite] % (2 * np.pi), radius[finite]
    if np.any(radius <= 0):
        raise ValueError("Regular contours require strictly positive input radii")
    # Collapse identical angles with equal weight per distinct angular cell.
    theta, inverse, counts = np.unique(theta, return_inverse=True, return_counts=True)
    radius_sum = np.bincount(inverse, weights=radius)
    radius = radius_sum / counts
    if len(theta) < 2 * modes + 1:
        raise ValueError(f"Need at least {2*modes+1} distinct angles for {modes} modes; got {len(theta)}")
    gaps = np.diff(np.r_[theta, theta[0] + 2 * np.pi])
    if gaps.max() > max_gap:
        raise ValueError(f"Angular gap {np.rad2deg(gaps.max()):.1f} deg exceeds limit {np.rad2deg(max_gap):.1f} deg")
    weights = (gaps + np.roll(gaps, 1)) / (4 * np.pi)
    m = np.arange(1, modes + 1)
    design = np.ones((len(theta), 2 * modes + 1))
    design[:, 1::2] = np.cos(theta[:, None] * m)
    design[:, 2::2] = np.sin(theta[:, None] * m)
    weighted = np.sqrt(weights)[:, None] * design
    singular = np.linalg.svd(weighted, compute_uv=False)
    if singular[-1] <= singular[0] * 1e-12:
        raise ValueError("Angular sampling is rank deficient or ill-conditioned")
    penalty = np.sqrt(regularization) * np.r_[0, np.repeat(m**2, 2)]
    coeffs = np.linalg.lstsq(
        np.vstack((weighted, np.diag(penalty))),
        np.r_[np.sqrt(weights) * radius, np.zeros(len(penalty))], rcond=None)[0]
    residual = design @ coeffs - radius
    fit = FourierContour(coeffs, {
        "sample_count": int(finite.sum()), "distinct_angles": len(theta),
        "modes": int(modes), "regularization": float(regularization),
        "max_gap_deg": float(np.rad2deg(gaps.max())),
        "condition_number": float(singular[0] / singular[-1]),
        "weighted_rms_m": float(np.sqrt(weights @ residual**2)),
        "max_residual_m": float(np.max(np.abs(residual))),
    })
    grid = np.linspace(0, 2*np.pi, max(4096, 64*modes), endpoint=False)
    if np.min(fit(grid)) <= 0:
        raise ValueError("Fourier fit has nonpositive radii on the validation grid")
    return fit
