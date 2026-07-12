"""Finite-difference helpers for scalar fields on periodic toroidal meshes."""

import numpy as np


def periodic_centered_difference(values, coordinates, axis):
    """Return a second-order centered derivative on a uniform periodic axis."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 1 or coordinates.size != values.shape[axis]:
        raise ValueError("Coordinate length must match the differentiated array axis")
    if coordinates.size < 3:
        raise ValueError("A periodic centered derivative requires at least three points")

    unwrapped = np.unwrap(coordinates)
    spacings = np.diff(unwrapped)
    seam_spacing = unwrapped[0] + 2.0*np.pi - unwrapped[-1]
    all_spacings = np.concatenate((spacings, [seam_spacing]))
    spacing = float(np.mean(all_spacings))
    if spacing <= 0.0 or not np.allclose(all_spacings, spacing, rtol=1e-6, atol=1e-12):
        raise ValueError("Periodic angular coordinates must be uniformly spaced")

    return (
        np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)
    ) / (2.0*spacing)


def scalar_gradient_periodic_angles(values, phi_degrees, theta_radians, radii):
    """Return derivatives in phi, theta, and radius for (phi, theta, rho) data."""
    values = np.asarray(values)
    phi_radians = np.radians(np.asarray(phi_degrees, dtype=np.float64))
    theta_radians = np.asarray(theta_radians, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)

    d_dphi = periodic_centered_difference(values, phi_radians, axis=0)
    d_dtheta = periodic_centered_difference(values, theta_radians, axis=1)
    d_drho = np.gradient(values, radii, axis=2, edge_order=2)
    return d_dphi, d_dtheta, d_drho
