import numpy as np
#import numba as nb

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


def RTP_to_XYZ(p_RTP, Rmajor=0.72):
    """Converts r-theta-phi coordinates to Cartesian coordinates.

    Args:
        p_RTP (array-like): Input point in r-theta-phi coordinates, shape (3,).
        Rmajor (float, optional): Major radius for the coordinate transformation. Defaults to 0.72.

    Returns:
        numpy.ndarray: Transformed point in Cartesian coordinates, shape (3,).

    Notes:
        When looking at a cross-section to the right of the +z axis, +theta is counterclockwise.
        +phi is clockwise when viewed from above.
    """
    r, theta, phi = p_RTP[:3]
    temp_ = (Rmajor + r * np.cos(theta))
    x = temp_ * np.cos(phi)
    y = (-1) * temp_ * np.sin(phi)
    z = r * np.sin(theta)
    p_XYZ = np.array([x, y, z])

    return p_XYZ

def XYZ_to_RTP(p_XYZ, Rmajor=0.72):
    """Converts Cartesian coordinates to r-theta-phi coordinates.

    Args:
        p_XYZ (array-like): Input point in Cartesian coordinates, shape (3,).
        Rmajor (float, optional): Major radius for the coordinate transformation. Defaults to 0.72.

    Returns:
        numpy.ndarray: Transformed point in r-theta-phi coordinates, shape (3,).

    Notes:
        When looking at a cross-section to the right of the +z axis, +theta is counterclockwise.
        +phi is clockwise when viewed from above.
    """
    x, y, z = p_XYZ[:3]
    x2plusy2 = x*x + y*y
    z2 = z*z
    R = np.sqrt(x2plusy2)
    r = np.sqrt(x2plusy2 + z2 + Rmajor*Rmajor - 2*Rmajor*R)

    den = R - Rmajor
    theta = np.arctan2(z, den)

    # arctan2 returns radians from (-pi to +pi)
    # here we shift the domain to (0 to 2*pi)
    if theta < 0:
        theta += 2 * np.pi

    phi = -np.arctan2(y, x)
    # arctan2 returns radians from (-pi to +pi)
    # here we shift the domain to (0 to 2*pi)
    if phi < 0:
        phi += 2 * np.pi

    p_RTP = np.array([r, theta, phi])

    return p_RTP

def XYZ_to_RTP2(p_XYZ, Rmajor=0.72):
    """Converts Cartesian coordinates to r-theta-phi coordinates.

    This function transforms a point or array of points from Cartesian (x, y, z)
    coordinates to r-theta-phi (RTP) coordinates, using a specified major radius.
    The conventions used are:
      - When looking at a cross-section to the right of the +z axis, +theta is counterclockwise.
      - +phi is clockwise when viewed from above.

    Args:
        p_XYZ (array-like or torch.Tensor): Input point(s) in Cartesian coordinates,
            shape (..., 3).
        Rmajor (float, optional): Major radius for the coordinate transformation.
            Defaults to 0.72.

    Returns:
        torch.Tensor: Transformed point(s) in r-theta-phi coordinates, same shape as input.
    """
    p_XYZ = torch.tensor(p_XYZ).to(device)
    p_RTP = torch.zeros(p_XYZ.shape, dtype=torch.float64).to(device)
    x, y, z = p_XYZ.T
    x2 = x*x
    y2 = y*y
    z2 = z*z
    R = torch.sqrt(x2 + y2)

    p_RTP.T[0] = torch.sqrt( x2 + y2 + z2 + Rmajor*Rmajor - 2*Rmajor*R )

    den = R - Rmajor
    theta = torch.arctan2(z,den) # arctan2 returns radians from (-pi to +pi)
    # here we shift the domain to (0 to 2*pi)
    p_RTP.T[1] = torch.where(theta < 0, theta + 2*torch.pi, theta)

    phi = (-1) * torch.arctan2(y,x) # arctan2 returns radians from (-pi to +pi)
    # here we shift the domain to (0 to 2*pi)
    p_RTP.T[2] = torch.where(phi<0, phi + 2*torch.pi, phi)

    return p_RTP


## ROTATE A CARTESIAN VECTOR BY ANGLE DELTA_PHI
def rot_vecXYZ_byPHI(vec_XYZ, delta_phi):
    """Rotates a 3D Cartesian vector around the z-axis by a given angle.
    Args:
        vec_XYZ (array-like): A 3-element array or list representing the Cartesian vector [x, y, z].
        delta_phi (float): The rotation angle in radians. Positive values rotate clockwise when viewed from above.
    Returns:
        numpy.ndarray: The rotated 3D Cartesian vector as a NumPy array.
    Notes:
        - The rotation is performed around the z-axis.
        - When looking at a cross-section to the right of the +z axis, theta is counterclockwise.
        - +phi is clockwise when viewed from above.
    """
    # Function takes in a cartesian vector and a phi angle
    # Returns the cartesian values of the vector rotated by phi degrees
    # convention: When looking at across-section to the right of the +z axis, theta is counterclockwise
    # convention: +phi is clockwise when viewed from above
    rotated_XYZ = np.zeros(3)
    xFormMatrix = np.array([[ np.cos(delta_phi), -np.sin(delta_phi), 0.0],
                            [ np.sin(delta_phi),  np.cos(delta_phi), 0.0],
                            [               0.0,                0.0, 1.0]])
    
    rotated_XYZ = np.dot(vec_XYZ, xFormMatrix)

    return rotated_XYZ

def RTP_XYZ_JAC(p_rtp, vec_xyz, form='xyz2rtp'):
    """
    Converts a vector between Cartesian (XYZ) and r-theta-phi (RTP) coordinates.

    Args:
        p_rtp (array-like): The r-theta-phi coordinates [r, theta, phi] of the point.
        vec_xyz (array-like): The vector to be transformed.
        form (str, optional): Transformation direction. 
            'xyz2rtp' converts from Cartesian to RTP.
            'rtp2xyz' converts from RTP to Cartesian.
            Defaults to 'xyz2rtp'.

    Returns:
        numpy.ndarray: The transformed vector.

    Raises:
        ValueError: If `form` is not 'rtp2xyz' or 'xyz2rtp'.
    """
    ctheta = np.cos(p_rtp[1])
    stheta = np.sin(p_rtp[1])
    cphi = np.cos(p_rtp[2])
    sphi = np.sin(p_rtp[2])

    if form == 'rtp2xyz':
        XformTranspose = np.array([[ctheta*cphi, -stheta*cphi, -sphi],
                                  [-ctheta*sphi,  stheta*sphi, -cphi],
                                  [      stheta,       ctheta,     0]])
        return np.dot(XformTranspose, vec_xyz)
    
    elif form == 'xyz2rtp':
        Xform = np.array([[ctheta*cphi, -ctheta*sphi, stheta],
                         [-stheta*cphi,  stheta*sphi, ctheta],
                         [       -sphi,        -cphi,     0]])
        return np.dot(Xform, vec_xyz)
    else:
        raise ValueError("form must be 'rtp2xyz' or 'xyz2rtp'")

def axisShift(theta, radius, d_theta, d_radius): 
    """
    Transforms polar coordinates from a original axis to a shifted axis.

    Given a point in polar coords, this function computes the corresponding
    coords about a shifted axis, offset by (d_theta, d_radius).

    Args:
        theta (float or np.ndarray): Angle(s) in radians about the geometric axis.
        radius (float or np.ndarray): Radial coordinate(s) about the geometric axis.
        d_theta (float): Angular offset in radians between axes.
        d_radius (float): Radial offset between axes.

    Returns:
        np.ndarray: A 2-element array containing:
            - theta_prime (np.ndarray): Angle(s) in radians about the shifted axis, in [0, 2π).
            - rad_prime (np.ndarray): Radial coordinate(s) about the shifted axis.
    """
    xprime = radius*np.cos(theta) - d_radius*np.cos(d_theta)
    zprime = radius*np.sin(theta) - d_radius*np.sin(d_theta)

    rad_prime = np.sqrt(xprime**2 + zprime**2)
    theta_prime = np.arctan2(zprime, xprime)
    #theta_prime[thetaprime <= 0] += 2 * np.pi
    theta_prime = np.where(theta_prime<=0, theta_prime + 2*np.pi, theta_prime)

    return np.array([theta_prime, rad_prime])

def align_z_to_vector(v):
    """Returns a rotation matrix that aligns the z-axis to the given vector.

    Args:
        v (np.ndarray): A 3-element array representing the target vector.

    Returns:
        np.ndarray: A 3x3 rotation matrix that rotates the z-axis to align with `v`.

    Raises:
        ValueError: If `v` is not a 3-element array.

    Notes:
        - If `v` is already aligned with the z-axis, the identity matrix is returned.
        - If `v` is anti-aligned with the z-axis, a 180-degree rotation matrix is returned.
    """
    # helper function to align the z-axis to a given vector
    z_axis = np.array([0, 0, 1])
    #v = v / np.linalg.norm(v)
    if np.allclose(v, z_axis):
        return np.eye(3)
    if np.allclose(v, -z_axis):
        # 180 degree rotation around any perpendicular axis
        return np.array([[-1,  0,  0],
                         [ 0, -1,  0],
                         [ 0,  0,  1]])
    axis = np.cross(z_axis, v)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.dot(z_axis, v))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R
