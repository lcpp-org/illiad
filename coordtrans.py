import numpy as np
import numba as nb

##
## TRANSFORM TO CARTESIAN COORDINATES
##
#@nb.jit(nb.types.Array(nb.float64, 1, "C")(nb.types.Array(nb.float64, 1, "C"), nb.float64), nopython=True)
def RTP_to_XYZ(p_RTP, Rmajor):
    r, theta, phi = p_RTP[:3]
    
    x = (Rmajor + r*np.cos(theta)) * np.cos(phi)
    y = (Rmajor + r*np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    p_XYZ = np.array([x, y, z])
    
    return p_XYZ


##
## TRANSFORM TO TOROIDAL COORDINATES
##
@nb.jit(nb.types.Array(nb.float64, 1, "C")(nb.types.Array(nb.float64, 1, "C"), nb.float64), nopython=True)
def XYZ_to_RTP(p_XYZ, Rmajor):
    x, y, z = p_XYZ

    r = np.sqrt( x**2 + y**2 + z**2 + Rmajor**2 - 2*Rmajor*np.sqrt(x**2 + y**2) )

    den = np.sqrt(x**2 + y**2) - Rmajor
    theta = np.arctan2(z,den)
    # arctan2 returns radians from (-pi to +pi)
    # here we shift the domain to (0 to 2*pi)
    if theta<0: theta += 2*np.pi

    phi = np.arctan2(y,x)
    # arctan2 returns radians from (-pi to +pi)
    # here we shift the domain to (0 to 2*pi)
    if phi<0: phi += 2*np.pi

    p_RTP = np.array([r, theta, phi])

    return p_RTP