import numpy as np
import numba as nb


## TRANSFORM TO CARTESIAN COORDINATES
##
@nb.jit(nb.types.Array(nb.float64, 1, "C")(nb.types.Array(nb.float64, 1, "A"), nb.float64), nopython=True)
def RTP_to_XYZ(p_RTP, Rmajor):
    # Function to take in a point defined in r-theta-phi coordinates
    # And return a point in Cartesian coordinates
    # convention: When looking at a cross-section to the right of the +z axis, +theta is counterclockwise
    # convention: +phi is clockwise when viewed from above
    r, theta, phi = p_RTP[:3]
    
    x = (Rmajor + r*np.cos(theta)) * np.cos(phi)
    y = (-1) * (Rmajor + r*np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    p_XYZ = np.array([x, y, z])
    
    return p_XYZ



## TRANSFORM TO TOROIDAL COORDINATES
##
@nb.jit(nb.types.Array(nb.float64, 1, "C")(nb.types.Array(nb.float64, 1, "C"), nb.float64), nopython=True)
def XYZ_to_RTP(p_XYZ, Rmajor):
    # Function to take in a point defined in Cartesian coordinates
    # And return a point in r-theta-phi coordinates
    # convention: When looking at a cross-section to the right of the +z axis, +theta is counterclockwise
    # convention: +phi is clockwise when viewed from above
    x, y, z = p_XYZ

    r = np.sqrt( x**2 + y**2 + z**2 + Rmajor**2 - 2*Rmajor*np.sqrt(x**2 + y**2) )

    den = np.sqrt(x**2 + y**2) - Rmajor
    theta = np.arctan2(z,den)
    # arctan2 returns radians from (-pi to +pi)
    # here we shift the domain to (0 to 2*pi)
    if theta<0: theta += 2*np.pi

    phi = (-1) * np.arctan2(y,x)
    # arctan2 returns radians from (-pi to +pi)
    # here we shift the domain to (0 to 2*pi)
    if phi<0: phi += 2*np.pi

    p_RTP = np.array([r, theta, phi])

    return p_RTP



@nb.jit(nb.types.Array(nb.float64, 1, "C")(nb.types.Array(nb.float64, 1, "C"), nb.float64), nopython=True)
def rot_vecXYZ_byPHI(vec_XYZ, delta_phi):
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


# transform from theta,r about (Geometric Axis)
# to (Magnetic Axis): theta=pi, r=0.0187m
def axisShift(rho, theta, r_delt_):

    #rprime = np.sqrt(r_delt_**2 + rho**2 + 2*rho*r_delt_*np.cos(theta))
    #thetaprime = np.arctan2( (rho*np.sin(theta)), r_delt_+rprime*np.cos(theta))
    #if thetaprime<0.: thetaprime += 2*np.pi
    #xformed_Coord = np.array([thetaprime, rprime])
    
    xformed_Coord = np.array([theta, rho])
    return xformed_Coord