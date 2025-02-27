import numpy as np
#import numba as nb

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

## TRANSFORM TO CARTESIAN COORDINATES
#@nb.jit(nb.types.Array(nb.float64, 1, "C")(nb.types.Array(nb.float64, 1, "A"), nb.float64), nopython=True)
def RTP_to_XYZ(p_RTP, Rmajor):
    # Function to take in a point defined in r-theta-phi coordinates
    # And return a point in Cartesian coordinates
    # convention: When looking at a cross-section to the right of the +z axis, +theta is counterclockwise
    # convention: +phi is clockwise when viewed from above
    r, theta, phi = p_RTP[:3]
    temp_ = (Rmajor + r*np.cos(theta))
    x = temp_ * np.cos(phi)
    y = (-1) * temp_ * np.sin(phi)
    z = r * np.sin(theta)
    p_XYZ = np.array([x, y, z])
    
    return p_XYZ


## TRANSFORM TO TOROIDAL COORDINATES
###@nb.jit(nb.types.Array(nb.float64, 1, "C")(nb.types.Array(nb.float64, 1, "C"), nb.float64), nopython=True)
def XYZ_to_RTP(p_XYZ, Rmajor):
    # Function to take in a point defined in Cartesian coordinates
    # And return a point in r-theta-phi coordinates
    # convention: When looking at a cross-section to the right of the +z axis, +theta is counterclockwise
    # convention: +phi is clockwise when viewed from above
    x, y, z = p_XYZ[:3]
    #x2 = x*x
    #y2 = y*y
    x2plusy2 = x*x + y*y
    z2 = z*z
    R = np.sqrt(x2plusy2)
    r = np.sqrt( x2plusy2 + z2 + Rmajor*Rmajor - 2*Rmajor*R )

    den = R - Rmajor
    theta = np.arctan2(z,den)

    # arctan2 returns radians from (-pi to +pi)
    # here we shift the domain to (0 to 2*pi)
    if theta<0: theta += 2*np.pi

    phi = -np.arctan2(y,x)
    # arctan2 returns radians from (-pi to +pi)
    # here we shift the domain to (0 to 2*pi)
    if phi<0: phi += 2*np.pi

    p_RTP = np.array([r, theta, phi])

    return p_RTP

def XYZ_to_RTP2(p_XYZ, Rmajor):
    # Function to take in a point defined in Cartesian coordinates
    # And return a point in r-theta-phi coordinates
    # convention: When looking at a cross-section to the right of the +z axis, +theta is counterclockwise
    # convention: +phi is clockwise when viewed from above
    p_XYZ = torch.tensor(p_XYZ).to(device)
    p_RTP = torch.zeros(p_XYZ.shape, dtype=torch.float64).to(device)
    x, y, z = p_XYZ.T
    #print(f'xyz_to_rto: {x=}')
    x2 = x*x
    y2 = y*y
    z2 = z*z
    R = torch.sqrt(x2 + y2)

    p_RTP.T[0] = torch.sqrt( x2 + y2 + z2 + Rmajor*Rmajor - 2*Rmajor*R )

    den = R - Rmajor
    theta = torch.arctan2(z,den) # arctan2 returns radians from (-pi to +pi)
    # here we shift the domain to (0 to 2*pi)
    p_RTP.T[1] = torch.where(theta<0, theta + 2*torch.pi, theta)
    #p_RTP.T[1] = theta

    phi = (-1) * torch.arctan2(y,x) # arctan2 returns radians from (-pi to +pi)
    # here we shift the domain to (0 to 2*pi)
    p_RTP.T[2] = torch.where(phi<0, phi + 2*torch.pi, phi)
    #p_RTP.T[2] = phi

    return p_RTP


## ROTATE A CARTESIAN VECTOR BY ANGLE DELTA_PHI
#@nb.jit(nb.types.Array(nb.float64, 1, "C")(nb.types.Array(nb.float64, 1, "C"), nb.float64), nopython=True)
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



def RTP_XYZ_JAC(p_rtp, vec_xyz):
    ctheta = np.cos(p_rtp[1])
    stheta = np.sin(p_rtp[1])
    cphi = np.cos(p_rtp[2])
    sphi = np.sin(p_rtp[2])
    Xform = np.array([[ctheta*cphi, -ctheta*sphi, stheta],
                     [-stheta*cphi,  stheta*sphi, ctheta],
                     [       -sphi,        -cphi,     0]])
    
    return np.dot(Xform, vec_xyz)

# TRANSFORM FROM THETA,R ABOUT (GEOMETRIC AXIS)
# TO (MAGNETIC AXIS): THETA=PI, R=0.0187M
def axisShift(rho, theta, rdel, thdel_, Rmaj=0.72): 

    xprime = rho*np.cos(theta) - rdel*np.cos(thdel_)
    zprime = rho*np.sin(theta) - rdel*np.sin(thdel_)

    rprime = np.sqrt(xprime**2 + zprime**2)
    thetaprime = np.arctan2(zprime, xprime)
    #thetaprime = np.where(thetaprime<=0, thetaprime + 2*np.pi, thetaprime)
    if thetaprime<=0: thetaprime += 2*np.pi
    return np.array([thetaprime, rprime])



