import numpy as np

# define a class with mesh information
class Mesh:
    # Class to store the mesh information
    #
    # r, theta, and phi are the 1D arrays with the grid points
    #
    #  r: minor radius coordinate, measured from the geometrical center of the poloidal cross section
    #  theta: poloidal angle [0,2pi)
    #  phi: toroidal angle [0,2pi)
    #
    # Rmaj: major radius of the tokamak
    # Rmin: minor radius of the tokamak
    #
    # dr: grid spacing in the r direction
    # dtheta: grid spacing in the theta direction
    # dphi: grid spacing in the phi direction

    def __init__(self):
        self.r = np.array([])
        self.theta = np.array([])
        self.phi = np.array([])
        self.R0 = 0.0 
        self.a = 0.0
        self.dr = 0.0
        self.dtheta = 0.0
        self.dphi = 0.0
        self.nr = 0.0
        self.ntheta = 0.0
        self.nphi = 0.0

    def setMeshValues(self, R0, a):
        self.R0 = R0
        self.a = a


    def XYZ_to_deltaWall(self, dum, p_XYZ):
        
        self.XYZ_to_deltaWall.terminal = True
        x, y, z = p_XYZ

        r = np.sqrt( x**2 + y**2 + z**2 + self.R0**2 - 2*self.R0*np.sqrt(x**2 + y**2) )
        return r - self.a



class Field(Mesh):
    #def __init__(self, Bx, By, Bz):

        #self.Bx = Bx
        #self.By = By
        #self.Bz = Bz
        #
        #self.nphi, self.ntheta, self.nr = Bx.shape
        #
        #self.dr     = self.a / (self.nr-1)
        #self.dtheta = 2*np.pi / (self.ntheta-11)
        #self.dphi   = 2*np.pi / (self.nphi-1)
        #
        #self.r     = np.linspace(0.0, self.Rmin, self.nr)
        #self.theta = np.linspace(0.0, 2*np.pi, self.ntheta)
        #self.phi   = np.linspace(0.0, 2*np.pi, self.nphi)

    def populateField(self, Bx, By, Bz):
        
        self.Bx = Bx
        self.By = By
        self.Bz = Bz

        self.nphi, self.ntheta, self.nr = Bx.shape

        self.dr     = self.a / (self.nr-1)
        self.dtheta = 2*np.pi / (self.ntheta-1)
        self.dphi   = 2*np.pi / (self.nphi-1)

        self.r     = np.linspace(0.0, self.a, self.nr)
        self.theta = np.linspace(0.0, 2*np.pi, self.ntheta)
        self.phi   = np.linspace(0.0, 2*np.pi, self.nphi)



#P_XYZ = np.array([0.0, 0.0, 0.0])

#P_RTP = cartesian2polioidal( P_XYZ, mesh, ...)