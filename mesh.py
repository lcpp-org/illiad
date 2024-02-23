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
		self.Rmaj = 0.0 
		self.Rmin = 0.0
		self.dr = 0.0
		self.dtheta = 0.0
		self.dphi = 0.0
		
def setMeshValues(mesh):
	mesh.Rmaj = 0.72
    mesh.Rmin = 0.19
	
class Field:
    def __init__(self, Bx, By, Bz, Bnorm):
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.Bnorm = Bnorm
		
P_XYZ = np.array([0.0, 0.0, 0.0])
	
P_RTP = cartesian2polioidal( P_XYZ, mesh, ...)