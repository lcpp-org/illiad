import numpy as np

# define a class with mesh information
class Mesh:
	def __init__(self, r, theta, phi, Bx, By, Bz, Bnorm):
        self.r = r
        self.theta = theta
        self.phi = phi
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.Bnorm = Bnorm
		self.Rmaj = 0.0 
		self.Rmin = 0.0
		
def setMeshValues(mesh):
	mesh.Rmaj = 0.72
    mesh.Rmin = 0.19