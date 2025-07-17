import numpy as np
import logging
#import phi_events


class Particle:
    particleCount = 0
    """
    Class to store particle properties, data, and push method(?)
    """
    def __init__(self, type):
        self.type=type
        types = ['fieldline', 'ion']
        if self.type not in types:
            raise ValueError("Invalid particle type. Expected one of: %s" % types)
        
        Particle.particleCount += 1
        self.particleID = Particle.particleCount
        self.charge = np.float32(0)
        self.mass = np.float32(0)
        self.charge_mass_ratio = np.float32(0)
        self.maxLife = np.float32(0)

        self.pos0_XYZ = np.zeros(3, dtype=np.float32)
        self.vel0_XYZ = np.zeros(3, dtype=np.float32)


class FieldLine(Particle):
    def __init__(self, init_XYZ, maxlength, direction=1.0):
        super().__init__('fieldline')
        self.pos0_XYZ = init_XYZ
        self.maxLife = maxlength
        self.direction = direction
        self.vel0_XYZ = 0
        self.charge = 0.
        self.mass = 0.

    def pushXYZ(self, t, p_XYZ, field):
        """Change in position of field line is the normalized field vector at its current position"""
        B, dum_ = field.interpField(p_XYZ[:3])
        dY = B / np.sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2])

        return dY * self.direction

    def storePath(self):
        pass


class Ion(Particle):
    def __init__(self, init_XYZ, mass_amu, charge_z, maxlife):
        super().__init__('ion')
        self.terminated = False

        self.pos0_XYZ = np.asarray(init_XYZ)
        self.vel0_XYZ = np.zeros(3, dtype=np.float32)
        self.maxLife = maxlife

        self.charge = charge_z * 1.60217663E-19 # Coulombs
        self.mass = mass_amu * 1.66053907E-27 # kilograms
        self.charge_mass_ratio = self.charge / self.mass

        self.pos_XYZ = []

    def initVelocity(self, v0_XYZ):
        self.vel0_XYZ = np.asarray(v0_XYZ)
        self.vel_XYZ = self.vel0_XYZ

    def initOutput(self, dt, tmax):
        #N = tmax // dt + 1
        #self.pos_XYZ = np.empty([int(N), 3]) #size output array
        #self.pos_XYZ = []
        pass

    def setPosition(self, index, value):
        #self.pos_XYZ[index] = np.copy(value)
        self.pos_XYZ.append(value)

    def setVelocity(self, index, value):
        #self.vel_XYZ[index] = np.copy(value)
        self.vel0_XYZ = np.copy(value)