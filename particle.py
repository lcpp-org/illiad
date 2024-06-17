import numpy as np
import logging



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
        self.charge: np.float32
        self.mass: np.float32
        self.pos0_XYZ = np.zeros(3, dtype=np.float64)
        self.vel0_XYZ = np.zeros(3, dtype=np.float64)
        self.pos_XYZ = np.zeros(3, dtype=np.float64)
        self.vel_XYZ = np.zeros(3, dtype=np.float64)
        self.maxLife: np.float32



class fieldLine(Particle):
    def __init__(self, init_XYZ, maxlength):
        super().__init__('fieldline')
        self.pos0_XYZ = init_XYZ
        self.maxLife = maxlength
        self.vel0_XYZ = 0
        self.charge = 0.
        self.mass = 0.

    def pushXYZ(self, t, p_XYZ, field):
        direction = 1
        B = np.zeros(3)
        B, dum_ = field.interpField(p_XYZ[:3])
        if field.errField==True:
            B[0] += 0.0002
            B[1] += -0.0002

        dY = direction * B / np.linalg.norm(B)

        return dY

    def storePath(self):
        pass



class Ion(Particle):
    def __init__(self, init_XYZ, mass_amu, charge_z, maxlife):
        super().__init__('ion')
        self.pos0_XYZ = np.asarray(init_XYZ)
        self.vel0_XYZ = np.zeros(3, dtype=np.float64)
        self.maxLife = maxlife
        self.charge = charge_z * 1.60217663E-19 # Coulombs
        self.mass = mass_amu * 1.66053907E-27 # kilograms
        self.charge_mass_ratio = self.charge / self.mass

    def initialize_velocity(self, v0_XYZ):
        self.vel0_XYZ = np.asarray(v0_XYZ)
        self.vel_XYZ = self.vel0_XYZ

    #def pushXYZ(self, t, p_XYZ, Bfield):
    def pushXYZ(self, t, state_XYZ, Bfield):
        #direction = 1
        B = np.zeros(3)
        E = np.zeros(3)
        pos = state_XYZ[:3]
        vel = state_XYZ[3:]

        B, dum_ = Bfield.interpField(pos)
        if Bfield.errField==True:
            B[0] += 0.0002
            B[1] += -0.0002

        dpos_dt = vel
        dvel_dt = self.charge_mass_ratio * (E + np.cross(vel, B))
        
        return np.concatenate((dpos_dt, dvel_dt))

    def storePath(self):
        pass
