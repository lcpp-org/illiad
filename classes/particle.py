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
        self.charge: np.float32
        self.mass: np.float32
        self.charge_mass_ratio: np.float32
        self.maxLife: np.float32

        self.pos0_XYZ = np.zeros(3, dtype=np.float32)
        self.vel0_XYZ = np.zeros(3, dtype=np.float32)
        #self.pos_XYZ = np.zeros(3, dtype=np.float64)
        #self.vel_XYZ = np.zeros(3, dtype=np.float64)


class fieldLine(Particle):
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
        pass
        #self.pos_XYZ = []

    def setPosition(self, index, value):
        #self.pos_XYZ[index] = np.copy(value)
        self.pos_XYZ.append(value)

    def setVelocity(self, index, value):
        #self.vel_XYZ[index] = np.copy(value)
        self.vel0_XYZ = np.copy(value)

#    def pushXYZ(self, t, state_XYZ, Bfield):
#        B = np.zeros(3)
#        pos = state_XYZ[:3]
#        vel = state_XYZ[3:]
#
#        B, dum_ = Bfield.interpField(pos)
#
#        # hard-code for MOAR SPEDE?
#        #if Bfield.errField==True:
#        B[0] += 0.0002
#        B[1] += -0.0002
#
#        dpos_dt = vel
#        dvel_dt = self.charge_mass_ratio * np.cross(vel, B)
#        # concatenate expensive?
#        #new_state = np.concatenate((dpos_dt, dvel_dt))
#        new_state = np.array([dpos_dt[0], dpos_dt[1], dpos_dt[2], 
#                              dvel_dt[0], dvel_dt[1], dvel_dt[2]]) 
#        return new_state