"""Particle, field-line, and ion state objects."""

import numpy as np
import logging
#import phi_events


class Particle:
    particleCount = 0
    """
    Stores particle properties, data, and push method.

    Attributes:
        type (str): Type of particle ('fieldline' or 'ion').
        particleID (int): Unique identifier for the particle.
        charge (float): Particle charge.
        mass (float): Particle mass.
        charge_mass_ratio (float): Charge-to-mass ratio.
        maxLife (float): Maximum lifetime of the particle.
        pos0_XYZ (ndarray): Initial position (x, y, z).
        vel0_XYZ (ndarray): Initial velocity (vx, vy, vz).
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
    """Represents a field line in the magnetic field.

    Args:
        init_XYZ (array-like): Initial position coordinates of the field line.
        maxlength (float): Maximum length or lifespan of the field line.
        direction (float, optional): Direction of the field line's movement. Defaults to 1.0.

    Attributes:
        pos0_XYZ (np.ndarray): Initial position coordinates.
        maxLife (float): Maximum length or lifespan.
        direction (float): Direction of movement.
        vel0_XYZ (int): Initial velocity, default is 0.
        charge (float): Particle charge, default is 0.0.
        mass (float): Particle mass, default is 0.0.
    """
    def __init__(self, init_XYZ, maxlength, direction=1.0):
        super().__init__('fieldline')
        self.pos0_XYZ = init_XYZ
        self.maxLife = maxlength
        self.direction = direction
        self.vel0_XYZ = 0
        self.charge = 0.
        self.mass = 0.

    def pushXYZ(self, t, p_XYZ, field):
        """
        Computes the normalized change in position of a field line at its current location.

        The change is determined by the direction of the normalized field vector at the given position.

        Args:
            t (float): The current time (unused in this method).
            p_XYZ (array-like): The current position in 3D space as a sequence of coordinates (x, y, z).
            field (object): An object with an interpField method that returns the field vector at a given position.

        Returns:
            numpy.ndarray: The normalized field vector at the current position, scaled by the particle's direction.
        """
        B = field.interpField(p_XYZ[:3])[0]
        dY = B / np.sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2])

        return dY * self.direction

    def storePath(self):
        pass


class Ion(Particle):
    """Represents an ion in the magnetic field."""
    def __init__(self, init_XYZ, mass_amu, charge_z, maxlife=0.0):
        """
        Initializes a new ion particle with specified position, mass, charge, and lifetime.

        Args:
            init_XYZ (array-like): Initial position of the particle in Cartesian coordinates (x, y, z).
            mass_amu (float): Mass of the particle in atomic mass units (amu).
            charge_z (float): Charge of the particle in units of elementary charge (e).
            maxlife (float, optional): Maximum lifetime of the particle in seconds. Defaults to 0.0.

        Attributes:
            terminated (bool): Indicates if the particle has been terminated.
            pos0_XYZ (np.ndarray): Initial position as a NumPy array.
            vel0_XYZ (np.ndarray): Initial velocity, set to zero vector.
            maxLife (float): Maximum lifetime of the particle.
            charge (float): Particle charge in Coulombs.
            mass (float): Particle mass in kilograms.
            charge_mass_ratio (float): Ratio of charge to mass.
            pos_XYZ (list): List to store particle positions over time.
        """
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
        """Initializes the velocity of the ion particle."""
        self.vel0_XYZ = np.asarray(v0_XYZ)
        self.vel_XYZ = self.vel0_XYZ

    def initOutput(self, dt, tmax):
        #N = tmax // dt + 1
        #self.pos_XYZ = np.empty([int(N), 3]) #size output array
        #self.pos_XYZ = []
        pass

    def setPosition(self, index, value):
        """Sets the position of the particle at a specific index."""
        #self.pos_XYZ[index] = np.copy(value)
        self.pos_XYZ.append(value)

    def setVelocity(self, index, value):
        """Sets the velocity of the particle at a specific index."""
        #self.vel_XYZ[index] = np.copy(value)
        self.vel0_XYZ = np.copy(value)
