"""Utility functions for analyzing field lines and particle trajectories in plasma physics simulations.

This module is designed to work with the Poincare and Mesh classes.

Functions:
    identifyLCFS: Identifies the Last-Closed Flux Surface (LCFS).
    boris_solver2: Implements a Boris solver for collisionless particle motion in magnetic and electric fields.
"""
import logging
from time import perf_counter
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
#plt.rcParams.update({'figure.autolayout':True})

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def identifyLCFS(LCFStype='inner', iconds=[0], t_maxs=[100], outputHandler=logging.getLogger(), num=11):
    """Returns the index of the Last-Closed Flux Surface (LCFS).

    Args:
        LCFStype (str): Method to identify LCFS. One of 'inner', 'outer', or 'input'.
        iconds (list): List of initial conditions (e.g., minor radii).
        t_maxs (list): List of connection lengths for each initial condition.
        outputHandler: Handler for logging and figure saving.
        num (int): Index to use if LCFStype is 'input'.

    Returns:
        int: Index of the identified LCFS.

    Raises:
        ValueError: If LCFStype is not one of ['inner', 'outer', 'input'].
    """
    LCFStypes = ['inner', 'outer', 'input']
    if LCFStype not in LCFStypes:
        raise ValueError("Invalid LCFS type. Expected one of: %s" % LCFStypes)

    elif LCFStype == 'input':
            ## Manually select LCFS index
            LCFS_index = num 

    elif LCFStype == 'inner':
        # Assuming surfaces are ordered from 'out' to 'in':
        ## This returns the LCFS 'inside' ALL open flux surfaces
        maxTime = np.max(t_maxs)

        openSurface_ind = [i for i, t in enumerate(t_maxs) if t != maxTime] # Get indices of open flux surfaces
        if openSurface_ind:
            LCFS_index = max(openSurface_ind) + 1
        else:
            LCFS_index = 1

        plt.figure()
        plt.plot(iconds, t_maxs, '-o', c='k')
        plt.plot(iconds[LCFS_index], maxTime, '^', c='b')

        plt.title(r'Connection length vs. $r_{initial} (@{}\phi=324\degree)$')
        plt.yscale('log')
        plt.grid(True, which='both')
        plt.xlabel('Minor radius [m]')
        plt.xlabel('Connection length [m]')
        outputHandler.saveFig('connectLengths')
        plt.close()

    elif LCFStype == 'outer':
        ## This returns the most 'outer' LCFS
        maxTime = np.max(t_maxs)
        LCFS_index = t_maxs.index(maxTime)

        outputHandler.log.info('LCFS_index={}'.format(LCFS_index))
        
        plt.figure()
        plt.plot(iconds, t_maxs, '-o', c='k')
        plt.plot(iconds[LCFS_index], maxTime, '^', c='b')

        plt.title(r'Connection length vs. $r_{initial} (@{}\phi=324\degree)$')
        plt.yscale('log')
        plt.grid(True, which='both')
        plt.xlabel('Minor radius [m]')
        plt.xlabel('Connection length [m]')
        outputHandler.saveFig('connectLengths')
        plt.close()

    outputHandler.log.info('LCFS_index = {}'.format(LCFS_index))

    return LCFS_index

def boris_solver2(ions, dt, tmax, Bfield, Efield=None, trace_IDs=[]):
    """
    Function to take in a particle and field object and solves the particle path until termination event or tmax
    using a fixed-step Boris-Buneman Solver, based on (Birdsall, 4-3&4).

    Parameters:
        -ions (list): List of ion objects containing initial conditions and properties.
        -dt (float): Time step for the solver.
        -tmax (float): Maximum simulation time.
        -Bfield (object): Magnetic field object providing field interpolation methods.
        -Efield (object, optional): Electric field object providing field interpolation methods. Defaults to None.
        -track_ID (list, optional): List of particle IDs to track. Defaults to [10, 20].
    Returns:
        -wallPts (torch.Tensor): XYZ Positions where particles terminate (e.g., hit the wall), shape (Nparticles, 3).
        -wallVelocities (torch.Tensor): Velocities of particles at termination, shape (Nparticles, 3).
        -maxStep (torch.Tensor): Step index at which each particle terminated, shape (Nparticles,).
    """
    
    log = logging.getLogger()
    log.info('Start ICs: {}-{}'.format(ions[0].particleID, ions[-1].particleID))
    t_startInd = perf_counter()

    Nparticles = len(ions)
    Nsteps = int((tmax // dt) + 1)
    #trace_output = torch.zeros([len(trace_IDs), Nsteps+1, 3], dtype=torch.float64, device=device)
    trace_output = torch.zeros([Nsteps+1, len(trace_IDs), 3], dtype=torch.float64, device=device)

    with torch.no_grad():
        wallPts = torch.zeros([Nparticles, 3], dtype=torch.float64, device=device)
        wallVelocities = torch.zeros([Nparticles, 3], dtype=torch.float64, device=device)
        maxStep = torch.zeros(Nparticles, dtype=torch.int, device=device)

        tvec = torch.empty([Nparticles, 3], dtype=torch.float64, device=device)
        qdt2m = torch.tensor([ion.charge_mass_ratio * dt / 2 for ion in ions], dtype=torch.float64, device=device)
        v_k = torch.tensor(np.array([ion.vel0_XYZ for ion in ions]), dtype=torch.float64, device=device)

        [ion.setPosition(0, ion.pos0_XYZ) for ion in ions]
        pos_k = torch.tensor(np.array([ion.pos0_XYZ for ion in ions]), dtype=torch.float64, device=device)

        # NEED v_n-1/2 TO START
        if Efield:
            Evec = (Efield.interpField(pos_k) * qdt2m).T
        else:
            Evec = torch.zeros([Nparticles, 3], dtype=torch.float64, device=device)

        tvec = (Bfield.interpField(pos_k) * qdt2m).T
        tmag = torch.linalg.norm(tvec, axis=-1)

        vminus = v_k + Evec
        vprime = vminus + torch.linalg.cross(vminus, tvec)
        svec = 2 * tvec / (1 + (tmag * tmag)[:, None])
        vplus = vminus - torch.linalg.cross(vprime, svec) / 2
        v_k = vplus + Evec

        x2 = pos_k.T[0] * pos_k.T[0]
        y2 = pos_k.T[1] * pos_k.T[1]
        z2 = pos_k.T[2] * pos_k.T[2]
        r_k = torch.sqrt(x2 + y2 + z2 + Bfield.R0 * Bfield.R0
                          - 2 * Bfield.R0 * torch.sqrt(x2 + y2))

        running = torch.arange(0, Nparticles, 1, dtype=torch.int, device=device)
        Nrunning = Nparticles

        # ADD SELECTED PARTICLE TRACING
        trace_output[0] = pos_k[trace_IDs]

        log.info('START STEPPING...')
        logging.basicConfig(level=logging.INFO)
        with logging_redirect_tqdm(loggers=[log]):
            pbar = tqdm(range(1, Nsteps), ncols=100, mininterval=2.0)
            for k in pbar:
                if Efield:
                    Evec[running] = (Efield.interpField(pos_k[running]) * qdt2m[running]).T
                tvec[running] = (Bfield.interpField(pos_k[running]) * qdt2m[running]).T
                tmag[running] = torch.linalg.norm(tvec[running], axis=-1)

                vminus[running] = v_k[running] + Evec[running]
                vprime[running] = vminus[running] + torch.linalg.cross(vminus[running], tvec[running])
                svec[running] = 2 * tvec[running] / (1 + (tmag[running] * tmag[running])[:, None])
                vplus[running] = vminus[running] + torch.linalg.cross(vprime[running], svec[running])
                v_k[running] = vplus[running] + Evec[running]

                pos_k[running] = pos_k[running] + v_k[running] * dt

                # ADD SELECTED PARTICLE TRACING
                trace_output[k] = pos_k[trace_IDs]

                x2[running] = pos_k[running].T[0]**2
                y2[running] = pos_k[running].T[1]**2
                z2[running] = pos_k[running].T[2]**2
                r_k[running] = torch.sqrt(x2[running] + y2[running] + z2[running] + Bfield.R0 * Bfield.R0
                                           - 2 * Bfield.R0 * torch.sqrt(x2[running] + y2[running]))

                running = torch.where(r_k < Bfield.a)[0]

                maxStep[running] = k # +1?
                Nrunning = running.size(0)
                if Nrunning == 0:
                    log.info('All particles terminated at step {}'.format(k))
                    break

                pbar.set_postfix({'#Particles running': Nrunning}, refresh=False)

        terminated = torch.where(r_k >= Bfield.a)[0]
        wallPts[terminated] = pos_k[terminated]
        wallVelocities[terminated] = v_k[terminated]

    t_stopInd = perf_counter()
    elapsed_timeInd = t_stopInd - t_startInd
    min_, sec_ = divmod(elapsed_timeInd, 60)
    hr_, min_ = divmod(min_, 60)

    log.info(
        'ELAPSED TIME({} Particles): {:02.0f}H:{:02.0f}M:{:02.3f}S'.format(
            Nparticles, hr_, min_, sec_
        )
    )

    return wallPts, wallVelocities, maxStep, trace_output