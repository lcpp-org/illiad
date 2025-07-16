
#if __name__ == '__main__':
import logging
from time import perf_counter
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout':True})

from functools import partial
import concurrent.futures as cf
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utility.coordtrans import XYZ_to_RTP, RTP_to_XYZ, axisShift
import phi_events


def identifyLCFS(LCFStype='inner', iconds=[0], t_maxs=[100], outputHandler=logging.getLogger(), num=11):
    """Function returns the index of the Last-Closed Flux Surface, with the option
        to input it directly, or determine it as the outermost confined surface, 
        or the confined surface innward from the first unconfined surface. """

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


def boris_wrapper(ion_list, b_hidra, ion_temp_eV, dt, tmax, dr_String):
    ion_ = ion_list[0]
    log = logging.getLogger()
    log.info('###########################################################################')
    log.info('RUNNING BORIS-BUNEMAN SOLVER WITH NEW ION SEED POINTS:')
    log.info('Initial Conditions:\t{} points'.format(len(ion_list)))
    log.info('Ions:\tmass={}[amu], q={}[Coulomb], ion temp.={}[eV]'#, initial velocity={:.0f} [m/s]'
             .format(ion_.mass, ion_.charge, ion_temp_eV))#, init_v_phi))
    log.info('Shells generated at delta-r(s) of {}mm from LCFS'.format(dr_String))
    log.info('SOLVER SETTINGS: tmax: {}sec., dt: {}sec., N={}pts'.format(tmax, dt, int(tmax/dt)))
    log.info('###########################################################################\n')


    ## PARALLELIZATION WITH CONCURRENT FUTURES 'MAP' OVER EACH PARTICLE
    boris_x = partial(boris_solver, dt=dt, tmax=tmax, Bfield=b_hidra)
    t_start = perf_counter()
    with cf.ProcessPoolExecutor(max_workers=6) as executor:
        boris_output_ = executor.map(boris_x, ion_list)#, chunksize=2)
    t_stop = perf_counter()
    tot_elapsed_time = t_stop - t_start
    log.info('ALL SOLVERS FINISHED IN {} seconds\n###############\n\n'.format(tot_elapsed_time))

    return boris_output_


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

def boris_solver(ion, dt, tmax, Bfield):
    """Function to take in a particle and field object and solves the particle path until termination even or tmax
       using a fixed-step Boris-Buneman Solver, based on (Birdsall, 4-3&4)"""
    log = logging.getLogger()
    log.info('Start IC: {}, {}'.format(ion.particleID, ion.pos0_XYZ))
    t_startInd = perf_counter()

    B = np.empty(3, dtype=np.float64)
    wallPt = np.zeros(3)
    N = int((tmax // dt) + 1)
    # Need particle parms: qdt2m, v0, p0
    qdt2m = ion.charge_mass_ratio * dt/2
    #print(f'{qdt2m=}')
    ion.setPosition(0, ion.pos0_XYZ)


    v_k = ion.vel0_XYZ

    # Need v_n-1/2 to start
    B, dum_ = Bfield.interpField(ion.pos_XYZ[0])

    tvec = qdt2m * B# tvec given by (4-4, Eq11)

    vprime = v_k + np.array([v_k[1]*tvec[2] - v_k[2]*tvec[1], 
                         v_k[2]*tvec[0] - v_k[0]*tvec[2],
                         v_k[0]*tvec[1] - v_k[1]*tvec[0]])   #np.cross(v_k, tvec)# vminus is incremented (4-4, Eq10), get vprime
    
    svec = 2*tvec / ( 1 + (np.linalg.norm(tvec)*np.linalg.norm(tvec)) )# svec given by (4-4, Eq13)

    vplus = v_k -  np.array([vprime[1]*svec[2] - vprime[2]*svec[1], 
                            vprime[2]*svec[0] - vprime[0]*svec[2],
                            vprime[0]*svec[1] - vprime[1]*svec[0]]) / 2 # from vminus, vprime, svec (4-4, Eq12), get vplus 

    v_k = vplus

    ## STEPPING THROUGH DTs
    for k in range(N-1):
        B, dum_ = Bfield.interpField(ion.pos_XYZ[k])

        tvec = qdt2m * B# tvec given by (4-4, Eq11)

        #vprime = v_k + np.cross(v_k, tvec)# vminus is incremented (4-4, Eq10), get vprime
        vprime = v_k + np.array([v_k[1]*tvec[2] - v_k[2]*tvec[1], 
                                 v_k[2]*tvec[0] - v_k[0]*tvec[2],
                                 v_k[0]*tvec[1] - v_k[1]*tvec[0]])   #np.cross(v_k, tvec)# vminus is incremented (4-4, Eq10), get vprime

        svec = 2*tvec / ( 1 + (np.linalg.norm(tvec)*np.linalg.norm(tvec)) )# svec given by (4-4, Eq13)

        #vplus = v_k + np.cross(vprime, svec)# from vminus, vprime, svec (4-4, Eq12), get vplus 
        vplus = v_k + np.array([vprime[1]*svec[2] - vprime[2]*svec[1], 
                                vprime[2]*svec[0] - vprime[0]*svec[2],
                                vprime[0]*svec[1] - vprime[1]*svec[0]]) # from vminus, vprime, svec (4-4, Eq12), get vplus 

        xplus = ion.pos_XYZ[k] + vplus*dt # from vplus, dt, get xplus
        v_k = vplus
        ion.setPosition(k+1, xplus)
        
        ion.maxLife = (k+1)*dt
        if phi_events.inVV(1, ion.pos_XYZ[k+1], Bfield) < 0.0:
            ion.terminated = True
            wallPt = ion.pos_XYZ[k+1]
            break

    t_stopInd = perf_counter()
    elapsed_timeInd = t_stopInd - t_startInd

    if ion.terminated:
        log.info('Success!: Particle {} of {} took {:.5f} sec.\tWall Event at t={:.5f}, k={}'
                 .format(ion.particleID, ion.particleCount, elapsed_timeInd, ion.maxLife, ion.maxLife//dt))
    else:
        log.info('Success!: Particle {} of {} took {:.5f} sec.\tWall Event at t='
                 .format(ion.particleID, ion.particleCount, elapsed_timeInd))
        
    return (wallPt, ion.pos_XYZ)



