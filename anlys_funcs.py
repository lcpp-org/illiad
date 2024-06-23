import numpy as np
import logging
from math import degrees
from time import perf_counter

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout':True})

from coordtrans import XYZ_to_RTP
#import phi_events


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
        LCFS_index = max(openSurface_ind) + 1

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


def Output_Poincare(iter, field_, Pdata, anlys_name, outputHandler=logging.getLogger(), saveData=True):
    """Function to output Poincare Plots and data set at a given Phi angle"""
    num_sets = len(Pdata)
    rminor = field_.a
    rmajor = field_.R0
    n, phi_ = iter

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    maxLength = 0
    for i in range(num_sets):
        maxLength = max(maxLength, len(Pdata[i][n]))

    # Looping over each initial condition
    scatter_points = np.full([num_sets, 2, maxLength], fill_value=np.nan)
    for i in range(num_sets):
        t_pts = Pdata[i][n]
        point_total = max(0, len(t_pts)-1)

        r_f = np.zeros(point_total)
        th_f = np.zeros(point_total)
        ph_f = np.zeros(point_total)

        for j in range(point_total):
            r_f[j], th_f[j], ph_f[j] = XYZ_to_RTP(t_pts[j][:3], rmajor)

        if saveData:
            scatter_points[i][0][:th_f.size] = th_f
            scatter_points[i][1][:r_f.size] = r_f
        else:
            pass

        plt.scatter(th_f, r_f, marker='.', s=1.5, c='k', linewidths=0.0)

    if saveData:
        f_output = scatter_points
        fname = anlys_name + '_{:03.0f}'.format(degrees(phi_))
        outputHandler.saveNumpyData(f_output, fname)
    else:
        pass 

    ax.set_rmax(rminor)
    ax.set_rticks(np.arange(0.0, 0.19, 0.02))
    ax.yaxis.set_tick_params(labelsize=5)
    ax.grid(linewidth = 0.25, linestyle=':', c='k')
    plt.title(r'Cross-section: $\phi$={:02.0f}$\degree$'.format(phi_*180/np.pi), loc='left')
    plot_name = anlys_name +'/'+ anlys_name + '_phi={:03.0f}.png'.format(phi_*180/np.pi)
    outputHandler.saveFig(plot_name)
    plt.close()

    return '\tPHI: {}'.format(phi_*(180/np.pi))


#def boris_solver(ion, dt, tmax, Bfield):
#    log = logging.getLogger()
#    log.info('Start IC: {}, {}'.format(ion.particleID, ion.pos0_XYZ))
#    t_startInd = perf_counter()
#    B = np.empty(3, dtype=np.float64)
#
#    wallPt = np.zeros(3)
#    N: np.int32v
#    N = int((tmax // dt) + 1)
#    # Need particle parms: qdt2m, v0, p0
#    qdt2m = ion.charge_mass_ratio * dt/2
#    ion.set_pos(0, ion.pos0_XYZ)
#    ion.set_vel(0, ion.vel0_XYZ)
#    #log.info('pos_XYZ[0]={}'.format(ion.pos_XYZ[0]))
#
#    ## Stepping through dt's until tmax (!!OR TERMINAL CONDITION!!):
#    for k in range(N-1):
#        B, dum_ = Bfield.interpField(ion.pos_XYZ[k])
#        
#        tvec = qdt2m * B# tvec given by (4-4, Eq11)
#        vprime = ion.vel_XYZ[k] + np.cross(ion.vel_XYZ[k], tvec)# vminus is incremented (4-4, Eq10), get vprime
#        svec = 2*tvec / ( 1 + (np.linalg.norm(tvec)*np.linalg.norm(tvec)) )# svec given by (4-4, Eq13)
#        #log.info('tvec:{}, vprime:{}, svec:{}'.format(tvec, vprime, svec))
#        #ion.vel_XYZ[k+1] = ion.vel_XYZ[k] + np.cross(vprime, svec)# from vminus, vprime, svec (4-4, Eq12), get vplus 
#        #ion.pos_XYZ[k+1] = np.copy(ion.pos_XYZ[k] + ion.vel_XYZ[k+1] * dt)# from vplus, dt, get xplus
#        vplus = ion.vel_XYZ[k] + np.cross(vprime, svec)# from vminus, vprime, svec (4-4, Eq12), get vplus 
#        xplus = ion.pos_XYZ[k] + vplus * dt# from vplus, dt, get xplus
#
#        ion.set_vel(k+1, vplus)
#        ion.set_pos(k+1, xplus)
#                    
#        ion.maxLife = (k+1)*dt
#        #log.info('ion.pos_XYZ[k+1]={}'.format(ion.pos_XYZ[k+1]))
#        if phi_events.inVV(1, ion.pos_XYZ[k+1], Bfield) < 0.0:
#            ion.terminated = True
#            wallPt = ion.pos_XYZ[k+1]
#            break
#
#    t_stopInd = perf_counter()
#    elapsed_timeInd = t_stopInd - t_startInd
#
#    #ion.pos_XYZ = ion.pos_XYZ[~np.isnan(ion.pos_XYZ)]
#    #ion.vel_XYZ = ion.vel_XYZ[~np.isnan(ion.vel_XYZ)]
#    log.info('pos_XYZ: {}'.format(ion.pos_XYZ))
#
#    if ion.terminated:
#        log.info('Success!: Particle {} of {} took {:.5f} sec.\tWall Event at t={:.5f}, k={}'
#                 .format(ion.particleID, ion.particleCount, elapsed_timeInd, ion.maxLife, ion.maxLife//dt))
#    else:
#        log.info('Success!: Particle {} of {} took {:.5f} sec.\tWall Event at t='
#                 .format(ion.particleID, ion.particleCount, elapsed_timeInd))
#    return wallPt
