import numpy as np
import logging
from math import degrees

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout':True})

from coordtrans import XYZ_to_RTP

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


    return LCFS_index





def Output_Poincare(iter, field_, Pdata, anlys_name, outputHandler=logging.getLogger(), saveData=True):
    """Function to output Poincare Plots and data set at a given Phi angle"""
    num_sets = len(Pdata)
    rminor = field_.a
    rmajor = field_.R0
    n, phi_ = iter    
    outputHandler.log.info('\tPHI: {}'.format(phi_*(180/np.pi)))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    maxLength = 0
    #num_sets = len(Pdata)
    print('len(Pdata)={}'.format(len(Pdata)))
    print('num_sets={}'.format(num_sets))
    for i in range(num_sets):
        maxLength = max(maxLength, len(Pdata[i][n]))

    # Looping over each initial condition
    scatter_points = np.full([num_sets, 2, maxLength], fill_value=np.nan)
    for i in range(num_sets):
        t_pts = Pdata[i][n]

        r_f = np.zeros(len(t_pts))
        th_f = np.zeros(len(t_pts))
        ph_f = np.zeros(len(t_pts))

        for j in range(len(t_pts)):
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
