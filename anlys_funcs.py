import numpy as np
import logging

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout':True})

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