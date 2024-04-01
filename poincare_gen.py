import numpy as np
import logging
from math import degrees

import phi_events
from ode import blines, solvePoincare
from coordtrans import XYZ_to_RTP

def Gen_Poincare(field_, ICs_XYZ, length, outputHandler, anlys_name):

    ## SOLVER SETUP
    Nlines = ICs_XYZ.shape[0]

    poincare_events = [ phi_events.inVV, 
                        phi_events.isphi9, 
                        phi_events.isphi18, 
                        phi_events.isphi27, 
                        phi_events.isphi36, 
                        phi_events.isphi45, 
                        phi_events.isphi54, 
                        phi_events.isphi63, 
                        phi_events.isphi72,
                        phi_events.isphi81, 
                        phi_events.isphi90, 
                        phi_events.isphi99, 
                        phi_events.isphi108, 
                        phi_events.isphi117, 
                        phi_events.isphi126, 
                        phi_events.isphi135, 
                        phi_events.isphi144,
                        phi_events.isphi153, 
                        phi_events.isphi162, 
                        phi_events.isphi171, 
                        phi_events.isphi180, 
                        phi_events.isphi189,
                        phi_events.isphi198,
                        phi_events.isphi207, 
                        phi_events.isphi216, 
                        phi_events.isphi225, 
                        phi_events.isphi234, 
                        phi_events.isphi243,
                        phi_events.isphi252, 
                        phi_events.isphi261, 
                        phi_events.isphi270, 
                        phi_events.isphi279, 
                        phi_events.isphi288,
                        phi_events.isphi297, 
                        phi_events.isphi306, 
                        phi_events.isphi315,
                        phi_events.isphi324, 
                        phi_events.isphi333,
                        phi_events.isphi342, 
                        phi_events.isphi351,
                        phi_events.isphi360
                    ]

    ## SOLVER
    ##########
    solver_output = [None]*Nlines

    from functools import partial
    import concurrent.futures
    from time import perf_counter

    solvePoincare_x = partial(solvePoincare, maxLength=length, field=field_, solver_events=poincare_events)

    t_start = perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        solver_output = executor.map(solvePoincare_x, ICs_XYZ)

    t_stop = perf_counter()
    tot_elapsed_time = t_stop - t_start
    outputHandler.log.info(f'ALL SOLVERS FINISHED IN {tot_elapsed_time} seconds\n###############\n\n')

    # Parse output into lists
    pathLength_=[]
    Poincare_output_ = []
    wall_output_ = []
    for pLngth, out in solver_output:
        pathLength_ += [pLngth]
        Poincare_output_ += [out[1:]]

        #print(f'{out[0]=}')
        if out[0].any():
            #print('reg output: ' +str( XYZ_to_RTP(out[0][0], field_.R0)) )
            wall_output_ += [XYZ_to_RTP(out[0][0], field_.R0)]
        #else:
        #    print('sub output: ' +str( np.array([-1., 0., 0.])) )
        #    wall_output_ += [ np.array([-1., 0., 0.])]
    #print(f'{pathLength_=}')


    ## POST-SOLVER OUTPUT
    ####################
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'figure.autolayout':True})

    phi_range = np.linspace( np.pi/20., 2*np.pi, 40)
    # Looping over each phi angle
    outputHandler.log.info('PLOTTING AND OUTPUTTING PHI-ANGLE DATA:')
    for n, phi_plot in enumerate(phi_range):
        outputHandler.log.info(f'\tPHI: {phi_plot*(180/np.pi)}')

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)

        # Looping over each initial condition
        for i in range(len(Poincare_output_)):
            t_pts = Poincare_output_[i][n] #skip wall event

            r_f = np.zeros(len(t_pts))
            th_f = np.zeros(len(t_pts))
            ph_f = np.zeros(len(t_pts))

            for j in range(len(t_pts)):
                r_f[j], th_f[j], ph_f[j] = XYZ_to_RTP(t_pts[j][:3], field_.R0)

            f_output = np.array([th_f, r_f])
            fname = f'Poincare_output_{degrees(phi_plot):03.0f}_{i}'
            outputHandler.saveNumpyData(f_output, fname)

            plt.scatter(th_f, r_f, marker='.', s=1.5, linewidths=0.0)

        ax.set_rmax(field_.a)
        plt.title(r'Poincare Plot, $\phi$={:02.0f}$\degree$'.format(phi_plot*180/np.pi))

        plot_name = anlys_name+'_phi={:03.0f}.png'.format(phi_plot*180/np.pi)
        outputHandler.saveFig(plot_name)
        plt.close()
    #plt.close('all')

    return pathLength_, Poincare_output_, wall_output_