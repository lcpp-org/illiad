import numpy as np
from math import degrees

import phi_events
from anlys_funcs import Output_Poincare
from ode import solvePoincare
from coordtrans import XYZ_to_RTP


def Gen_Poincare(field_, ICs_XYZ, length, outputHandler, saveData, anlys_name, solvr, rtl_, atl_):
    
    outputHandler.createSubDir(anlys_name)

    ## SOLVER SETUP
    Nlines = ICs_XYZ.shape[0]


    # avert your eyes
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
    from functools import partial
    import concurrent.futures as cf
    from time import perf_counter

    solvePoincare_x = partial(solvePoincare, maxLength=length, field=field_, solver=solvr, rtl= rtl_, atl=atl_, solver_events=poincare_events)

    outputHandler.log.info('Begin running {} Initial Conditions for max. {} spins...'.format(Nlines, int(length/(2*np.pi * field_.R0))))

    t_start = perf_counter()
    with cf.ProcessPoolExecutor(max_workers=24) as executor:
        solver_output = executor.map(solvePoincare_x, ICs_XYZ)

    t_stop = perf_counter()
    tot_elapsed_time = t_stop - t_start
    outputHandler.log.info('ALL SOLVERS FINISHED IN {} seconds\n###############\n\n'.format(tot_elapsed_time))

    # Parse output into lists
    pathLength_=[]
    Poincare_output_ = []
    wall_output_ = []

    for pLngth, out in solver_output:
        pathLength_ += [pLngth]
        Poincare_output_ += [out[1:]]

        if out[0].any():
            wall_output_ += [XYZ_to_RTP(out[0][0], field_.R0)]


    ## POST-SOLVER OUTPUT
    ####################
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'figure.autolayout':True})


    phi_range = np.linspace( np.pi/20., 2*np.pi, 40)

    # Looping over each phi angle
    outputHandler.log.info('PLOTTING AND OUTPUTTING PHI-ANGLE DATA:')

    num_sets = len(Poincare_output_)

    iter_in = enumerate(phi_range)
    Output_Poincare_x = partial(Output_Poincare, field_=field_, Pdata=Poincare_output_, anlys_name=anlys_name, outputHandler=outputHandler)
    with cf.ProcessPoolExecutor(max_workers=24) as executor:
        outs = executor.map(Output_Poincare_x, iter_in)
    
    for out in outs:
        outputHandler.log.info(out)

    return pathLength_, Poincare_output_, wall_output_