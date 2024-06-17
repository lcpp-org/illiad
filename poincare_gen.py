import numpy as np
from math import degrees
from functools import partial
import concurrent.futures as cf
from time import perf_counter

import phi_events
from anlys_funcs import Output_Poincare
from ode import solvePoincare
from coordtrans import XYZ_to_RTP
from particle import Particle

def Gen_Poincare(field_, fieldlines, outputHandler, anlys_name, solvr, rtl_, atl_, saveData=True):
    outputHandler.createSubDir(anlys_name)

    ## SOLVER SETUP
    #Nlines = ICs_XYZ.shape[0]
    Nlines = Particle.particleCount

    """
    #jon = partial(phi_events.isAngle, phi_deg=9)
    #fred = partial(phi_events.isAngle, phi_deg=18.)

    #event_angles = np.linspace(9., 351, 39)
    #angle_events = [partial(phi_events.isAngle, phi_deg=angle) for angle in event_angles]

    #testphi9 = phi_events.make_event(9)
    #jon = testphi9(10., np.array([0., 0., 0.]), field_)
    #print(f'{jon=}')

    #poincare_events = [phi_events.inVV]
    #poincare_events.append(angle_events)
    #poincare_events.append(phi_events.isphi360)

    #poincare_events = [phi_events.inVV, jon, fred, phi_events.isphi360]
    """
    
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
    length = fieldlines[0].maxLife
    outputHandler.log.info('Begin running {} Initial Conditions for max. {} spins...'.format(Nlines, int(length/(2*np.pi * field_.R0))))
    #solvePoincare_x = partial(solvePoincare, maxLength=length, field=field_, solver=solvr, rtl= rtl_, atl=atl_, solver_events=poincare_events)
    solvePoincare_x = partial(solvePoincare, field=field_, solver=solvr, rtl= rtl_, atl=atl_, solver_events=poincare_events)

    t_start = perf_counter()
    with cf.ProcessPoolExecutor(max_workers=80) as executor:
        solver_output = executor.map(solvePoincare_x, fieldlines)
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

    # Looping over each phi angle
    outputHandler.log.info('PLOTTING AND OUTPUTTING PHI-ANGLE DATA:')

    #num_sets = len(Poincare_output_)
    phi_range = np.linspace( np.pi/20., 2*np.pi, 40)

    iter_in = enumerate(phi_range)
    Output_Poincare_x = partial(Output_Poincare, field_=field_, Pdata=Poincare_output_, anlys_name=anlys_name, outputHandler=outputHandler, saveData=saveData)
    with cf.ProcessPoolExecutor(max_workers=40) as executor:
        outs = executor.map(Output_Poincare_x, iter_in)
    
    for out in outs:
        outputHandler.log.info(out)

    return pathLength_, Poincare_output_, wall_output_