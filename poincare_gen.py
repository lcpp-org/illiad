import numpy as np
from math import degrees
from functools import partial
import concurrent.futures as cf
from time import perf_counter

from phi_events import eventsAndRange
from anlys_funcs import Output_Poincare
from ode import solvePoincare
from coordtrans import XYZ_to_RTP
from particle import Particle

def Gen_Poincare(field_, fieldlines, outputHandler, anlys_name, solvr, rtl_, atl_, saveData=True):
    outputHandler.createSubDir(anlys_name)

    poincare_events, phi_range = eventsAndRange()
    
    ## SOLVER SETUP
    Nlines = Particle.particleCount

    ## SOLVER
    length = fieldlines[0].maxLife
    solvePoincare_x = partial(solvePoincare, field=field_, solver=solvr, rtl= rtl_, atl=atl_, solver_events=poincare_events)

    ## PARALLELIZATION WITH CONCURRENT FUTURES 'MAP' OVER EACH PARTICLE
    outputHandler.log.info('Begin running {} Initial Conditions for max. {} spins...'.format(Nlines, int(length/(2*np.pi * field_.R0))))
    t_start = perf_counter()
    with cf.ProcessPoolExecutor(max_workers=40) as executor:
        solver_output = executor.map(solvePoincare_x, fieldlines)
    t_stop = perf_counter()
    tot_elapsed_time = t_stop - t_start
    outputHandler.log.info('ALL SOLVERS FINISHED IN {} seconds\n###############\n\n'.format(tot_elapsed_time))

    ## PARSE OUTPUT INTO LISTS
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

    outputHandler.log.info('PLOTTING AND OUTPUTTING PHI-ANGLE DATA:')
    
    # Looping over each phi angle
    iter_in = enumerate(phi_range)
    Output_Poincare_x = partial(Output_Poincare, field_=field_, Pdata=Poincare_output_, anlys_name=anlys_name, outputHandler=outputHandler, saveData=saveData)
    with cf.ProcessPoolExecutor(max_workers=40) as executor:
        outs = executor.map(Output_Poincare_x, iter_in)
    
    # EXECUTE GENERATOR
    for out in outs:
        outputHandler.log.info(out)


    return pathLength_, Poincare_output_, wall_output_