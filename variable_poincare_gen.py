import numpy as np
from math import degrees
from functools import partial
import concurrent.futures as cf
from time import perf_counter

import phi_events_5 
from anlys_funcs import Output_Poincare
from ode import solvePoincare
from coordtrans import XYZ_to_RTP
from particle import Particle

def Gen_Poincare(field_, fieldlines, outputHandler, anlys_name, solvr, rtl_, atl_, saveData=True):
    outputHandler.createSubDir(anlys_name)

    ## SOLVER SETUP
    Nlines = Particle.particleCount

    ## EVENT LIST (avert your eyes)
    poincare_events = [ phi_events_5.inVV, 
                        phi_events_5.isphi5, 
                        phi_events_5.isphi10, 
                        phi_events_5.isphi15, 
                        phi_events_5.isphi20, 
                        phi_events_5.isphi25, 
                        phi_events_5.isphi30, 
                        phi_events_5.isphi35, 
                        phi_events_5.isphi40, 
                        phi_events_5.isphi45, 
                        phi_events_5.isphi50, 
                        phi_events_5.isphi55, 
                        phi_events_5.isphi60, 
                        phi_events_5.isphi65, 
                        phi_events_5.isphi70, 
                        phi_events_5.isphi75, 
                        phi_events_5.isphi80, 
                        phi_events_5.isphi85, 
                        phi_events_5.isphi90, 
                        phi_events_5.isphi95, 
                        phi_events_5.isphi100, 
                        phi_events_5.isphi105, 
                        phi_events_5.isphi110, 
                        phi_events_5.isphi115, 
                        phi_events_5.isphi120, 
                        phi_events_5.isphi125, 
                        phi_events_5.isphi130, 
                        phi_events_5.isphi135, 
                        phi_events_5.isphi140, 
                        phi_events_5.isphi145, 
                        phi_events_5.isphi150, 
                        phi_events_5.isphi155, 
                        phi_events_5.isphi160, 
                        phi_events_5.isphi165, 
                        phi_events_5.isphi170, 
                        phi_events_5.isphi175, 
                        phi_events_5.isphi180, 
                        phi_events_5.isphi185, 
                        phi_events_5.isphi190, 
                        phi_events_5.isphi195, 
                        phi_events_5.isphi200, 
                        phi_events_5.isphi205, 
                        phi_events_5.isphi210, 
                        phi_events_5.isphi215, 
                        phi_events_5.isphi220, 
                        phi_events_5.isphi225, 
                        phi_events_5.isphi230, 
                        phi_events_5.isphi235, 
                        phi_events_5.isphi240, 
                        phi_events_5.isphi245, 
                        phi_events_5.isphi250, 
                        phi_events_5.isphi255, 
                        phi_events_5.isphi260, 
                        phi_events_5.isphi265, 
                        phi_events_5.isphi270, 
                        phi_events_5.isphi275, 
                        phi_events_5.isphi280, 
                        phi_events_5.isphi285, 
                        phi_events_5.isphi290, 
                        phi_events_5.isphi295, 
                        phi_events_5.isphi300, 
                        phi_events_5.isphi305, 
                        phi_events_5.isphi310, 
                        phi_events_5.isphi315, 
                        phi_events_5.isphi320, 
                        phi_events_5.isphi325, 
                        phi_events_5.isphi330, 
                        phi_events_5.isphi335, 
                        phi_events_5.isphi340, 
                        phi_events_5.isphi345, 
                        phi_events_5.isphi350, 
                        phi_events_5.isphi355, 
                        phi_events_5.isphi360] 
 
 
    ## SOLVER
    length = fieldlines[0].maxLife
    solvePoincare_x = partial(solvePoincare, field=field_, solver=solvr, rtl= rtl_, atl=atl_, solver_events=poincare_events)

    ## PARALLELIZATION WITH CONCURRENT FUTURES 'MAP' OVER EACH PARTICLE
    outputHandler.log.info('Begin running {} Initial Conditions for max. {} spins...'.format(Nlines, int(length/(2*np.pi * field_.R0))))
    t_start = perf_counter()
    with cf.ProcessPoolExecutor(max_workers=16) as executor:
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
    phi_range = np.linspace( np.pi/36., 2*np.pi, 72) 

    # Looping over each phi angle
    iter_in = enumerate(phi_range)
    Output_Poincare_x = partial(Output_Poincare, field_=field_, Pdata=Poincare_output_, anlys_name=anlys_name, outputHandler=outputHandler, saveData=saveData)
    with cf.ProcessPoolExecutor(max_workers=72) as executor: 
        outs = executor.map(Output_Poincare_x, iter_in)
    
    # EXECUTE GENERATOR
    for out in outs:
        outputHandler.log.info(out)


    return pathLength_, Poincare_output_, wall_output_