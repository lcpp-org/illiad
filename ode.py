import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from time import perf_counter
import logging

from coordtrans import *

## ==================================== ##-
## FIELD LINE SOLVER (FROM "Bfield.py") ##
## ==================================== ##
def blines(t, p_XYZ, field):
    B = np.zeros(3)
    direction = 1

    B = field.interpField(p_XYZ[:3])

    # hard-coded, hacky error field implementation
    #B[0] += 0.0002
    #B[1] += -0.0002

    dY = direction * B / np.linalg.norm(B)

    return dY


##===============##
## DEFINE SOLVER ##
##===============##
def solvePoincare(init_cond, maxLength, field, solver_events):
    log = logging.getLogger()

    init_cond_rtp = XYZ_to_RTP(init_cond, field.R0)
    log.info(f'Start IC: {init_cond_rtp}')

    span = (0.0, maxLength)

    t_startInd = perf_counter()
    fieldlines = solve_ivp(blines, span, init_cond, args = ([field]),
            dense_output=False,
            events = solver_events, 
            method='RK45', rtol=1e-9, atol=1e-9) #3e-4                    # 1
            #method='RK45', rtol=1e-10, atol=1e-10) #3e-4                  # 2

    t_stopInd = perf_counter()
    elapsed_timeInd = t_stopInd - t_startInd


    tmax = np.max(fieldlines.t)
    #pathLength = np.hstack((init_cond_rtp, tmax))

    if fieldlines.status == 0: #solver ran to max. time
        wallSpot = np.array([-1, 0., 0.]) # filter on negative r values later

        log.info(f'Success!: IC={init_cond}\tTook {elapsed_timeInd} sec.\tWall Event at t= {fieldlines.t_events[0]}')

    elif fieldlines.status == 1: #termination event
        log.info(f'Success!: IC={init_cond}\tTook {elapsed_timeInd} sec.\tWall Event at t= {fieldlines.t_events[0]}')
        wallSpot = XYZ_to_RTP(fieldlines.y_events[0][0], field.R0) # first point in wall event

        #log.info(f'Result: IC(rtp)={init_cond_rtp}, tmax={tmax}, wallPt(rtp)={wallSpot}')

    else: #solver failure
        log.critical(f'FAILURE!: IC:{init_cond}')

    data = fieldlines.y_events

    #log.info(f'Wall Event at t= {fieldlines.t_events[0]}')

    #return pathLength, data
    return tmax, data