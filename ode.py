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

    B, dum_ = field.interpField(p_XYZ[:3])

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

    t_startInd = perf_counter()
    fieldlines = solve_ivp(blines, (0.0, maxLength), init_cond, args = ([field]),
            dense_output=False,
            events = solver_events, 
            #method='RK45', rtol=1e-7, atol=1e-14)
            method='LSODA', rtol=1e-7, atol=1e-14)#, first_step=1e-5)
            #method='DOP853', rtol=1e-6, atol=1e-14)
    t_stopInd = perf_counter()
    elapsed_timeInd = t_stopInd - t_startInd

    tmax = np.max(fieldlines.t)

    if fieldlines.status == 0: #solver ran to max. time
        log.info(f'Success!: IC\tTook {elapsed_timeInd:0.3f} sec.\tWall Event at t={fieldlines.t_events[0]}')
    elif fieldlines.status == 1: #termination event
        log.info(f'Success!: IC\tTook {elapsed_timeInd:0.3f} sec.\tWall Event at t={fieldlines.t_events[0]}')
    else: #solver failure
        log.critical(f'FAILURE!: IC:{init_cond}')

    data = fieldlines.y_events[:]

    return tmax, data