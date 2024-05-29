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
    direction = 1
    B = np.zeros(3)
    B, dum_ = field.interpField(p_XYZ[:3])

    # hard-coded, hacky error field implementation
    B[0] += 0.0002
    B[1] += -0.0002

    dY = direction * B / np.linalg.norm(B)

    return dY


##===============##
## DEFINE SOLVER ##
##===============##
def solvePoincare(init_cond, maxLength, field, solver, rtl, atl, solver_events):
    log = logging.getLogger()

    init_cond_rtp = XYZ_to_RTP(init_cond, field.R0)
    log.info('Start IC: {}'.format(init_cond_rtp))

    t_startInd = perf_counter()
    fieldlines = solve_ivp(blines, (0.0, maxLength), init_cond, args = ([field]),
            dense_output=False,
            events = solver_events, 
            method=solver, rtol=rtl, atol=atl)
    t_stopInd = perf_counter()
    elapsed_timeInd = t_stopInd - t_startInd

    tmax = np.max(fieldlines.t)

    if fieldlines.status == 0: #solver ran to max. time
        #log.info(f'Success!: IC\tTook {:0.3f} sec.\tWall Event at t={}'.format(elapsed_timeInd, fieldlines.t_events[0]))
        log.info('Success!:\tSolver took {:.4f} sec.\tWall Event at t={}'.format(elapsed_timeInd, fieldlines.t_events[0]))
    elif fieldlines.status == 1: #termination event
        log.info('Success!:\tSolver took {:.4f} sec.\tWall Event at t={}'.format(elapsed_timeInd, fieldlines.t_events[0]))
    else: #solver failure
        log.critical('FAILURE!: IC:{}'.format(init_cond))

    data = fieldlines.y_events[:]

    return tmax, data